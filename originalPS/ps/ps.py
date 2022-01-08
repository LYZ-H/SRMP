"""ps-worker connection."""
import logging
import random
import sys
import time

import yaml
from .conn import RxThread, TxThread
from .interpolation import ConstantInterpolation, \
    ClockWeightedInterpolation, LossInterpolation

INTERPOLATION_METHODS = {
    'constant': ConstantInterpolation,
    'clock': ClockWeightedInterpolation,
    'loss': LossInterpolation,
}

LOGGER = logging.getLogger(__name__)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        return 'Struct: ' + repr(self.__dict__)


class psConfiguration:
    def __init__(self, config_file):
        self.yaml = yaml.safe_load(open(config_file, 'rt'))
        self.config = {}
        for c in self.yaml:
            k = list(c.keys())[0]
            self.config[k] = c[k]

    def get_nodes(self):
        return self.config['nodes']

    def get_interpolation(self):
        interpolation = self.config['interpolation']
        return (interpolation, self.config[interpolation])

    def get_timeoutms(self):
        return self.config['timeout_ms']

    def get_fetch_probability(self):
        return self.config['fetch_probability']

    def get_divergence_threshold(self):
        return self.config['divergence_threshold']

    def get_is_para(self):
        return self.config['is_para']

    def get_bw(self):
        return self.config['bw']


class psConnection:
    def __init__(self, name, config_file):
        self.name = name
        # The clock is used to keep track of the model's age in terms of
        # training samples trained so far (increase by 1 in update_send())
        self.clock = 0
        self.config = psConfiguration(config_file)
        self.nodes = self.config.get_nodes()
        self.fetching = False
        self.fetch_probability = self.config.get_fetch_probability()
        self.para = self.config.get_is_para()
        self.bw = self.config.get_bw()

        # Initialize the list of peers
        self.peers = []
        if self.name == 'ps':
            for node in self.nodes:
                node = Struct(**node)
                if node.name == name:
                    self.me = node
                elif node.name != 'ps':
                    self.peers += [node]
                    # Create the client/server threads
            timeout_ms = self.config.get_timeoutms()
            self.rx = RxThread(self.me.host, self.me.port, timeout_ms)
            self.rx.start()
            self.tx = {}  # a tx thread for each worker
            for peer in self.peers:
                self.tx[peer.name] = TxThread(timeout_ms, self.me.name, self.para, self.bw)
                self.tx[peer.name].add_peer(peer.name, peer.host, peer.port)
                self.tx[peer.name].start()

        else:
            for node in self.nodes:
                node = Struct(**node)
                if node.name == name:
                    self.me = node
                elif node.name == 'ps':
                    self.peers += [node]

                    # Create the client/server threads
            timeout_ms = self.config.get_timeoutms()
            self.rx = RxThread(self.me.host, self.me.port, timeout_ms)
            self.tx = TxThread(timeout_ms, self.me.name, False, self.bw)
            # Add all the peers
            for peer in self.peers:
                self.add_peer(peer.name, peer.host, peer.port)

            # Start the threads
            self.rx.start()
            self.tx.start()

    def add_peer(self, name, host, port):
        self.tx.add_peer(name, host, port)

    def remove_peer(self, name):
        self.tx.remove_peer(name)

    def _bernouli_trial(self, probability):
        return random.random() < probability

    def grad_send(self, grad, step):
        """Initiate an update to the cluster.

        Performs 2 things:
        1. Updates the local server with the latest parameters, so other peers could fetch them
        2. Initiate a fetch parameters request to a random peer.
        """
        # Serve the new parameters
        state = {'clock': self.clock, 'step': step, 'name': self.name}
        self.tx.set_current_state(state, grad)

        if self._bernouli_trial(self.fetch_probability):
            LOGGER.debug("update_send(): starting fetch parameters request")
            self.fetching = True
            self.tx.fetch_send()
        else:
            self.fetching = False

    def ps_send_syn(self, peer, data):
        state = {'clock': self.clock, 'name': self.name}
        self.tx[peer.name].set_current_state(state, data)
        self.tx[peer.name].fetch_send()

    def ps_send(self, parameters, peer_name):
        """ps sends updated paramters to the pionted worker
        """
        # Serve the new parameters
        state = {'clock': self.clock, 'name': self.name}
        self.tx[peer_name].set_current_state(state, parameters)
        self.tx[peer_name].fetch_send()

    def enable_fetching(self):
        self.fetching = True

    def disable_fetching(self):
        self.fetching = False

    def update_wait(self):
        """Waits for the paramters from ps;
           Waits for the gradients from workers
        """
        if not self.fetching:
            return None, None
        peer_state, peer_payload = self.rx.fetch_wait()
        self.fetching = False
        # There may be no peers listening
        if peer_payload is None:
            return None, None

        # Increase the clock value
        self.clock += 1
        peer_clock = peer_state['clock']
        peer_name = peer_state['name']
        # print("update_wait(): (clock={0}, peer_clock={1}, peer_name={2})".format(self.clock, peer_clock, peer_name))
        LOGGER.debug("update_wait(): (clock=%s, peer_clock=%s, peer_name=%s)", self.clock, peer_clock, peer_name)
        return peer_payload, peer_state

    def ps_sendToAll(self, parameters):
        """ps sends updated paramters to the pionted worker
        """
        # Serve the new parameters
        state = {'clock': self.clock, 'name': self.name}
        for peer in self.peers:
            self.tx[peer.name].set_current_state(state, parameters)
            self.tx[peer.name].fetch_send()
            # time.sleep(sys.getsizeof(parameters) / (2 * 1024 * 1024))

    def ps_receive(self):
        peer_state, peer_grads = self.rx.fetch_wait()
        peer_clock = peer_state['clock']
        peer_name = peer_state['name']
        return peer_grads

    def ps_receiveAll(self):
        """
        Waits for the models from workers (blocking)
        """
        if not self.fetching:
            return None, None

        rev_list = []
        # initialize barrier
        barrier = {}
        for peer in self.peers:
            barrier[peer.name] = 0

        # check received models at last clock
        while True:
            peer_state, peer_grads = self.rx.fetch_wait()
            # There may be no peers listening
            if peer_grads is None:
                continue

            # time.sleep(sys.getsizeof(peer_grads) / (32 * 1024 * 1024))

            peer_clock = peer_state['clock']
            peer_name = peer_state['name']
            LOGGER.debug("ps_receive(): (clock=%s, peer_clock=%s, peer_name=%s)", self.clock, peer_clock, peer_name)
            # print("ps_receive(): (clock=%{0}, peer_clock=%{1}, peer_name=%{2})".format(self.clock, peer_clock, peer_name))
            # check whether the peer is fake peer
            if peer_name in barrier:
                rev_list.append(peer_grads)
                del barrier[peer_name]

            # check whether all the peers' models are received at this clock
            if len(barrier) == 0:
                break

                # Increase the clock value
        self.clock += 1
        return rev_list
