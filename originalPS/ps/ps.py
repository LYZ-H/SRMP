"""ps-worker connection."""
import logging
import random
import yaml
import socket
from .conn import RxThread, TxThread
from .conn_one import RxThread as RxThread_one, TxThread as TxThread_one
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

    def get_transmit_type(self):
        return self.config['broadcast']

    def get_protocal_type(self):
        return self.config['udp']

    def get_bandwidth(self):
        return self.config['bandwidth']


class psConnection:
    def __init__(self, name, config_file, alr, feedback):
        self.name = name
        # The clock is used to keep track of the model's age in terms of
        # training samples trained so far (increase by 1 in update_send())
        self.clock = 0
        self.config = psConfiguration(config_file)
        self.nodes = self.config.get_nodes()
        self.fetching = False
        self.fetch_probability = self.config.get_fetch_probability()
        self.broadcast = self.config.get_transmit_type()
        self.is_udp = self.config.get_protocal_type()
        self.bandwidth = 1024 * self.config.get_bandwidth()

        self.acceptable_loss_rate = alr
        self.feedback = feedback

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

        else:
            for node in self.nodes:
                node = Struct(**node)
                if node.name == name:
                    self.me = node
                elif node.name == 'ps':
                    self.peers += [node]

                    # Create the client/server threads
            timeout_ms = self.config.get_timeoutms()

        if self.me.name == 'ps':
            sock_one = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock_one.bind(('', self.me.port))

            self.rx_one = RxThread_one(self.me.port,
                                       channel_name=self.me.name,
                                       bw_in_kbps=self.bandwidth,
                                       local_host=self.me.host,
                                       socket_one=sock_one)
            self.rx_one.start()

            self.tx = TxThread('tx',
                               '239.0.0.1',
                               5007,
                               channel_name=self.me.name,
                               bw_in_kbps=self.bandwidth,
                               type=self.broadcast,
                               acceptable_loss_rate=self.acceptable_loss_rate,
                               local_port=self.me.port,
                               is_udp=self.is_udp,
                               socket_one=sock_one)
            for peer in self.peers:
                self.tx.add_peer(self.peers.index(peer) + 1, peer.name)

        else:
            sock_one = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rx = RxThread('239.0.0.1',
                               5007,
                               channel_name=self.me.name,
                               bw_in_kbps=self.bandwidth,
                               socket_one=sock_one,
                               acceptable_loss_rate=self.acceptable_loss_rate,
                               feedback=self.feedback)

            self.rx.start()

            self.tx_one = {}  # a tx thread for each worker
            for peer in self.peers:
                self.tx_one[peer.name] = TxThread_one(
                    'tx',
                    channel_name=self.me.name,
                    bw_in_kbps=self.bandwidth,
                    type=self.broadcast,
                    acceptable_loss_rate=self.acceptable_loss_rate,
                    is_udp=self.is_udp,
                    socket_one=sock_one)

                self.tx_one[peer.name].add_peer(
                    'r{0}'.format(self.peers.index(peer) + 1),
                    host=peer.host,
                    port=peer.port)

    def add_peer(self, id, name):

        self.tx.add_peer(id, name)

    def add_peer_one(self, name, host, port):
        self.tx_one.add_peer(name, host, port)

    def remove_peer(self, name):
        self.tx.remove_peer(name)

    def _bernouli_trial(self, probability):
        return random.random() < probability

    def data_send(self, data, step):
        """Initiate an update to the cluster.

        Performs 2 things:
        1. Updates the local server with the latest parameters, so other peers could fetch them
        2. Initiate a fetch parameters request to a random peer.
        """
        # Serve the new parameters
        state = {'clock': self.clock, 'step': step, 'name': self.name}
        if self._bernouli_trial(self.fetch_probability):
            self.fetching = True
            for peer in self.peers:
                self.tx_one[peer.name].send_chunks(clock=self.clock,
                                                   chunks=data)
        else:
            self.fetching = False

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
        """Waits for the model paramters or gradients from ps, depending on the ps
        """
        if not self.fetching:
            return None, None

        peer_state = self.rx.fetch_wait()

        peer_grads = peer_state['chunks']
        peer_clock = peer_state['clock']
        peer_name = 'ps'

        self.fetching = False
        # There may be no peers listening
        if peer_grads is None:
            return None, None
        # Increase the clock value
        self.clock += 1

        return peer_grads, peer_state

    def ps_sendToAll(self, parameters):
        """ps sends updated paramters to the pionted worker
        """
        # Serve the new parameters
        state = {'clock': self.clock, 'name': self.name}

        self.tx.send_chunks(clock=self.clock,
                            chunks=parameters)

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
            try:
                peer_state = self.rx_one.fetch_wait()
                peer_grads = peer_state['chunks']
                peer_clock = peer_state['clock']
                peer_name = peer_state['name']

                # There may be no peers listening
                if peer_grads is None:
                    continue

                LOGGER.debug(
                    "ps_receive(): (clock=%s, peer_clock=%s, peer_name=%s)",
                    self.clock, peer_clock, peer_name)
                if peer_name in barrier:

                    rev_list.append(peer_grads)
                    del barrier[peer_name]

                # check whether all the peers' models are received at this clock
                if len(barrier) == 0:
                    break
            except Exception as e:
                print(e)

                # Increase the clock value
        self.clock += 1
        return rev_list
