import logging
import pickle
import time

import torch
import numpy as np
from ..ps import psConnection, psConfiguration, Struct

LOGGER = logging.getLogger(__name__)

TYPE_CONVERSION = {
    'torch.cuda.FloatTensor': np.float32,
    'torch.FloatTensor': np.float32
}


def _tensor_to_buffer(t):
    return bytes(t.cpu().numpy())


def _tensor_from_buffer_like(buf, t):
    n = np.frombuffer(buf, dtype=TYPE_CONVERSION[t.type()]).copy()
    result = torch.from_numpy(n).view(t.size())
    if t.is_cuda:
        result = result.cuda()
    return result


def _serialize_bytes_dict(params):
    return pickle.dumps(params)


def _deserialize_bytes_dict(blob):
    return pickle.loads(blob)


class psPyTorchAdapter:
    def __init__(self, net, name, config_file):
        self._net = net
        self._conn = psConnection(name, config_file)

    # ----------------functions for workers---------------------

    # def check_send(self):
    #     return self._conn.check_send()

    def grad_send(self, step):
        """Initiate an update to the ps.

        Performs 2 things:
        1. Updates the local server with the latest parameters, so other peers could fetch them
        2. Initiate a fetch parameters request to a random peer.
        """
        grad = {}
        for name, param in self._net.named_parameters():
            grad[name] = _tensor_to_buffer(param.grad.data)
        blob = _serialize_bytes_dict(grad)
        self._conn.grad_send(blob, step)

    def update_wait(self, bid):
        """Waits for the parameters from ps
        """
        self._conn.enable_fetching()
        blob, state = self._conn.update_wait()
        if blob is None:
            return
        try:
            bytes.decode(blob) == 'allow'
        except UnicodeDecodeError:
            other_params = _deserialize_bytes_dict(blob)
            for name, param in self._net.named_parameters():
                t = _tensor_from_buffer_like(other_params[name], param.data)
                param.data = t
            data = str.encode('re')
            self._conn.grad_send(data, None)
        else:
            self.grad_send(bid)

    # -------------------functions for ps------------------------
    def ps_send(self, peer_name):
        '''
        ps send updated parameters to a given worker
        '''
        params = {}
        for name, param in self._net.named_parameters():
            params[name] = _tensor_to_buffer(param.data)
        blob = _serialize_bytes_dict(params)
        self._conn.ps_send(blob, peer_name)

    def ps_asynUpdate(self, optimizer):
        """Waits for the cluster update to finish.

        Waiting for the fetch parameters request to complete (blocking)
        """
        self._conn.enable_fetching()
        blob, state = self._conn.update_wait()
        if blob is None:
            return
        optimizer.zero_grad()
        other_grad = _deserialize_bytes_dict(blob)
        for name, param in self._net.named_parameters():
            t = _tensor_from_buffer_like(other_grad[name], param.data)
            param.grad.data = t
        optimizer.step()
        peer_name = state['name']
        self.ps_send(peer_name)

    def ps_sendToAll(self):
        '''
        ps send updated parameters to a given worker
        '''
        params = {}
        for name, param in self._net.named_parameters():
            params[name] = _tensor_to_buffer(param.data)
        blob = _serialize_bytes_dict(params)
        self._conn.ps_sendToAll(blob)

    def ps_msynUpdate(self, optimizer):
        data = str.encode('allow')
        self.nodes = self._conn.config.get_nodes()
        self.peers = []
        other_grads = []
        for node in self.nodes:
            node = Struct(**node)
            if node.name != 'ps':
                self.peers += [node]
        for peer in self.peers:
            self._conn.enable_fetching()
            self._conn.ps_send(data, peer.name)
            while True:
                blob, state = self._conn.update_wait()
                if blob is not None:
                    break
            other_grad = _deserialize_bytes_dict(blob)
            other_grads.append(other_grad)
        optimizer.zero_grad()

        for name, param in self._net.named_parameters():
            # average the same keys of all the models
            """
            for i in range(peer_num):
                # i-th model
                other_grad = other_grads[i]
                if i == 0:
                   t = _tensor_from_buffer_like(other_grad[name], param.data)
                t += _tensor_from_buffer_like(other_grad[name], param.data)
            """
            t = sum(
                map(
                    lambda other_grad: _tensor_from_buffer_like(other_grad[name], param.data),
                    other_grads
                )
            )
            param.grad.data = t / len(other_grads)
        optimizer.step()
        self.ps_sendToAll()
        count = 0
        while True:
            self._conn.enable_fetching()
            blob, state = self._conn.update_wait()
            if blob is not None:
                try:
                    bytes.decode(blob) == 're'
                except UnicodeDecodeError:
                    print('err')
                else:
                    count += 1
            if count >= len(self.peers):
                break

        # time.sleep(10)

    def ps_synUpdate(self, optimizer):
        self._conn.enable_fetching()

        rev_list = self._conn.ps_receiveAll()
        other_grads = []
        # decode received models
        for blob in rev_list:
            other_grad = _deserialize_bytes_dict(blob)
            other_grads.append(other_grad)
        # average models
        peer_num = len(other_grads)
        factor = 1 / float(peer_num)
        optimizer.zero_grad()
        for name, param in self._net.named_parameters():
            # average the same keys of all the models
            """
            for i in range(peer_num):
                # i-th model
                other_grad = other_grads[i]
                if i == 0:
                   t = _tensor_from_buffer_like(other_grad[name], param.data)
                t += _tensor_from_buffer_like(other_grad[name], param.data)
            """
            t = sum(
                map(
                    lambda other_grad: _tensor_from_buffer_like(other_grad[name], param.data),
                    other_grads
                )
            )
            param.grad.data = factor * t
        optimizer.step()
        # send back to workers
        self.ps_sendToAll()
