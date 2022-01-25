import logging
import pickle
import torch
import numpy as np
from ..ps import psConnection
import math

LOGGER = logging.getLogger(__name__)

TYPE_CONVERSION = {
    'torch.cuda.FloatTensor': np.float32,
    'torch.FloatTensor': np.float32
}


def _serialize_obj_to_bytes(params):
    return pickle.dumps(params)


def _deserialize_bytes_to_obj(blob):
    return pickle.loads(blob)


MSG_CHUNK_SIZE = 2**10


def get_chunks_from_model(net, is_grad=False):
    chunks = []
    if is_grad:
        # TODO: compress param.grad.data
        get_buf = lambda param: param.grad.data.cpu().numpy().tobytes()
    else:
        get_buf = lambda param: param.data.cpu().numpy().tobytes()

    for name, param in net.named_parameters():
        buf = get_buf(param)
        j = 0
        while len(buf) > j:
            payload = buf[j:j + MSG_CHUNK_SIZE]
            j += len(payload)
            chunks.append(payload)

    return chunks


def get_param_chunks(net):
    return get_chunks_from_model(net, is_grad=False)


def get_grad_chunks(net):
    return get_chunks_from_model(net, is_grad=True)


def set_model_with_chunks(net, chunks, is_grad=True):
    old_chunks = get_chunks_from_model(net, is_grad=is_grad)
    offset = 0
    for name, param in net.named_parameters():
        t = param.data
        buf_len = len(t.numpy().tobytes())
        chunk_num = math.ceil(buf_len / MSG_CHUNK_SIZE)

        for i in range(offset, offset + chunk_num):
            if chunks[i] is None:
                chunks[i] = old_chunks[i]

        buf = b''.join(chunks[offset:offset + chunk_num])
        if len(buf) == buf_len:
            # assert len(buf) == buf_len
            n = np.frombuffer(buf, dtype=np.float32).copy()
            new_tensor = torch.from_numpy(n).view_as(t)
            if t.is_cuda:
                new_tensor = new_tensor.cuda()

            if is_grad:
                param.grad.data = new_tensor
            else:
                param.data = new_tensor
            offset += chunk_num


def set_grads_with_chunks(net, chunks):
    return set_model_with_chunks(net, chunks, is_grad=True)


def set_param_with_chunks(net, chunks):
    return set_model_with_chunks(net, chunks, is_grad=False)


class psPyTorchAdapter:
    def __init__(self, net, name, config_file, alr, fb):
        self._net = net
        self._conn = psConnection(name, config_file, alr, fb)
        self.name = name
        self.count = 0
        if self.name == 'ps':
            chunks = get_param_chunks(self._net)
            self.aggregated_param_chunks_nparray = []
            for chunk in chunks:
                n = np.frombuffer(chunk, dtype=np.float32).copy()
                self.aggregated_param_chunks_nparray.append(n)

    # ----------------functions for workers---------------------
    def grad_send(self, step):
        """
        send model gradients to the peer
        """
        chunks = get_grad_chunks(self._net)

        self._conn.data_send(chunks, step)

    def send_zero_grad(self, step):
        """
        send zero gradients to the peer
        """
        chunks = get_param_chunks(self._net)

        zero_chunks = []
        for chunk in chunks:
            n = np.frombuffer(chunk, dtype=np.float32)
            z = np.zeros_like(n)
            zero_chunks.append(z.tobytes())

        self._conn.data_send(zero_chunks, step)

    def param_send(self, step):
        """
        send model parameters to the peer
        """
        chunks = get_param_chunks(self._net)
        blob = _serialize_obj_to_bytes(chunks)
        self._conn.data_send(blob, step)

    def update_wait(self, is_grad=False):
        """Waits for the new model parameters from ps
        """
        blob, state = self._conn.update_wait()
        if blob is None:
            return
        set_param_with_chunks(self._net, blob)

    # -------------------functions for ps------------------------
    def ps_asynUpdate(self, optimizer=None):
        """Waits for the cluster update to finish.

        Waiting for the fetch parameters request to complete (blocking)
        """
        self._conn.enable_fetching()
        blob, state = self._conn.update_wait()
        if blob is None:
            return

        chunks = blob
        if not hasattr(self, 'aggregated_chunks'):
            self.aggregated_chunks = chunks
        else:
            new_aggregated_chunks = []
            for c1, c2 in zip(self.aggregated_chunks, chunks):
                n1 = np.frombuffer(c1, dtype=np.float32)
                n2 = np.frombuffer(c2, dtype=np.float32)
                ret = n1 * 0.5 + n2 * 0.5
                new_aggregated_chunks.append(ret.tobytes())
            self.aggregated_chunks = new_aggregated_chunks

        # send to a worker
        blob = _serialize_obj_to_bytes(self.aggregated_chunks)
        peer_name = state['name']
        self._conn.ps_send(blob, peer_name)

    def ps_synUpdate(self, optimizer=None):
        old_chunks = get_chunks_from_model(self._net, is_grad=True)
        self._conn.enable_fetching()
        grads_list = self._conn.ps_receiveAll()
        print('------------------')

        r = 1. / len(grads_list)

        # decode received model gradients

        optimizer.zero_grad()

        offset = 0
        for name, param in self._net.named_parameters():
            t = param.data
            buf_len = len(t.numpy().tobytes())
            chunk_num = math.ceil(buf_len / MSG_CHUNK_SIZE)
            for chunks in grads_list:
                for i in range(offset, offset + chunk_num):
                    if chunks[i] is None:
                        chunks[i] = old_chunks[i]
                buf = b''.join(chunks[offset:offset + chunk_num])
                assert len(buf) == buf_len
                n = np.frombuffer(buf, dtype=np.float32).copy()
                new_tensor = torch.from_numpy(n).view_as(t)
                if t.is_cuda:
                    new_tensor = new_tensor.cuda()
                param.grad.data += new_tensor * r

            offset += chunk_num

        optimizer.step()
        self._conn.ps_sendToAll(get_param_chunks(self._net))
