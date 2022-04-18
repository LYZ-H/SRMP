#
# All workers communicate with each other via essentially a single RPC:
# fetch_parameters() which essentially requests the latest parameters
# from another worker.
#
# A flow-control mechanism is used to select the target to fetch parameters
# from. A timeout value is used to limit the time spent waiting for a
# chunk of data to arrive. Whenever the timeout expires, we decrease the score
# of the target, and we increase it any time we succeed. The score is bounded,
# and used combined with a random term to select the target.
#
# Every worker have 2 threads:
# RxThread:
#       Holds the most up-to-date model parameters and state,
#       it serves requests from the other workers.
# TxThread:
#       Initiating a fetch request to a random peer each time
#
"""Async Client/Server implementation."""
from copy import deepcopy
from collections import OrderedDict
import time
import gc
import queue
import threading
import multiprocessing
from xml.sax import parseString
import posix_ipc
import pickle
from .messages import P2PMSG


def _serialize_obj_to_bytes(params):
    return pickle.dumps(params)


def _deserialize_bytes_to_obj(blob):
    return pickle.loads(blob)


class RxProtocol:
    """
    handle received SYN, CHUNK, and FIN messages/datagrams
    """
    def __init__(self, rx_queue, channel_name=None, bw_in_kbps=8 * 8 * 1024):
        # all packets assgined with the same local port are handled by this protocol/protocol
        # super().__init__()
        self.rx_queue = rx_queue
        self.buf = {}

        self.channel_name = channel_name
        self.sem = posix_ipc.Semaphore(channel_name,
                                       flags=posix_ipc.O_CREAT,
                                       initial_value=1)
        self.bw_in_kbps = bw_in_kbps

    def __del__(self):
        self.sem.unlink()
        # self.sem.close()

    def connection_made(self, transport):
        self.transport = transport

    def get_cur_time(self):
        return time.time()

    def sendto(self, data, addr):
        if self.channel_name is None:
            return self.transport.sendto(data, addr)
        else:
            self.sem.acquire()
            assert self.sem.value == 0
            self.transport.sendto(data, addr)
            self.sem.release()

    def datagram_received(self, data, addr):
        reply = None
        cur_time = self.get_cur_time()
        try:
            msg = P2PMSG(frombuffer=data)

        except Exception as e:
            return

        if msg.is_CHUNK:

            if msg.peer in self.buf:
                ref = self.buf[msg.peer]

                if ref['chunks'][msg.chunk_seq] is None:
                    ref['chunks'][msg.chunk_seq] = msg.chunk_data
                    ref['received'] += 1
                else:
                    pass
                # gen ACK
                reply = P2PMSG(peer=self.channel_name,
                               mtype=P2PMSG.CHUNK_ACK,
                               seq=msg.seq,
                               chunk_seq=msg.chunk_seq)
        elif msg.is_SYN:

            if msg.peer not in self.buf:
                t1 = self.get_cur_time()
                chunks = [None for _ in range(msg.chunk_num)]

                self.buf[msg.peer] = dict(chunks=chunks,
                                          chunk_dtype=msg.chunk_dtype,
                                          chunk_bytes=msg.chunk_bytes,
                                          clock=msg.clock,
                                          received=0,
                                          total=msg.chunk_num,
                                          start_time=cur_time,
                                          name=msg.peer)

                # TODO: count the number of received chunk
                t2 = self.get_cur_time()
            else:
                pass
            # gen SYN_ACK
            reply = P2PMSG(peer=self.channel_name,
                           mtype=P2PMSG.SYN_ACK,
                           clock=msg.clock,
                           chunk_num=msg.chunk_num,
                           chunk_bytes=msg.chunk_bytes,
                           chunk_dtype=msg.chunk_dtype)
        elif msg.is_FIN:
            if msg.peer in self.buf:
                # with RX_QUEUE_lock:
                it = self.buf[msg.peer]
                it['finish_time'] = cur_time
                self.rx_queue.put(self.buf.pop(msg.peer))
                gc.collect()
            else:
                pass
            # gen FIN_ACK
            reply = P2PMSG(peer=self.channel_name,
                           mtype=P2PMSG.FIN_ACK,
                           clock=msg.clock)

        elif msg.is_CHUNK_ACK:
            pass
        else:
            return
        if reply is not None:
            try:
                self.sendto(reply.tobytes(), addr)
            except Exception as e:
                pass


class RxThread(threading.Thread):
    """
    listen on a port to receive chunks
    """
    def __init__(self,
                 local_port,
                 channel_name=None,
                 bw_in_kbps=8 * 8 * 1024,
                 local_host='',
                 socket_one=None):
        super().__init__()
        self.local_host = local_host
        self.local_port = local_port
        self.sock = socket_one

        self.rx_queue = queue.Queue()

        self.channel_name = channel_name
        self.bw_in_kbps = bw_in_kbps
        self.protocol = RxProtocol(rx_queue=self.rx_queue,
                                   channel_name=self.channel_name,
                                   bw_in_kbps=self.bw_in_kbps)
        self.protocol.connection_made(self.sock)
        self.running = True

    def run(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(P2PMSG.MAX_SIZE)
            except Exception as e:
                pass
            self.protocol.datagram_received(data, addr)

    def fetch_wait(self):
        """fetch a model. if not received, wait"""
        ref = self.rx_queue.get()
        return ref

    def shutdown(self):
        self.running = False


#
# Sender
#


class TxProtocol:
    SYN = 1
    SENDING = 2
    FIN = 3
    DRAINING = 4
    COMPLETED = 5

    CC_STATE_SLOW_START = 1
    CC_STATE_CONGESTION_AVOIDANCE = 2

    def __init__(
            self,
            tx_queue,
            clock,
            chunks,
            chunk_dtype=P2PMSG.FLOAT32,
            init_cwnd=100,
            min_cwnd=50,
            max_cwnd=1e9,
            init_rto=1,
            min_rto=1e-2,
            init_sshreshold=2**30,
            min_sshreshold=100,
            max_rate_in_mbps=2e3,
            tx_name=None,
            rx_name=None,
            acceptable_loss_rate=0.0,
            # note that, because of the error of estimation,
            # the acceptable loss rate should be slightly larger than the loss rate of the channel.
            channel_name=None,
            bw_in_kbps=8 * 8 * 1024,
            type=0,
            is_udp=0,
            **param):

        super().__init__()

        self.tx_queue = tx_queue
        self.channel_name = channel_name
        self.bw_in_kbps = bw_in_kbps
        self.sem = posix_ipc.Semaphore(channel_name,
                                       flags=posix_ipc.O_CREAT,
                                       initial_value=1)
        if self.sem.value == 0:
            self.sem.release()

        self.type = type
        self.is_udp = is_udp
        self.tx_name = tx_name
        self.rx_name = rx_name

        self.clock = clock
        self.chunk_dtype = chunk_dtype
        self.chunks = chunks

        # -------------------------------
        # cwnd, sshreshold, and cc_state can be reused for sequential transmistion tasks
        # --------------------------
        self.cwnd = init_cwnd
        self.sshreshold = init_sshreshold

        self.cc_state = self.CC_STATE_SLOW_START

        self.min_cwnd = min_cwnd
        self.max_cwnd = max_cwnd
        self.max_rate_in_mbps = max_rate_in_mbps
        self.max_rate_in_chunk_per_second = self.max_rate_in_mbps * 1e8 / 8 / P2PMSG.AVG_SIZE

        self.connection_stage = self.SYN
        self.min_sshreshold = min_sshreshold
        self.rto = init_rto
        self.min_rto = min_rto

        # ------------------------------
        self.LOSS_DETECT_THRESHOLD = 3
        self.last_cutdown_time = -1

        self.sent_yet_unacked = OrderedDict()
        self.to_resend = OrderedDict()
        self.next_chunk_seq = 0
        self.total_chunk_num = len(self.chunks)

        self.pkt_index = 0
        # self.tx_tokens = Queue()
        # self.tx_is_blocked = False
        self.rtt = None
        self.rttdev = 0
        self.rtt_alpha = 0.125
        self.rtt_1_alpha = 1 - self.rtt_alpha
        self.rtt_beta = 0.25
        self.rtt_1_beta = 1 - self.rtt_beta

        self.fast_draining_factor = 5
        self.perf_metrics = dict(resend_cnt=0,
                                 timeout_cnt=0,
                                 start_time=time.time(),
                                 chunk_num=len(self.chunks))
        self.last_receved_msg = None
        self.cc_state_trace = []
        self.last_fin_sent_time = -1

        self.acceptable_loss_rate = acceptable_loss_rate
        self.total_received_chunk_from_syn = 0
        self.estimated_loss_in_last_rtt = 0
        self.estimated_loss_reset_time = 0
        self.syn_time = 0

        self.met_congestion = False
        self.loss_count = 0
        self.total_loss = 0

    def update_rtt(self, rtt_sample):
        # http://blough.ece.gatech.edu/4110/TCPTimers.pdf
        if self.rtt is None:
            self.rtt = rtt_sample
        e = rtt_sample - self.rtt
        self.rtt += self.rtt_alpha * e
        # """
        ee = e if e > 0 else -e
        self.rttdev += self.rtt_beta * (ee - self.rttdev)
        self.rto = max(self.min_rto, self.rtt + 4 * self.rttdev)
        self.transport.settimeout(self.rto)
        # """

    def connection_made(self, transport, addr=None):
        self.transport = transport
        self.addr = addr
        self.transmit()

    def __del__(self):
        self.sem.unlink()
        # self.sem.close()

    def get_cur_time(self):
        return time.time()

    def sendto(self, data, addr):
        try:
            if self.channel_name is None:
                return self.transport.sendto(data, addr)
            else:
                self.sem.acquire()
                assert self.sem.value == 0

                self.transport.sendto(data, addr)
                self.sem.release()
        except Exception as e:
            if self.sem.value == 0:
                self.sem.release()
            self.sendto(data, addr)

    def handle_ack_timeout(self):
        self.perf_metrics['timeout_cnt'] += 1
        self.sshreshold = max(self.min_sshreshold, self.cwnd / 2)
        self.cwnd = self.min_cwnd
        # self.met_congestion = True

        for seq, v in self.sent_yet_unacked.items():
            self.to_resend[seq] = v
            # if self.tx_is_blocked:
            #    self.tx_tokens.put(1)
        self.sent_yet_unacked.clear()
        self.transmit()

    def datagram_received(self, data, addr):
        cur_time = self.get_cur_time()
        try:
            msg = P2PMSG(frombuffer=data)
            self.last_receved_msg = msg
        except Exception as e:
            return

        if self.connection_stage == self.SENDING:
            if msg.is_CHUNK_ACK:
                # with self.cwnd_queue_lock:
                self.update_rtt_cc_cwnd_and_buf(msg)
                if self.next_chunk_seq >= self.total_chunk_num:
                    self.connection_stage = self.FIN
                    self.perf_metrics['finish_time'] = cur_time

        elif self.connection_stage == self.SYN:
            if msg.is_SYN_ACK:
                self.connection_stage = self.SENDING
                self.syn_time = cur_time

        elif self.connection_stage == self.FIN:
            if msg.is_FIN_ACK:
                self.connection_stage = self.COMPLETED
                self.tx_queue.put(
                    dict(tx_name=self.tx_name,
                         rx_name=self.rx_name,
                         cwnd=self.cwnd,
                         sshreshold=self.sshreshold))
        elif self.connection_stage == self.COMPLETED:
            pass
        else:
            pass
        self.transmit()

    def get_chunk_to_send(self):
        if self.next_chunk_seq == 0:
            loss_r = 0
        else:
            loss_r = len(self.to_resend.keys()) / self.next_chunk_seq
        if (loss_r > self.acceptable_loss_rate and self.connection_stage
                == self.SENDING) or self.connection_stage != self.SENDING:
            seq, v = self.to_resend.popitem(last=False)
            chunk = v['chunk']
        elif self.next_chunk_seq < self.total_chunk_num:
            seq = self.next_chunk_seq
            chunk = self.chunks[seq]
            self.next_chunk_seq += 1
        elif 0 < len(self.sent_yet_unacked) < self.fast_draining_factor:
            seq, v = self.sent_yet_unacked.popitem(last=False)
            chunk = v['chunk']
        else:
            seq, chunk = None, None
        return seq, chunk

    def update_rtt_cc_cwnd_and_buf(self, msg):

        cur_time = self.get_cur_time()
        acked_chunk_seq = msg.chunk_seq

        v = self.to_resend.pop(acked_chunk_seq,
                               None) or self.sent_yet_unacked.pop(
                                   acked_chunk_seq, None)

        if v is None:
            return None

        if msg.seq == v['seq']:
            rtt_sample = cur_time - v['sent_time']
            self.update_rtt(rtt_sample)

        # detect packet loss
        newly_to_resend = []
        for seq, v in self.sent_yet_unacked.items():
            if seq > acked_chunk_seq:
                break
            v['reorder_cnt'] += 1
            if v['reorder_cnt'] >= self.LOSS_DETECT_THRESHOLD:
                newly_to_resend.append(seq)
        for seq in newly_to_resend:
            v = self.sent_yet_unacked.pop(seq)
            self.to_resend[seq] = v

        self.total_received_chunk_from_syn += 1
        # packet loss
        self.met_congestion = False
        resent_num = len(newly_to_resend)

        if resent_num > 0:
            self.perf_metrics['resend_cnt'] += resent_num

            self.estimated_loss_in_last_rtt += resent_num
            estimated_packets_per_rtt = self.total_received_chunk_from_syn * self.rtt / (
                cur_time - self.syn_time)
            estimated_loss_rate = self.estimated_loss_in_last_rtt / (
                self.estimated_loss_in_last_rtt + estimated_packets_per_rtt)

            if estimated_loss_rate >= self.acceptable_loss_rate:
                self.met_congestion = True

        if self.estimated_loss_reset_time + self.rtt < cur_time:

            self.estimated_loss_reset_time = cur_time
            self.estimated_loss_in_last_rtt = 0
            # raise ValueError

        # update cc and cwnd
        if self.cc_state == self.CC_STATE_CONGESTION_AVOIDANCE:
            self.cwnd += 1
            if self.met_congestion and self.last_cutdown_time + self.rtt < cur_time:
                self.sshreshold = self.cwnd / 2
                self.cwnd = 1
                self.last_cutdown_time = cur_time
            elif resent_num > 0:
                pass  # TODO:
            else:
                pass  # TODO:

        elif self.cc_state == self.CC_STATE_SLOW_START:
            self.cwnd *= 2
            if self.met_congestion:
                self.cc_state = self.CC_STATE_CONGESTION_AVOIDANCE
                self.sshreshold = self.cwnd / 2
                self.cwnd = 1
                self.last_cutdown_time = cur_time
            elif self.cwnd > self.sshreshold:
                self.cc_state = self.CC_STATE_CONGESTION_AVOIDANCE
            elif resent_num > 0:
                pass  # TODO: xxx

        self.cwnd = max(
            self.min_cwnd,
            min(self.cwnd, self.max_cwnd,
                self.rtt * self.max_rate_in_chunk_per_second))

    def transmit(self):
        if self.connection_stage == self.SENDING:
            msg_num = int(self.cwnd) - len(self.sent_yet_unacked)
            if self.is_udp == 1:
                while True:
                    seq, chunk = self.get_chunk_to_send()
                    if seq is None:
                        break

                    msg = P2PMSG(peer=self.channel_name,
                                 mtype=P2PMSG.CHUNK,
                                 seq=self.pkt_index,
                                 chunk_seq=seq,
                                 chunk_data=chunk)

                    self.sendto(msg.tobytes(), self.addr)

                    cur_time = self.get_cur_time()
                    self.sent_yet_unacked[seq] = dict(seq=msg.seq,
                                                      sent_time=cur_time,
                                                      chunk_seq=msg.chunk_seq,
                                                      chunk=msg.chunk_data,
                                                      reorder_cnt=0)
                    self.pkt_index += 1

            else:
                for _ in range(msg_num):

                    seq, chunk = self.get_chunk_to_send()
                    if seq is None:
                        break

                    msg = P2PMSG(peer=self.channel_name,
                                 mtype=P2PMSG.CHUNK,
                                 seq=self.pkt_index,
                                 chunk_seq=seq,
                                 chunk_data=chunk)

                    self.sendto(msg.tobytes(), self.addr)

                    cur_time = self.get_cur_time()
                    self.sent_yet_unacked[seq] = dict(seq=msg.seq,
                                                      sent_time=cur_time,
                                                      chunk_seq=msg.chunk_seq,
                                                      chunk=msg.chunk_data,
                                                      reorder_cnt=0)
                    self.pkt_index += 1

        elif self.connection_stage == self.SYN:
            syn_msg = P2PMSG(mtype=P2PMSG.SYN,
                             clock=self.clock,
                             chunk_num=len(self.chunks),
                             chunk_bytes=len(self.chunks[0]),
                             chunk_dtype=self.chunk_dtype,
                             peer=self.channel_name)

            self.sendto(syn_msg.tobytes(), self.addr)

        elif self.connection_stage == self.FIN:
            cur_time = self.get_cur_time()
            if self.last_fin_sent_time + self.rtt < cur_time:
                fin_msg = P2PMSG(mtype=P2PMSG.FIN,
                                 clock=self.clock,
                                 peer=self.channel_name)
                self.sendto(fin_msg.tobytes(), self.addr)
                self.sendto(fin_msg.tobytes(), self.addr)
                self.last_fin_sent_time = self.get_cur_time()

        else:
            pass

    def is_active(self):
        return self.connection_stage != self.COMPLETED

    def get_cur_time(self):
        return time.time()

    def shutdown(self):
        self.connection_stage = 'stopped'


class TxThread(threading.Thread):
    def __init__(self,
                 name,
                 channel_name=None,
                 bw_in_kbps=8 * 8 * 1024,
                 use_mp=True,
                 acceptable_loss_rate=0,
                 is_udp=0,
                 type=0,
                 local_host=None,
                 local_port=None,
                 socket_one=None,
                 **param):
        super().__init__()
        self.peers = {}
        self.name = name
        self.param = param
        self.clock = None
        self.chunks = None
        self.use_mp = use_mp
        self.channel_name = channel_name
        self.bw_in_kbps = bw_in_kbps
        self.is_udp = is_udp
        self.acceptable_loss_rate = acceptable_loss_rate
        self.type = type
        self.sock = socket_one

        if self.use_mp:
            self.proc = multiprocessing.Process
        else:
            self.proc = threading.Thread

    def add_peer(self, name, host, port):
        conn_state = dict(
            rx_name=name, host=host, port=port,
            tx_name=self.name)  # , cwnd=None, cc_state=None, sshreshold=None)
        self.peers[name] = conn_state

    def remove_peer(self, name):
        peer = self.peers[name]
        # TODO:
        del self.peers[name]

    def start_transfer(self, name):
        conn_state = self.peers[name]
        protocol = TxProtocol(self.tx_queue,
                              clock=self.clock,
                              chunks=self.chunks,
                              channel_name=self.channel_name,
                              bw_in_kbps=self.bw_in_kbps,
                              acceptable_loss_rate=self.acceptable_loss_rate,
                              is_udp=self.is_udp,
                              type=self.type,
                              **conn_state)
        protocol.connection_made(self.sock,
                                 addr=(conn_state['host'], conn_state['port']))

        self.sock.settimeout(protocol.rto)
        while protocol.is_active():
            try:
                data, addr = self.sock.recvfrom(P2PMSG.MAX_SIZE)
                protocol.datagram_received(data, addr)
            except Exception as e:
                # timeout error
                protocol.handle_ack_timeout()
                continue

        # update self.peers with new init_cwnd and sshrehold
        # update cwnd=None, sshreshold=None, which would be used for next ....
        self.peers[name]["init_cwnd"] = protocol.cwnd
        self.peers[name]["init_sshreshold"] = protocol.sshreshold

    def send_chunks(self, clock, chunks):

        self.clock = clock
        self.chunks = chunks

        self.transfers = []
        self.tx_queue = queue.Queue()
        # send payload to all the peers
        transfers = [
            self.proc(target=lambda: self.start_transfer(name))
            for name in self.peers
        ]
        for t in transfers:
            t.start()

        # waiting all transfers to complete
        for t in transfers:
            t.join()

        gc.collect()
        return True

    def run(self):
        pass

    def fetch_wait(self):

        # update cwnd=None, sshreshold=None, which would be used for next ....
        for _ in range(len(self.peers)):
            d = self.tx_queue.get()

            name = d['rx_name']
            self.peers[name]["init_cwnd"] = d['cwnd']
            self.peers[name]["init_sshreshold"] = d['sshreshold']

    def shutdown(self):
        pass