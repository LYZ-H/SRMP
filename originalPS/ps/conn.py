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
import logging
import socket
from collections import OrderedDict, deque
import time
import gc
import queue
import struct
import threading
import multiprocessing
import posix_ipc
from operator import xor
import _thread
import queue

LOGGER = logging.getLogger(__name__)

from .messages import P2MMSG, P2PMSG


class RxProtocol:
    """
    handle received SYN, CHUNK, and FIN messages/datagrams
    """
    def __init__(self,
                 rx_queue,
                 channel_name=None,
                 bw_in_kbps=800 * 1024,
                 acceptable_loss_rate=0.0,
                 feedback = 1):
        # def __init__(self, rx_queue, source_peer, channel_name=None, bw_in_kbps=800 * 1024):
        # all packets assgined with the same local port are handled by this protocol/protocol
        #super().__init__()
        self.rx_queue = rx_queue
        self.buf = {}
        self.acceptable_loss_rate = acceptable_loss_rate
        # self.source_peer = source_peer

        self.channel_name = channel_name
        self.sem = posix_ipc.Semaphore(channel_name,
                                       flags=posix_ipc.O_CREAT,
                                       initial_value=1)
        self.bw_in_kbps = bw_in_kbps
        self.i = 0

        self.fb_count = 1
        self.max_chunk_seq = 0
        
        self.feedback = feedback

        # TODO: (maybe) support dynamicall join and leave in the future
        """
        self.joined = False
        self.join_sent_time = None
        self.mcast_source = None
        self.join_timeout = 2
        """

    def __del__(self):
        self.sem.unlink()
        #self.sem.close()

    def connection_made(self, transport, transport_one):
        LOGGER.info('Protocol started!')
        self.transport = transport
        self.transport_one = transport_one

    def get_cur_time(self):
        return time.time()

    def send_join(self):
        msg = P2MMSG()  # join
        self.sendto(msg.tobytes(), self.source_peer)
        self.join_sent_time = self.get_cur_time()

    def check_join(self):
        if self.joined:
            return
        cur_time = self.get_cur_time()
        if self.join_sent_time + self.join_timeout < cur_time:
            self.join_sent_time = cur_time
            self.send_join()

    def sendto(self, data, addr):
        try:
            if self.channel_name is None:
                return self.transport.sendto(data, addr)
            else:
                self.sem.acquire()
                self.transport.sendto(data, addr)
                self.sem.release()
        except Exception as e:
            self.transport.sendto(data, addr)

    def check_completed(self, addr):
        if self.buf[addr] != None:
            gr = self.buf[addr]['received'] / self.buf[addr]['total']
            print("### {} {}".format(self.channel_name, gr))

    def bitwise_xor_bytes(self, a, b):
        result_int = int.from_bytes(a, byteorder="big") ^ int.from_bytes(
            b, byteorder="big")
        return result_int.to_bytes(max(len(a), len(b)), byteorder="big")

    def datagram_received(self, data, addr):

        cur_time = self.get_cur_time()
        try:
            msg = P2MMSG(frombuffer=data)
            # msg: is_SYN, is_FIN, is_CHUNK, pkt_index, chunk_seq
        except Exception as e:
            print('a22' + str(e))
            LOGGER.info('message decode error mult: %s', str(e))
            return

        reply = None
        reply_fin = None
        reply_feedback = None

        if addr not in self.buf and msg.mtype == 2:
            chunks = [None for _ in range(msg.total_chunk)]
            self.buf[addr] = dict(
                chunks=chunks,
                clock=msg.clock,
                received=0,
                total=msg.total_chunk,
                start_time=cur_time,
                max_received_chunk_seq=-1,
                lost_chunks=set(),
            )
            # TODO: count the number of received chunk
        elif addr not in self.buf and msg.mtype == 6:
            reply_fin = P2PMSG(peer=self.channel_name,
                              mtype=P2PMSG.FIN_ACK_MUL,
                              clock=int(time.time()))

        if addr in self.buf:
            ref = self.buf[addr]
            if self.buf[addr]['received'] / self.buf[addr][
                    'total'] >= 1 - self.acceptable_loss_rate and msg.mtype != 6:
                reply = P2PMSG(
                    peer=self.channel_name,
                    mtype=P2PMSG.PRE_FIN,
                )

            elif msg.mtype == 2:
                chunk_data = msg.chunk_data
                chunk_seqs = msg.chunk_seqs
                chunk_seq = msg.chunk_seq

                if ref['chunks'][chunk_seq] is None:
                    ref['chunks'][chunk_seq] = chunk_data
                    ref['received'] += 1
                if chunk_seq in ref['lost_chunks']:
                    ref['lost_chunks'].discard(chunk_seq)
                else:
                    ref['lost_chunks'].update(
                        range(ref['max_received_chunk_seq'] + 1,
                              msg.chunk_seq))
                reply = P2PMSG(peer=self.channel_name,
                              mtype=P2PMSG.CHUNK_ACK,
                              seq=msg.seq,
                              chunk_seq=chunk_seq)

                if self.feedback == 1:
                    if msg.chunk_seq > self.max_chunk_seq:
                        self.max_chunk_seq = msg.chunk_seq
                    # if msg.chunk_seq >= (self.max_chunk_seq / 4 * 3) and msg.chunk_seq // 1000 == self.fb_count:
                    if msg.chunk_seq >= self.max_chunk_seq and msg.chunk_seq % 1000 == 0:
                        uncover = []
                        for i in range(0, self.max_chunk_seq):
                            if ref['chunks'][i] is not None:
                                # chunk_data = xor(chunk_data, ref['chunks'][i])
                                chunk_data = self.bitwise_xor_bytes(
                                    chunk_data, ref['chunks'][i])
                            else:
                                uncover.append(i)
                        if len(uncover) == 1:
                            chunk_seq = uncover[0]
                            if ref['chunks'][chunk_seq] is None:
                                #ref['chunks'][msg.chunk_seq] = msg.chunk_data
                                ref['chunks'][chunk_seq] = chunk_data
                                ref['received'] += 1
                                if chunk_seq in ref['lost_chunks']:
                                    ref['lost_chunks'].discard(chunk_seq)
                                else:
                                    ref['lost_chunks'].update(
                                        range(ref['max_received_chunk_seq'] + 1,
                                            msg.chunk_seq))
                            else:
                                LOGGER.debug(
                                    '# Debug, duplicated chunk... {0} from peer {1}'
                                    .format(msg.chunk_seq, addr))
                        elif len(uncover) >= 40:
                            print("x")
                            chunk_seqs = uncover
                            reply_feedback = P2PMSG(
                                peer=self.channel_name,
                                mtype=P2PMSG.CHUNK_FEEDBACK,
                                seq=msg.seq,
                                chunk_seqs=chunk_seqs,
                                chunk_seqs_len=len(chunk_seqs),
                                total_chunk=msg.total_chunk,
                                last_received_chunk_seq=chunk_seq,
                                clock=int(time.time()),
                                receiver_id=0)
                            if self.channel_name == 'w2':
                                addr = ('10.0.0.1', 42000)
                                self.sendto(reply_feedback.tobytes(), addr)
                            elif self.channel_name == 'w3':
                                addr = ('10.0.0.1', 43000)
                                self.sendto(reply_feedback.tobytes(), addr)
                            elif self.channel_name == 'w4':
                                addr = ('10.0.0.1', 44000)
                                self.sendto(reply_feedback.tobytes(), addr)
                            elif self.channel_name == 'w5':
                                addr = ('10.0.0.1', 45000)
                                self.sendto(reply_feedback.tobytes(), addr)

                        self.fb_count += 1

                self.buf[addr] = ref

            elif msg.mtype == 6:
                if addr in self.buf:
                    self.check_completed(addr)
                    self.rx_queue.put(self.buf.pop(addr))
                    gc.collect()
                reply_fin = P2PMSG(peer=self.channel_name,
                                  mtype=P2PMSG.FIN_ACK_MUL,
                                  clock=int(time.time()))

                self.fb_count = 1
                self.max_chunk_seq = 0

        if self.channel_name == 'w1':
            addr = ('10.0.0.1', 41000)
            if reply_fin is not None:
                self.sendto(reply_fin.tobytes(), addr)
            else:
                self.sendto(reply.tobytes(), addr)
            self.i += 1
        elif self.channel_name == 'w2' and reply_fin is not None:
            addr = ('10.0.0.1', 42000)
            self.sendto(reply_fin.tobytes(), addr)
        elif self.channel_name == 'w3' and reply_fin is not None:
            addr = ('10.0.0.1', 43000)
            self.sendto(reply_fin.tobytes(), addr)
        elif self.channel_name == 'w4' and reply_fin is not None:
            addr = ('10.0.0.1', 44000)
            self.sendto(reply_fin.tobytes(), addr)
        elif self.channel_name == 'w5' and reply_fin is not None:
            addr = ('10.0.0.1', 45000)
            self.sendto(reply_fin.tobytes(), addr)



class RxThread(threading.Thread):
    """
    listen on a port to receive chunks
    """
    def __init__(self,
                 mcast_grp,
                 mcast_port,
                 channel_name=None,
                 source_peer=None,
                 bw_in_kbps=800 * 1024,
                 socket_one=None,
                 acceptable_loss_rate=0.0,
                 feedback = 1):
        super().__init__()
        #self.source_peer = source_peer
        self.mcast_grp = mcast_grp
        self.mcast_port = mcast_port
        # TODO: can setup multiple socket if needed.
        self.rx_queue = queue.Queue()
        self.sock_one = socket_one
        self.acceptable_loss_rate = acceptable_loss_rate
        self.feedback = feedback
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                                      socket.IPPROTO_UDP)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # on this port, receives ALL multicast groups
            self.sock.bind(('', self.mcast_port))
        except Exception as e:
            print('a3' + str(e))
            LOGGER.info('RxThread ... init error: %s', str(e))
        self.channel_name = channel_name
        self.bw_in_kbps = bw_in_kbps
        self.protocol = RxProtocol(
            rx_queue=self.rx_queue,
            channel_name=self.channel_name,
            bw_in_kbps=self.bw_in_kbps,
            acceptable_loss_rate=self.acceptable_loss_rate,
            feedback = self.feedback)
        self.protocol.connection_made(self.sock, self.sock_one)
        self.running = True
        LOGGER.info("RxThread starts to receve data..... at %s",
                    str(self.sock.getsockname()))

    def __del__(self):
        self.sock.setsockopt(
            socket.SOL_IP, socket.IP_DROP_MEMBERSHIP,
            socket.inet_aton(self.mcast_grp) + socket.inet_aton('0.0.0.0'))
        #print('# Done!')

    def run(self):
        LOGGER.info('rx is running now')
        mreq = struct.pack("4sl", socket.inet_aton(self.mcast_grp),
                           socket.INADDR_ANY)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        #self.protocol.send_join()
        # TODO: maybe support dynamically join in the future
        while self.running:
            try:
                data, addr = self.sock.recvfrom(P2MMSG.MAX_SIZE)
                addr = ('10.0.0.1', 40000)
                self.protocol.datagram_received(data, addr)
                # set and check timeout
            except Exception as e:
                print('a4' + str(e))
                pass
            """
            if not self.protocol.joined:
                self.protocol.check_join()
            """

    def fetch_wait(self):
        """fetch a model. if not received, wait"""
        ref = self.rx_queue.get()
        return ref

    def shutdown(self):
        self.running = False


#
# Sender
#

packets = queue.Queue()


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
            receivers,
            clock,
            chunks,
            chunk_dtype=P2MMSG.FLOAT32,
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
        self.receivers = receivers
        self.receivers_list = list(self.receivers)
        self.next_rid_index = -1

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

        # self.init_cwnd = init_cwnd
        self.min_cwnd = min_cwnd
        self.max_cwnd = max_cwnd
        self.max_rate_in_mbps = max_rate_in_mbps
        self.max_rate_in_chunk_per_second = self.max_rate_in_mbps * 1e8 / 8 / P2MMSG.AVG_SIZE

        self.connection_stage = self.SENDING
        # self.init_sshreshold = init_sshreshold
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

        self.fin_arr = [False] * 5

        self.total_received = 0

        self.packets = queue.Queue()
        self.met_err = False

        self.rec = 0

        self.sended = []

        self.feedback_seq = {}

        self.chunks_to_resend = set()

        self.recv = set()

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
        LOGGER.info('Connected!')
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
        self.met_congestion = True
        try:
            for seq, v in self.sent_yet_unacked.items():
                self.to_resend[seq] = v
                # if self.tx_is_blocked:
                #    self.tx_tokens.put(1)
        except Exception as e:
            pass
        self.sent_yet_unacked.clear()
        self.transmit()

    def chunk_to_feedback(self):

        total_peer = len(self.feedback_seq.keys())
        chunk_total_feedback = set()
        for chunks in self.feedback_seq.values():
            chunk_total_feedback.update(chunks)

        self.chunks_to_resend = set()
        print(total_peer)
        for chunk in chunk_total_feedback:
            i = 0
            for chunks in self.feedback_seq.values():
                if chunk in chunks:
                    i += 1
                if i >= total_peer - 4:
                    self.chunks_to_resend.add(chunk)
        for i, seq in enumerate(self.chunks_to_resend):
            if i % 50 == 0:
                time.sleep(0.001)
            msg = P2MMSG(
                mtype=P2MMSG.CHUNK,
                clock=int(time.time()),
                total_chunk=self.total_chunk_num,
                receiver_id=0,
                seq=self.pkt_index,
                chunk_seq=seq,
                chunk_data=self.chunks[seq])
            self.sendto(msg.tobytes(), self.addr)

    def datagram_received(self, data, addr):
        cur_time = self.get_cur_time()
        try:
            msg = P2PMSG(frombuffer=data)
        except Exception as e:
            print('a2' + str(e))
            LOGGER.info('message decode error mult: %s', str(e))
            return

        if self.connection_stage == self.SENDING and msg.peer == 'w1':

            if msg.mtype == 9:
                self.chunk_to_feedback()
                self.connection_stage = self.FIN
                self.perf_metrics['finish_time'] = cur_time
            elif msg.mtype == 4:

                self.update_rtt_cc_cwnd_and_buf(msg)

        elif self.connection_stage == self.SENDING:
            if msg.CHUNK_FEEDBACK:
                if msg.peer not in self.feedback_seq:
                    self.feedback_seq[msg.peer] = set()
                self.feedback_seq[msg.peer].update(set(msg.chunk_seqs))

        elif self.connection_stage == self.FIN:
            if msg.FIN_ACK_MUL and msg.peer != 1:

                self.fin_arr[int(msg.peer[1:]) - 1] = True

                if not False in self.fin_arr:
                    self.connection_stage = self.COMPLETED

        elif self.connection_stage == self.COMPLETED:
            pass
        else:
            LOGGER.info('error meesage: %s', str(msg))
        if (self.connection_stage == self.SENDING
                and msg.peer == 'w1') or self.connection_stage != self.SENDING:
            self.transmit()

    def get_chunk_to_send(self):

        if self.next_chunk_seq < self.total_chunk_num:
            seq = self.next_chunk_seq
            chunk = self.chunks[seq]
            self.next_chunk_seq += 1
        elif len(self.to_resend) > 0:
            seq, v = self.to_resend.popitem(last=False)
            chunk = v['chunk']
        elif 0 < len(self.sent_yet_unacked) < self.fast_draining_factor:
            seq, v = self.sent_yet_unacked.popitem(last=False)
            chunk = v['chunk']
        elif len(self.sent_yet_unacked) > 0:
            seq, v = self.sent_yet_unacked.popitem(last=False)
            chunk = v['chunk']
        else:
            seq, chunk = None, None
        self.sended.append(seq)
        return seq, chunk

    def update_rtt_cc_cwnd_and_buf(self, msg):
        self.total_received += 1
        cur_time = self.get_cur_time()
        acked_chunk_seq = msg.chunk_seq
        if acked_chunk_seq not in self.recv:
            self.recv.add(acked_chunk_seq)
        v = self.to_resend.pop(acked_chunk_seq,
                               None) or self.sent_yet_unacked.pop(
                                   acked_chunk_seq, None)

        if v is None:
            return None

        if msg.seq == v['pkt_seq']:
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
            try:
                v = self.sent_yet_unacked.pop(seq)
                self.to_resend[seq] = v
            except KeyError:
                pass

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

    def get_rid_to_ack(self):
        if len(self.receivers_list) == 0:
            return None
        self.next_rid_index = (1 + self.next_rid_index) % len(
            self.receivers_list)
        return self.receivers_list[self.next_rid_index]

    def transmit(self):
        LOGGER.debug('cwnd: %f, in_flight_pkt: %d, sshreshold: %f', self.cwnd,
                     len(self.sent_yet_unacked), self.sshreshold)

        msg_num = int(self.cwnd) - len(self.sent_yet_unacked)

        if self.connection_stage == self.SENDING:
            for _ in range(msg_num):
                chunk_seq, chunk = self.get_chunk_to_send()
                # rid = self.get_rid_to_ack()
                rid = 0
                if rid is None or chunk_seq is None:
                    break

                msg = P2MMSG(
                    mtype=P2MMSG.CHUNK,
                    clock=int(time.time()),
                    total_chunk=self.total_chunk_num,
                    receiver_id=rid,
                    seq=self.pkt_index,
                    # chunk_seqs=self.sended,
                    chunk_seq=chunk_seq,
                    chunk_data=chunk)

                self.sendto(msg.tobytes(), self.addr)

                cur_time = self.get_cur_time()

                self.sent_yet_unacked[chunk_seq] = dict(
                    pkt_seq=self.pkt_index,
                    sent_time=cur_time,
                    chunk_seqs=msg.chunk_seqs,
                    chunk=msg.chunk_data,
                    reorder_cnt=0)

                self.pkt_index += 1
        elif self.connection_stage == self.FIN:
            cur_time = self.get_cur_time()
            # if self.last_fin_sent_time + self.rtt < cur_time:
            fin_msg = P2MMSG(mtype=P2MMSG.FIN, clock=self.clock)
            self.sendto(fin_msg.tobytes(), self.addr)
            # self.last_fin_sent_time = self.get_cur_time()

    def is_active(self):
        return self.connection_stage != self.COMPLETED

    def get_cur_time(self):
        return time.time()

    def shutdown(self):
        self.connection_stage = 'stopped'

    def show_perf(self):
        LOGGER.info("Flow <%s, %s>, %s", self.tx_name, self.rx_name,
                    str(self.perf_metrics))
        LOGGER.info("rtt %f, rttdev %f", self.rtt, self.rttdev)
        LOGGER.info(
            'AVG Rate in Mbps %f', self.next_chunk_seq * 8 / 1000 /
            (self.perf_metrics['finish_time'] -
             self.perf_metrics['start_time']))
        # with open('cc_state_trace.txt', 'w') as f:
        #    print(self.cc_state_trace, file=f)


rec_data = queue.Queue()


class TxThread(threading.Thread):
    def __init__(self,
                 name,
                 mcast_grp,
                 mcast_port,
                 local_port=None,
                 channel_name=None,
                 bw_in_kbps=800 * 1024,
                 use_mp=True,
                 socket_one=None,
                 acceptable_loss_rate=0,
                 **param):
        super().__init__()
        self.peers = {}
        self.name = name
        self.mcast_grp = mcast_grp
        self.mcast_port = mcast_port
        self.local_port = local_port
        self.local_host = ''
        self.param = param
        self.clock = None
        self.chunks = None
        self.use_mp = use_mp
        self.channel_name = channel_name
        self.bw_in_kbps = bw_in_kbps
        self.sock_one = socket_one
        self.acceptable_loss_rate = acceptable_loss_rate

        self.receivers = {}
        self.next_receiver_id_to_ack = 0

        if self.use_mp:
            self.proc = multiprocessing.Process
        else:
            self.proc = threading.Thread

        # regarding socket.IP_MULTICAST_TTL
        # ---------------------------------
        # for all packets sent, after two hops on the network the packet will not
        # be re-sent/broadcast (see https://www.tldp.org/HOWTO/Multicast-HOWTO-6.html)
        # self.sock_one = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.sock_one.bind((self.local_host, self.local_port))
        MULTICAST_TTL = 10

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                                  socket.IPPROTO_UDP)

        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL,
                             MULTICAST_TTL)

        # For Python 3, change next line to 'sock.sendto(b"robot", ...' to avoid the
        # "bytes-like object is required" msg (https://stackoverflow.com/a/42612820)
        #sock.sendto(b"robot", (MCAST_GRP, MCAST_PORT))

    def add_peer(self, receiver_id, name=None):
        LOGGER.debug("Adding peer %s (%d)...", name, receiver_id)
        self.receivers[receiver_id] = dict(name=name)
        LOGGER.debug("peer %s added.", name)

    def remove_peer(self, receiver_id, name=None):
        LOGGER.debug("Removing peer %s...", name)
        del self.receivers[receiver_id]
        LOGGER.debug("peer %s removed.", name)

    def recvf(self, protocol, sock):
        global rec_data

        while protocol.is_active():
            try:
                data, addr = sock.recvfrom(P2PMSG.MAX_SIZE)
            except Exception as e:
                protocol.handle_ack_timeout()
                continue
            rec_data.put([data, addr])

    def start_transfer(self):
        self.conn_state = {}
        protocol = TxProtocol(self.tx_queue,
                              receivers=self.receivers,
                              clock=self.clock,
                              chunks=self.chunks,
                              channel_name=self.channel_name,
                              bw_in_kbps=self.bw_in_kbps,
                              acceptable_loss_rate=self.acceptable_loss_rate,
                              **self.conn_state)
        lock = _thread.allocate_lock()
        protocol.connection_made(self.sock,
                                 addr=(self.mcast_grp, self.mcast_port))
        self.sock.settimeout(protocol.rto)

        socket_recs = []
        for i in range(41000, 46000, 1000):
            sock_x = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock_x.bind(('', i))
            sock_x.settimeout(protocol.rto)
            socket_recs.append(sock_x)

        t_arr = []
        for sock in socket_recs:
            t = threading.Thread(target=self.recvf, args=(
                protocol,
                sock,
            ))
            t_arr.append(t)

        for t in t_arr:
            t.start()

        while protocol.is_active():

            data_r = rec_data.get()
            protocol.datagram_received(data_r[0], data_r[1])
        # update self.peers with new init_cwnd and sshrehold
        # update cwnd=None, sshreshold=None, which would be used for next ....
        self.conn_state["init_cwnd"] = protocol.cwnd
        self.conn_state["init_sshreshold"] = protocol.sshreshold

    def send_chunks(self, clock, chunks, show_perf=False):
        self.clock = clock
        self.chunks = chunks

        self.transfers = []
        self.tx_queue = queue.Queue()

        # multicast payload to all the peers
        transfer = self.proc(target=self.start_transfer)

        transfer.start()
        # waiting all transfers to complete
        transfer.join()
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
