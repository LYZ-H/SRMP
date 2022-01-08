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
from collections import OrderedDict
import time
import gc
import queue
import threading
import multiprocessing


LOGGER = logging.getLogger(__name__)


from messages import MSG


class RxProtocol:
    """
    handle received SYN, CHUNK, and FIN messages/datagrams
    """
    def __init__(self, rx_queue):
        # all packets assgined with the same local port are handled by this protocol/protocol
        #super().__init__()
        self.rx_queue = rx_queue
        self.buf = {}

    def connection_made(self, transport):
        LOGGER.info('Protocol started!')
        self.transport = transport
        
    def get_cur_time(self):
        return time.time()

    def datagram_received(self, data, addr):
        cur_time = self.get_cur_time()
        try:
            msg = MSG(frombuffer=data)
            # msg: is_SYN, is_FIN, is_CHUNK, pkt_index, chunk_seq
            # LOGGER.debug('# Rx gets msg {0} from {1}'.format(msg, peer))
        except Exception as e:
            LOGGER.info('message decode error: %s', str(e))
            return
        LOGGER.debug('rx gets msg %s', str(msg))
        if msg.is_CHUNK:
            ref = self.buf[addr]
            if ref['chunks'][msg.chunk_seq] is None:
                #ref['chunks'][msg.chunk_seq] = msg.chunk_data
                ref['chunks'][msg.chunk_seq] = msg.chunk_data
                ref['received'] += 1
            else:
                LOGGER.debug('# Debug, duplicated chunk... {0} from peer {1}'.format(msg.chunk_seq, addr))
            # gen ACK
            reply = MSG(mtype=MSG.CHUNK_ACK, seq=msg.seq, chunk_seq=msg.chunk_seq)
        elif msg.is_SYN:
            if addr not in self.buf:
                t1 = self.get_cur_time()
                LOGGER.info('SYN: %d, %d', msg.chunk_num, msg.chunk_bytes)
                chunks = [None for _ in range(msg.chunk_num)]
                #chunks = np.zeros(msg.chunk_num * msg.chunk_bytes >> 2, dtype=np.float32) # assume float32
                #mm = mmap.mmap(-1, msg.chunk_num * (msg.chunk_bytes + 4) + 100)
                self.buf[addr] = dict(chunks=chunks, chunk_dtype=msg.chunk_dtype, chunk_bytes=msg.chunk_bytes, 
                    #mm=mm,
                    clock=msg.clock, received=0, total=msg.chunk_num, start_time=cur_time)
                # TODO: count the number of received chunk
                t2 = self.get_cur_time()
                LOGGER.info('t2 - t1: %f', t2 - t1)
            else:
                LOGGER.debug('#duplicated SYN {0} from {1}'.format(msg, addr))
            # gen SYN_ACK
            reply = MSG(mtype=MSG.SYN_ACK, clock=msg.clock, chunk_num=msg.chunk_num, chunk_bytes=msg.chunk_bytes, chunk_dtype=msg.chunk_dtype)
        elif msg.is_FIN:
            if addr in self.buf:
                #with RX_QUEUE_lock:
                it = self.buf[addr]
                it['finish_time'] = cur_time
                LOGGER.info('AVG Rate in Mbps %f', it['received'] * 8 / 1000 / (it['finish_time'] - it['start_time']))
                self.rx_queue.put(self.buf.pop(addr))
                gc.collect()
            else:
                LOGGER.debug('#duplicated FIN {0} from {1}'.format(msg, addr))
            # gen FIN_ACK
            reply = MSG(mtype=MSG.FIN_ACK, clock=msg.clock)
        else:
            LOGGER.info('unknown msg type: %s', str(msg))
            return
        #LOGGER.debug('reply {0} to peer {1}'.format(reply, peer))
        try:
            self.transport.sendto(reply.tobytes(), addr)
            LOGGER.debug('rx sends reply %s', str(reply))
        except Exception as e:
            LOGGER.info('Fail to send msg: %s', str(e))
            #raise e
        if msg.seq is not None and msg.seq % 2000 == 0:
            t2 = time.time()
            it = self.buf[addr]
            LOGGER.info('# ack deley on Rx: %f, %d', t2 - cur_time, msg.chunk_seq)
            LOGGER.info('AVG Rate in Mbps %f', it['received'] * 8 / 1000 / (t2 - it['start_time']))


class RxThread(threading.Thread):
    """
    listen on a port to receive chunks
    """
    def __init__(self, local_port, local_host=''):
        super().__init__()
        self.local_host = local_host
        self.local_port = local_port
        # TODO: can setup multiple socket if needed.
        self.rx_queue = queue.Queue() 
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.local_host, self.local_port))
        except Exception as e:
            LOGGER.info('RxThread ... init error: %s', str(e))
        self.protocol = RxProtocol(rx_queue=self.rx_queue)
        self.protocol.connection_made(self.sock)
        self.running = True
        LOGGER.info("RxThread starts to receve data..... at %s", str(self.sock.getsockname()))

    def run(self):
        LOGGER.info('rx is running now')
        while self.running:
            try:
                data, addr = self.sock.recvfrom(MSG.MAX_SIZE)
            except Exception as e:
                LOGGER.info('rx error on receving %s', str(e))
            self.protocol.datagram_received(data, addr)

    def fetch_wait(self):
        """fetch a model. if not received, wait"""
        ref = self.rx_queue.get()
        return ref['clock'], ref['chunks']

    def shutdown(self):
        self.running = False


#
# Sender
#

class TxProtocol:
    SYN = 1
    SENDING = 2
    FIN =3
    DRAINING = 4
    COMPLETED = 5

    CC_STATE_SLOW_START = 1
    CC_STATE_CONGESTION_AVOIDANCE = 2

    def __init__(
            self, 
            tx_queue,
            clock, chunks, chunk_dtype=MSG.FLOAT32, 
            init_cwnd=100, min_cwnd=50, max_cwnd=1e9, 
            init_rto=1, min_rto=1e-2, 
            init_sshreshold=2**30, min_sshreshold=100, 
            max_rate_in_mbps=2e3, 
            tx_name=None, rx_name=None,
            acceptable_loss_rate=0.0, 
            # note that, because of the error of estimation, 
            # the acceptable loss rate should be slightly larger than the loss rate of the channel.
            **param):

        super().__init__()

        self.tx_queue = tx_queue

        self.tx_name = tx_name
        self.rx_name = rx_name

        self.clock = clock
        self.chunk_dtype = chunk_dtype
        self.chunks = chunks

        #-------------------------------
        # cwnd, sshreshold, and cc_state can be reused for sequential transmistion tasks
        #--------------------------
        self.cwnd = init_cwnd
        self.sshreshold = init_sshreshold

        self.cc_state = self.CC_STATE_SLOW_START
        
        #self.init_cwnd = init_cwnd
        self.min_cwnd = min_cwnd
        self.max_cwnd = max_cwnd
        self.max_rate_in_mbps = max_rate_in_mbps
        self.max_rate_in_chunk_per_second = self.max_rate_in_mbps * 1e6 / 8 / MSG.AVG_SIZE

        self.connection_stage = self.SYN
        #self.init_sshreshold = init_sshreshold
        self.min_sshreshold = min_sshreshold
        self.rto = init_rto
        self.min_rto = min_rto

        #------------------------------ 
        self.LOSS_DETECT_THRESHOLD = 3
        self.last_cutdown_time = -1

        self.sent_yet_unacked = OrderedDict()
        self.to_resend = OrderedDict()
        self.next_chunk_seq = 0
        self.total_chunk_num = len(self.chunks)

        self.pkt_index = 0
        #self.tx_tokens = Queue()
        #self.tx_is_blocked = False
        self.rtt = None
        self.rttdev = 0
        self.rtt_alpha = 0.125
        self.rtt_1_alpha = 1 - self.rtt_alpha
        self.rtt_beta = 0.25
        self.rtt_1_beta = 1 - self.rtt_beta

        self.fast_draining_factor = 5
        self.perf_metrics = dict(resend_cnt=0, timeout_cnt=0, start_time=time.time(), chunk_num=len(self.chunks))
        self.last_receved_msg = None
        self.cc_state_trace = []
        self.last_fin_sent_time = -1

        self.acceptable_loss_rate = acceptable_loss_rate
        self.total_received_chunk_from_syn = 0
        self.estimated_loss_in_last_rtt = 0
        self.estimated_loss_reset_time = 0
        self.syn_time = 0
        
    def update_rtt(self, rtt_sample):
        #http://blough.ece.gatech.edu/4110/TCPTimers.pdf
        if self.rtt is None:
            self.rtt = rtt_sample
        e = rtt_sample - self.rtt
        self.rtt += self.rtt_alpha * e
        #"""
        ee = e if e > 0 else -e
        self.rttdev += self.rtt_beta * (ee - self.rttdev)
        self.rto = max(self.min_rto, self.rtt + 4 * self.rttdev)
        self.transport.settimeout(self.rto)
        #"""

    def connection_made(self, transport, addr=None):
        LOGGER.info('Connected!')
        self.transport = transport
        self.addr = addr
        self.transmit()

    def handle_ack_timeout(self):
        LOGGER.info(" sent-yet-unacked: %d, resent: %d, cwnd: %f, ssreshold: %f, cc-state: %d", 
                len(self.to_resend.keys()), len(self.sent_yet_unacked.keys()), self.cwnd, self.sshreshold, self.cc_state)
        self.perf_metrics['timeout_cnt'] += 1
        self.sshreshold = max(self.min_sshreshold, self.cwnd / 2)
        self.cwnd = self.min_cwnd        
        LOGGER.info(" sent-yet-unacked: %d, resent: %d, cwnd: %f, ssreshold: %f, cc-state: %d", 
                len(self.to_resend.keys()), len(self.sent_yet_unacked.keys()), self.cwnd, self.sshreshold, self.cc_state)
        for seq, v in self.sent_yet_unacked.items():
            self.to_resend[seq] = v
            #if self.tx_is_blocked:
            #    self.tx_tokens.put(1)
        self.sent_yet_unacked.clear()
        #LOGGER.info(str(self.to_resend.keys()))
        self.transmit()

    def datagram_received(self, data, addr):
        cur_time = self.get_cur_time()
        try:
            msg = MSG(frombuffer=data)
            self.last_receved_msg = msg
        except Exception as e:
            LOGGER.info('broken msg, error during parsing: %s', str(e))
            return
        LOGGER.debug('recv msg %s', str(msg))

        if self.connection_stage == self.SENDING:
            if msg.is_CHUNK_ACK:
                self.last_ack = msg
                LOGGER.debug('# get chunk_ack chunk_seq: {0}, seq: {1}'.format(msg.chunk_seq, msg.seq))
                #with self.cwnd_queue_lock:
                self.update_rtt_cc_cwnd_and_buf(msg)
                if self.all_chunks_delivered():
                    self.connection_stage = self.FIN
                    self.perf_metrics['finish_time'] = cur_time
                    self.show_perf()

        elif self.connection_stage == self.SYN:
            if msg.is_SYN_ACK:
                # TODO： 
                self.connection_stage = self.SENDING
                self.syn_time = cur_time
        elif self.connection_stage == self.FIN:
            if msg.is_FIN_ACK:
                self.connection_stage = self.COMPLETED
                self.show_perf()
                self.tx_queue.put(dict(tx_name=self.tx_name, rx_name=self.rx_name, cwnd=self.cwnd, sshreshold=self.sshreshold))
        elif self.connection_stage == self.COMPLETED:
            pass
        else:
            LOGGER.info('error meesage: %s', str(msg))
        self.transmit()
        
    def all_chunks_delivered(self):
        # TODO: to be updated
        #if self.next_chunk_seq > 10 * 1000:
        #    return 1
        return len(self.sent_yet_unacked) == 0 and self.next_chunk_seq >= self.total_chunk_num and len(self.to_resend) == 0

    def get_chunk_to_send(self):
        if len(self.to_resend) > 0:
            seq, v = self.to_resend.popitem(last=False)
            chunk = v['chunk']
        elif self.next_chunk_seq < self.total_chunk_num:
            seq = self.next_chunk_seq
            chunk = self.chunks[seq]
            self.next_chunk_seq += 1
        elif 0 < len(self.sent_yet_unacked) < self.fast_draining_factor:
            LOGGER.debug('# fast draining .. {0}'.format(self.sent_yet_unacked.keys()))
            seq, v = self.sent_yet_unacked.popitem(last=False)
            chunk = v['chunk']
        else:
            seq, chunk = None, None
        return seq, chunk

    def update_rtt_cc_cwnd_and_buf(self, msg):
        cur_time = self.get_cur_time()
        acked_chunk_seq = msg.chunk_seq
        v = self.to_resend.pop(acked_chunk_seq, None) or self.sent_yet_unacked.pop(acked_chunk_seq, None)
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
        met_congestion = False
        resent_num = len(newly_to_resend)
        if resent_num > 0:
            self.perf_metrics['resend_cnt'] += resent_num
            LOGGER.debug('newly_to_resend: ' + str(newly_to_resend))

            self.estimated_loss_in_last_rtt += resent_num
            estimated_packets_per_rtt = self.total_received_chunk_from_syn * self.rtt / (cur_time - self.syn_time) 
            estimated_loss_rate = self.estimated_loss_in_last_rtt / (self.estimated_loss_in_last_rtt + estimated_packets_per_rtt)
            if estimated_loss_rate >= self.acceptable_loss_rate:
                met_congestion = True
                LOGGER.debug('estimated_loss_rate: %f, at time: %f, rtt %f, lossnum %f %f', estimated_loss_rate, cur_time, self.rtt,  self.estimated_loss_in_last_rtt, estimated_packets_per_rtt)
            #LOGGER.info('acceptable_loss_rate %f', self.acceptable_loss_rate)
            
        
        if self.estimated_loss_reset_time  + self.rtt < cur_time:
            #LOGGER.info('elr reset %d, %f', self.estimated_loss_in_last_rtt, cur_time)
            self.estimated_loss_reset_time = cur_time
            self.estimated_loss_in_last_rtt = 0
            #raise ValueError

        # update cc and cwnd
        if self.cc_state == self.CC_STATE_CONGESTION_AVOIDANCE:
            self.cwnd += 1. / self.cwnd
            if met_congestion and self.last_cutdown_time + self.rtt < cur_time:
                self.sshreshold = self.cwnd
                self.cwnd = self.cwnd / 2
                self.last_cutdown_time = cur_time
            elif resent_num > 0:
                pass # TODO:
            else:
                pass # TODO:

        elif self.cc_state == self.CC_STATE_SLOW_START:
            self.cwnd += 1
            if met_congestion:
                self.cc_state = self.CC_STATE_CONGESTION_AVOIDANCE
                self.sshreshold = self.cwnd
                self.cwnd = self.cwnd / 2
                self.last_cutdown_time = cur_time
            elif self.cwnd > self.sshreshold:
                self.cc_state = self.CC_STATE_CONGESTION_AVOIDANCE
            elif resent_num > 0:
                pass # TODO: xxx

        self.cwnd = max(self.min_cwnd, min(self.cwnd, self.max_cwnd, self.rtt * self.max_rate_in_chunk_per_second))
        #LOGGER.info("# rem is 0: %d, %d, %d", self.remaining_cwnd_queue.qsize(), len(self.sent_yet_unacked), int(self.cwnd))
        #self.cc_state_trace.append((cur_time, self.cwnd, self.rtt, self.sshreshold, self.cc_state))

    def transmit(self):
        LOGGER.debug('cwnd: %f, in_flight_pkt: %d, sshreshold: %f', self.cwnd, len(self.sent_yet_unacked), self.sshreshold)
        if self.connection_stage == self.SENDING:
            msg_num = int(self.cwnd) - len(self.sent_yet_unacked)
            for _ in range(msg_num):
                LOGGER.debug('# remain task size {0} {1}'.format(len(self.to_resend), self.total_chunk_num - self.next_chunk_seq))
                seq, chunk = self.get_chunk_to_send()
                if seq is None:
                    break
                msg = MSG(mtype=MSG.CHUNK, seq=self.pkt_index, chunk_seq=seq, chunk_data=chunk)
                self.transport.sendto(msg.tobytes(), self.addr)
                LOGGER.debug('Send chunk {0} {1}'.format(msg.chunk_seq, msg.seq))
                cur_time = self.get_cur_time()
                self.sent_yet_unacked[seq] = dict(seq=msg.seq, sent_time=cur_time, chunk_seq=msg.chunk_seq, chunk=msg.chunk_data, reorder_cnt=0)
                self.pkt_index += 1

        elif self.connection_stage == self.SYN:
            syn_msg = MSG(mtype=MSG.SYN, clock=self.clock, chunk_num=len(self.chunks), chunk_bytes=len(self.chunks[0]), chunk_dtype=self.chunk_dtype) 
            #syn_msg = MSG(mtype=MSG.SYN, clock=self.clock, chunk_num=10*1000, chunk_dtype=self.chunk_dtype) 
            self.transport.sendto(syn_msg.tobytes(), self.addr)
            self.transport.sendto(syn_msg.tobytes(), self.addr)
            LOGGER.debug('tx sends SYN: %s to %s', str(syn_msg), str(self.addr))
        
        elif self.connection_stage == self.FIN:
            cur_time = self.get_cur_time()
            if self.last_fin_sent_time + self.rtt < cur_time:
                fin_msg = MSG(mtype=MSG.FIN, clock=self.clock) 
                self.transport.sendto(fin_msg.tobytes(), self.addr)
                self.transport.sendto(fin_msg.tobytes(), self.addr)
                self.last_fin_sent_time = self.get_cur_time()
                LOGGER.debug('tx sends FIN: %s to %s', str(fin_msg), str(self.addr))
        else:
            pass
        
    def is_active(self):
        return self.connection_stage != self.COMPLETED

    def get_cur_time(self):
        return time.time()

    def shutdown(self):
        self.connection_stage = 'stopped'

    def show_perf(self):
        LOGGER.info("Flow <%s, %s>, %s", self.tx_name, self.rx_name, str(self.perf_metrics))
        LOGGER.info("rtt %f, rttdev %f", self.rtt, self.rttdev)
        LOGGER.info('AVG Rate in Mbps %f', self.next_chunk_seq * 8 / 1000 / (self.perf_metrics['finish_time'] - self.perf_metrics['start_time']))
        #with open('cc_state_trace.txt', 'w') as f:
        #    print(self.cc_state_trace, file=f)


class TxThread(threading.Thread):
    def __init__(self, name, use_mp=True, **param):
        super().__init__()
        self.peers = {}
        self.name = name
        self.param = param
        self.clock = None
        self.chunks = None
        self.use_mp = use_mp

        if self.use_mp:
            self.proc = multiprocessing.Process
        else:
            self.proc = threading.Thread
        
    def add_peer(self, name, host, port):
        LOGGER.debug("Adding peer %s (%s:%d)...", name, host, port)
        conn_state = dict(rx_name=name, host=host, port=port, tx_name=self.name)#, cwnd=None, cc_state=None, sshreshold=None)
        self.peers[name] = conn_state
        LOGGER.debug("peer %s added.", name)

    def remove_peer(self, name):
        LOGGER.debug("Removing peer %s...", name)
        peer = self.peers[name]
        # TODO:
        del self.peers[name]
        LOGGER.debug("peer %s removed.", name)
    
    def start_transfer(self, name):
        conn_state = self.peers[name]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        protocol = TxProtocol(
            self.tx_queue, clock=self.clock, chunks=self.chunks,
            acceptable_loss_rate=self.param.get('acceptable_loss_rate', 0.0), 
            **conn_state)
        protocol.connection_made(sock, addr=(conn_state['host'], conn_state['port']))
        sock.settimeout(protocol.rto)        
        while protocol.is_active():
            try:
                #sock.settimeout(protocol.rto)
                data, addr = sock.recvfrom(MSG.MAX_SIZE)
            except Exception as e:
                # timeout error
                LOGGER.info('# exception when waiting for chunk acks: {0}, last ack {1}'.format(e, protocol.last_receved_msg))
                protocol.handle_ack_timeout()
                continue
            protocol.datagram_received(data, addr)
        # update self.peers with new init_cwnd and sshrehold
        # update cwnd=None, sshreshold=None, which would be used for next ....
        self.peers[name]["init_cwnd"] = protocol.cwnd
        self.peers[name]["init_sshreshold"] = protocol.sshreshold
        protocol.show_perf()

    def send_chunks(self, clock, chunks, show_perf=False):
        self.clock = clock
        self.chunks = chunks
        
        self.transfers = []
        self.tx_queue = queue.Queue()

        # send payload to all the peers
        transfers = [self.proc(target=lambda:self.start_transfer(name)) for name in self.peers]
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

if __name__ == '__main__':
    # simple test
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--rx', action='store_true')
    parser.add_argument('--tx', action='store_true')
    parser.add_argument('--rx-ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=10000)
    parser.add_argument('--tx-num', type=int, default=1)
    parser.add_argument('--acceptable-loss-rate', type=float, default=0.3, help='Due to estimation errors, the acceptable loss rate should be slightly larger than the loss rate of the channel')
    

    args = parser.parse_args()

    import sys
    logging.basicConfig(format="%(thread)d:%(message)s", stream=sys.stdout, level=logging.INFO)

    port = args.port
    if args.rx:
        r1 = RxThread(port)
        r1.start()
    
    if args.tx:
        chunks = [bytes(1024) for _ in range(1000 * 1000)] #0 * 1000)]#1000)]
        # note that  because of the error of estimation, 
        # the acceptable loss rate should be slightly larger than the loss rate of the channel.
        tx = TxThread('tx', acceptable_loss_rate=args.acceptable_loss_rate) 
        for i in range(args.tx_num):
            tx.add_peer('r{0}'.format(i+1), host=args.rx_ip, port=port)

        tx.send_chunks(clock=1, chunks=chunks, show_perf=True)
        print('tx done')

    if args.rx:
        while True:
            ret = r1.fetch_wait()
            print('# Get clock', ret[0], 'with chunks num', len(ret[1]))
            #r1.running = False
            #r1.shutdown()
