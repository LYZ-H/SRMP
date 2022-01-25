import logging
import struct

# packet struct
# <pkt_type, seq, chunk_offset, pkt_payload>


class P2MMSG:
    MAX_SIZE = 150000
    AVG_SIZE = 105000

    # at the begin, each  receiver joins the task by sending a join message specifying its receiver id
    # On getting the chunk indiciating itself to ack, the receiver knows that it joined successful
    JOIN = 1  # REQ
    # when sending a chunk, the receiver_to_ack is set randomly,
    # or respecting the newly joined receiver, if there are.
    CHUNK = 2
    # same tp chunk, however, the payload of chunk_fix might be a xor of multiple chunks.
    CHUNK_FIX = 3  # list of chunk_seq, chunk_value
    # on receiving the chunk, the selected_receiver will ack, indicting the current received chunk_seq,
    # along with all the missed chunk_seq from last ack
    CHUNK_FEEDBACK = 4

    LEAVE = 5

    FIN = 6

    PRE_FB_FIN = 7

    ENCODE_METHODS = {}
    DECODE_METHODS = {}

    INT8 = 1
    UINT8 = 2
    INT16 = 3
    UINT16 = 4
    INT32 = 5
    UINT32 = 6
    INT64 = 7
    UINT64 = 8
    FLOAT16 = 9
    FLOAT32 = 10

    def __init__(self,
                 mtype=None,
                 clock=None,
                 receiver_id=None,
                 total_chunk=None,
                 seq=None,
                 chunk_seq=None,
                 chunk_seqs=[],
                 chunk_data=None,
                 chunk_bytes=None,
                 chunk_dtype=None,
                 frombuffer=None):

        self.mtype = mtype
        self.clock = clock
        self.receiver_id = receiver_id

        self.chunk_dtype = chunk_dtype
        self.total_chunk = total_chunk

        self.chunk_seq = chunk_seq
        self.chunk_seqs = chunk_seqs
        self.chunk_seqs_len = len(chunk_seqs)

        self.chunk_data = chunk_data
        self.chunk_bytes = chunk_bytes
        self.seq = seq
        self.buf = frombuffer

        if self.buf is not None:
            self.decode()
        else:
            self.encode()

    def __str__(self):
        return "{0}, {1}, {2}".format(self.mtype, self.seq, self.chunk_seq)

    def __getattr__(self, name):
        if name not in self.__dict__:
            return False
        return self.__dict__[name]

    def tobytes(self):
        return self.buf

    def encode(self):
        if self.buf is None:
            self.ENCODE_METHODS[self.mtype](self)
        return self.buf

    def decode(self):
        mtype = struct.unpack('!B', self.buf[:1])[0]
        self.DECODE_METHODS[mtype](self)

    def encode_chunk(self):

        # mtype, clock, total_chunk, receiver_to_ack, seq, chunk_offset, number_of_fix_chunk, chunk_seqss, chunk_data
        h1 = struct.pack("!BIIIIII", self.mtype, self.clock, self.total_chunk,
                         self.receiver_id, self.seq, self.chunk_seq,
                         self.chunk_seqs_len)
        h2 = struct.pack("I" * self.chunk_seqs_len, *self.chunk_seqs)
        self.buf = h1 + h2 + self.chunk_data

    ENCODE_METHODS[CHUNK] = encode_chunk

    CHUNK_FIX_HEADER1_OFFSET = struct.calcsize("!BIIIIII")

    def decode_chunk(self):
        self.is_CHUNK_FIX = True
        self.mtype, self.clock, self.total_chunk, self.receiver_id, self.seq, self.chunk_seq, self.chunk_seqs_len = struct.unpack(
            "!BIIIIII", self.buf[:self.CHUNK_FIX_HEADER1_OFFSET])
        struct_format_str = "I" * self.chunk_seqs_len
        offset = struct.calcsize(
            'I') * self.chunk_seqs_len + self.CHUNK_FIX_HEADER1_OFFSET
        self.chunk_seqs = struct.unpack(
            struct_format_str, self.buf[self.CHUNK_FIX_HEADER1_OFFSET:offset])
        self.chunk_data = self.buf[offset:]

    DECODE_METHODS[CHUNK] = decode_chunk

    def encode_chunk_feedback(self):
        # mtype, clock, receiver_id, last_received_seq, number_of_lost, list_of_lost,
        self.buf = (
            struct.pack("!BIIII", self.mtype, self.clock, self.receiver_id,
                        self.last_received_chunk_seq, self.lost_chunk_num) +
            struct.pack("I" * self.lost_chunk_num, *self.list_of_lost_chunk))

    ENCODE_METHODS[CHUNK_FEEDBACK] = encode_chunk_feedback
    ENCODE_METHODS[JOIN] = encode_chunk_feedback

    CHUNK_FEEADBACK_HEADER_OFFSET = struct.calcsize("!BIIII")

    def decode_chunk_feedback(self):
        self.is_CHUNK_FEEDBACK = True
        self.mtype, self.clock, self.receiver_id, self.last_received_chunk_seq, self.lost_chunk_num = struct.unpack(
            "!BIIII", self.buf[:self.CHUNK_FEEADBACK_HEADER_OFFSET])
        self.list_of_lost_chunk = struct.unpack(
            'I' * self.lost_chunk_num,
            self.buf[self.CHUNK_FEEADBACK_HEADER_OFFSET:])

    DECODE_METHODS[CHUNK_FEEDBACK] = decode_chunk_feedback

    def encode_fin(self):
        self.buf = struct.pack("!BII", self.mtype, self.clock, self.peer)

    ENCODE_METHODS[FIN] = encode_fin

    def decode_fin(self):
        self.mtype, self.clock, self.peer = struct.unpack("!BII", self.buf)

    DECODE_METHODS[FIN] = decode_fin

    def decode_join(self):
        self.decode_chunk_feedback()
        self.is_JOIN = True

    DECODE_METHODS[JOIN] = decode_join

    def encode_leave(self):
        self.buf = struct.pack("!BII", self.mtype, self.clock,
                               self.receiver_id)

    ENCODE_METHODS[LEAVE] = encode_leave

    def decode_leave(self):
        self.is_LEAVE = True
        self.mtype, self.clock, self.receiver_id = struct.unpack(
            "!BII", self.buf)

    DECODE_METHODS[LEAVE] = decode_leave

    def encode_pre_fb_fin(self):
        self.buf = struct.pack("!B", self.mtype)

    ENCODE_METHODS[PRE_FB_FIN] = encode_pre_fb_fin

    def decode_pre_fb_fin(self):
        self.mtype = struct.unpack("!B", self.buf)[0]

    DECODE_METHODS[PRE_FB_FIN] = decode_pre_fb_fin


class P2PMSG:
    MAX_SIZE = 150000
    AVG_SIZE = 105000

    SYN = 1
    SYN_ACK = 2
    CHUNK = 3
    CHUNK_ACK = 4
    FIN = 5
    FIN_ACK = 6
    FIN_ACK_MUL = 7
    CHUNK_FEEDBACK = 8
    PRE_FIN = 9
    PRE_FB_ACK = 10
    ENCODE_METHODS = {}
    DECODE_METHODS = {}

    INT8 = 1
    UINT8 = 2
    INT16 = 3
    UINT16 = 4
    INT32 = 5
    UINT32 = 6
    INT64 = 7
    UINT64 = 8
    FLOAT16 = 9
    FLOAT32 = 10

    def __init__(self,
                 name=None,
                 mtype=None,
                 clock=None,
                 seq=None,
                 chunk_seq=None,
                 chunk_data=None,
                 chunk_bytes=None,
                 chunk_num=None,
                 chunk_dtype=None,
                 frombuffer=None,
                 chunk_seqs=None,
                 chunk_seqs_len=None,
                 total_chunk=None,
                 last_received_chunk_seq=None,
                 receiver_id=None,
                 peer=None):

        self.mtype = mtype
        self.name = name
        self.clock = clock
        self.chunk_dtype = chunk_dtype
        self.chunk_num = chunk_num

        self.chunk_data = chunk_data
        self.chunk_seq = chunk_seq
        self.chunk_bytes = chunk_bytes
        self.seq = seq
        self.buf = frombuffer

        self.peer = peer

        self.receiver_id = receiver_id

        self.chunk_seqs = chunk_seqs
        self.chunk_seqs_len = chunk_seqs_len

        self.lost_chunk_num = self.chunk_seqs_len
        self.list_of_lost_chunk = self.chunk_seqs

        self.total_chunk = total_chunk

        self.last_received_chunk_seq = last_received_chunk_seq

        if self.buf is not None:
            self.decode()
        else:
            self.encode()

    def __str__(self):
        return "{0}, {1}, {2}, {3}, {6}".format(self.mtype, self.seq,
                                                self.chunk_seq, self.peer,
                                                self.chunk_bytes, self.clock,
                                                self.chunk_dtype)

    def __getattr__(self, name):
        if name not in self.__dict__:
            return False
        return self.__dict__[name]

    def name_to_int(self):
        if self.peer == 'ps':
            self.peer = 0
        else:
            self.peer = int(self.peer[1:])

    def int_to_name(self):
        if self.peer == 0:
            self.peer = 'ps'
        else:
            self.peer = "w" + str(self.peer)

    def tobytes(self):
        return self.buf

    def encode(self):
        if self.buf is None:
            self.ENCODE_METHODS[self.mtype](self)
        return self.buf

    def decode(self):
        mtype = struct.unpack('!B', self.buf[:1])[0]
        self.DECODE_METHODS[mtype](self)

    def encode_syn(self):
        self.name_to_int()

        self.buf = struct.pack("!BIIIBI", self.mtype, self.clock,
                               self.chunk_num, self.chunk_bytes,
                               self.chunk_dtype, self.peer)

    ENCODE_METHODS[SYN] = encode_syn

    def decode_syn(self):

        self.is_SYN = True

        self.mtype, self.clock, self.chunk_num, self.chunk_bytes, self.chunk_dtype, self.peer = struct.unpack(
            "!BIIIBI", self.buf)

        self.int_to_name()

    DECODE_METHODS[SYN] = decode_syn

    def encode_syn_ack(self):
        self.buf = struct.pack("!BI", self.mtype, self.clock)

    ENCODE_METHODS[SYN_ACK] = encode_syn_ack

    def decode_syn_ack(self):
        self.is_SYN_ACK = True
        self.mtype, self.clock = struct.unpack("!BI", self.buf)

    DECODE_METHODS[SYN_ACK] = decode_syn_ack

    def encode_pre_fb_ack(self):
        self.name_to_int()
        self.buf = struct.pack("!BI", self.mtype, self.peer)

    ENCODE_METHODS[PRE_FB_ACK] = encode_pre_fb_ack

    def decode_pre_fb_ack(self):
        self.mtype, self.peer = struct.unpack("!BI", self.buf)
        self.int_to_name()

    DECODE_METHODS[PRE_FB_ACK] = decode_pre_fb_ack

    def encode_pre_fin(self):
        self.name_to_int()
        self.buf = struct.pack("!BI", self.mtype, self.peer)

    ENCODE_METHODS[PRE_FIN] = encode_pre_fin

    def decode_pre_fin(self):
        self.mtype, self.peer = struct.unpack("!BI", self.buf)
        self.int_to_name()

    DECODE_METHODS[PRE_FIN] = decode_pre_fin

    def encode_fin(self):
        self.name_to_int()

        self.buf = struct.pack("!BII", self.mtype, self.clock, self.peer)

    ENCODE_METHODS[FIN] = encode_fin

    def decode_fin(self):
        self.is_FIN = True
        self.mtype, self.clock, self.peer = struct.unpack("!BII", self.buf)
        self.int_to_name()

    DECODE_METHODS[FIN] = decode_fin

    def encode_fin_ack(self):
        self.buf = struct.pack("!BI", self.mtype, self.clock)

    ENCODE_METHODS[FIN_ACK] = encode_fin_ack

    def decode_fin_ack(self):
        self.is_FIN_ACK = True
        self.mtype, self.clock = struct.unpack("!BI", self.buf)

    DECODE_METHODS[FIN_ACK] = decode_fin_ack

    def encode_fin_ack_mul(self):
        self.name_to_int()
        self.buf = struct.pack("!BII", self.mtype, self.peer, self.clock)

    ENCODE_METHODS[FIN_ACK_MUL] = encode_fin_ack_mul

    def decode_fin_ack_mul(self):
        self.is_FIN_ACK_MUL = True
        self.mtype, self.peer, self.clock = struct.unpack("!BII", self.buf)
        self.int_to_name()

    DECODE_METHODS[FIN_ACK_MUL] = decode_fin_ack_mul

    def encode_chunk(self):
        self.name_to_int()

        # HITS:  self.clock,
        # self.buf = struct.pack("!BII", self.mtype, self.seq, self.chunk_seq) + self.chunk_data.to_bytes(2, byteorder='big')
        self.buf = struct.pack("!BIII", self.mtype, self.peer, self.seq,
                               self.chunk_seq) + self.chunk_data

    ENCODE_METHODS[CHUNK] = encode_chunk

    CHUNK_HEADER_OFFSET = struct.calcsize("!BIII")

    def decode_chunk(self):

        self.is_CHUNK = True
        self.mtype, self.peer, self.seq, self.chunk_seq = struct.unpack(
            "!BIII", self.buf[:self.CHUNK_HEADER_OFFSET])
        self.int_to_name()
        self.chunk_data = self.buf[self.CHUNK_HEADER_OFFSET:]

    DECODE_METHODS[CHUNK] = decode_chunk

    def encode_chunk_ack(self):
        self.name_to_int()
        self.buf = struct.pack("!BIII", self.mtype, self.peer, self.seq,
                               self.chunk_seq)

    ENCODE_METHODS[CHUNK_ACK] = encode_chunk_ack

    def decode_chunk_ack(self):

        self.is_CHUNK_ACK = True

        self.mtype, self.peer, self.seq, self.chunk_seq = struct.unpack(
            "!BIII", self.buf)
        self.int_to_name()

    DECODE_METHODS[CHUNK_ACK] = decode_chunk_ack

    def encode_chunk_feedback(self):
        self.name_to_int()
        #print(self.chunk_seqs)
        self.buf = (
            struct.pack("!BIIIIII", self.mtype, self.clock, self.receiver_id,
                        self.peer, self.last_received_chunk_seq,
                        self.total_chunk, self.chunk_seqs_len) +
            struct.pack("I" * self.chunk_seqs_len, *self.chunk_seqs))

    ENCODE_METHODS[CHUNK_FEEDBACK] = encode_chunk_feedback

    CHUNK_FEEADBACK_HEADER_OFFSET = struct.calcsize("!BIIIIII")

    def decode_chunk_feedback(self):
        self.is_CHUNK_FEEDBACK = True
        self.mtype, self.clock, self.receiver_id, self.peer, self.last_received_chunk_seq, self.total_chunk, self.chunk_seqs_len = struct.unpack(
            "!BIIIIII", self.buf[:self.CHUNK_FEEADBACK_HEADER_OFFSET])
        self.chunk_seqs = struct.unpack(
            'I' * self.chunk_seqs_len,
            self.buf[self.CHUNK_FEEADBACK_HEADER_OFFSET:])
        self.int_to_name()

    DECODE_METHODS[CHUNK_FEEDBACK] = decode_chunk_feedback
