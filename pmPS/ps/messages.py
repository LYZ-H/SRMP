import logging
import struct

# packet struct
# <pkt_type, seq, chunk_offset, pkt_payload>

class MSG:
    MAX_SIZE = 1500
    AVG_SIZE = 1050

    SYN = 1
    SYN_ACK = 2
    CHUNK = 3
    CHUNK_ACK = 4
    FIN = 5
    FIN_ACK = 6
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
            mtype=None, clock=None, seq=None,
            chunk_seq=None, chunk_data=None, chunk_bytes=None,
            chunk_num=None, chunk_dtype=None,
            frombuffer=None):

        self.mtype = mtype

        self.clock = clock
        self.chunk_dtype = chunk_dtype
        self.chunk_num = chunk_num
        
        self.chunk_data = chunk_data
        self.chunk_seq = chunk_seq
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
        
    def encode_syn(self):
        self.buf = struct.pack("!BIIIB", self.mtype, self.clock, self.chunk_num, self.chunk_bytes, self.chunk_dtype)
    ENCODE_METHODS[SYN] = encode_syn

    def decode_syn(self):
        self.is_SYN = True
        self.mtype, self.clock, self.chunk_num, self.chunk_bytes, self.chunk_dtype = struct.unpack("!BIIIB", self.buf)
    DECODE_METHODS[SYN] = decode_syn

    def encode_syn_ack(self):
        self.buf = struct.pack("!BI", self.mtype, self.clock)
    ENCODE_METHODS[SYN_ACK] = encode_syn_ack

    def decode_syn_ack(self):
        self.is_SYN_ACK = True
        self.mtype, self.clock = struct.unpack("!BI", self.buf)        
    DECODE_METHODS[SYN_ACK] = decode_syn_ack

    def encode_fin(self):
        #print(self.mtype, self.clock, self.chunk_num, self.chunk_dtype)
        self.buf = struct.pack("!BI", self.mtype, self.clock)
    ENCODE_METHODS[FIN] = encode_fin

    def decode_fin(self):
        self.is_FIN = True
        self.mtype, self.clock = struct.unpack("!BI", self.buf)
    DECODE_METHODS[FIN] = decode_fin

    def encode_fin_ack(self):
        self.buf = struct.pack("!BI", self.mtype, self.clock)
    ENCODE_METHODS[FIN_ACK] = encode_fin_ack

    def decode_fin_ack(self):
        self.is_FIN_ACK = True
        self.mtype, self.clock = struct.unpack("!BI", self.buf)        
    DECODE_METHODS[FIN_ACK] = decode_fin_ack

    def encode_chunk(self):
        # HITS:  self.clock,
        self.buf = struct.pack("!BII", self.mtype, self.seq, self.chunk_seq) + self.chunk_data
    ENCODE_METHODS[CHUNK] = encode_chunk
    
    CHUNK_HEADER_OFFSET = struct.calcsize("!BII")
    def decode_chunk(self):
        self.is_CHUNK = True
        self.mtype, self.seq, self.chunk_seq = struct.unpack("!BII", self.buf[: self.CHUNK_HEADER_OFFSET])
        self.chunk_data = self.buf[self.CHUNK_HEADER_OFFSET:]
    DECODE_METHODS[CHUNK] = decode_chunk

    def encode_chunk_ack(self):
        self.buf = struct.pack("!BII", self.mtype, self.seq, self.chunk_seq)
    ENCODE_METHODS[CHUNK_ACK] = encode_chunk_ack
    
    def decode_chunk_ack(self):
        self.is_CHUNK_ACK = True
        self.mtype, self.seq, self.chunk_seq = struct.unpack("!BII", self.buf)
    DECODE_METHODS[CHUNK_ACK] = decode_chunk_ack
