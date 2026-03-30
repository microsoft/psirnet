import ismrmrd
import struct
import io
from typing import BinaryIO, Iterable, Callable, Union
import numpy as np
from ismrmrd import Image
from enum import Enum
from dataclasses import dataclass

_DTYPE_MAP = {
    1:  np.uint16,
    2:  np.int16,
    3:  np.uint32,
    4:  np.int32,
    5:  np.float32,
    6:  np.float64,
    7:  np.complex64,
    8:  np.complex128,
}


@dataclass
class NDArray:
    """
    Represents a NumPy array for MRD communication.
    """
    version: int
    data_type: int
    dims: tuple[int, ...]
    data: np.ndarray


class MessageType(Enum):
    """
    Enumeration of possible message types for MRD communication.
    """
    UNPEEKED = 0
    CONFIG_FILE = 1
    CONFIG_TEXT = 2
    HEADER = 3
    CLOSE = 4
    TEXT = 5
    ACQUISITION = 1008
    IMAGE = 1022
    WAVEFORM = 1026
    NDARRAY = 1030


Message = Union[
    ismrmrd.xsd.ismrmrdHeader, ismrmrd.Acquisition,
    ismrmrd.Image, ismrmrd.Waveform, NDArray, str
]


class IsmrmrdSink(io.BytesIO):
    def __init__(self, output_writer: BinaryIO) -> None:
        super().__init__()
        self._output_writer = output_writer

    def _write_type(self, message_type: MessageType) -> None:
        self._output_writer.write(struct.pack('H', message_type.value))

    def _serialize_ndarray(self, ndarray: NDArray) -> None:
        self._output_writer.write(struct.pack('<HHH',
                                              ndarray.data_type,
                                              ndarray.version,
                                              len(ndarray.dims)))
        self._output_writer.write(
            struct.pack(f'<{len(ndarray.dims)}Q', *ndarray.dims)
        )
        self._output_writer.write(ndarray.data.tobytes(order='F'))

    def _run(self, seq: Iterable[Message]) -> Iterable:
        for message in seq:
            if isinstance(message, Image):
                self._write_type(MessageType.IMAGE)
                message.serialize_into(self._output_writer.write)
            elif isinstance(message, NDArray):
                self._write_type(MessageType.NDARRAY)
                self._serialize_ndarray(message)
            else:
                raise Exception(f"Unexpected message type {type(message)}")

        self._write_type(MessageType.CLOSE)
        self._output_writer.flush()
        return
        yield


class IsmrmrdSource(io.BytesIO):
    def __init__(self, input_reader: BinaryIO) -> None:
        super().__init__()
        self._input_reader = input_reader

    @staticmethod
    def _deserialize_ndarray(read: Callable[[int], bytes]) -> NDArray:
        """
        Deserialize an ISMRMRD NDArray message.
        """
        header = read(6)
        if len(header) != 6:
            raise EOFError(
                "Unexpected end of stream while reading NDArray header"
            )
        data_type, version, ndim = struct.unpack('<HHH', header)

        dims_bytes = read(8 * ndim)
        if len(dims_bytes) != 8 * ndim:
            raise EOFError(
                "Unexpected end of stream while reading NDArray dimensions"
            )
        dims = struct.unpack(f'<{ndim}Q', dims_bytes)

        dtype = _DTYPE_MAP[data_type]
        nbytes = int(np.prod(dims)) * np.dtype(dtype).itemsize
        payload = read(nbytes)
        if len(payload) != nbytes:
            raise EOFError(
                "Unexpected end of stream while reading NDArray data"
            )

        data = np.frombuffer(payload, dtype=dtype).reshape(dims, order='F')
        out = NDArray(
            version=version,
            data_type=data_type,
            dims=dims,
            data=data
        )
        return out

    def source(self) -> Iterable[Message]:
        while True:
            message_type = MessageType(
                struct.unpack('H', self._input_reader.read(2))[0]
            )
            if message_type == MessageType.HEADER:
                header_size = struct.unpack('I', self._input_reader.read(4))[0]
                doc = self._input_reader.read(header_size)
                yield ismrmrd.xsd.CreateFromDocument(doc)
            elif message_type == MessageType.ACQUISITION:
                yield ismrmrd.Acquisition.deserialize_from(
                    self._input_reader.read
                )
            elif message_type == MessageType.IMAGE:
                yield ismrmrd.Image.deserialize_from(
                    self._input_reader.read
                )
            elif message_type == MessageType.WAVEFORM:
                yield ismrmrd.Waveform.deserialize_from(
                    self._input_reader.read
                )
            elif message_type == MessageType.NDARRAY:
                yield self._deserialize_ndarray(self._input_reader.read)
            elif message_type in (MessageType.CONFIG_FILE,
                                  MessageType.CONFIG_TEXT,
                                  MessageType.TEXT):
                # all string-like messages share the same <len><payload> layout
                (txt_len,) = struct.unpack('<I', self._input_reader.read(4))
                txt = self._input_reader.read(txt_len).decode('utf-8')
                yield txt
            elif message_type == MessageType.CLOSE:
                if len(self._input_reader.read(1)) != 0:
                    raise Exception("Unexpected data after close message")
                return
            else:
                raise Exception(f"Unexpected message {message_type}")
