from google.protobuf import any_pb2 as _any_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FromImageRequest(_message.Message):
    __slots__ = ["init_img_binary_data", "params"]
    class ParamsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    INIT_IMG_BINARY_DATA_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    init_img_binary_data: bytes
    params: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, init_img_binary_data: _Optional[bytes] = ..., params: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class FromImageandTextRequest(_message.Message):
    __slots__ = ["caption", "init_img_binary_data", "params"]
    class ParamsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    CAPTION_FIELD_NUMBER: _ClassVar[int]
    INIT_IMG_BINARY_DATA_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    caption: str
    init_img_binary_data: bytes
    params: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, init_img_binary_data: _Optional[bytes] = ..., caption: _Optional[str] = ..., params: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class FromImgOnlyRequest(_message.Message):
    __slots__ = ["init_img_binary_data"]
    INIT_IMG_BINARY_DATA_FIELD_NUMBER: _ClassVar[int]
    init_img_binary_data: bytes
    def __init__(self, init_img_binary_data: _Optional[bytes] = ...) -> None: ...

class FromImgandMaskandTextRequest(_message.Message):
    __slots__ = ["caption", "init_img_binary_data", "mask", "params"]
    class ParamsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    CAPTION_FIELD_NUMBER: _ClassVar[int]
    INIT_IMG_BINARY_DATA_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    caption: str
    init_img_binary_data: bytes
    mask: bytes
    params: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, init_img_binary_data: _Optional[bytes] = ..., mask: _Optional[bytes] = ..., caption: _Optional[str] = ..., params: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class FromTextRequest(_message.Message):
    __slots__ = ["caption", "params"]
    class ParamsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    CAPTION_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    caption: str
    params: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, caption: _Optional[str] = ..., params: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class ImageReply(_message.Message):
    __slots__ = ["binary_data", "h", "w"]
    BINARY_DATA_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    binary_data: bytes
    h: int
    w: int
    def __init__(self, w: _Optional[int] = ..., h: _Optional[int] = ..., binary_data: _Optional[bytes] = ...) -> None: ...

class IntReply(_message.Message):
    __slots__ = ["c"]
    C_FIELD_NUMBER: _ClassVar[int]
    c: int
    def __init__(self, c: _Optional[int] = ...) -> None: ...

class TestRequest(_message.Message):
    __slots__ = ["str"]
    STR_FIELD_NUMBER: _ClassVar[int]
    str: str
    def __init__(self, str: _Optional[str] = ...) -> None: ...

class TextReply(_message.Message):
    __slots__ = ["caption"]
    CAPTION_FIELD_NUMBER: _ClassVar[int]
    caption: str
    def __init__(self, caption: _Optional[str] = ...) -> None: ...
