"""Read external-data references recorded inside an ONNX model without the onnx package.

ONNX Runtime は TensorProto.external_data の location を見て重みを読む. ファイル名からの推測
（model.onnx → model.onnx.data）は改名で外れるため、署名には記録された参照先を使う.
onnx パッケージはランタイム依存に含めないので、必要なフィールドだけを読む最小の protobuf 走査を行う.
"""

from __future__ import annotations

import mmap
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterator

# protobuf フィールド番号（onnx.proto3）
_MODEL_GRAPH = 7
_GRAPH_NODE, _GRAPH_INITIALIZER, _GRAPH_SPARSE_INITIALIZER = 1, 5, 15
_NODE_ATTRIBUTE = 5
_ATTR_TENSOR, _ATTR_GRAPH, _ATTR_TENSORS, _ATTR_GRAPHS = 5, 6, 10, 11
_ATTR_SPARSE_TENSOR, _ATTR_SPARSE_TENSORS = 22, 23
_SPARSE_VALUES, _SPARSE_INDICES = 1, 2
_TENSOR_EXTERNAL_DATA = 13
_ENTRY_KEY, _ENTRY_VALUE = 1, 2


def _varint(buf: memoryview, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(buf) or shift > 63:
            raise ValueError("truncated or invalid ONNX varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def _fields(buf: memoryview) -> Iterator[tuple[int, memoryview | None]]:
    """Yield (field number, payload) for length-delimited fields; other wire types are skipped."""
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            _, pos = _varint(buf, pos)
            yield number, None
        elif wire == 1:
            pos += 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            if pos + length > len(buf):
                raise ValueError("truncated ONNX field")
            yield number, buf[pos : pos + length]
            pos += length
        elif wire == 5:
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
    if pos != len(buf):
        raise ValueError("truncated ONNX message")


def _tensor_locations(tensor: memoryview, out: set[str]) -> None:
    for number, payload in _fields(tensor):
        if number != _TENSOR_EXTERNAL_DATA or payload is None:
            continue
        entry = {field: bytes(value).decode("utf-8") for field, value in _fields(payload) if value is not None}
        if entry.get(_ENTRY_KEY) == "location" and _ENTRY_VALUE in entry:
            out.add(entry[_ENTRY_VALUE])


def _sparse_locations(sparse: memoryview, out: set[str]) -> None:
    for number, payload in _fields(sparse):
        if number in (_SPARSE_VALUES, _SPARSE_INDICES) and payload is not None:
            _tensor_locations(payload, out)


def _graph_locations(graph: memoryview, out: set[str]) -> None:
    for number, payload in _fields(graph):
        if payload is None:
            continue
        if number == _GRAPH_INITIALIZER:
            _tensor_locations(payload, out)
        elif number == _GRAPH_SPARSE_INITIALIZER:
            _sparse_locations(payload, out)
        elif number == _GRAPH_NODE:
            for node_field, attribute in _fields(payload):
                if node_field == _NODE_ATTRIBUTE and attribute is not None:
                    _attribute_locations(attribute, out)


def _attribute_locations(attribute: memoryview, out: set[str]) -> None:
    for number, payload in _fields(attribute):
        if payload is None:
            continue
        if number in (_ATTR_TENSOR, _ATTR_TENSORS):
            _tensor_locations(payload, out)
        elif number in (_ATTR_GRAPH, _ATTR_GRAPHS):
            _graph_locations(payload, out)
        elif number in (_ATTR_SPARSE_TENSOR, _ATTR_SPARSE_TENSORS):
            _sparse_locations(payload, out)


def _validate_location(location: str) -> str:
    """ORT と同様にモデルディレクトリ外を指す参照は拒否する."""
    for flavour in (PurePosixPath(location), PureWindowsPath(location)):
        if not location or flavour.is_absolute() or flavour.drive or ".." in flavour.parts:
            raise ValueError(f"ONNX external data location must stay inside the model directory: {location!r}")
    return location


def _scan_graphs(mapped: mmap.mmap, found: set[str]) -> None:
    for number, payload in _fields(memoryview(mapped)):
        if number == _MODEL_GRAPH and payload is not None:
            _graph_locations(payload, found)


def _scan_model(mapped: mmap.mmap) -> set[str]:
    """
    mmap 由来の memoryview が残ると close 時に BufferError になる. 例外のトレースバックもフレーム経由で
    memoryview を保持するため、メッセージだけ取り出して except を抜け、参照を解放してから送出し直す.
    """
    found: set[str] = set()
    error: str | None = None
    try:
        _scan_graphs(mapped, found)
    except (ValueError, IndexError, UnicodeDecodeError) as exc:
        error = f"invalid ONNX model: {exc}"
    if error is not None:
        raise ValueError(error)
    return found


@lru_cache(maxsize=8)
def _locations_for(path: str, size: int, mtime_ns: int, ctime_ns: int) -> tuple[str, ...]:
    if size == 0:
        raise ValueError(f"ONNX model is empty: {path}")
    # 重み埋め込みモデルは数 GB になり得るため、全体を読み込まず mmap 上で長さ付きフィールドを読み飛ばす.
    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        found = _scan_model(mapped)
    return tuple(sorted(_validate_location(location) for location in found))


def external_data_locations(model_path: str | Path) -> tuple[str, ...]:
    """Return sorted relative locations referenced by the model; empty when all weights are embedded."""
    path = Path(model_path).expanduser().resolve()
    stat = path.stat()
    return _locations_for(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


__all__ = ["external_data_locations"]
