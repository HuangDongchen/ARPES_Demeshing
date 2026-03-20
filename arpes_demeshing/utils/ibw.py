"""
IBW (Igor Binary Wave v5) <-> NumPy conversion.
Public API: ibw_to_mat(filepath) and mat_to_ibw(name, path, mat, xs, xd, ys, yd, zs, zd).
"""
from __future__ import annotations
import os, struct
import numpy as np
from typing import List, Tuple

MAXDIMS = 4
NT_CMPLX     = 0x01
NT_FP32      = 0x02
NT_FP64      = 0x04
NT_I8        = 0x08
NT_I16       = 0x10
NT_I32       = 0x20
NT_UNSIGNED  = 0x40

WAVEHEADER5_SIZE = 320
BINHEADER5_SIZE = 64

def _dtype_to_typecode(dtype: np.dtype):
    dt = np.dtype(dtype)
    if dt == np.dtype('float32'):
        return NT_FP32, 4
    if dt == np.dtype('float64'):
        return NT_FP64, 8
    if dt == np.dtype('int8'):
        return NT_I8, 1
    if dt == np.dtype('int16'):
        return NT_I16, 2
    if dt == np.dtype('int32'):
        return NT_I32, 4
    if dt == np.dtype('uint8'):
        return NT_I8 | NT_UNSIGNED, 1
    if dt == np.dtype('uint16'):
        return NT_I16 | NT_UNSIGNED, 2
    if dt == np.dtype('uint32'):
        return NT_I32 | NT_UNSIGNED, 4
    raise ValueError(f"Unsupported dtype for IBW v5: {dt}")

def _compute_checksum(bh: bytes, wh: bytes) -> int:
    data = bh + wh
    words = struct.unpack('<' + 'h' * ((BINHEADER5_SIZE + WAVEHEADER5_SIZE)//2), data[:(BINHEADER5_SIZE + WAVEHEADER5_SIZE)])
    s = sum(words) & 0xFFFF
    checksum = (-s) & 0xFFFF
    return checksum

def mat_to_ibw(name: str, path: str, mat: np.ndarray,
               xs: float=0, xd: float=1, ys: float=0, yd: float=1, zs: float=0, zd: float=1,
               dtype: str | None = None) -> str:
    if dtype is not None:
        mat = mat.astype(dtype, copy=False)
    mat = np.asarray(mat, order='F')
    shape = list(mat.shape)
    ndim = len(shape)
    if not (1 <= ndim <= MAXDIMS):
        raise ValueError("Only 1..4D arrays are supported for IBW v5")
    dims = shape + [0] * (MAXDIMS - ndim)

    igor_type, bpp = _dtype_to_typecode(mat.dtype)
    npnts = int(np.prod(shape))

    wh = bytearray(WAVEHEADER5_SIZE)
    wh[12:16] = struct.pack('<i', npnts)          # npnts
    wh[16:18] = struct.pack('<H', igor_type)      # type
    wh[26:28] = struct.pack('<H', 1)              # whVersion
    wname_bytes = name.encode('utf-8')[:31]
    wh[28:60] = wname_bytes.ljust(32, b'\x00')
    for i in range(MAXDIMS):
        struct.pack_into('<i', wh, 68 + 4*i, dims[i])
    sfA = [xd, yd, zd, 1.0]
    sfB = [xs, ys, zs, 0.0]
    for i in range(MAXDIMS):
        struct.pack_into('<d', wh, 84 + 8*i, float(sfA[i]))
        struct.pack_into('<d', wh, 116 + 8*i, float(sfB[i]))

    data_bytes = npnts * bpp
    bh = bytearray(BINHEADER5_SIZE)
    struct.pack_into('<h', bh, 0, 5)                 # version
    struct.pack_into('<i', bh, 4, WAVEHEADER5_SIZE + data_bytes)  # wfmSize
    for off in (8, 12, 16):
        struct.pack_into('<i', bh, off, 0)
    for i in range(8):
        struct.pack_into('<i', bh, 20 + 4*i, 0)
    struct.pack_into('<i', bh, 52, 0)
    struct.pack_into('<i', bh, 56, 0)
    struct.pack_into('<i', bh, 60, 0)

    chk = _compute_checksum(bytes(bh), bytes(wh))
    struct.pack_into('<H', bh, 2, chk)

    os.makedirs(path, exist_ok=True)
    fullpath = os.path.join(path, f"{name}.ibw")
    with open(fullpath, 'wb') as f:
        f.write(bh); f.write(wh); f.write(np.ascontiguousarray(mat).tobytes(order='F'))
    return fullpath

def ibw_to_mat(filepath: str):
    with open(filepath, 'rb') as f:
        bh = f.read(BINHEADER5_SIZE); wh = f.read(WAVEHEADER5_SIZE)
        dims = [struct.unpack('<i', wh[68 + 4*i:72 + 4*i])[0] for i in range(MAXDIMS)]
        dims = [d for d in dims if d > 0]
        ndim = len(dims)
        npnts = struct.unpack('<i', wh[12:16])[0]
        data_type = struct.unpack('<H', wh[16:18])[0]

        is_unsigned = bool(data_type & 0x40)
        base_type = data_type & ~(0x01 | 0x40)

        if base_type == NT_FP32:
            dtype = np.dtype('<f4')
        elif base_type == NT_FP64:
            dtype = np.dtype('<f8')
        elif base_type == NT_I8:
            dtype = np.dtype('<u1' if is_unsigned else '<i1')
        elif base_type == NT_I16:
            dtype = np.dtype('<u2' if is_unsigned else '<i2')
        elif base_type == NT_I32:
            dtype = np.dtype('<u4' if is_unsigned else '<i4')
        else:
            raise ValueError(f"Unsupported IBW data type bitmask: {data_type}")
        points = int(np.prod(dims)) if ndim>0 else npnts
        data = f.read(points * dtype.itemsize)

    arr = np.frombuffer(data, dtype=dtype)
    mat = arr.reshape(dims, order='F') if ndim>0 else arr

    axes = []
    for i in range(len(dims)):
        delta = struct.unpack('<d', wh[84 + 8*i : 92 + 8*i])[0]
        offset = struct.unpack('<d', wh[116 + 8*i : 124 + 8*i])[0]
        axes.append(offset + delta * np.arange(dims[i]))
    return mat, axes
