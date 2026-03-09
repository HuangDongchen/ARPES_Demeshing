# -*- coding: UTF-8 -*-
"""PXT (Igor Pro Packed Experiment Text) file reading utilities.

Provides functions to load .pxt files commonly used in ARPES experiments,
extract 2D spectrum data and wave names, and convert to standard txt format.

Dependencies: igor2 (pip install igor2)
"""
import os

import numpy as np


def load_pxt(path):
    """Read a .pxt file and return (2D_data, wave_name).

    Uses igor2 to parse the Igor Pro packed experiment binary format.
    Extracts the first wave's data array and its name.

    Args:
        path: Path to the .pxt file.

    Returns:
        tuple: (data, wave_name)
            - data: 2D numpy array of spectrum intensity values.
            - wave_name: str, the Igor wave name embedded in the file.
              Falls back to filename (without extension) if no name is found.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no wave data is found in the file.
    """
    import igor2.packed as packed

    if not os.path.exists(path):
        raise FileNotFoundError(f"PXT file not found: {path}")

    with open(path, 'rb') as f:
        records = packed.load(f)

    # records structure:
    #   records[0]: list of WaveRecord objects
    #   records[1]: filesystem dict {'root': {b'WaveName': WaveRecord, ...}}
    filesystem = records[1]
    root = filesystem.get('root', filesystem)

    for key, wave_rec in root.items():
        wave_name = key.decode('utf-8') if isinstance(key, bytes) else str(key)

        # Navigate: WaveRecord.wave -> {'version': int, 'wave': {'wData': ndarray, ...}}
        if hasattr(wave_rec, 'wave'):
            inner = wave_rec.wave
            wave_dict = inner.get('wave', {}) if isinstance(inner, dict) else {}
            if 'wData' in wave_dict:
                data = np.array(wave_dict['wData'])
                if data.ndim == 2:
                    return data, wave_name

    # Fallback: no valid 2D wave found
    raise ValueError(f"No 2D wave data found in PXT file: {path}")


def pxt_to_txt(pxt_path, txt_path=None):
    """Convert a .pxt file to standard txt format.

    Output format:
        - Line 1: wave name (header)
        - Remaining lines: tab-separated numeric data

    Args:
        pxt_path: Path to the source .pxt file.
        txt_path: Path for output .txt file. If None, replaces .pxt extension with .txt.

    Returns:
        str: Path to the written txt file.
    """
    data, wave_name = load_pxt(pxt_path)

    if txt_path is None:
        txt_path = os.path.splitext(pxt_path)[0] + '.txt'

    with open(txt_path, 'w') as f:
        f.write(wave_name + '\n')
        np.savetxt(f, data, fmt='%.6f', delimiter='\t')

    return txt_path
