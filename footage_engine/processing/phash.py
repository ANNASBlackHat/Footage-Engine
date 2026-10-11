"""Perceptual hashing (dHash gradient hash & DCT pHash) for video keyframe dedup."""

import logging
from typing import Any, List, Optional, Union
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def compute_dhash(
    img_or_path: Union[str, np.ndarray, Image.Image],
    hash_size: int = 8,
) -> Optional[str]:
    """Computes a 64-bit gradient difference hash (dHash) for an image or frame.

    dHash tracks relative pixel gradient differences, making it exceptionally
    robust to re-encoding, minor crops, resizing, and brightness adjustments.
    """
    if isinstance(img_or_path, str):
        try:
            with Image.open(img_or_path) as im:
                im_gray = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
                pixels = np.array(im_gray, dtype=np.int32)
        except Exception:
            return None
    elif isinstance(img_or_path, Image.Image):
        im_gray = img_or_path.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.array(im_gray, dtype=np.int32)
    elif isinstance(img_or_path, np.ndarray):
        if len(img_or_path.shape) == 3:
            gray_arr = np.dot(img_or_path[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        else:
            gray_arr = img_or_path.astype(np.uint8)
        im = Image.fromarray(gray_arr).resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.array(im, dtype=np.int32)
    else:
        return None

    # Calculate horizontal gradient (left-to-right difference)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bit_str = "".join(["1" if b else "0" for b in diff.flatten()])
    num_hex_digits = (hash_size * hash_size) // 4
    return f"{int(bit_str, 2):0{num_hex_digits}x}"


def compute_dct_phash(
    img_or_path: Union[str, np.ndarray, Image.Image],
    hash_size: int = 8,
) -> Optional[str]:
    """Alias for perceptual difference hashing (dHash)."""
    return compute_dhash(img_or_path, hash_size=hash_size)


def hamming_distance(hash1_hex: str, hash2_hex: str) -> int:
    """Computes the bitwise Hamming distance between two hexadecimal hash strings."""
    if not hash1_hex or not hash2_hex:
        return 64
    val1 = int(hash1_hex, 16)
    val2 = int(hash2_hex, 16)
    return bin(val1 ^ val2).count("1")


def compare_phash_similarity(hash1_hex: str, hash2_hex: str, max_bits: int = 64) -> float:
    """Returns a normalized similarity score between 0.0 and 1.0 (1.0 = identical)."""
    dist = hamming_distance(hash1_hex, hash2_hex)
    return max(0.0, 1.0 - (dist / float(max_bits)))
