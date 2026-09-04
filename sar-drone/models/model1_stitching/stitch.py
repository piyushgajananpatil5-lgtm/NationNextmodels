"""Panorama stitching for overlapping drone frames."""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


class Stitcher:
    """Wrap OpenCV's Stitcher with input validation and a callable API."""

    def __init__(self, mode: int = cv2.Stitcher_PANORAMA) -> None:
        self._stitcher = cv2.Stitcher_create(mode)

    def __call__(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """Return a stitched BGR panorama from at least two non-empty frames.

        Raises ValueError for unusable input and RuntimeError when OpenCV cannot
        find a panorama. The original frame arrays are never modified.
        """
        images = [frame for frame in frames if frame is not None and frame.size]
        if len(images) < 2:
            raise ValueError("stitching requires at least two non-empty frames")
        status, panorama = self._stitcher.stitch(images)
        if status != cv2.Stitcher_OK or panorama is None or panorama.size == 0:
            raise RuntimeError(f"OpenCV stitching failed with status {status}")
        return panorama

    def stitch(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """Named alias for callers that prefer an explicit method."""
        return self(frames)


def stitch_frames(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Convenience function that stitches a frame collection once."""
    return Stitcher()(frames)
