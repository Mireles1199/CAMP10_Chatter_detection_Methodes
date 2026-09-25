
from __future__ import annotations
import logging
from typing import List, Sequence, Tuple, Union
import numpy as np
from ..utils.opr import segment_opr, segment_signal_raw
from .entropy import entropy_from_segments, EntropyEstimator
from ..models.maxent import fit_maxent_gaussians, MaxEntModels

logger = logging.getLogger(__name__)

ArrayOrPieces = Union[np.ndarray, Sequence[np.ndarray]]


def _as_piece_list(arr: ArrayOrPieces) -> List[np.ndarray]:
    """Wrap a bare array as a single-piece list; pass an existing list through."""
    return [arr] if isinstance(arr, np.ndarray) else list(arr)


def _windows_and_entropy_pool(
    pieces: List[np.ndarray],
    pieces_t: List[np.ndarray],
    piece_ids: List[str] | None,
    N_seg: int,
    step: int | None,
    segmentation: str,
    N_samples_per_seg: int | None,
    estimator: EntropyEstimator | None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Window each piece independently, compute entropy per piece, and pool the
    resulting entropy/time arrays. Never concatenates raw signal across
    pieces -- that would let a fixed-length window straddle the seam between
    two physically-unrelated time spans (see COMMON_TEMPLATE.md §10).
    """
    H_parts: List[np.ndarray] = []
    t_mid_parts: List[np.ndarray] = []
    n_windows: List[int] = []

    for i, (arr, arr_t) in enumerate(zip(pieces, pieces_t)):
        pid = piece_ids[i] if piece_ids else f"piece[{i}]"
        if segmentation == "raw":
            segs, segs_t = segment_signal_raw(
                arr, arr_t, N_samples_per_seg=N_samples_per_seg, step=step
            )
        else:
            segs, segs_t = segment_opr(arr, arr_t, N_seg=N_seg, step=step)

        if len(segs) == 0:
            logger.warning(
                "piece %s shorter than one window (%d samples) - skipped", pid, len(arr)
            )
            n_windows.append(0)
            continue

        H_parts.append(entropy_from_segments(segs, estimator=estimator))
        t_mid_parts.append(np.array([np.mean(seg_t) for seg_t in segs_t]))
        n_windows.append(len(segs))

    H = np.concatenate(H_parts) if H_parts else np.array([])
    t_mid = np.concatenate(t_mid_parts) if t_mid_parts else np.array([])
    return H, t_mid, n_windows


def offline_train_maxent_sprt(
    opr_free: ArrayOrPieces,
    opr_chat: ArrayOrPieces,
    opr_t_free: ArrayOrPieces,
    opr_t_chat: ArrayOrPieces,
    N_seg: int,
    estimator: EntropyEstimator | None = None,
    step: int | None = None,
    segmentation: str = "opr",
    N_samples_per_seg: int | None = None,
    piece_ids_free: List[str] | None = None,
    piece_ids_chat: List[str] | None = None,
) -> Tuple[MaxEntModels, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Train Maximum Entropy SPRT models on offline OPR (Operating Point Range) data.

    This function segments operational data into windows, computes entropy metrics for each segment,
    and fits maximum entropy Gaussian models to distinguish between free and chatter states.

    :param opr_free: OPR signal representing the stable reference condition. A
        bare array is treated as one continuous piece; a list of arrays is
        treated as several physically-disjoint pieces, each windowed
        independently so no window straddles the seam between two pieces.
    :param opr_chat: OPR signal representing the chatter reference condition. Same shape rules as ``opr_free``.
    :param opr_t_free: Time vector(s) aligned with ``opr_free`` (same shape: bare array or list, matching).
    :param opr_t_chat: Time vector(s) aligned with ``opr_chat`` (same shape: bare array or list, matching).
    :param N_seg: Number of OPR samples grouped into each training segment (``segmentation="opr"``).
    :param estimator: Segment-to-entropy estimator. If ``None``, the default entropy estimator is used.
    :param step: Hop size in domain samples between consecutive segment starts.
        ``None`` (default) is equivalent to ``step = N_seg`` (no overlap).
    :param segmentation: ``"opr"`` (default) – use OPR-decimated arrays with
        :func:`segment_opr`; ``"raw"`` – use raw signal arrays directly with
        :func:`segment_signal_raw` (no prior OPR decimation needed).
    :param N_samples_per_seg: Length of each raw block in samples.  Required
        when ``segmentation="raw"``; ignored otherwise.
    :param piece_ids_free: Optional identifier per stable piece, used only for
        the too-short-for-one-window warning message.
    :param piece_ids_chat: Optional identifier per chatter piece, mirrors ``piece_ids_free``.

    Returns:
        Tuple[MaxEntModels, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
        Trained Gaussian models, pooled entropy sequence for stable segments,
        pooled entropy sequence for chatter segments, pooled midpoint
        timestamps for both segment sets, and the per-piece window count for
        each side (stable, chatter).

    Raises
    ------
    ValueError
        If insufficient segments are generated from the input data. Check N_seg and OPR data length.
    """

    if segmentation == "raw" and N_samples_per_seg is None:
        raise ValueError("N_samples_per_seg must be provided when segmentation='raw'.")

    free_pieces, free_pieces_t = _as_piece_list(opr_free), _as_piece_list(opr_t_free)
    chat_pieces, chat_pieces_t = _as_piece_list(opr_chat), _as_piece_list(opr_t_chat)

    # 1) Segmentation per piece (OPR or raw)  2) entropy per piece  -- pooled after
    H_free, t_mid_free, n_windows_free = _windows_and_entropy_pool(
        free_pieces, free_pieces_t, piece_ids_free, N_seg, step, segmentation, N_samples_per_seg, estimator
    )
    H_chat, t_mid_chat, n_windows_chat = _windows_and_entropy_pool(
        chat_pieces, chat_pieces_t, piece_ids_chat, N_seg, step, segmentation, N_samples_per_seg, estimator
    )

    if H_free.size == 0 or H_chat.size == 0:
        raise ValueError("Insufficient segments generated for training. Check N_seg and OPR data length.")

    # 3) Fitting of pdfs p0(H) and p1(H) on the pooled entropy values
    models = fit_maxent_gaussians(H_free, H_chat)

    return models, H_free, H_chat, t_mid_free, t_mid_chat, n_windows_free, n_windows_chat
