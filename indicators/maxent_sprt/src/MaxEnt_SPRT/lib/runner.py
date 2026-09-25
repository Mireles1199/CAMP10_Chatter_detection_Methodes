from __future__ import annotations
import logging
import math
from typing import Any, Callable, Dict, List, Sequence, Optional, Tuple, Union

from collections import defaultdict

from ..logging_setup import _section
from ..utils.types import SignalData, IndicatorResult
from ..lib.detector import MaxEntSPRTConfig, MaxEntSPRTDetector
from ..lib.entropy import GaussianMaxEntEstimator, EmpiricalHistogramEntropyEstimator, entropy_from_segments
from ..utils.opr import sample_opr

import numpy as np

IndicatorFunc = Callable[..., IndicatorResult]
logger = logging.getLogger(__name__)

# ── Keys forwarded unchanged from params_physical to the native pipeline ──────
_MAXENT_PASS_THROUGH_PARAMS: frozenset = frozenset({
    "t_stable_total", "alpha", "beta", "reset_on_H0",
    "cut_start_time", "cut_end_time", "ratio_sampling", "step_seg", "segmentation",
    "use_sprt", "H_threshold", "training_intervals", "t_theorical",  # for debug/plots, not used in detection
})


def _resolve_physical_params_maxent(
    param_mode: str,
    params_physical: Dict[str, Any],
    fs: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Translate a physical parameter specification into native MaxEnt-SPRT parameters.

    Supports two physical modes (see ``indicators/COMMON_TEMPLATE.md`` for the
    contract shared with rms_cv/ssq_chatter/green_integral):

    * ``by_revolution`` – the analysis segment spans ``N_rev_window`` spindle
      revolutions.  ``T_rev`` (s) sets the spindle rotation period; ``step_rev``
      (revolutions) is the mandatory hop between consecutive segments.

    * ``by_modal`` – the segment duration spans ``N_modal_window`` modal
      (chatter) periods.  ``T_modal`` (s) sets the modal period; ``step_modal``
      (modal periods) is the mandatory hop. ``T_rev``, if given, is purely
      informational here and does not affect the window size.

    ``segmentation`` (default ``"opr"``) controls how fractional windows are
    handled: in ``"opr"`` mode ``N_*_window``/``step_*`` must be exact integers
    (one OPR sample per cycle), and a non-integer value raises ``ValueError``
    instead of being silently truncated; in ``"raw"`` mode fractional values
    are accepted and converted to a raw sample count via ``ceil``.

    All pass-through parameters (``alpha``, ``beta``, ``reset_on_H0``,
    ``t_stable_total``, ``cut_start_time``, ``cut_end_time``, ``ratio_sampling``,
    etc. – see ``_MAXENT_PASS_THROUGH_PARAMS``) are forwarded unchanged.

    :param param_mode: Either ``"by_revolution"`` or ``"by_modal"``.
    :param params_physical: Dictionary of physical parameters (mode-specific
        keys plus optional pass-through keys).
    :param fs: Sampling frequency of the signal.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            *native_params* – kwargs ready for ``_maxent_sprt_pipeline``;
            *trace* – traceability record (physical inputs, resolved native
            values, window unit bookkeeping, quantisation notes).

    Raises
    ------
    ValueError
        If required keys are missing or have inadmissible values.
    """
    # ── pass-through: copy keys the pipeline accepts directly ────────────────
    native_params: Dict[str, Any] = {
        k: v for k, v in params_physical.items()
        if k in _MAXENT_PASS_THROUGH_PARAMS
    }
    quantization_notes: List[str] = []

    # ── segmentation mode (default "opr" for backward compat) ────────────────
    segmentation: str = params_physical.get("segmentation", "opr")
    if segmentation not in ("opr", "raw"):
        raise ValueError(
            f"segmentation must be 'opr' or 'raw', got '{segmentation}'."
        )

    def _window_value(name: str, value: Any) -> float:
        value = float(value)
        if segmentation == "opr" and not value.is_integer():
            raise ValueError(
                f"{name} must be an integer when segmentation='opr' "
                f"(one OPR sample per cycle), got {value}."
            )
        return value

    if param_mode == "by_revolution":
        for key in ("T_rev", "N_rev_window", "step_rev"):
            if key not in params_physical:
                raise ValueError(
                    f"by_revolution mode requires '{key}' in params_physical."
                )

        T_rev = float(params_physical["T_rev"])
        if T_rev <= 0.0:
            raise ValueError(f"T_rev must be > 0, got {T_rev}.")

        N_win = _window_value("N_rev_window", params_physical["N_rev_window"])
        step  = _window_value("step_rev", params_physical["step_rev"])
        if N_win < 1:
            raise ValueError(f"N_rev_window must be >= 1, got {N_win}.")
        if not (0 < step <= N_win):
            raise ValueError(
                f"step_rev must satisfy 0 < step_rev <= N_rev_window={N_win}, got {step}."
            )

        rpm      = 60.0 / T_rev
        t_seg    = N_win * T_rev
        unit_name = "rev"
        T_unit    = T_rev

        native_params["rpm"]      = rpm
        native_params["N_seg"]    = int(N_win)
        native_params["step_seg"] = int(step) if segmentation == "opr" else step

        # raw segmentation: convert revolution counts to raw sample counts
        if segmentation == "raw":
            samples_per_rev          = fs * T_rev
            N_samples_per_seg        = int(math.ceil(N_win * samples_per_rev))
            step_samples             = int(math.ceil(step * samples_per_rev))
            native_params["N_seg"]    = int(math.ceil(N_win))
            native_params["N_samples_per_seg"] = N_samples_per_seg
            native_params["step_seg"] = step_samples             # override: hop in raw samples
            quantization_notes.append(
                f"raw mode: N_samples_per_seg = ceil({N_win} x {samples_per_rev:.1f}) = {N_samples_per_seg}"
            )
            quantization_notes.append(
                f"raw mode: step_samples = ceil({step} x {samples_per_rev:.1f}) = {step_samples}"
            )

        quantization_notes.append(
            f"t_seg = {N_win} x {T_rev:.6f} s = {t_seg:.6f} s"
        )
        quantization_notes.append(
            f"step_rev = {step}  (overlap = {1.0 - step/N_win:.1%})"
        )

    elif param_mode == "by_modal":
        for key in ("T_modal", "N_modal_window", "step_modal"):
            if key not in params_physical:
                raise ValueError(
                    f"by_modal mode requires '{key}' in params_physical."
                )

        T_modal = float(params_physical["T_modal"])
        if T_modal <= 0.0:
            raise ValueError(f"T_modal must be > 0, got {T_modal}.")

        # T_rev is optional in by_modal: informational only, not used for windowing.
        T_rev_raw = params_physical.get("T_rev")
        if T_rev_raw is not None:
            T_rev = float(T_rev_raw)
            if T_rev <= 0.0:
                raise ValueError(f"T_rev must be > 0, got {T_rev}.")
            quantization_notes.append(
                f"T_rev = {T_rev:.6f} s (informational only, not used for windowing)"
            )

        N_win = _window_value("N_modal_window", params_physical["N_modal_window"])
        step  = _window_value("step_modal", params_physical["step_modal"])
        if N_win < 1:
            raise ValueError(f"N_modal_window must be >= 1, got {N_win}.")
        if not (0 < step <= N_win):
            raise ValueError(
                f"step_modal must satisfy 0 < step_modal <= N_modal_window={N_win}, got {step}."
            )

        rpm_modal    = 60.0 / T_modal
        t_seg_target = N_win * T_modal
        t_seg_real   = math.ceil(t_seg_target * fs) / fs
        quant_err_s  = t_seg_real - t_seg_target
        quant_err_pct = abs(quant_err_s) / t_seg_target * 100.0
        unit_name = "modal"
        T_unit    = T_modal

        native_params["rpm"]      = rpm_modal
        native_params["N_seg"]    = int(N_win)
        native_params["step_seg"] = int(step) if segmentation == "opr" else step

        # raw segmentation: convert modal-period counts to raw sample counts
        if segmentation == "raw":
            samples_per_modal        = T_modal * fs
            N_samples_per_seg        = int(math.ceil(N_win * samples_per_modal))
            step_samples             = int(math.ceil(step * samples_per_modal))
            native_params["N_seg"]    = int(math.ceil(N_win))
            native_params["N_samples_per_seg"] = N_samples_per_seg
            native_params["step_seg"] = step_samples             # override: hop in raw samples
            quantization_notes.append(
                f"raw mode: N_samples_per_seg = ceil({N_win} x {samples_per_modal:.1f}) = {N_samples_per_seg}"
            )
            quantization_notes.append(
                f"raw mode: step_samples = ceil({step} x {samples_per_modal:.1f}) = {step_samples}"
            )

        quantization_notes.append(f"N_modal_window = {N_win}")
        quantization_notes.append(
            f"t_seg_target={t_seg_target:.6f} s | t_seg_real={t_seg_real:.6f} s"
            f" | delta={quant_err_s:+.6f} s ({quant_err_pct:.2f}%)"
        )
        quantization_notes.append(
            f"step_modal = {step}  (overlap = {1.0 - step/N_win:.1%})"
        )

    else:
        raise ValueError(
            f"Unknown param_mode '{param_mode}'. "
            "Valid options: 'native', 'by_revolution', 'by_modal'."
        )

    trace: Dict[str, Any] = {
        "physical_params_input":  dict(params_physical),
        "native_params_resolved": {"rpm": native_params["rpm"],
                                   "N_seg": native_params["N_seg"],
                                   "step_seg": native_params["step_seg"],
                                   "segmentation": segmentation,
                                   **({"N_samples_per_seg": native_params["N_samples_per_seg"]}
                                      if "N_samples_per_seg" in native_params else {})},
        "unit_name": unit_name,
        "T_unit": T_unit,
        "N_win": N_win,
        "step": step,
        "quantization_notes":     "; ".join(quantization_notes),
    }
    return native_params, trace


def run_maxent_sprt(signal: SignalData, INDICATOR_CONFIG: dict ) -> IndicatorResult:
    """
    Execute the Maximum Entropy Sequential Probability Ratio Test (MaxEnt SPRT) indicator.

    This function serves as a wrapper that retrieves the appropriate analysis function
    from the configuration and executes it with the provided signal data and parameters.

    ``INDICATOR_CONFIG["param_mode"]`` selects how the analysis window is
    specified (see ``indicators/COMMON_TEMPLATE.md`` for the full contract
    shared with rms_cv/ssq_chatter/green_integral):

    * ``"native"`` (default) – native kwargs for ``_maxent_sprt_pipeline`` go in
      ``INDICATOR_CONFIG["params"]``.
    * ``"by_revolution"`` / ``"by_modal"`` – physical parameters
      (``T_rev``/``N_rev_window``/``step_rev`` or
      ``T_modal``/``N_modal_window``/``step_modal``) go in
      ``INDICATOR_CONFIG["params_physical"]`` and are resolved to native kwargs
      via ``_resolve_physical_params_maxent``.

    :param signal: Input signal bundle containing the analysis array, aligned time axis, sampling frequency, and optional metadata.
    :param INDICATOR_CONFIG: Dispatcher dictionary containing the selected function under ``func``, the mode under ``param_mode``, its keyword arguments under ``params`` (native mode) or ``params_physical`` (physical modes), and an optional ``reference_signal`` (``SignalData`` or ``List[SignalData]``) to calibrate P0/stable training against externally-labelled "stable" reference piece(s) instead of internally-derived cuts — see ``indicators/COMMON_TEMPLATE.md``. ``reference_signal_chatter`` is MaxEnt's own mirror of it for P1/chatter (not part of the shared contract).

    Returns:
        IndicatorResult: Result object returned by the selected indicator
        function.

    Raises
    ------
    KeyError
        If required keys are missing from ``INDICATOR_CONFIG``.
    TypeError
        If the selected function is not callable or the signal is invalid.
    """

    param_mode: str = INDICATOR_CONFIG.get("param_mode", "native")
    fs = signal.fs

    func: IndicatorFunc = INDICATOR_CONFIG["func"]
    if func == "Default":
        func = _maxent_sprt_pipeline

    trace: Optional[Dict[str, Any]] = None
    params_physical: Dict[str, Any] = {}

    if param_mode == "native":
        params: Dict[str, Any] = INDICATOR_CONFIG.get("params", {})
    else:
        params_physical = INDICATOR_CONFIG["params_physical"]
        params, trace = _resolve_physical_params_maxent(param_mode, params_physical, fs)
        _phys_display = {
            k: v for k, v in trace["physical_params_input"].items()
            if k not in _MAXENT_PASS_THROUGH_PARAMS
        }
        logger.debug(
            "Physical parametrization [%s] -> native:\n"
            "  Physical input : %s\n"
            "  Native resolved: %s\n"
            "  Quantization   : %s",
            param_mode,
            _phys_display,
            trace["native_params_resolved"],
            trace["quantization_notes"],
        )

    reference_signal: Optional[Union[SignalData, List[SignalData]]] = INDICATOR_CONFIG.get("reference_signal")
    if reference_signal is not None:
        logger.debug("reference_signal provided: overrides internally-derived stable/training region.")
    # reference_signal_chatter is a MaxEnt-specific mirror of reference_signal for the
    # chatter/P1 side (not part of the shared COMMON_TEMPLATE contract).
    reference_signal_chatter: Optional[Union[SignalData, List[SignalData]]] = INDICATOR_CONFIG.get("reference_signal_chatter")
    extra_kwargs: Dict[str, Any] = {}
    if reference_signal is not None:
        extra_kwargs["reference_signal"] = reference_signal
    if reference_signal_chatter is not None:
        extra_kwargs["reference_signal_chatter"] = reference_signal_chatter

    result: IndicatorResult = func(signal, **params, **extra_kwargs)

    # ── Traceability: attach mode + standard meta keys (see COMMON_TEMPLATE.md) ──
    if param_mode == "native":
        f_cycle      = (params["rpm"] / 60.0) if params.get("rpm") else float("nan")
        unit_name    = "native"
        T_unit       = float("nan")
        step_cycles  = params.get("step_seg", float("nan"))
        reset_on_H0  = params.get("reset_on_H0", "n/a")
    else:
        unit_name   = trace["unit_name"]
        T_unit      = trace["T_unit"]
        f_cycle     = 1.0 / T_unit
        step_cycles = trace["step"]
        reset_on_H0 = params_physical.get("reset_on_H0", "n/a")

    result.meta["param_mode"]   = param_mode
    result.meta["unit_name"]    = unit_name
    result.meta["T_unit"]       = T_unit
    result.meta["f_cycle"]      = f_cycle
    result.meta["N_cycles"]     = result.meta.get("N_seg")
    result.meta["step_cycles"]  = step_cycles
    result.meta["reset_on_H0"]  = reset_on_H0
    result.meta["Total_window"] = result.meta.get("N_seg")
    result.meta.setdefault(
        "training_source", "external_reference" if reference_signal is not None else "internal"
    )
    result.meta.setdefault(
        "chatter_source", "external_reference" if reference_signal_chatter is not None else "internal"
    )

    if trace is not None:
        result.meta["physical_params_input"]  = trace["physical_params_input"]
        result.meta["native_params_resolved"] = trace["native_params_resolved"]
        result.meta["quantization_notes"]     = trace["quantization_notes"]


    if result.t_d.size > 0:
        logger.info(_section("CHATTER INDICATOR - MaxEnt-SPRT"))
        logger.info("  %-24s %s",     "Indicador:",         result.name)
        logger.info("  %-24s %s",     "Modo config:",       result.meta["param_mode"])
        logger.info("  %-24s %s",     "Segmentation:",       result.meta.get("segmentation", "n/a"))
        logger.info("  %-24s %.3f Hz",   "Frecuency Cycle:", result.meta["f_cycle"])
        logger.info("  %-24s %d",     "Entropie Windows:", result.meta.get("N_seg", "n/a"))
        logger.info("  %-24s %d",      "Total Windows",           result.meta["Total_window"])

        logger.info("  %-24s %.3f",   "Step:", result.meta["step_cycles"])
        logger.info("  %-24s %s",     "ON SPRT:",         result.meta.get("use_sprt", "n/a"))
        logger.info("  %-24s %s",     "Reset on H0:",         result.meta.get("reset_on_H0", "n/a"))
        logger.info(
            "  %-24s mu: %.3f, sigma: %.3f",
            "Training P0:",
            result.meta.get("P0_mu", float("nan")),
            result.meta.get("P0_sigma", float("nan")),
        )

        logger.info(
            "  %-24s mu: %.3f, sigma: %.3f",
            "Training P1:",
            result.meta.get("P1_mu", float("nan")),
            result.meta.get("P1_sigma", float("nan")),
        )
        logger.info("  %-24s %.10f", "A:",     result.meta["sprt_result"].a)
        logger.info("  %-24s %.10f", "B:",      result.meta["sprt_result"].b)

        logger.info("  %-24s %.3f s", "Primera deteccion:", result.t_d[0])
        if result.t_d_no_FAR.size > 0:
            logger.info("  %-24s %.3f s", "Primera Detecion Non Far:", result.t_d_no_FAR[0])
        else:
            logger.info("  %-24s %s", "Primera Detecion Non Far:", "n/a (sin t_theorical)")
        logger.info("  %-24s %d",     "Total detecciones:", result.t_d.size)
        logger.info("  %-24s %.4f, %.4f ms", "Tiempo I[0], I[1]:", result.t[0]*1000, result.t[1]*1000)
        logger.info("  %-24s %.4f, %.4f ms", "Hop[0], H[1] ", result.t[1]*1000 - result.t[0]*1000, result.t[2]*1000 - result.t[1]*1000 )
    else:
        logger.info(_section("CHATTER INDICATOR - MaxEnt-SPRT"))
        logger.info("  Indicador: %s  |  Modo: %s",
                    result.name, param_mode)


    return result

def _cut_signal( t,x , time_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Restrict a time series to a closed time interval.

    :param t: Time vector of the signal.
    :param x: Signal values aligned with ``t``.
    :param time_range: Start and end times delimiting the interval to keep.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Time and signal arrays restricted to the
        requested interval.
    """
    start_time, end_time = time_range
    mask = (t >= start_time) & (t <= end_time)
    return t[mask], x[mask]


def _extract_training_segments(
    t: np.ndarray,
    x: np.ndarray,
    intervals,
):
    """
    Split a signal into per-interval training pieces given a list of
    labelled time intervals -- WITHOUT concatenating same-label intervals.

    Each interval stays its own isolated piece so a downstream fixed-length
    window never straddles the seam between two discontinuous intervals
    (see COMMON_TEMPLATE.md §10: "ventanear cada tramo por separado, nunca
    concatenar la señal cruda entre tramos discontinuos").

    :param t: Full time axis of the signal.
    :param x: Full signal values aligned with ``t``.
    :param intervals: Sequence of ``(t_start, t_end, label)`` tuples where
        ``label`` is either ``"stable"`` or ``"chatter"``.

    Returns:
        ``(stable_pieces, chatter_pieces)``, each a list of
        ``(t_piece, x_piece, piece_id)`` tuples, one per interval of that label.
    """
    stable_pieces:  list = []
    chatter_pieces: list = []

    for i, entry in enumerate(intervals):
        t0, t1, label = float(entry[0]), float(entry[1]), str(entry[2]).lower().strip()
        if not (label.startswith("stable") or label.startswith("chatter")):
            raise ValueError(
                f"training_intervals label must start with 'stable' or 'chatter', got '{label}'."
            )
        mask = (t >= t0) & (t <= t1)
        piece = (t[mask], x[mask], f"interval[{i}]:{label}:{t0:.3f}-{t1:.3f}")
        (stable_pieces if label.startswith("stable") else chatter_pieces).append(piece)

    return stable_pieces, chatter_pieces


def _pieces_from_signaldata(ref) -> list:
    """Normalize a ``SignalData`` or ``List[SignalData]`` into ``(t, x, piece_id)`` pieces."""
    ref_list = [ref] if isinstance(ref, SignalData) else list(ref)
    return [
        (sd.t_analysis, sd.signal_analysis, str(sd.meta.get("signal_id", sd.path)))
        for sd in ref_list
    ]


def _reference_meta_list(ref) -> Optional[List[dict]]:
    """Per-piece ``.meta`` dicts for a ``reference_signal``/``reference_signal_chatter`` value."""
    if ref is None:
        return None
    ref_list = [ref] if isinstance(ref, SignalData) else list(ref)
    return [dict(sd.meta) for sd in ref_list]


def _maxent_sprt_pipeline(
    signal: SignalData,
    rpm: float,
    N_seg: int,
    t_stable_total: float,
    alpha: float,
    beta: float,
    reset_on_H0: bool,
    ratio_sampling: Optional[float] = None,
    cut_start_time: Optional[float] = None,
    cut_end_time: Optional[float] = None,
    step_seg: Optional[int] = None,
    segmentation: str = "opr",
    N_samples_per_seg: Optional[int] = None,
    use_sprt: bool = True,
    H_threshold: Optional[float] = None,
    training_intervals = None,
    t_theorical: Optional[float] = None,  # for debug/plots, not used in detection
    reference_signal: Optional[Union[SignalData, List[SignalData]]] = None,
    reference_signal_chatter: Optional[Union[SignalData, List[SignalData]]] = None,

    ) -> IndicatorResult:
    """
    Execute the Maximum Entropy Sequential Probability Ratio Test (MaxEnt SPRT) pipeline for chatter detection.

    This function performs a complete chatter detection workflow consisting of offline training and online detection phases.
    It processes a signal into stable (chatter-free) and chatter-included segments, trains a Gaussian maximum entropy
    estimator on sampled Operating Point Response (OPR) data, and applies SPRT for online chatter detection.

    :param signal: Full input bundle containing the time axis, analysis signal, sampling frequency, and any source metadata.
    :param rpm: Spindle speed in revolutions per minute.
    :param N_seg: Number of revolutions or OPR samples grouped into one analysis segment.
    :param t_stable_total: Duration in seconds from the beginning of the record assumed to be chatter-free and used for stable-state training.
    :param alpha: Target false-alarm probability used to derive SPRT thresholds.
    :param beta: Target missed-detection probability used to derive SPRT thresholds.
    :param reset_on_H0: Whether the cumulative SPRT statistic is reset after accepting the stable hypothesis.
    :param ratio_sampling: Optional OPR sampling ratio used during online detection.
    :param cut_start_time: Optional lower time bound applied before splitting the signal into stable and chatter portions.
    :param cut_end_time: Optional upper time bound applied before splitting the signal into stable and chatter portions.
    :param step_seg: Hop size in OPR samples between consecutive segment starts for both offline training and online
        detection. ``None`` (default) is equivalent to ``step_seg = N_seg`` (no overlap).
    :param segmentation: ``"opr"`` (default) – OPR decimation + ``segment_opr``;
        ``"raw"`` – skip OPR decimation, use ``segment_signal_raw`` on the
        full-rate signal. When ``"raw"``, entropy is estimated from
        ``N_samples_per_seg`` raw samples per block (much larger than N_seg OPR
        samples), yielding a lower-variance Gaussian fit.
    :param N_samples_per_seg: Block length in raw samples used when
        ``segmentation="raw"``.  Resolved automatically for physical modes;
        must be provided explicitly for native mode.
    :param reference_signal: Optional externally-labelled ``SignalData`` or
        ``List[SignalData]`` whose pieces replace the internally-derived
        stable/training region (``training_intervals`` or the legacy
        ``cut_start_time``/``t_stable_total`` cut). Each piece is windowed
        independently and only the resulting entropy values are pooled, so a
        window never straddles the seam between two physically-disjoint
        pieces. When given without ``reference_signal_chatter``, the chatter
        counterpart still comes from ``training_intervals``/the legacy cut if
        those are also provided (falls back to empty otherwise). See
        ``indicators/COMMON_TEMPLATE.md`` §10 for the shared contract.
    :param reference_signal_chatter: Optional externally-labelled ``SignalData``
        or ``List[SignalData]`` (MaxEnt-specific, not part of the shared
        contract) whose pieces replace the internally-derived chatter/P1
        training region, mirroring ``reference_signal`` for the stable/P0
        side. Takes priority over ``training_intervals``/the legacy cut for
        the chatter side when given.

    Returns:
        IndicatorResult: Result object with segment timestamps, SPRT statistic
        history, detected chatter times, and rich metadata containing the
        intermediate models, signals, detector state, and derived quantities.

    Notes
    -----
    The pipeline consists of three main phases:

    1. Signal Preparation: Splits the input signal into stable and chatter-included portions.
    2. Offline Training: Samples OPR from both signal portions and trains a Gaussian MaxEnt estimator.
    3. Online Detection: Applies SPRT on the entire signal in segments and identifies chatter points.

    Chatter points are identified where the SPRT statistic S exceeds the detection threshold (b).
    """

    t_analysis = signal.t_analysis
    signal_analysis = signal.signal_analysis
    fs = signal.fs

    fr: float = rpm / 60.0       # Hz, frequency of rotation
    t_total = t_analysis[-1]-t_analysis[0]

    # ── Training signal split ─────────────────────────────────────────────────
    # Each side (stable/P0, chatter/P1) independently comes from its own
    # externally-supplied reference SignalData/List[SignalData] when given,
    # else from training_intervals/the legacy cut on the analyzed signal
    # (unchanged behaviour when neither reference is given). Every source is
    # normalized into a list of (t, x, piece_id) "pieces" -- one per
    # physically-disjoint span -- so pieces are windowed independently
    # further down and never concatenated as raw signal across a seam
    # (see COMMON_TEMPLATE.md §10).
    training_source = "external_reference" if reference_signal is not None else "internal"
    chatter_source  = "external_reference" if reference_signal_chatter is not None else "internal"

    if reference_signal is not None and reference_signal_chatter is not None:
        stable_pieces = _pieces_from_signaldata(reference_signal)
        chatter_pieces = _pieces_from_signaldata(reference_signal_chatter)
    elif reference_signal is not None:
        stable_pieces = _pieces_from_signaldata(reference_signal)
        if training_intervals is not None:
            _, chatter_pieces = _extract_training_segments(t_analysis, signal_analysis, training_intervals)
        elif cut_end_time is not None:
            t_c, x_c = _cut_signal(t_analysis, signal_analysis, (t_stable_total, cut_end_time))
            chatter_pieces = [(t_c, x_c, "legacy_cut:chatter")]
        else:
            chatter_pieces = []
    elif reference_signal_chatter is not None:
        chatter_pieces = _pieces_from_signaldata(reference_signal_chatter)
        if training_intervals is not None:
            stable_pieces, _ = _extract_training_segments(t_analysis, signal_analysis, training_intervals)
        elif cut_start_time is not None:
            t_s, x_s = _cut_signal(t_analysis, signal_analysis, (cut_start_time, t_stable_total))
            stable_pieces = [(t_s, x_s, "legacy_cut:stable")]
        else:
            stable_pieces = []
    elif training_intervals is not None:
        # General mode: arbitrary list of [(t0, t1, "stable"|"chatter"), ...]
        stable_pieces, chatter_pieces = _extract_training_segments(t_analysis, signal_analysis, training_intervals)
    else:
        # Legacy mode: [cut_start_time, t_stable_total] = stable
        #              [t_stable_total, cut_end_time]   = chatter
        t_s, x_s = _cut_signal(t_analysis, signal_analysis, (cut_start_time, t_stable_total))
        t_c, x_c = _cut_signal(t_analysis, signal_analysis, (t_stable_total, cut_end_time))
        stable_pieces = [(t_s, x_s, "legacy_cut:stable")]
        chatter_pieces = [(t_c, x_c, "legacy_cut:chatter")]

    if reference_signal is not None or reference_signal_chatter is not None:
        logger.info_plus(
            "  %-24s %s", " - reference_signal priority:",
            f"stable={training_source}, chatter={chatter_source} "
            "(training_intervals/legacy cut only used for whichever side has no external reference).",
        )

    size_stable  = sum(x.size for _, x, _ in stable_pieces)
    size_chatter = sum(x.size for _, x, _ in chatter_pieces)

    logger.info_plus(_section("Signal loaded:"))
    logger.info_plus("  %-24s %s", " - Samples:", f"{signal_analysis.size}")
    logger.info_plus("  %-24s %s", " - Duration:", f"{t_analysis[-1]-t_analysis[0]:.2f} s")
    logger.info_plus("  %-24s %s", " - Sampling freq.:", f"{fs:.1f} Hz")
    logger.info_plus("  %-24s %s", " - Rotation freq.:", f"{fr:.1f} Hz")
    logger.info_plus("  %-24s %s", " - Segmentation:", segmentation)
    if segmentation == "raw":
        _nsamp = N_samples_per_seg or 0
        logger.info_plus("  %-24s %s", " - N_samples_per_seg:", f"{_nsamp} raw samp  ({_nsamp/fs*1e3:.2f} ms)")
        logger.info_plus("  %-24s %s", " - Total segments approx:", f"{int(len(signal_analysis) // (_nsamp or 1))}")
    else:
        logger.info_plus("  %-24s %s", f" - Segments of {N_seg} revolutions:", f"{N_seg/fr:.2f} s each")
        logger.info_plus("  %-24s %s", " - Total segments available:", f"{int(t_total*fr/N_seg)}")

    logger.info_plus(_section("Generated chatter-free and chatter-included signals."))
    logger.info_plus("  %-24s %s", "Size of signal free:", f"{size_stable} samples ({len(stable_pieces)} pieces)")
    logger.info_plus("  %-24s %s", "Size of signal chatter:", f"{size_chatter} samples ({len(chatter_pieces)} pieces)")


    # =========== Fase Offline : OPR / raw segmentation training (per piece) ==========
    ids_free  = [pid for _, _, pid in stable_pieces]
    ids_chat  = [pid for _, _, pid in chatter_pieces]

    if segmentation == "raw":
        # Skip OPR decimation — train directly on raw signal blocks, per piece
        train_free, t_train_free   = [x for _, x, _ in stable_pieces], [t for t, _, _ in stable_pieces]
        train_chat, t_train_chat   = [x for _, x, _ in chatter_pieces], [t for t, _, _ in chatter_pieces]
        opr_free = opr_chat = t_opr_free = t_opr_chat = None
        logger.info_plus("  %-24s %s", "Segmentation (raw):",
                         f"N_samples_per_seg={N_samples_per_seg}, "
                         f"free={size_stable} samp, "
                         f"chat={size_chatter} samp.")
    else:
        opr_pairs_free = [sample_opr(x, t, fs=fs, fr=fr) for t, x, _ in stable_pieces]
        opr_pairs_chat = [sample_opr(x, t, fs=fs, fr=fr) for t, x, _ in chatter_pieces]
        opr_free      = [pair[0] for pair in opr_pairs_free]
        t_opr_free    = [pair[1] for pair in opr_pairs_free]
        opr_chat      = [pair[0] for pair in opr_pairs_chat]
        t_opr_chat    = [pair[1] for pair in opr_pairs_chat]
        train_free, t_train_free = opr_free, t_opr_free
        train_chat, t_train_chat = opr_chat, t_opr_chat
        logger.info_plus("  %-24s %s", "Sampled OPR:",
                         f"{sum(a.size for a in opr_free)} samples free, "
                         f"{sum(a.size for a in opr_chat)} samples chatter.")

    # ============ Offline Phase:END-TO-END GAUSSIAN ===========
    detector_cfg = MaxEntSPRTConfig(alpha=alpha, beta=beta, reset_on_H0=reset_on_H0)
    gaussian_estimator = GaussianMaxEntEstimator()
    detector = MaxEntSPRTDetector(config=detector_cfg, estimator=gaussian_estimator)

    # Offline phase: OPR / raw Training -- each piece windowed independently,
    # only the resulting entropy values are pooled (see offline.py).
    detector.fit_offline_from_opr(
        opr_free=train_free,
        opr_t_free=t_train_free,
        opr_chat=train_chat,
        opr_t_chat=t_train_chat,
        N_seg=N_seg,
        step=step_seg,
        segmentation=segmentation,
        N_samples_per_seg=N_samples_per_seg,
        piece_ids_free=ids_free,
        piece_ids_chat=ids_chat,
    )

    models_trained = detector._check_models()
    logger.info_plus(_section("OFFLINE MODEL (Gaussian MaxEnt):"))
    logger.info_plus("  %-24s %s", "FREE:", f"mu0={models_trained.p0.mu:.5f}, sigma0={models_trained.p0.sigma:.5f}")
    logger.info_plus("  %-24s %s", "CHAT:", f"mu1={models_trained.p1.mu:.5f}, sigma1={models_trained.p1.sigma:.5f}")

    sprt_result_raw, H_seq_online, t_mid_segments = detector.detect_online_from_signal(
        y_online=signal_analysis,
        t_online=t_analysis,
        rpm=rpm,
        ratio_sampling=ratio_sampling,
        N_seg=N_seg,
        fs=fs,
        step=step_seg,
        segmentation=segmentation,
        N_samples_per_seg=N_samples_per_seg,
    )

    H_arr = np.asarray(H_seq_online)

    # =========== Decision strategy: SPRT vs per-segment threshold ==========
    if use_sprt:
        # ── Standard SPRT accumulation ──────────────────────────────────────
        sprt_result     = sprt_result_raw
        I_t_result      = sprt_result.S_history
        _H_thr_used     = None
        name_result     = "MaxEnt_SPRT"
        mask = np.where(sprt_result.S_history >= sprt_result.b)[0]
        chatter_points_time   = t_mid_segments[mask] if mask.size > 0 else np.array([])

        if t_theorical is not None:
            t_d_no_FAR_idx = np.where(chatter_points_time > t_theorical)[0]
            t_d_no_FAR = chatter_points_time[t_d_no_FAR_idx] if t_d_no_FAR_idx.size > 0 else np.array([])
        else:
            t_d_no_FAR = np.array([])

        chatter_points_values = sprt_result.S_history[mask] if mask.size > 0 else np.array([])
        logger.info_plus("  %-24s %s", "ONLINE FINAL STATE:",
                         f"{sprt_result.final_state}, decision at segment {sprt_result.decision_index}")

    else:
        # ── Per-segment threshold on H  (no memory / no accumulation) ───────
        # Auto-threshold = Bayes-optimal midpoint between P0 and P1 means
        _H_thr_used = H_threshold if H_threshold is not None \
                      else (models_trained.p0.mu + models_trained.p1.mu) / 2.0

        class _SegThresholdResult:
            """Minimal SPRT-compatible container for per-segment threshold mode."""
            S_history      = H_arr
            b              = _H_thr_used
            a              = -np.inf
            final_state    = "threshold"
            decision_index = int(np.where(H_arr >= _H_thr_used)[0][0]) \
                             if np.any(H_arr >= _H_thr_used) else None

        sprt_result       = _SegThresholdResult()
        I_t_result        = H_arr
        name_result       = "MaxEnt_threshold"
        _thr_mask         = H_arr >= _H_thr_used
        chatter_points_time   = t_mid_segments[_thr_mask]
        chatter_points_values = H_arr[_thr_mask]

        if t_theorical is not None:
            t_d_no_FAR_idx = np.where(chatter_points_time > t_theorical)[0]
            t_d_no_FAR = chatter_points_time[t_d_no_FAR_idx] if t_d_no_FAR_idx.size > 0 else np.array([])
        else:
            t_d_no_FAR = np.array([])
        
        logger.info_plus("  %-24s %s", "ONLINE MODE (no SPRT):",
                         f"per-segment threshold  H_thr = {_H_thr_used:.5f}")
        logger.info_plus("  %-24s %s", "DETECTIONS:",
                         f"{_thr_mask.sum()} / {len(H_arr)} segments above threshold")
    

    #%%
    # ============ Online Phase: Results visualization ===========

    result = IndicatorResult(
        name=name_result,
        t=t_mid_segments,
        I_t=I_t_result,
        t_d=chatter_points_time,
        t_d_no_FAR=t_d_no_FAR,
        meta={
            "Samples": signal_analysis.size,
            "Duration": t_total,
            "fs": fs,
            "Rotational_Frequency_Hz": fr,
            "N_seg": N_seg,
            "step_seg": step_seg if step_seg is not None else N_seg,
            "overlap_pct": 1.0 - (step_seg if step_seg is not None else N_seg) / N_seg,
            "segmentation": segmentation,
            "N_samples_per_seg": N_samples_per_seg,
            "alpha": alpha,
            "beta": beta,
            "rpm": rpm,
            "training_source": training_source,
            "chatter_source": chatter_source,
            "reference_signal_meta": _reference_meta_list(reference_signal),
            "reference_signal_chatter_meta": _reference_meta_list(reference_signal_chatter),
            "reference_n_pieces": len(reference_signal) if isinstance(reference_signal, list)
                                  else (1 if reference_signal is not None else 0),
            "reference_n_pieces_chatter": len(reference_signal_chatter) if isinstance(reference_signal_chatter, list)
                                  else (1 if reference_signal_chatter is not None else 0),
            "n_windows_per_piece_free": detector.n_windows_free,
            "n_windows_per_piece_chat": detector.n_windows_chat,
            "ratio_sampling": ratio_sampling,
            "use_sprt": use_sprt,
            "H_threshold_used": _H_thr_used,
            "Total_segments": int(t_total*fr/N_seg),
            "Size_signal_free": size_stable,
            "Size_signal_chatter": size_chatter,
            "Sampled OPR free":    sum(a.size for a in opr_free) if opr_free is not None else None,
            "Sampled OPR chatter": sum(a.size for a in opr_chat) if opr_chat is not None else None,
            "P0_mu": models_trained.p0.mu,
            "P0_sigma": models_trained.p0.sigma,
            "P1_mu": models_trained.p1.mu,
            "P1_sigma": models_trained.p1.sigma,
            "detector": detector,
            "gaussian_estimator": gaussian_estimator,
            "sprt_result": sprt_result,
            "models_trained": models_trained,
            "H_seq_online": H_seq_online,
            "chatter_points_values": chatter_points_values,
            # Flat concatenated views for plotting only (display-only, computed
            # after windowing/entropy already ran correctly per piece above).
            "t_stable": np.concatenate([t for t, _, _ in stable_pieces]) if stable_pieces else np.array([]),
            "signal_analysis_stable": np.concatenate([x for _, x, _ in stable_pieces]) if stable_pieces else np.array([]),
            "t_chatter": np.concatenate([t for t, _, _ in chatter_pieces]) if chatter_pieces else np.array([]),
            "signal_analysis_chatter": np.concatenate([x for _, x, _ in chatter_pieces]) if chatter_pieces else np.array([]),
            "t_opr_free": np.concatenate(t_opr_free) if t_opr_free else None,
            "opr_free":   np.concatenate(opr_free) if opr_free else None,
            "t_opr_chat": np.concatenate(t_opr_chat) if t_opr_chat else None,
            "opr_chat":   np.concatenate(opr_chat) if opr_chat else None,
            "train_free": np.concatenate(train_free) if train_free else np.array([]),
            "train_chat": np.concatenate(train_chat) if train_chat else np.array([]),
            "training_intervals": training_intervals,

        },
    )

    return result
