"""
sweep/basis.py
==============
Physical basis descriptor for the discrete-parameter sweep study.

A ``StudyBasis`` encapsulates the choice of synchronisation unit
(revolution vs. modal period) together with the process parameters that
derive the physical time scale.  Every other sweep module receives a
``StudyBasis`` instance so that all time-unit conversions are consistent
and centralised.

Usage
-----
    from sweep.basis import StudyBasis

    basis = StudyBasis("by_modal", f_modal=150.0, rpm=12_000.0)
    print(basis.T_unit)      # 1 / 150.0 ≈ 6.667 ms
    print(basis.T_rev)       # 60 / 12000 = 5 ms
    print(basis.maxent_opr_valid)  # False — f_modal > rpm/120
"""

from __future__ import annotations
from typing import Optional

__all__ = ["StudyBasis"]

_VALID_MODES = ("by_modal", "by_revolution")


class StudyBasis:
    """
    Physical basis descriptor for one sweep study.

    Parameters
    ----------
    mode : str
        Synchronisation unit: ``"by_modal"`` (modal period T_modal = 1/f_modal)
        or ``"by_revolution"`` (revolution period T_rev = 60/rpm).
    f_modal : float
        Dominant chatter frequency [Hz].  Must be > 0.
    rpm : float
        Spindle speed [rev/min].  Must be > 0.

    Attributes (computed)
    ----------------------
    T_rev : float
        Revolution period [s] = 60 / rpm.
    T_modal : float
        Modal period [s] = 1 / f_modal.
    T_unit : float
        The natural time unit for this basis: T_modal (by_modal) or T_rev (by_revolution).
    unit_name : str
        Human-readable name of the basis unit: ``"modal_period"`` or ``"revolution"``.
    maxent_opr_valid : bool
        True only when OPR sampling can resolve the chatter frequency.
        Auto-computed as f_modal < rpm/120 (OPR Nyquist) unless overridden
        explicitly via the constructor argument.
        For f_modal=150 Hz and rpm=12000 rpm → OPR Nyquist=100 Hz < 150 Hz → False.
    """

    def __init__(
        self,
        mode: str,
        f_modal: float,
        rpm: float,
        maxent_opr_valid: Optional[bool] = None,
    ) -> None:
        """
        Parameters
        ----------
        mode : str
            ``"by_modal"`` or ``"by_revolution"``.
        f_modal : float
            Dominant chatter frequency [Hz].
        rpm : float
            Spindle speed [rev/min].
        maxent_opr_valid : bool or None, optional
            Override whether OPR segmentation is valid for MaxEnt-SPRT.
            ``None`` (default) → computed automatically from Nyquist criterion
            (f_modal < rpm / 120).  Pass ``True`` or ``False`` to force a value
            regardless of the signal parameters.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES!r}, got {mode!r}."
            )
        if f_modal <= 0.0:
            raise ValueError(f"f_modal must be > 0, got {f_modal}.")
        if rpm <= 0.0:
            raise ValueError(f"rpm must be > 0, got {rpm}.")
        if maxent_opr_valid is not None and not isinstance(maxent_opr_valid, bool):
            raise TypeError(
                f"maxent_opr_valid must be bool or None, got {type(maxent_opr_valid).__name__!r}."
            )

        self._mode                      = mode
        self._f_modal                   = f_modal
        self._rpm                       = rpm
        self._maxent_opr_valid_override = maxent_opr_valid

    # ── read-only properties ──────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """Synchronisation mode string: ``"by_modal"`` or ``"by_revolution"``."""
        return self._mode

    @property
    def f_modal(self) -> float:
        """Dominant chatter frequency [Hz]."""
        return self._f_modal

    @property
    def rpm(self) -> float:
        """Spindle speed [rev/min]."""
        return self._rpm

    @property
    def T_rev(self) -> float:
        """Revolution period [s]."""
        return 60.0 / self._rpm

    @property
    def T_modal(self) -> float:
        """Modal period [s]."""
        return 1.0 / self._f_modal

    @property
    def T_unit(self) -> float:
        """Natural time unit for this basis [s]."""
        return self.T_modal if self._mode == "by_modal" else self.T_rev

    @property
    def unit_name(self) -> str:
        """Human-readable unit name."""
        return "modal_period" if self._mode == "by_modal" else "revolution"

    @property
    def maxent_opr_valid(self) -> bool:
        """
        Whether OPR-based segmentation can resolve the chatter frequency.

        Auto-computed: OPR Nyquist = rpm / 120 [Hz].  Valid iff f_modal < rpm/120.
        Can be overridden via the constructor argument ``maxent_opr_valid``.

        If False, ``by_revolution`` mode for MaxEnt-SPRT will use
        ``segmentation="raw"`` (see config_builder.py).
        """
        if self._maxent_opr_valid_override is not None:
            return self._maxent_opr_valid_override
        opr_nyquist = self._rpm / 120.0
        return self._f_modal < opr_nyquist

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"StudyBasis(mode={self._mode!r}, f_modal={self._f_modal} Hz, "
            f"rpm={self._rpm} rpm, T_unit={self.T_unit*1e3:.4f} ms, "
            f"maxent_opr_valid={self.maxent_opr_valid})"
        )
