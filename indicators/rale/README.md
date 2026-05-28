RALE indicator packaged from Areas_Indicator_V1.py

This package provides an independent implementation of the RALE (Regenerative-Area-Lyapunov-Empirical) indicator, matching the original logic in `Areas_Indicator_V1.py`.

Usage:
- Use `rale.run_rale(t, q, q_o, cfg)` where `cfg` is a simple config object (see example).

The package includes an example that loads the same HDF5 data used in the original `Areas_Indicator_V1.py` and compares outputs.