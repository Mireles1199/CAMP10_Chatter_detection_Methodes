# ssq-chatter

SSQ-STFT utilities and pipelines for chatter monitoring and time–frequency analysis.  
Designed for robust, dependency-light usage with optional FFTW and visualization extras.

---

## Features

- **Local, custom SSQ-STFT core** (`ssq_stft_T`) with NumPy/Numba
- Clean separation: `utils/` (core & helpers) and `lib/` (wrappers, pipelines)
- Optional **FFTW** backend (via `pyfftw`) and plotting helpers
- Installable with modern `pyproject.toml` (PEP 517), `src/` layout
- Typed code, Spanish comments (code), English identifiers

---

## Requirements

- Python **≥ 3.9**
- Runtime deps (see `pyproject.toml`):
  - `numpy`, `scipy`, `numba`
  - Optional: `matplotlib` (`[viz]` extra), `pyfftw` (`[fftw]` extra; skip on Windows)

---

## Installation

Editable install for development:

```bash
pip install -e .
