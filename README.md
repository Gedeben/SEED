# SEED

Stochastic Effusivity-based Evolution of Diffusion

---

## Overview

SEED is a stochastic particle-based framework for the simulation of diffusion processes in heterogeneous media with discontinuous material properties.

The method relies on a probabilistic treatment of interface conditions derived from effusivity, ensuring consistency with both transient behavior and equilibrium partitioning without requiring additional tuning parameters.

---

## Features

* Stochastic random-walk formulation of diffusion
* Effusivity-based interface treatment
* Applicability to heterogeneous media with sharp discontinuities
* Consistency with analytical solutions
* Linear computational complexity with respect to particle number
* Parallel implementation using multiprocessing
* Python implementation based on NumPy

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/seed-diffusion.git
cd seed-diffusion
pip install -r requirements.txt
```

---

## Usage

### 1D interface validation

```bash
python unsteady_1D.py
```

### Effective diffusivity in heterogeneous media

```bash
python diffusion_3D.py
```

### Performance analysis

```bash
python scaling.py
```

---

## Requirements

* Python 3.x
* NumPy
* Matplotlib

---

## Repository structure

* *.py files : core implementation

* `docs/` : softwarex_seed.pdf

---

## License

GNU General Public License v3.0

---

## Citation

If you use this code, please cite:

Debenest, G., Horgue, P., Guibert, R.
*A stochastic particle-based method for diffusion in heterogeneous media with effusivity-driven interface conditions*

---

## Contact

[gerald.debenest@toulouse-inp.fr](mailto:gerald.debenest@toulouse-inp.fr)
