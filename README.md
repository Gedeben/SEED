# SEED 🌱

**Stochastic Effusivity-based Evolution of Diffusion**

---

## 🔬 Overview

SEED is a stochastic particle-based framework for simulating diffusion in heterogeneous media with discontinuous material properties.

The method introduces a **physically consistent, parameter-free interface condition based on effusivity**, enabling accurate modeling of transport across sharp material interfaces.

This repository accompanies the scientific publication:

> *A stochastic particle-based method for diffusion in heterogeneous media with effusivity-driven interface conditions*

---

## ✨ Key Features

* 🔁 Random-walk particle diffusion
* ⚡ Effusivity-driven interface condition
* 🧱 Handles discontinuous heterogeneous media
* 📊 Validated against analytical solutions
* 🚀 Linear scaling with number of particles (O(N))
* 🧵 Parallelizable with multiprocessing
* 🐍 Fully implemented in Python (NumPy)

---

## 🧠 Method Summary

Particle motion follows a Brownian random walk:

$$
\Delta x \sim \mathcal{N}(0, 2D \Delta t)
$$

At interfaces, transmission probability is:

$$
P_{i \to j} = \frac{E_j}{E_i + E_j}
$$

where (E = \rho C_p \sqrt{D}) is the **effusivity**.

This ensures:

* correct transient behavior
* detailed balance
* equilibrium partitioning

---

## 📦 Installation

```bash
git clone https://github.com/your-username/seed-diffusion.git
cd seed-diffusion
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1D interface validation

```bash
python src/unsteady_1D.py
```

Monte Carlo vs analytical solution comparison
(see: )

---

### 3D heterogeneous diffusion

```bash
python src/3D_Dvar.py
```

* Generates porous medium
* Computes effective diffusivity
  (see: )

---

### Performance scaling

```bash
python src/scaling.py
```

* Particle scaling
* Parallel scaling
  (see: )

---

## 📊 Results

* ✔ Excellent agreement with analytical solutions
* ✔ Correct equilibrium partitioning
* ✔ Accurate effective diffusivity in porous media
* ✔ Linear computational scaling

---

## 🧪 Requirements

* Python ≥ 3.8
* NumPy
* Matplotlib

```bash
pip install numpy matplotlib
```

---

## 📁 Structure

```
src/        → core simulation code  
examples/   → usage scripts  
figures/    → output plots  
docs/       → paper  
```

---

## 📜 License

GNU GPL

---

## 📬 Contact

Gérald Debenest
IMFT – Toulouse INP
📧 [gerald.debenest@toulouse-inp.fr](mailto:gerald.debenest@toulouse-inp.fr)

---

## 📖 Citation

If you use SEED, please cite:

```
Debenest et al.,
A stochastic particle-based method for diffusion in heterogeneous media with effusivity-driven interface conditions
```

---

## 🌱 Why SEED?

SEED provides a **physics-driven alternative** to classical methods:

* ❌ No artificial flux matching

* ❌ No numerical tuning

* ✅ Interface behavior emerges naturally

* ✅ Fully consistent with diffusion theory

* ✅ Robust for complex heterogeneous systems

---

## 🚀 Future Work

* Advection–diffusion coupling
* Reactive transport
* GPU acceleration
* Large-scale 3D simulations

---
