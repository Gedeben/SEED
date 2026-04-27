# ============================================
# validation_1D_interface_rhoCp_time.py
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================
n_particles = 300000
nsteps = 4000
dt = 0.005

D1 = 1.0
D2 = 1.0

x0 = -2.0

ratios = [1, 10, 100]

# =========================
# TIME ANALYSIS
# =========================
t = nsteps * dt
t_star = x0**2 / D1
t_ratio = t / t_star

print("\n=== TIME SCALES ===")
print(f"Physical time t       = {t:.3f}")
print(f"Characteristic time t* = {t_star:.3f}")
print(f"Dimensionless time t/t* = {t_ratio:.3f}")

# =========================
# MONTE CARLO FUNCTION
# =========================
def run_simulation(C1, C2):

    E1 = C1 * np.sqrt(D1)
    E2 = C2 * np.sqrt(D2)

    x = np.ones(n_particles) * x0

    for step in range(nsteps):

        D_local = np.where(x < 0, D1, D2)
        sigma = np.sqrt(2 * D_local * dt)

        dx = np.random.normal(0, sigma, size=n_particles)
        x_new = x + dx

        cross = (x * x_new) < 0

        # no crossing
        x[~cross] = x_new[~cross]

        # crossing
        idx = np.where(cross)[0]

        if len(idx) > 0:

            xi = x[idx]
            dx_i = dx[idx]

            # fraction jusqu'à interface
            f = np.abs(xi) / np.abs(dx_i)

            # reste du déplacement
            dx_remain = (1 - f) * dx_i

            # probabilités
            P = np.zeros_like(xi)

            mask_lr = xi < 0
            P[mask_lr] = E2 / (E1 + E2)

            mask_rl = xi > 0
            P[mask_rl] = E1 / (E1 + E2)

            rand = np.random.rand(len(idx))

            # passage
            pass_mask = rand < P
            x[idx[pass_mask]] = dx_remain[pass_mask]

            # réflexion
            refl_mask = ~pass_mask
            x[idx[refl_mask]] = -dx_remain[refl_mask]

    bins = np.linspace(-10, 10, 500)
    hist, edges = np.histogram(x, bins=bins, density=True)
    xc = 0.5 * (edges[:-1] + edges[1:])

    return xc, hist


# =========================
# ANALYTICAL FUNCTION
# =========================
def analytical_solution(xc, C1, C2):

    E1 = C1 * np.sqrt(D1)
    E2 = C2 * np.sqrt(D2)

    def G(x, D):
        return (1/np.sqrt(4*np.pi*D1*t)) * np.exp(-(x**2)/(4*D1*t))

    R = (E1 - E2) / (E1 + E2)
    Tcoef = 2 * E2 / (E1 + E2)

    T = np.zeros_like(xc)

    mask_left = xc < 0
    T[mask_left] = G(xc[mask_left] - x0, D1) + R * G(xc[mask_left] + x0, D1)

    mask_right = xc >= 0
    T[mask_right] = Tcoef * G(xc[mask_right] - x0, D2)

    return T


# =========================
# RUN SIMULATIONS
# =========================
plt.figure(figsize=(7,5))

for ratio in ratios:

    C1 = 1.0
    C2 = ratio

    xc, hist = run_simulation(C1, C2)
    T = analytical_solution(xc, C1, C2)

    # normalisation
    hist /= np.trapz(hist, xc)
    T /= np.trapz(T, xc)

    plt.plot(xc, hist, label=f"MC ratio={ratio}", linewidth=2)
    plt.plot(xc, T, '--', label=f"Analytical ratio={ratio}")

# interface
plt.axvline(0, color='k', linestyle=':')

# labels
plt.xlabel("x")
plt.ylabel("T(x,t)")
plt.title(
    rf"Effect of $\rho C_p$ ratio (t/t* = {t_ratio:.2f})"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("rhoCp_scan_time.pdf")
plt.savefig("rhoCp_scan_time.png", dpi=300)

plt.show()