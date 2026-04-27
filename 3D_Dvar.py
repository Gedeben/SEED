import numpy as np

# =========================
# OPTIONAL PLOTTING
# =========================
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except:
    HAS_PLOT = False
    print("[WARNING] matplotlib not found → plots disabled")

# =========================
# PARAMETERS
# =========================
N = 128
R_big, R_small = 8, 4
beta = 0.85

batch_size = 100000
n_batches = 10

nsteps = 15000
dt = 0.01

D_g = 1.0
Ds_ratios = [1, 0.1, 0.01, 0.001, 0.0001]

np.random.seed(0)

# =========================
# BUILD GRID
# =========================
def add_spheres(grid, centers, R):
    r = np.arange(-R, R+1)
    X,Y,Z = np.meshgrid(r,r,r, indexing='ij')
    mask = X**2 + Y**2 + Z**2 <= R**2

    for c in centers:
        cx,cy,cz = c.astype(int)

        x0,x1 = max(cx-R,0), min(cx+R+1,N)
        y0,y1 = max(cy-R,0), min(cy+R+1,N)
        z0,z1 = max(cz-R,0), min(cz+R+1,N)

        sub = grid[x0:x1,y0:y1,z0:z1]

        mx0 = x0-(cx-R); mx1 = mx0+sub.shape[0]
        my0 = y0-(cy-R); my1 = my0+sub.shape[1]
        mz0 = z0-(cz-R); mz1 = mz0+sub.shape[2]

        sub_mask = mask[mx0:mx1,my0:my1,mz0:mz1]
        sub[sub_mask] = 1


def build_grid():
    centers_big, centers_small = [], []
    min_big = beta*2*R_big
    min_small = beta*2*R_small

    for i in range(200000):

        c = np.random.randint(R_big, N-R_big, size=3)
        if len(centers_big)==0 or all(np.sum((cb-c)**2)>=min_big**2 for cb in centers_big):
            centers_big.append(c)

        c2 = np.random.randint(R_small, N-R_small, size=3)
        if len(centers_small)==0 or all(np.sum((cs-c2)**2)>=min_small**2 for cs in centers_small):
            centers_small.append(c2)

        if i % 5000 == 0 and len(centers_big)>50:
            grid = np.zeros((N,N,N))
            add_spheres(grid, centers_big, R_big)
            add_spheres(grid, centers_small, R_small)
            if grid.mean()>0.58:
                break

    grid = np.zeros((N,N,N))
    add_spheres(grid, centers_big, R_big)
    add_spheres(grid, centers_small, R_small)

    print("[INFO] solid fraction =", grid.mean())
    return grid.astype(np.uint8)

grid = build_grid()

# =========================
# INIT
# =========================
def init_positions(grid, n_particles):
    fluid = np.argwhere(grid == 0)
    idx = np.random.choice(len(fluid), size=n_particles)
    return fluid[idx].astype(float)

# =========================
# MSD
# =========================
def compute_msd(pos, pos0):
    return ((pos - pos0)**2).sum(axis=1).mean()

# =========================
# SIMULATION
# =========================
def run_simulation(grid, Ds, n_particles):

    pos = init_positions(grid, n_particles)
    pos0 = pos.copy()

    msd_hist = []

    for step in range(nsteps):

        ix = np.clip(pos.astype(int),0,N-1)
        phase = grid[ix[:,0],ix[:,1],ix[:,2]]

        D = np.where(phase==1, Ds, D_g)

        disp = np.random.normal(
            0,
            np.sqrt(2 * D * dt)[:, None],
            (n_particles,3)
        )

        pos_new = pos + disp

        ix_new = np.clip(pos_new.astype(int),0,N-1)
        phase_new = grid[ix_new[:,0],ix_new[:,1],ix_new[:,2]]

        cross = phase != phase_new

        pos[~cross] = pos_new[~cross]

        idx = np.where(cross)[0]

        if len(idx) > 0:

            if Ds == 0:
                blocked = (phase[idx] == 0) & (phase_new[idx] == 1)
                pos[idx[~blocked]] = pos_new[idx[~blocked]]
                pos[idx[blocked]] -= disp[idx[blocked]]

            else:
                p = phase[idx]

                E_s = np.sqrt(Ds)
                E_g = np.sqrt(D_g)

                P = np.zeros(len(idx))
                P[p==1] = E_g/(E_s+E_g)
                P[p==0] = E_s/(E_s+E_g)

                rand = np.random.rand(len(idx))
                pass_mask = rand < P

                pos[idx[pass_mask]] = pos_new[idx[pass_mask]]

                refl = disp[idx[~pass_mask]]
                refl[:,0] *= -1
                pos[idx[~pass_mask]] += refl

        msd_hist.append(compute_msd(pos, pos0))

    return np.array(msd_hist)

# =========================
# BATCH
# =========================
def run_batched(grid, Ds):

    msd_acc = None

    for b in range(n_batches):

        print(f"[INFO] Ds/Dg={Ds} batch {b+1}")

        msd = run_simulation(grid, Ds, batch_size)

        if msd_acc is None:
            msd_acc = msd
        else:
            L = min(len(msd_acc), len(msd))
            msd_acc = msd_acc[:L] + msd[:L]

    return msd_acc / n_batches

# =========================
# RUN ALL
# =========================
Deff = {}
msd_all = {}

for ratio in Ds_ratios:

    Ds = ratio * D_g

    print(f"\n=== Ds/Dg = {ratio} ===")

    msd = run_batched(grid, Ds)
    msd_all[ratio] = msd

    t = np.arange(len(msd))*dt
    start = int(len(msd)*0.5)

    slope = np.polyfit(t[start:], msd[start:], 1)[0]
    Deff[ratio] = slope / 6

    print(f"[RESULT] Deff = {Deff[ratio]:.6f}")

# =========================
# OPTIONAL PLOTS
# =========================
if HAS_PLOT:

    plt.figure()
    for r in Ds_ratios:
        t = np.arange(len(msd_all[r]))*dt
        plt.plot(t, msd_all[r], label=f"{r}")

    plt.xlabel("t")
    plt.ylabel("MSD")
    plt.legend(title="Ds/Dg")
    plt.title("MSD")
    plt.show()

    plt.figure()
    plt.plot(Ds_ratios, [Deff[r] for r in Ds_ratios], marker='o')
    plt.xscale("log")
    plt.xlabel("Ds/Dg")
    plt.ylabel("Deff")
    plt.title("Effective diffusion")
    plt.show()

# =========================
# RESULTS
# =========================
print("\n=== FINAL RESULTS ===")
for r in Deff:
    print(f"Ds/Dg={r} -> Deff={Deff[r]:.6f}")