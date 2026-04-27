import numpy as np
import multiprocessing as mp
import os
import time

# =========================
# PARAMETERS
# =========================
N = 128
R_big, R_small = 8, 4
beta = 0.85

batch_size = 200000
n_batches = 5

nsteps = 20000
dt = 0.01

D_s, D_g = 1.0, 1.0
rhoCp_ratios = [1, 10, 100, 1000, 10000]

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

    np.random.seed(0)

    centers_big, centers_small = [], []
    min_big = beta*2*R_big
    min_small = beta*2*R_small

    for i in range(200000):

        if i % 5000 == 0:
            print(f"[DEBUG] iter={i}, big={len(centers_big)}, small={len(centers_small)}")

        c = np.random.randint(R_big, N-R_big, size=3)
        if len(centers_big)==0 or all(np.sum((cb-c)**2) >= min_big**2 for cb in centers_big):
            centers_big.append(c)

        c2 = np.random.randint(R_small, N-R_small, size=3)
        if len(centers_small)==0 or all(np.sum((cs-c2)**2) >= min_small**2 for cs in centers_small):
            centers_small.append(c2)

        if i % 5000 == 0 and len(centers_big) > 50:

            grid = np.zeros((N,N,N), dtype=np.uint8)
            add_spheres(grid, centers_big, R_big)
            add_spheres(grid, centers_small, R_small)

            frac = grid.mean()
            print(f"[DEBUG] solid fraction = {frac:.3f}")

            if frac > 0.58:
                print("[INFO] target reached")
                break

    grid = np.zeros((N,N,N), dtype=np.uint8)
    add_spheres(grid, centers_big, R_big)
    add_spheres(grid, centers_small, R_small)

    print("[INFO] final solid fraction =", grid.mean())
    return grid

# =========================
# MSD
# =========================
def compute_msd(pos):
    return ((pos - N/2)**2).sum(axis=1).mean()

# =========================
# SIMULATION
# =========================
def run_simulation(grid, ratio, n_particles, D_s_local, D_g_local):

    if ratio is None:
        E_s = E_g = 1.0
    else:
        E_s = ratio*np.sqrt(D_s_local)
        E_g = np.sqrt(D_g_local)

    pos = np.ones((n_particles,3))*N/2

    Ts_hist, Tg_hist, msd_hist = [], [], []

    for step in range(nsteps):

        ix = np.clip(pos.astype(int),0,N-1)
        phase = grid[ix[:,0],ix[:,1],ix[:,2]]

        D = np.where(phase==1, D_s_local, D_g_local)
        disp = np.random.normal(0,np.sqrt(2*D*dt)[:,None],(n_particles,3))
        pos_new = pos + disp

        ix_new = np.clip(pos_new.astype(int),0,N-1)
        phase_new = grid[ix_new[:,0],ix_new[:,1],ix_new[:,2]]

        cross = phase!=phase_new
        pos[~cross] = pos_new[~cross]

        idx = np.where(cross)[0]
        if len(idx)>0:
            p = phase[idx]

            P = np.zeros(len(idx))
            P[p==1] = E_g/(E_s+E_g)
            P[p==0] = E_s/(E_s+E_g)

            rand = np.random.rand(len(idx))
            pass_mask = rand < P

            pos[idx[pass_mask]] = pos_new[idx[pass_mask]]

            refl = disp[idx[~pass_mask]]
            refl[:,0]*=-1
            pos[idx[~pass_mask]] += refl

        ix = np.clip(pos.astype(int),0,N-1)
        phase = grid[ix[:,0],ix[:,1],ix[:,2]]

        Ts_hist.append(np.mean(phase==1))
        Tg_hist.append(np.mean(phase==0))
        msd_hist.append(compute_msd(pos))

        if step > 5000 and abs(Ts_hist[-1]-Tg_hist[-1]) < 1e-3:
            break

    return np.array(Ts_hist), np.array(Tg_hist), np.array(msd_hist)

# =========================
# WORKER
# =========================
def worker_batch(args):
    grid, ratio, batch_id, Ds, Dg = args
    np.random.seed()
    print(f"[INFO] batch {batch_id+1}/{n_batches}")
    return run_simulation(grid, ratio, batch_size, Ds, Dg)

# =========================
# PARALLEL
# =========================
def run_batched_parallel(grid, ratio, Ds, Dg):

    with mp.Pool(mp.cpu_count()) as pool:
        args = [(grid, ratio, b, Ds, Dg) for b in range(n_batches)]
        results = pool.map(worker_batch, args)

    Ts_acc, Tg_acc, msd_acc = None, None, None

    for Ts, Tg, msd in results:
        if Ts_acc is None:
            Ts_acc, Tg_acc, msd_acc = Ts, Tg, msd
        else:
            L = min(len(Ts_acc), len(Ts))
            Ts_acc = Ts_acc[:L] + Ts[:L]
            Tg_acc = Tg_acc[:L] + Tg[:L]
            msd_acc = msd_acc[:L] + msd[:L]

    return Ts_acc/n_batches, Tg_acc/n_batches, msd_acc/n_batches

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    grid = build_grid()

    results = {}
    Deff = {}

    cases = [
        ("heterogeneous", r, D_s, D_g) for r in rhoCp_ratios
    ] + [
        ("homogeneous", None, 1.0, 1.0)
    ]

    for name, r, Ds, Dg in cases:

        print(f"\n=== Running {name} {r} ===")

        Ts, Tg, msd = run_batched_parallel(grid, r, Ds, Dg)

        key = f"{name}_{r}" if r is not None else "homogeneous"
        results[key] = (Ts, Tg, msd)

        t = np.arange(len(msd)) * dt
        start = int(len(msd)*0.5)

        slope = np.polyfit(t[start:], msd[start:], 1)[0]
        Deff[key] = slope / 6

        print(f"[RESULT] {key} -> Deff={Deff[key]:.6f}")

    # =========================
    # SAFE SAVE
    # =========================
    filename = f"Deff_results_{int(time.time())}.csv"

    with open(filename, "w") as f:
        f.write("case,Deff\n")
        for k in Deff:
            f.write(f"{k},{Deff[k]}\n")

    print(f"\n[INFO] Results saved to {filename}")