import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import D_Deff


def run_particle_scaling(grid, ratio):
    print("\n[INFO] Particle scaling")

    total_particles_list = [50000, 100000, 200000, 400000]
    times = []

    for Ntot in total_particles_list:
        print(f"\n[TEST] total_particles = {Ntot}")

        # 👉 on fixe batch_size et on ajuste nb_batches
        batch_size = 50000
        n_batches = max(1, Ntot // batch_size)

        D_Deff.batch_size = batch_size

        t0 = time.time()

        for b in range(n_batches):
            print(f"[INFO] batch {b+1}/{n_batches}")
            D_Deff.worker_batch((grid, ratio, b, D_Deff.D_s, D_Deff.D_g))

        elapsed = time.time() - t0
        print(f"[TIME] {elapsed:.2f} s")

        times.append(elapsed)

    return total_particles_list, times


def run_core_scaling(grid, ratio):
    print("\n[INFO] Core scaling")

    n_cores_list = [1, 2, 4]
    times = []

    total_particles = 200000
    batch_size = 50000
    n_batches = total_particles // batch_size

    D_Deff.batch_size = batch_size

    for nproc in n_cores_list:
        print(f"\n[TEST] nproc = {nproc}")

        t0 = time.time()

        with mp.Pool(nproc) as pool:
            args = [(grid, ratio, b, D_Deff.D_s, D_Deff.D_g) for b in range(n_batches)]
            pool.map(D_Deff.worker_batch, args)

        elapsed = time.time() - t0
        print(f"[TIME] {elapsed:.2f} s")

        times.append(elapsed)

    return n_cores_list, times


def plot_results(N_list, times_particles, cores, times_cores):
    # --- particle scaling ---
    plt.figure()
    plt.plot(N_list, times_particles, 'o-')
    plt.xlabel("Total number of particles")
    plt.ylabel("CPU time (s)")
    plt.title("Scaling with number of particles")
    plt.grid()
    plt.savefig("scaling_particles.png", dpi=300)

    # --- core scaling ---
    speedup = times_cores[0] / np.array(times_cores)

    plt.figure()
    plt.plot(cores, speedup, 'o-')
    plt.xlabel("Number of cores")
    plt.ylabel("Speedup")
    plt.title("Parallel scaling")
    plt.grid()
    plt.savefig("scaling_cores.png", dpi=300)

    plt.show()


def main():
    print("[INFO] Building grid...")
    grid = D_Deff.build_grid()

    ratio = 10

    N_list, times_particles = run_particle_scaling(grid, ratio)
    cores, times_cores = run_core_scaling(grid, ratio)

    plot_results(N_list, times_particles, cores, times_cores)


# ⚠️ obligatoire Windows
if __name__ == "__main__":
    mp.freeze_support()
    main()