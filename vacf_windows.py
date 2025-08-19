# -*- coding: utf-8 -*-
"""
streaming window-averaged VACF for large lammps velocity trajectories
- no COM/drift corrections
- FFT-based Wiener–Khinchin per window (for speed)
- reads whole atom set for each window (no batching)
- caps window length -> MAX_WINDOW_FRAMES (default 100,000)
- individually saves each window VACF to its own .txt for inspection**** 
"""

import argparse
import numpy as np
import os

MAX_WINDOW_FRAMES = 100_000  # hard cap on frames per window


# functions for FFT transform for VACF calculation
def _next_pow_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def vacf_total_fft(vel):
    """Compute total vector VACF via FFT (Wiener–Khinchin)."""
    L, N, _ = vel.shape
    X = vel.reshape(L, N * 3)
    n_fft = _next_pow_two(2 * L - 1)
    F = np.fft.rfft(X, n=n_fft, axis=0)
    S = F * F.conj()
    ac = np.fft.irfft(S, n=n_fft, axis=0)[:L, :]
    ac /= np.arange(L, 0, -1).reshape(L, 1)  # unbiased
    vacf_total = ac.reshape(L, N, 3).sum(axis=2).mean(axis=1)
    return vacf_total


# reading in the lammpstrj velocity dump file (vx,vy,vz)
def read_lammpstrj_velocities(files, start_frame, L, num_atoms, header_info):
    """Reads a block of L consecutive frames starting at start_frame."""
    vx_i, vy_i, vz_i = header_info["vx"], header_info["vy"], header_info["vz"]
    out = np.empty((L, num_atoms, 3), dtype=np.float64)
    cur_frame = -1
    filled = 0

    for path in files:
        if filled >= L:
            break
        with open(path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.startswith("ITEM: TIMESTEP"):
                    continue
                cur_frame += 1
                f.readline()  # skip timestep value

                # NUMBER OF ATOMS
                while True:
                    line = f.readline()
                    if line.startswith("ITEM: NUMBER OF ATOMS"):
                        n_atoms_this = int(f.readline().strip())
                        if n_atoms_this != num_atoms:
                            raise ValueError("Atom count mismatch.")
                        break

                # BOX BOUNDS
                while True:
                    line = f.readline()
                    if line.startswith("ITEM: BOX BOUNDS"):
                        for _ in range(3):
                            f.readline()
                        break

                # ATOMS header
                while True:
                    line = f.readline()
                    if line.startswith("ITEM: ATOMS"):
                        break

                # Skip frames outside target range
                if cur_frame < start_frame or cur_frame >= start_frame + L:
                    for _ in range(num_atoms):
                        f.readline()
                    continue

                frame_idx = cur_frame - start_frame
                for j in range(num_atoms):
                    parts = f.readline().split()
                    out[frame_idx, j, 0] = float(parts[vx_i])
                    out[frame_idx, j, 1] = float(parts[vy_i])
                    out[frame_idx, j, 2] = float(parts[vz_i])
                filled += 1

    if filled != L:
        raise ValueError(f"Requested window [{start_frame}:{start_frame+L}) not fully read.")
    return out


def scan_first_frame(files):
    """Detect atom count, total frames, and velocity column indices."""
    num_atoms = None
    total_frames = 0
    header_info = None
    found_header = False

    for path in files:
        with open(path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith("ITEM: TIMESTEP"):
                    total_frames += 1
                    f.readline()
                elif line.startswith("ITEM: NUMBER OF ATOMS"):
                    num_atoms = int(f.readline().strip())
                elif line.startswith("ITEM: BOX BOUNDS"):
                    for _ in range(3):
                        f.readline()
                elif line.startswith("ITEM: ATOMS") and not found_header:
                    headers = line.strip().split()[2:]
                    idx = {k: None for k in ("vx", "vy", "vz")}
                    for k in idx:
                        if k in headers:
                            idx[k] = headers.index(k)
                    if None in idx.values():
                        raise ValueError("vx/vy/vz not found in ATOMS header.")
                    header_info = idx
                    found_header = True
                    for _ in range(num_atoms):
                        f.readline()

    if num_atoms is None or total_frames == 0:
        raise ValueError("Failed to detect atom count or frame count.")
    return num_atoms, total_frames, header_info


# window averaging VACF settings
def plan_windows(T, n_windows=5, overlap=0.5):
    """Plan windows with fractional overlap, capped at MAX_WINDOW_FRAMES."""
    n_windows = max(1, int(n_windows))
    overlap = float(np.clip(overlap, 0.0, 0.9))
    L = int(np.floor(T / (1.0 + (n_windows - 1) * (1.0 - overlap))))
    L = max(2, L)
    L = min(L, MAX_WINDOW_FRAMES)  # capping window length
    stride = max(1, int(round(L * (1.0 - overlap))))
    starts = [i * stride for i in range(n_windows)]
    if starts and starts[-1] + L > T:
        shift = (starts[-1] + L) - T
        starts = [max(0, s - shift) for s in starts]
    windows = [(s, L) for s in starts if s + L <= T]
    if not windows:
        windows = [(0, min(L, T))]
    return windows, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--compare-vacf", action="store_true")
    args = ap.parse_args()

    num_atoms, total_frames, header_info = scan_first_frame(args.input)
    windows, L = plan_windows(total_frames, args.windows, args.overlap)

    vacf_windows = []
    for idx, (start, Lw) in enumerate(windows):
        print(f"Processing window {idx+1}/{len(windows)}: frames {start}–{start+Lw} (len={Lw})")
        vel_block = read_lammpstrj_velocities(args.input, start, Lw, num_atoms, header_info)
        vacf_w = vacf_total_fft(vel_block)
        vacf_windows.append(vacf_w)

        # save window immediately
        time_axis_w = np.arange(Lw) * args.dt * 10
        window_out = args.output.replace(".txt", f"_window{idx+1}.txt")
        np.savetxt(window_out, np.column_stack((time_axis_w, vacf_w)),
                   header=f"Time(fs)\tVACF_window{idx+1}")
        print(f"Saved VACF for window {idx+1} to {window_out}")

    # average all windows
    vacf_avg = np.mean(np.vstack(vacf_windows), axis=0)
    time_axis = np.arange(L) * args.dt * 10
    out_path = args.output.replace(".txt", "_windowavg.txt")
    np.savetxt(out_path, np.column_stack((time_axis, vacf_avg)),
               header="Time(fs)\tVACF_windowavg")
    print(f"Saved window-averaged VACF to {out_path}")

    if args.compare_vacf:
        import matplotlib.pyplot as plt
        for idx, vacf_w in enumerate(vacf_windows):
            plt.plot(np.arange(len(vacf_w)) * args.dt * 10, vacf_w, alpha=0.4, label=f"Window {idx+1}")
        plt.plot(time_axis, vacf_avg, 'k', lw=2, label="Window-avg")
        plt.xlabel("Time (fs)")
        plt.ylabel("VACF (A^2/fs^2)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output.replace(".txt", "_windowavg.png"), dpi=300)


if __name__ == "__main__":
    main()

