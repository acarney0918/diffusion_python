# msd_streaming_multi_origin.py
import numpy as np
from math import gcd

############################
# Parsing & auto-detection #
############################

def detect_atoms_stride(file_paths):
    """
    Scan dump headers to detect:
      - natoms
      - per-frame timesteps list (across files)
      - inferred dump stride (GCD of timestep diffs, in MD steps)
      - column indices for id,x,y,z (xu/yu/zu preferred; fallback x/y/z)
    Only reads headers + counts; does not parse atom data payload fully.
    """
    timesteps = []
    natoms_first = None
    cols = None  # indices dict: {'id':i,'x':i,'y':i,'z':i}
    for path in file_paths:
        with open(path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith("ITEM: TIMESTEP"):
                    ts_line = f.readline()
                    if not ts_line:
                        break
                    try:
                        timesteps.append(int(ts_line.strip()))
                    except ValueError:
                        pass
                    # number of atoms
                    # We expect the formal sequence, but be defensive:
                    # keep reading until we hit "ITEM: NUMBER OF ATOMS"
                    while True:
                        l2 = f.readline()
                        if not l2:
                            break
                        if l2.startswith("ITEM: NUMBER OF ATOMS"):
                            nat = int(f.readline().strip())
                            if natoms_first is None:
                                natoms_first = nat
                            # skip box bounds section (typically 3 lines)
                            while True:
                                l3 = f.readline()
                                if not l3:
                                    break
                                if l3.startswith("ITEM: ATOMS"):
                                    headers = l3.strip().split()[2:]
                                    # prefer unwrapped xu/yu/zu, else x/y/z
                                    want = {}
                                    try:
                                        want['id'] = headers.index('id')
                                    except ValueError:
                                        raise ValueError("Dump must include 'id'.")
                                    for xyz in [('xu','yu','zu'), ('x','y','z')]:
                                        try:
                                            want['x'] = headers.index(xyz[0])
                                            want['y'] = headers.index(xyz[1])
                                            want['z'] = headers.index(xyz[2])
                                            cols = want
                                            break
                                        except ValueError:
                                            continue
                                    if cols is None:
                                        raise ValueError("Dump must include either xu/yu/zu or x/y/z.")
                                    # skip atom payload for this frame
                                    for _ in range(nat):
                                        f.readline()
                                    break  # done with this frame
                            break  # done from NUMBER OF ATOMS to ATOMS for this frame
    if not timesteps:
        raise ValueError("No TIMESTEP records found.")
    timesteps.sort()
    diffs = [b - a for a, b in zip(timesteps[:-1], timesteps[1:]) if (b - a) > 0]
    dump_stride = diffs[0] if diffs else 1
    for d in diffs[1:]:
        dump_stride = gcd(dump_stride, d)
    return natoms_first, dump_stride, cols

def frame_reader(file_paths, natoms, cols, assume_sorted_by_id=True):
    """
    Stream frames across files, yielding (frame_index, ids, coords3).
    coords3 is (natoms, 3) float64 array.
    Assumes each frame begins with the standard LAMMPS headers.
    """
    k = 0
    for path in file_paths:
        with open(path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.startswith("ITEM: TIMESTEP"):
                    continue
                # consume timestep line
                _ = f.readline()
                # find NUMBER OF ATOMS
                while True:
                    l2 = f.readline()
                    if not l2:
                        break
                    if l2.startswith("ITEM: NUMBER OF ATOMS"):
                        nat = int(f.readline().strip())
                        if nat != natoms:
                            raise ValueError(f"natoms changed within files: {nat} vs {natoms}")
                        # skip box bounds (3 lines typical; handle arbitrary count)
                        while True:
                            l3 = f.readline()
                            if not l3:
                                break
                            if l3.startswith("ITEM: ATOMS"):
                                # parse atom lines
                                ids = np.empty(natoms, dtype=np.int64)
                                xyz = np.empty((natoms, 3), dtype=np.float64)
                                id_i = cols['id']; x_i = cols['x']; y_i = cols['y']; z_i = cols['z']
                                for i in range(natoms):
                                    parts = f.readline().split()
                                    ids[i] = int(parts[id_i])
                                    xyz[i, 0] = float(parts[x_i])
                                    xyz[i, 1] = float(parts[y_i])
                                    xyz[i, 2] = float(parts[z_i])
                                if not assume_sorted_by_id:
                                    order = np.argsort(ids)
                                    ids = ids[order]
                                    xyz = xyz[order]
                                else:
                                    # quick sanity: if not monotonic, sort
                                    if np.any(np.diff(ids) < 0):
                                        order = np.argsort(ids)
                                        ids = ids[order]
                                        xyz = xyz[order]
                                yield k, ids, xyz
                                k += 1
                                break
                        break

########################################
# Streaming multi-origin MSD (1 pass)  #
########################################

def msd_streaming(file_paths,
                  timestep_fs=0.25,
                  max_tau_frames=100000,
                  max_active_origins=8,
                  desired_atom_id=1,
                  assume_sorted_by_id=True):
    """
    Single-pass, memory-light MSD.
    We keep <= max_active_origins origin frames. For each new frame, we
    update MSD(tau) where tau = current_k - origin_k for all active origins,
    for 1 <= tau <= max_tau_frames.

    Returns: time_fs[1:K], msd[1:K], single_atom_msd[1:K], dump_stride
    """
    # Auto-detect natoms and columns + dump stride
    natoms, dump_stride, cols = detect_atoms_stride(file_paths)

    # Output accumulators
    K = max_tau_frames
    sum_msd = np.zeros(K + 1, dtype=np.float64)
    sum_msd_single = np.zeros(K + 1, dtype=np.float64)
    cnt = np.zeros(K + 1, dtype=np.int64)

    # Origins management
    origins = []  # list of dicts: {'k':int, 'coords':(natoms,3), 'single':(3,), 'id_to_idx':optional}
    # Choose origin insertion stride so that active origins <= max_active_origins
    # Roughly keep ~max_active_origins origins spanning K frames:
    origin_stride = max(1, K // max_active_origins)

    # Map desired atom id to index after we see first frame
    atom_index = None

    # Stream frames
    for k, ids, xyz in frame_reader(file_paths, natoms, cols, assume_sorted_by_id=assume_sorted_by_id):
        if atom_index is None:
            # ids are sorted; find index of desired atom quickly
            # If desired id not present, default to 0 (first atom)
            loc = np.searchsorted(ids, desired_atom_id)
            if loc < len(ids) and ids[loc] == desired_atom_id:
                atom_index = int(loc)
            else:
                atom_index = 0

        # Drop origins that are too old (tau > K)
        keep = []
        for org in origins:
            tau = k - org['k']
            if tau <= K:
                keep.append(org)
        origins = keep

        # Possibly start a new origin at this frame
        if (k % origin_stride) == 0:
            origins.append({
                'k': k,
                'coords': xyz.copy(),                   # store full origin coords
                'single': xyz[atom_index].copy(),       # store single atom origin coord
            })

        if not origins:
            continue  # until we have the first origin

        # Vectorized MSD update for all active origins
        for org in origins:
            tau = k - org['k']
            if tau <= 0 or tau > K:
                continue
            disp = xyz - org['coords']                 # (natoms,3)
            msd_tau = np.mean(np.einsum('ij,ij->i', disp, disp))  # fast |disp|^2 mean
            sdisp = xyz[atom_index] - org['single']
            s_msd_tau = float(np.dot(sdisp, sdisp))

            sum_msd[tau] += msd_tau
            sum_msd_single[tau] += s_msd_tau
            cnt[tau] += 1

    # Finalize
    valid = cnt > 0
    msd = np.zeros_like(sum_msd)
    s_msd = np.zeros_like(sum_msd_single)
    msd[valid] = sum_msd[valid] / cnt[valid]
    s_msd[valid] = sum_msd_single[valid] / cnt[valid]

    # Build time axis (skip tau=0)
    time_fs = np.arange(len(msd), dtype=np.float64) * (timestep_fs * dump_stride)
    return time_fs[1:][valid[1:]], msd[1:][valid[1:]], s_msd[1:][valid[1:]], dump_stride

def save_msd_to_txt(time_fs, msd, single, filename="msd_results_streaming.txt"):
    with open(filename, "w") as f:
        f.write("Time(fs) MSD SingleAtomMSD\n")
        for t, m, s in zip(time_fs, msd, single):
            f.write(f"{t:.6f} {m:.10e} {s:.10e}\n")

if __name__ == "__main__":
    file_paths = [
        "dump_unwrapped_stage1.lammpstrj"
    ]
    # Tunables for huge runs:
    timestep_fs = 0.25
    max_tau_frames = 100_000      # longest lag to compute (frames)
    max_active_origins = 8        # memory knob (<=8 is usually plenty)
    desired_atom_id = 1

    time_fs, msd, s_msd, dump_stride = msd_streaming(
        file_paths,
        timestep_fs=timestep_fs,
        max_tau_frames=max_tau_frames,
        max_active_origins=max_active_origins,
        desired_atom_id=desired_atom_id,
        assume_sorted_by_id=True,   # set False if your dump isn't sorted by id
    )

    print(f"Inferred dump stride: {dump_stride} MD steps")
    print(f"Computed MSD up to {time_fs[-1]:.1f} fs with {len(msd)} lags.")

    save_msd_to_txt(time_fs, msd, s_msd)
    print("Saved: msd_results_streaming.txt")

