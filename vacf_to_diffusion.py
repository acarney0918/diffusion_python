import glob
import numpy as np

files = sorted(glob.glob("v2/rep*/vacf_trial*.txt"))
dt_fs = 2.5  # 0.25 fs × 10 steps

diffusion_constants = []

for f in files:
    try:
        data = np.loadtxt(f)
        vacf = data[:, 1]
        D = np.trapz(vacf, dx=dt_fs) / 3.0
        diffusion_constants.append(D)
    except Exception as e:
        print(f"Failed loading {f}: {e}")

if diffusion_constants:
    diffusion_constants = np.array(diffusion_constants)
    print(f"Diffusion coefficient from VACF: {np.mean(diffusion_constants):.3e} ± {np.std(diffusion_constants):.3e} Å²/fs")
else:
    print("No valid VACF data found for diffusion calculation.")
