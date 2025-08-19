import glob
import numpy as np
import matplotlib.pyplot as plt

files = sorted(glob.glob("v2/rep*/vacf_trial*.txt"))
all_vacf = []

for f in files:
    try:
        data = np.loadtxt(f)
        if data.ndim == 2 and data.shape[1] >= 2:
            all_vacf.append(data)
    except:
        continue

if not all_vacf:
    print("No VACF data found.")
    exit(0)

time = all_vacf[0][:, 0]
vacf_values = np.array([x[:, 1] for x in all_vacf])
mean_vacf = np.mean(vacf_values, axis=0)
std_vacf = np.std(vacf_values, axis=0)

plt.figure(figsize=(8, 5))
plt.plot(time / 1000, mean_vacf, label="Mean VACF", color="blue")
plt.fill_between(time, mean_vacf - std_vacf, mean_vacf + std_vacf,
                 alpha=0.3, color="blue", label="± Std Dev")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Time (ps)")
plt.ylabel("VACF (Å²/fs²)")
plt.title("Mean VACF(t) ± Std Dev Across Trials")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vacf_avg_plot.png")
plt.show()
