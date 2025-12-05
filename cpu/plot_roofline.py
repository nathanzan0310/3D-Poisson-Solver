import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse peak values from the command line
parser = argparse.ArgumentParser()
parser.add_argument("--peak-gflops", type=float, required=True,
                    help="Peak compute performance in GFLOP/s")
parser.add_argument("--peak-bw", type=float, required=True,
                    help="Peak memory bandwidth in GB/s")
args = parser.parse_args()

PEAK_GFLOPS = args.peak_gflops
PEAK_BW_GBps = args.peak_bw

# Read your measured kernel point(s)
intensities = []
gflops_measured = []

with open("roofline_jacobi.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        intensities.append(float(row["intensity"]))
        gflops_measured.append(float(row["GFLOPs"]))

intensities = np.array(intensities)
gflops_measured = np.array(gflops_measured)

# Build roofline curves
I_min = 1e-3
I_max = 1e2
I = np.logspace(np.log10(I_min), np.log10(I_max), 200)

# Bandwidth roof: P = I * B_peak
P_bw = I * PEAK_BW_GBps

# Compute roof is flat at PEAK_GFLOPS
P_peak = PEAK_GFLOPS * np.ones_like(I)

# Roofline is the min of the two
P_roof = np.minimum(P_bw, P_peak)

plt.figure(figsize=(8, 6))

# Roofline
plt.loglog(I, P_bw, "--", label="Memory roof (B_peak)")
plt.loglog(I, P_peak, "--", label="Compute roof (P_peak)")
plt.loglog(I, P_roof, "k-", label="Roofline")

# Your measured point(s)
plt.loglog(intensities, gflops_measured, "o", markersize=8,
           label="Jacobi kernel")

for Ipt, Ppt in zip(intensities, gflops_measured):
    plt.annotate(f"I={Ipt:.2f}\nP={Ppt:.1f} GF/s",
                 (Ipt, Ppt),
                 textcoords="offset points",
                 xytext=(5, 5),
                 fontsize=8)

plt.xlabel("Operational intensity [FLOPs / byte]")
plt.ylabel("Performance [GFLOP/s]")
plt.title("Roofline Analysis: 3D Jacobi Poisson Solver (CPU)")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("roofline_poisson.png", dpi=200)
plt.show()
