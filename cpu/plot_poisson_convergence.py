import csv
import matplotlib.pyplot as plt

iters = []
residuals = []
errors = []

with open("poisson_stats.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        iters.append(int(row["iter"]))
        residuals.append(float(row["residual_l2"]))
        errors.append(float(row["error_l2"]))

plt.figure()
plt.semilogy(iters, residuals, label="Residual L2")
plt.semilogy(iters, errors, label="Error L2")

plt.xlabel("Iteration")
plt.ylabel("L2 norm")
plt.title("Poisson Solver Convergence")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("poisson_convergence.png", dpi=200)
plt.show()
