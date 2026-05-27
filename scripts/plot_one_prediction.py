import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--idx", type=int, default=0)
parser.add_argument("--save", type=str, default=None)
args = parser.parse_args()

d = np.load(args.path, allow_pickle=True)

past = d["past_abs"][args.idx]
future = d["future_abs"][args.idx]
best = d["best_ade_abs"][args.idx]
pred20 = d["pred20_abs"][args.idx]

plt.figure(figsize=(6, 6))

# 20 candidate predictions, thin lines
for k in range(pred20.shape[0]):
    plt.plot(pred20[k, :, 0], pred20[k, :, 1], linewidth=0.5, alpha=0.25)

# main lines
plt.plot(past[:, 0], past[:, 1], marker="o", linewidth=2, label="Observed past")
plt.plot(future[:, 0], future[:, 1], marker="o", linewidth=2, label="GT future")
plt.plot(best[:, 0], best[:, 1], marker="o", linewidth=2, label="PPT best-ADE pred")

plt.axis("equal")
plt.legend()
plt.title(f"Sample {args.idx}")

if args.save:
    plt.savefig(args.save, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {args.save}")
else:
    plt.show()