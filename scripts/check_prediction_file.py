import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()

d = np.load(args.path, allow_pickle=True)

past = d["past_rel"]
future = d["future_rel"]
pred20 = d["pred20_rel"]
best_ade = d["best_ade_rel"]

print("Loaded:", args.path)
print("past_rel     :", past.shape)
print("future_rel   :", future.shape)
print("pred20_rel   :", pred20.shape)
print("best_ade_rel :", best_ade.shape)

# Recompute minADE20
dist = np.linalg.norm(pred20 - future[:, None, :, :], axis=-1)  # [N, 20, 12]
ade_each = dist.mean(axis=-1)                                  # [N, 20]
minade = ade_each.min(axis=1).mean()

# Recompute minFDE20
fde_each = dist[:, :, -1]                                      # [N, 20]
minfde = fde_each.min(axis=1).mean()

print("Recomputed minADE20:", minade)
print("Recomputed minFDE20:", minfde)
print("Saved ADE:", float(d["ade_48s"]))
print("Saved FDE:", float(d["fde_48s"]))