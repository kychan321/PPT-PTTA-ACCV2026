import csv
from pathlib import Path
from collections import defaultdict

RESULT_DIR = Path("results/robust")
SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]

all_rows = []

for scene in SCENES:
    path = RESULT_DIR / f"{scene}.csv"
    if not path.exists():
        print(f"[Warning] Missing file: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)

if len(all_rows) == 0:
    raise RuntimeError("No robust results found.")

# group by robust_tag
groups = defaultdict(list)
for row in all_rows:
    tag = row.get("robust_tag", "unknown")
    groups[tag].append(row)

summary_path = RESULT_DIR / "summary_robust.csv"

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["robust_tag", "num_scenes", "avg_ade_48s", "avg_fde_48s"])

    for tag, rows in sorted(groups.items()):
        ades = [float(r["ade_48s"]) for r in rows]
        fdes = [float(r["fde_48s"]) for r in rows]

        writer.writerow([
            tag,
            len(rows),
            sum(ades) / len(ades),
            sum(fdes) / len(fdes),
        ])

print(f"[Saved summary] {summary_path}")

print("")
print("Robust summary")
print("--------------")
for tag, rows in sorted(groups.items()):
    ades = [float(r["ade_48s"]) for r in rows]
    fdes = [float(r["fde_48s"]) for r in rows]
    print(f"{tag:>24s} | n={len(rows):2d} | ADE={sum(ades)/len(ades):.4f} | FDE={sum(fdes)/len(fdes):.4f}")