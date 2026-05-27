import os
import csv
from pathlib import Path

RESULT_DIR = Path("results/clean")
SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]

rows = []

for scene in SCENES:
    csv_path = RESULT_DIR / f"{scene}.csv"

    if not csv_path.exists():
        print(f"[Warning] Missing file: {csv_path}")
        continue

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if len(reader) == 0:
        print(f"[Warning] Empty file: {csv_path}")
        continue

    # 가장 마지막 실행 결과 사용
    last = reader[-1]

    rows.append({
        "scene": scene,
        "ade_48s": float(last["ade_48s"]),
        "fde_48s": float(last["fde_48s"]),
        "model_path": last["model_path"],
        "info": last["info"],
    })

if len(rows) == 0:
    raise RuntimeError("No clean evaluation results found.")

avg_ade = sum(r["ade_48s"] for r in rows) / len(rows)
avg_fde = sum(r["fde_48s"] for r in rows) / len(rows)

summary_path = RESULT_DIR / "summary_clean.csv"

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["scene", "ade_48s", "fde_48s", "model_path", "info"])

    for r in rows:
        writer.writerow([
            r["scene"],
            r["ade_48s"],
            r["fde_48s"],
            r["model_path"],
            r["info"],
        ])

    writer.writerow([
        "avg",
        avg_ade,
        avg_fde,
        "-",
        "average over available scenes",
    ])

print(f"[Saved summary] {summary_path}")
print("")
print("Clean baseline summary")
print("----------------------")
for r in rows:
    print(f"{r['scene']:>6s} | ADE: {r['ade_48s']:.4f} | FDE: {r['fde_48s']:.4f}")
print(f"{'avg':>6s} | ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f}")