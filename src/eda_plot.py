from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "data.yaml") as f:
    class_names = yaml.safe_load(f)["names"]

counts = {c: 0 for c in class_names}
for split in ["train", "valid", "test"]:
    for label_file in (DATA_DIR / split / "labels").glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                if line.strip():
                    counts[class_names[int(line.split()[0])]] += 1

df = pd.DataFrame(list(counts.items()), columns=["classe", "n"]).sort_values("n", ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(df["classe"], df["n"], color="#4C78A8")
plt.gca().invert_yaxis()
plt.xlabel("Nombre d'annotations (train+valid+test)")
plt.title("Distribution des classes - FieldPlant")
plt.tight_layout()
plt.savefig(FIG_DIR / "class_distribution.png", dpi=130)
print("Saved class_distribution.png")
print(df.to_string(index=False))
