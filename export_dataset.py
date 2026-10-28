import csv
import json
import sys
from retrain import generate_dataset, RetrainConfig

# Initialize config and generate the exact same dataset used for training
cfg = RetrainConfig()
train_ds, eval_ds = generate_dataset(cfg)

# Combine them back into one list
all_data = list(train_ds) + list(eval_ds)

# 1. Save as JSON
with open("synthetic_dataset.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2)

# 2. Save as CSV (easier to open in Excel)
with open("synthetic_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["buggy", "fixed"])
    writer.writeheader()
    writer.writerows(all_data)

print(f"Success! Extracted {len(all_data)} examples.")
print("Saved to: synthetic_dataset.csv and synthetic_dataset.json")
