import json
from tqdm import tqdm
from collections import Counter

TUI_CATEGORIES = {
    "ENFERMEDAD": {"T047", "T046", "T048", "T191", "T037", "T049"},
    "SINTOMA": {"T184", "T033", "T034"},
    "MEDICAMENTO": {"T121", "T109", "T195", "T200", "T114"},
    "ANATOMIA": {"T029", "T023", "T030", "T024"},
    "PROCEDIMIENTO": {"T060", "T061", "T059"}
}

ALL_RELEVANT_TUIS = set()
for tuis in TUI_CATEGORIES.values():
    ALL_RELEVANT_TUIS.update(tuis)

jsonl_path = "Datasets/datasets/d5e593bc2d8adeee7754be423cd64f5d331ebf26272074a2575616be55697632.0660f30a60ad00fffd8bbf084a18eb3f462fd192ac5563bf50940fc32a850a3c.umls_2022_ab_cat0129.jsonl"

relevant_count = 0
total_count = 0

try:
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f):
            total_count += 1
            data = json.loads(line)
            tuis = set(data.get("types", []))
            if any(tui in ALL_RELEVANT_TUIS for tui in tuis):
                relevant_count += 1
except Exception as e:
    print(f"Error: {e}")

print(f"Total: {total_count}")
print(f"Relevant: {relevant_count}")
print(f"Percentage: {relevant_count/total_count:.2%}")
