"""
Merges new COCO-format datasets into data/Avocado Augmneted_Dataset/.

Label mapping:
  palta_categoria_1, palta_madura  -> Healthy
  palta_descarte                   -> Disease
  Defectos-y-madurez-Paltas-Hass   -> Disease  (all annotated images)

AVOCADO UNO (RP1-RP5) is intentionally skipped: class semantics unclear.
"""
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "Avocado Augmneted_Dataset"
HASSIA_DB = BASE_DIR / "Hassia Database" / "extracted"

PALTAS_MAP = {
    "palta_categoria_1": "Healthy",
    "palta_madura": "Healthy",
    "palta_descarte": "Disease",
    # "paltas" generic bbox → skip
}

DEFECTOS_MAP = {
    "Defectos-y-madurez-Paltas-Hass": "Disease",
}


def copy_from_coco(dataset_dir: Path, cat_map: dict, split: str, prefix: str) -> dict:
    ann_path = dataset_dir / split / "_annotations.coco.json"
    if not ann_path.exists():
        return {}

    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)

    cats = {c["id"]: c["name"] for c in data["categories"]}
    img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    # Collect class votes per image from annotations
    img_votes: dict = defaultdict(list)
    for ann in data["annotations"]:
        cname = cats.get(ann["category_id"], "")
        our_class = cat_map.get(cname)
        if our_class:
            img_votes[ann["image_id"]].append(our_class)

    # For single-class datasets (Defectos) assign all images that have any annotation
    copied: dict = Counter()
    for img_id, fname in img_id_to_file.items():
        if img_id not in img_votes:
            continue
        dominant = Counter(img_votes[img_id]).most_common(1)[0][0]
        src = dataset_dir / split / fname
        if not src.exists():
            continue
        dst_dir = DATA_DIR / split / dominant
        dst_dir.mkdir(parents=True, exist_ok=True)
        # Use a prefix + original name to avoid collisions
        safe_name = f"{prefix}_{fname}"
        dst = dst_dir / safe_name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied[dominant] += 1

    return dict(copied)


def main():
    grand_total = 0
    for split in ["train", "valid", "test"]:
        print(f"\n=== {split} ===")

        counts = copy_from_coco(
            HASSIA_DB / "Paltas por categoria",
            PALTAS_MAP, split, prefix="palcat"
        )
        print(f"  Paltas por categoria: {counts}")
        grand_total += sum(counts.values())

        counts = copy_from_coco(
            HASSIA_DB / "Defectos y madurez - Paltas Hass",
            DEFECTOS_MAP, split, prefix="defect"
        )
        print(f"  Defectos y madurez:   {counts}")
        grand_total += sum(counts.values())

    print(f"\nTotal new images copied: {grand_total}")

    print("\n=== Final dataset counts ===")
    for split in ["train", "valid", "test"]:
        for cls in ["Disease", "Healthy"]:
            p = DATA_DIR / split / cls
            count = len(list(p.glob("*.*"))) if p.exists() else 0
            print(f"  {split:5s} / {cls:8s}: {count}")


if __name__ == "__main__":
    main()
