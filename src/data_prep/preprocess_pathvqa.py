import os
import json
from pathlib import Path
from PIL import Image
from datasets import load_dataset

def preprocess_pathvqa(raw_dir="./data/raw", processed_dir="./data/processed"):
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    ds = load_dataset("flaviagiammarino/path-vqa")

    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue

        records = []
        for idx, item in enumerate(ds[split]):
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((224, 224))

            fname = f"{split}_{idx:06d}.jpg"
            save_path = os.path.join(processed_dir, fname)
            img.save(save_path)

            records.append({
                "image_path": save_path,
                "question": str(item["question"]),
                "answer": str(item["answer"]),
                "answer_type": item.get("answer_type", "other"),
                "split": split
            })
        
        out_json = os.path.join(processed_dir, f"{split}.json")
        with open(out_json, "w") as f:
            json.dump(records, f, indent=2)

        print(f"[{split}] Saved {len(records)} records to {out_json}")

if __name__ == "__main__":
    preprocess_pathvqa()