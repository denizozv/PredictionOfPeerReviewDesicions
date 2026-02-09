import os
import json
import glob


class ReviewExtractor:
    """Extracts id, title and abstract from PeerRead review JSON files."""

    def __init__(self, reviews_dir: str, output_dir: str = "extracted_data", test: bool = False):
        self.reviews_dir = reviews_dir
        self.output_dir = output_dir
        self.test = test

    def get_review_files(self) -> list[str]:
        """Return a list of all .json file paths in the reviews directory."""
        pattern = os.path.join(self.reviews_dir, "*.json")
        files = glob.glob(pattern)
        print(f"[INFO] Found {len(files)} review JSON file(s) in '{self.reviews_dir}'")
        return files

    def extract(self) -> list[tuple[str, dict]]:
        """Read each review JSON and extract id, title, and abstract."""
        files = self.get_review_files()
        extracted = []

        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            filename = os.path.basename(filepath)
            record = {
                "id": data.get("id", os.path.splitext(filename)[0]),
                "title": data.get("title", ""),
                "abstract": data.get("abstract", ""),
                "accepted": data.get("accepted", ""),
                "test": self.test,
            }
            extracted.append((filename, record))

        print(f"[INFO] Extracted title & abstract from {len(extracted)} file(s).")
        return extracted

    def save(self) -> list[tuple[str, dict]]:
        """Extract and save each record as a separate JSON file. Returns the records.
        If the file already exists, only updates base fields (id, title, abstract, accepted)
        and preserves everything else (zeroShot, fewShot, article, etc.) as-is."""
        os.makedirs(self.output_dir, exist_ok=True)

        records = self.extract()

        for filename, record in records:
            output_path = os.path.join(self.output_dir, filename)

            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                # Only update base fields, keep the rest untouched
                existing["id"] = record["id"]
                existing["title"] = record["title"]
                existing["abstract"] = record["abstract"]
                existing["accepted"] = record["accepted"]
                existing["test"] = record["test"]
                record = existing

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved {len(records)} review file(s) to '{self.output_dir}/'")
        return records
