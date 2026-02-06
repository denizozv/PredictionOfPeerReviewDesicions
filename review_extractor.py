import os
import json
import glob


class ReviewExtractor:
    """Extracts id, title and abstract from PeerRead review JSON files."""

    def __init__(self, reviews_dir: str, output_dir: str = "extracted_data"):
        self.reviews_dir = reviews_dir
        self.output_dir = output_dir

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
                "zeroShot": [],
            }
            extracted.append((filename, record))

        print(f"[INFO] Extracted title & abstract from {len(extracted)} file(s).")
        return extracted

    def save(self) -> list[tuple[str, dict]]:
        """Extract and save each record as a separate JSON file. Returns the records.
        If the file already exists, preserves the existing zeroShot data."""
        os.makedirs(self.output_dir, exist_ok=True)

        records = self.extract()

        for filename, record in records:
            output_path = os.path.join(self.output_dir, filename)

            # Preserve existing zeroShot data if the file already exists
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                record["zeroShot"] = existing.get("zeroShot", [])

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved {len(records)} review file(s) to '{self.output_dir}/'")
        return records
