import os
import json


EXCLUDED_HEADINGS = [
    "introduction",
    "related work",
    "related works",
    "conclusion",
    "conclusions",
    "discussion",
    "future work",
    "references",
    "bibliography",
    "appendix",
    "appendices",
    "acknowledgment",
    "abstract",
]


class ArticleExtractor:
    """Extracts article sections from parsed PDF JSONs, matched by review IDs."""

    def __init__(self, parsed_pdfs_dir: str, output_dir: str = "extracted_data"):
        self.parsed_pdfs_dir = parsed_pdfs_dir
        self.output_dir = output_dir

    def _get_parsed_pdf_path(self, review_filename: str) -> str:
        """Convert review filename (e.g. 330.json) to parsed_pdf path (e.g. 330.pdf.json)."""
        base = os.path.splitext(review_filename)[0]
        return os.path.join(self.parsed_pdfs_dir, f"{base}.pdf.json")

    @staticmethod
    def _should_exclude(heading: str | None) -> bool:
        """Check if the heading matches any excluded heading keyword."""
        if not heading:
            return True
        heading_lower = heading.lower()
        return any(keyword in heading_lower for keyword in EXCLUDED_HEADINGS)

    def _extract_sections(self, parsed_pdf_path: str) -> list[dict]:
        """Extract heading and text from metadata.sections in a parsed PDF JSON."""
        if not os.path.exists(parsed_pdf_path):
            print(f"[WARN] Parsed PDF not found: {parsed_pdf_path}")
            return []

        with open(parsed_pdf_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sections = data.get("metadata", {}).get("sections", [])
        return [
            {
                "heading": s.get("heading") or "",
                "text": s.get("text") or "",
                "exclude": self._should_exclude(s.get("heading")),
            }
            for s in sections
        ]

    def enrich(self, records: list[tuple[str, dict]]) -> None:
        """Add article sections to existing extracted records and update the JSON files.
        If the file already exists, preserves the existing zeroShot data."""
        enriched_count = 0

        for filename, record in records:
            parsed_pdf_path = self._get_parsed_pdf_path(filename)
            sections = self._extract_sections(parsed_pdf_path)
            record["article"] = sections

            output_path = os.path.join(self.output_dir, filename)

            # Preserve existing zeroShot data if the file already exists
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                record["zeroShot"] = existing.get("zeroShot", [])

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

            enriched_count += 1

        print(f"[OK] Enriched {enriched_count} file(s) with article sections in '{self.output_dir}/'")

