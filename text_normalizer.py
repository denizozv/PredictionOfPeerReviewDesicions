import os
import re
import json
import glob
import unicodedata


# (pattern, replacement, flags)
# Order matters: more specific patterns should come before generic ones.
LATEX_PATTERNS = [
    # Inline math: $...$
    (r"\$[^$]+\$", " ", 0),
    # Display math: \[...\] and \(...\)
    (r"\\\[.*?\\\]", " ", re.DOTALL),
    (r"\\\(.*?\\\)", " ", re.DOTALL),
    # Keep inner text for formatting commands: \textbf{content} -> content
    (r"\\(?:textbf|textit|emph|texttt|mathrm|mathbf|mathit)\{([^}]*)\}", r"\1", 0),
    # Remove reference/citation commands entirely
    (r"\\(?:cite|ref|label|footnote|url)\{[^}]*\}", "", 0),
    # Remove environment markers: \begin{...}, \end{...}
    (r"\\(?:begin|end)\{[^}]*\}", "", 0),
    # Remove any remaining \command{...}
    (r"\\[a-zA-Z]+\{[^}]*\}", "", 0),
    # Remove any remaining \command
    (r"\\[a-zA-Z]+", " ", 0),
    # Remove curly braces
    (r"[{}]", "", 0),
]

HTML_PATTERNS = [
    (r"<[^>]+>", " ", 0),
]


class TextNormalizer:
    """Cleans HTML/LaTeX tags and normalizes special characters in article texts."""

    def __init__(self, data_dir: str = "extracted_data"):
        self.data_dir = data_dir

    @staticmethod
    def _apply_patterns(text: str, patterns: list[tuple]) -> str:
        """Apply a list of (pattern, replacement, flags) to the text in order."""
        for pattern, replacement, flags in patterns:
            text = re.sub(pattern, replacement, text, flags=flags)
        return text

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize unicode characters to their closest ASCII equivalents."""
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse multiple spaces/newlines into a single space and strip."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize(self, text: str) -> str:
        """Apply all normalization steps to a given text."""
        text = self._apply_patterns(text, HTML_PATTERNS)
        text = self._apply_patterns(text, LATEX_PATTERNS)
        text = self._normalize_unicode(text)
        text = self._normalize_whitespace(text)
        return text

    def process(self) -> None:
        """Normalize all article texts in the extracted data directory."""
        pattern = os.path.join(self.data_dir, "*.json")
        files = glob.glob(pattern)
        print(f"[INFO] Normalizing texts in {len(files)} file(s) from '{self.data_dir}/'")

        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            for section in data.get("article", []):
                section["text"] = self.normalize(section.get("text") or "")

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Normalized texts in {len(files)} file(s).")
