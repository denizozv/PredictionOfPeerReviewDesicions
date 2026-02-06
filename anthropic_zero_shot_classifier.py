import os
import re
import json
import glob
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
import anthropic

load_dotenv()

SYSTEM_PROMPT = """Sen deneyimli bir akademik editörsün. Aşağıdaki bilimsel makaleyi inceleyerek konferansta (ICML/ICLR/NeurIPS gibi) kabul edilip edilmeyeceği kararını ver.

Değerlendirme kriterleri:
- Kapsam uyumu (dergi/konferans ile alakalı mı?)
- Metodolojik kalite (yöntem güvenilir mi?)
- Örneklem büyüklüğü (yeterli mi?)
- Özgünlük (yeni bir katkı var mı?)

Çıktını kesinlikle aşağıdaki JSON formatında ver, başka hiçbir şey yazma:
{"rejection": true/false, "confidence": 0.00-1.0, "primary_reason": "Short Desc"}"""


class AnthropicZeroShotClassifier:
    """Sends extracted paper data to Anthropic Claude for zero-shot desk rejection classification."""

    def __init__(self, data_dir: str = "extracted_data", model: str = "claude-sonnet-4-20250514",
                 error_log: str = "anthropic_failed_ids.json", limit: int = 0):
        self.data_dir = data_dir
        self.model = model
        self.error_log = error_log
        self.limit = limit  # 0 = process all, >0 = process only N files (for debugging)
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
        self.failed: list[dict] = []

    def _build_user_prompt(self, record: dict) -> str:
        """Build the user prompt from title, abstract, and non-excluded article sections."""
        parts = []

        parts.append(f"Title: {record.get('title', '')}")
        parts.append(f"\nAbstract: {record.get('abstract', '')}")

        sections = [
            s for s in record.get("article", [])
            if not s.get("exclude", False)
        ]

        if sections:
            parts.append("\n--- Article Sections ---")
            for section in sections:
                heading = section.get("heading", "")
                text = section.get("text", "")
                if heading:
                    parts.append(f"\n## {heading}")
                if text:
                    parts.append(text)

        return "\n".join(parts)

    def _classify(self, user_prompt: str) -> tuple[dict | None, dict | None, float]:
        """Send the prompt to Anthropic Claude and return (decision, token_usage, elapsed_seconds)."""
        start = time.time()

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
        except Exception as e:
            print(f"[ERROR] Anthropic API call failed: {e}")
            return None, None, time.time() - start

        elapsed = time.time() - start

        usage = response.usage
        token_info = {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
        } if usage else {}

        content = response.content[0].text if response.content else ""
        content = re.sub(r"^```(?:json)?\s*\n?", "", content.strip())
        content = re.sub(r"\n?```\s*$", "", content.strip())

        try:
            decision = json.loads(content)
        except json.JSONDecodeError:
            print(f"[WARN] Could not parse model response as JSON: {content}")
            return None, token_info, elapsed

        return decision, token_info, elapsed

    def run(self) -> None:
        """Process all extracted JSON files and append zero-shot classification results."""
        pattern = os.path.join(self.data_dir, "*.json")
        files = sorted(glob.glob(pattern))
        if self.limit > 0:
            files = files[:self.limit]
        print(f"[INFO] Running Anthropic zero-shot classification on {len(files)} file(s) with model '{self.model}'")

        for i, filepath in enumerate(files, start=1):
            filename = os.path.basename(filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            existing_models = [e.get("model") for e in data.get("zeroShot", [])]
            if self.model in existing_models:
                print(f"[SKIP] {filename} already has results for model '{self.model}'")
                continue

            user_prompt = self._build_user_prompt(data)
            decision, token_info, elapsed = self._classify(user_prompt)

            if decision is None:
                print(f"[WARN] Skipping {filename} due to classification error.")
                self.failed.append({
                    "id": data.get("id", filename),
                    "filename": filename,
                    "time": datetime.now(timezone.utc).isoformat(),
                })
                continue

            entry = {
                "model": self.model,
                "decision": decision,
                "token": token_info,
                "time": datetime.now(timezone.utc).isoformat(),
            }

            data.setdefault("zeroShot", []).append(entry)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"[{i}/{len(files)}] {filename} -> rejection={decision.get('rejection')} "
                  f"(confidence={decision.get('confidence')}) [{elapsed:.1f}s]")

            time.sleep(0.2)

        self._save_failed()
        print(f"[OK] Anthropic zero-shot classification completed.")

    def _save_failed(self) -> None:
        """Save failed IDs to a JSON file. Remove the file if there are no failures."""
        if self.failed:
            with open(self.error_log, "w", encoding="utf-8") as f:
                json.dump(self.failed, f, indent=2, ensure_ascii=False)
            print(f"[WARN] {len(self.failed)} failed ID(s) saved to '{self.error_log}'")
        elif os.path.exists(self.error_log):
            os.remove(self.error_log)
