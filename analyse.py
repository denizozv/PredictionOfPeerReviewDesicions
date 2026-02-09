import os
import json
import glob
import math
from datetime import datetime
from collections import defaultdict


class Analyse:
    """
    Tüm extracted_data JSON dosyalarını okuyarak model ve yaklaşım bazında
    kapsamlı bir analiz raporu üretir.  Veriler train / test split'lerine
    göre ayrılarak her split için bağımsız metrikler hesaplanır.

    Hesaplanan metrikler (all / train / test ayrı ayrı):
      - Token istatistikleri (prompt, completion, total)
      - Süre istatistikleri (ilk/son zaman damgası, toplam süre)
      - Confusion matrix (TP, TN, FP, FN)
      - Accuracy, Precision, Recall, F1, Specificity, MCC
      - Confidence istatistikleri (avg, min, max, std)
      - Kabul/ret dağılımları
    """

    SPLITS = ["all", "train", "test"]

    def __init__(self, data_dir: str = "extracted_data", output_dir: str = "analysis"):
        self.data_dir = data_dir
        self.output_dir = output_dir

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        return round(a / b, 4) if b != 0 else 0.0

    @staticmethod
    def _std(values: list[float], mean: float) -> float:
        if len(values) < 2:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return round(math.sqrt(variance), 4)

    @staticmethod
    def _parse_time(ts: str) -> datetime | None:
        """ISO formatındaki zaman damgasını parse eder."""
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------ split helpers
    @staticmethod
    def _split_records(records: list[dict]) -> dict[str, list[dict]]:
        """Kayıtları all / train / test olarak ayırır."""
        train = [r for r in records if not r.get("test", False)]
        test = [r for r in records if r.get("test", False)]
        return {"all": records, "train": train, "test": test}

    @staticmethod
    def _split_pairs(
        pairs: list[tuple[dict, dict]],
    ) -> dict[str, list[tuple[dict, dict]]]:
        """(record, entry) çiftlerini all / train / test olarak ayırır."""
        train = [(rec, ent) for rec, ent in pairs if not rec.get("test", False)]
        test = [(rec, ent) for rec, ent in pairs if rec.get("test", False)]
        return {"all": pairs, "train": train, "test": test}

    # ------------------------------------------------------------------ ground truth
    def _ground_truth(self, records: list[dict]) -> dict:
        total = len(records)
        accepted = sum(1 for r in records if r.get("accepted") is True)
        rejected = total - accepted
        return {
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": self._safe_div(accepted, total),
        }

    # ------------------------------------------------------------------ core
    def run(self) -> dict:
        """Analizi çalıştırır, JSON dosyasına yazar ve sonucu döndürür."""
        pattern = os.path.join(self.data_dir, "*.json")
        files = sorted(glob.glob(pattern))

        if not files:
            print("[WARN] Hiç JSON dosyası bulunamadı.")
            return {}

        # --- Tüm verileri oku ---
        all_records: list[dict] = []
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                all_records.append(json.load(f))

        # --- Split'lere ayır ---
        split_records = self._split_records(all_records)

        # --- Ground truth dağılımı (split bazında) ---
        ground_truth = {s: self._ground_truth(split_records[s]) for s in self.SPLITS}

        # --- Yaklaşım + model bazında grupla ---
        groups: dict[tuple[str, str], list[tuple[dict, dict]]] = defaultdict(list)
        approaches_found: set[str] = set()

        for record in all_records:
            for approach in ("zeroShot", "fewShot"):
                entries = record.get(approach, [])
                for entry in entries:
                    model = entry.get("model", "unknown")
                    groups[(approach, model)].append((record, entry))
                    approaches_found.add(approach)

        # --- Her grup için split bazında metrikleri hesapla ---
        model_results: list[dict] = []

        for (approach, model), pairs in sorted(groups.items()):
            split_pairs = self._split_pairs(pairs)
            result = {
                "approach": approach,
                "model": model,
            }
            for s in self.SPLITS:
                result[s] = self._compute_group_metrics(approach, model, split_pairs[s], include_time=(s == "all"))
            model_results.append(result)

        # --- Genel özet ---
        report = {
            "summary": {
                "total_files": len(all_records),
                "train_files": ground_truth["train"]["total"],
                "test_files": ground_truth["test"]["total"],
                "ground_truth": ground_truth,
                "approaches_found": sorted(approaches_found),
                "total_model_approach_combinations": len(model_results),
            },
            "per_model_approach": model_results,
            "cross_comparison": self._cross_compare(model_results),
        }

        # --- Dosyaya yaz ---
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "analysis_report.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"[OK] Analiz raporu '{output_path}' dosyasına yazıldı.")
        self._print_summary(report)
        return report

    # ------------------------------------------------------------------ group metrics
    def _compute_group_metrics(self, approach: str, model: str,
                               pairs: list[tuple[dict, dict]],
                               include_time: bool = True) -> dict:
        """Tek bir (approach, model, split) grubu için tüm metrikleri hesaplar."""
        n = len(pairs)

        if n == 0:
            return self._empty_metrics()

        # Token değerleri
        prompt_tokens: list[int] = []
        completion_tokens: list[int] = []
        total_tokens: list[int] = []

        # Confidence değerleri
        confidences: list[float] = []

        # Zaman damgaları (sadece all için)
        timestamps: list[datetime] = [] if include_time else None

        # Confusion matrix sayaçları
        # Pozitif sınıf = Acceptance (accepted=True, rejection=False)
        tp = tn = fp = fn = 0
        skipped = 0

        # Dağılım
        predicted_reject = 0
        predicted_accept = 0

        for record, entry in pairs:
            # Token
            token = entry.get("token", {})
            if token:
                prompt_tokens.append(token.get("prompt_tokens", 0))
                completion_tokens.append(token.get("completion_tokens", 0))
                total_tokens.append(token.get("total_tokens", 0))

            # Zaman (sadece all split için)
            if include_time:
                ts = self._parse_time(entry.get("time", ""))
                if ts:
                    timestamps.append(ts)

            # Decision
            decision = entry.get("decision", {})
            pred_rejection = decision.get("rejection")
            confidence = decision.get("confidence")
            actual_accepted = record.get("accepted")

            if confidence is not None:
                confidences.append(float(confidence))

            if pred_rejection is None or actual_accepted is None:
                skipped += 1
                continue

            if pred_rejection:
                predicted_reject += 1
            else:
                predicted_accept += 1

            # Confusion matrix (pozitif sınıf = acceptance)
            pred_accepted = not pred_rejection

            if actual_accepted and pred_accepted:
                tp += 1  # Gerçekten accepted, model de accepted demiş
            elif not actual_accepted and not pred_accepted:
                tn += 1  # Gerçekten rejected, model de rejected demiş
            elif not actual_accepted and pred_accepted:
                fp += 1  # Gerçekten rejected ama model accepted demiş
            elif actual_accepted and not pred_accepted:
                fn += 1  # Gerçekten accepted ama model rejected demiş

        # --- Metrik hesaplamaları ---
        accuracy = self._safe_div(tp + tn, tp + tn + fp + fn)
        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)
        f1 = self._safe_div(2 * precision * recall, precision + recall)
        specificity = self._safe_div(tn, tn + fp)

        # Matthews Correlation Coefficient
        mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = self._safe_div((tp * tn) - (fp * fn), mcc_denom)

        # Token istatistikleri
        sum_prompt = sum(prompt_tokens)
        sum_completion = sum(completion_tokens)
        sum_total = sum(total_tokens)
        avg_prompt = self._safe_div(sum_prompt, len(prompt_tokens)) if prompt_tokens else 0
        avg_completion = self._safe_div(sum_completion, len(completion_tokens)) if completion_tokens else 0
        avg_total = self._safe_div(sum_total, len(total_tokens)) if total_tokens else 0

        # Confidence istatistikleri
        avg_conf = self._safe_div(sum(confidences), len(confidences)) if confidences else 0
        min_conf = round(min(confidences), 4) if confidences else 0
        max_conf = round(max(confidences), 4) if confidences else 0
        std_conf = self._std(confidences, avg_conf) if confidences else 0

        # Süre hesaplama (sadece all split için)
        time_info = {}
        if include_time and timestamps:
            timestamps.sort()
            first_ts = timestamps[0]
            last_ts = timestamps[-1]
            total_elapsed = (last_ts - first_ts).total_seconds()
            time_info = {
                "first_timestamp": first_ts.isoformat(),
                "last_timestamp": last_ts.isoformat(),
                "total_elapsed_seconds": round(total_elapsed, 2),
                "total_elapsed_formatted": self._format_duration(total_elapsed),
                "avg_seconds_per_file": round(total_elapsed / max(n - 1, 1), 2),
            }

        result = {
            "total_classified": n,
            "skipped_invalid": skipped,
            "tokens": {
                "total_prompt_tokens": sum_prompt,
                "total_completion_tokens": sum_completion,
                "total_tokens": sum_total,
                "avg_prompt_tokens": avg_prompt,
                "avg_completion_tokens": avg_completion,
                "avg_total_tokens": avg_total,
                "min_total_tokens": min(total_tokens) if total_tokens else 0,
                "max_total_tokens": max(total_tokens) if total_tokens else 0,
            },
            "confusion_matrix": {
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "note": "Pozitif sinif = Acceptance (accepted=True)",
            },
            "classification_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "specificity": specificity,
                "mcc": mcc,
            },
            "confidence_stats": {
                "mean": avg_conf,
                "min": min_conf,
                "max": max_conf,
                "std": std_conf,
            },
            "prediction_distribution": {
                "predicted_reject": predicted_reject,
                "predicted_accept": predicted_accept,
                "rejection_rate": self._safe_div(predicted_reject, predicted_reject + predicted_accept),
            },
        }

        if include_time and time_info:
            result["time"] = time_info

        return result

    def _empty_metrics(self) -> dict:
        """Boş split için sıfır metrik döndürür."""
        return {
            "total_classified": 0,
            "skipped_invalid": 0,
            "tokens": {
                "total_prompt_tokens": 0, "total_completion_tokens": 0,
                "total_tokens": 0, "avg_prompt_tokens": 0,
                "avg_completion_tokens": 0, "avg_total_tokens": 0,
                "min_total_tokens": 0, "max_total_tokens": 0,
            },
            "confusion_matrix": {
                "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                "note": "Pozitif sinif = Acceptance (accepted=True)",
            },
            "classification_metrics": {
                "accuracy": 0, "precision": 0, "recall": 0,
                "f1_score": 0, "specificity": 0, "mcc": 0,
            },
            "confidence_stats": {"mean": 0, "min": 0, "max": 0, "std": 0},
            "prediction_distribution": {
                "predicted_reject": 0, "predicted_accept": 0, "rejection_rate": 0,
            },
        }

    # ------------------------------------------------------------------ cross compare
    def _cross_compare(self, model_results: list[dict]) -> list[dict]:
        """Aynı modelin zeroShot vs fewShot performansını karşılaştırır (split bazında)."""
        by_model: dict[str, dict[str, dict]] = defaultdict(dict)
        for r in model_results:
            by_model[r["model"]][r["approach"]] = r

        comparisons = []
        for model, approaches in sorted(by_model.items()):
            if len(approaches) < 2:
                continue

            zero = approaches.get("zeroShot", {})
            few = approaches.get("fewShot", {})

            if not zero or not few:
                continue

            comparison = {"model": model}

            for s in self.SPLITS:
                zm = zero.get(s, {}).get("classification_metrics", {})
                fm = few.get(s, {}).get("classification_metrics", {})

                comparison[s] = {
                    "zeroShot_accuracy": zm.get("accuracy", 0),
                    "fewShot_accuracy": fm.get("accuracy", 0),
                    "accuracy_diff": round(fm.get("accuracy", 0) - zm.get("accuracy", 0), 4),
                    "zeroShot_f1": zm.get("f1_score", 0),
                    "fewShot_f1": fm.get("f1_score", 0),
                    "f1_diff": round(fm.get("f1_score", 0) - zm.get("f1_score", 0), 4),
                    "zeroShot_precision": zm.get("precision", 0),
                    "fewShot_precision": fm.get("precision", 0),
                    "zeroShot_recall": zm.get("recall", 0),
                    "fewShot_recall": fm.get("recall", 0),
                    "zeroShot_total_tokens": zero.get(s, {}).get("tokens", {}).get("total_tokens", 0),
                    "fewShot_total_tokens": few.get(s, {}).get("tokens", {}).get("total_tokens", 0),
                    "token_overhead_pct": round(
                        self._safe_div(
                            few.get(s, {}).get("tokens", {}).get("total_tokens", 0)
                            - zero.get(s, {}).get("tokens", {}).get("total_tokens", 0),
                            zero.get(s, {}).get("tokens", {}).get("total_tokens", 0),
                        ) * 100, 2
                    ),
                }

            comparisons.append(comparison)

        return comparisons

    # ------------------------------------------------------------------ formatting
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Saniyeyi okunabilir formata çevirir."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        if minutes < 60:
            return f"{minutes}m {secs:.0f}s"
        hours = int(minutes // 60)
        mins = minutes % 60
        return f"{hours}h {mins}m {secs:.0f}s"

    def _print_summary(self, report: dict) -> None:
        """Konsola okunabilir bir özet basar."""
        summary = report["summary"]
        gt = summary["ground_truth"]

        print(f"\n{'='*70}")
        print(f"  ANALİZ RAPORU ÖZETİ")
        print(f"{'='*70}")
        print(f"  Toplam Dosya       : {summary['total_files']}")
        print(f"    Train            : {summary['train_files']}")
        print(f"    Test             : {summary['test_files']}")

        for s in self.SPLITS:
            g = gt[s]
            print(f"\n  Ground Truth ({s}):")
            print(f"    Accepted         : {g['accepted']}  ({g['acceptance_rate']*100:.1f}%)")
            print(f"    Rejected         : {g['rejected']}  ({(1-g['acceptance_rate'])*100:.1f}%)")

        print(f"\n  Yaklaşımlar        : {', '.join(summary['approaches_found'])}")
        print(f"  Model-Yaklaşım     : {summary['total_model_approach_combinations']} kombinasyon")
        print(f"{'='*70}")

        for r in report["per_model_approach"]:
            print(f"\n  [{r['approach']}] {r['model']}")
            print(f"  {'='*50}")

            for s in self.SPLITS:
                m = r[s]
                cm = m["classification_metrics"]
                tk = m["tokens"]
                tm = m.get("time", {})
                cs = m["confidence_stats"]

                print(f"\n    --- {s.upper()} (n={m['total_classified']}) ---")
                print(f"    Accuracy           : {cm['accuracy']:.4f}")
                print(f"    Precision          : {cm['precision']:.4f}")
                print(f"    Recall             : {cm['recall']:.4f}")
                print(f"    F1 Score           : {cm['f1_score']:.4f}")
                print(f"    Specificity        : {cm['specificity']:.4f}")
                print(f"    MCC                : {cm['mcc']:.4f}")
                print(f"    Toplam Token       : {tk['total_tokens']:,}")
                print(f"    Ort. Token/Dosya   : {tk['avg_total_tokens']:,.1f}")
                print(f"    Ort. Confidence    : {cs['mean']:.4f}  (min={cs['min']}, max={cs['max']}, std={cs['std']})")
                if tm:
                    print(f"    Toplam Süre        : {tm.get('total_elapsed_formatted', 'N/A')}")
                    print(f"    Ort. Süre/Dosya    : {tm.get('avg_seconds_per_file', 0):.2f}s")

        # Cross comparison
        if report.get("cross_comparison"):
            print(f"\n{'='*70}")
            print(f"  CROSS-COMPARISON (zeroShot vs fewShot)")
            print(f"{'='*70}")
            for c in report["cross_comparison"]:
                print(f"\n  Model: {c['model']}")
                for s in self.SPLITS:
                    sc = c.get(s, {})
                    diff_symbol = "+" if sc.get("accuracy_diff", 0) > 0 else ("-" if sc.get("accuracy_diff", 0) < 0 else "=")
                    f1_symbol = "+" if sc.get("f1_diff", 0) > 0 else ("-" if sc.get("f1_diff", 0) < 0 else "=")
                    print(f"    [{s.upper()}]")
                    print(f"    Accuracy  : {sc.get('zeroShot_accuracy', 0):.4f} -> {sc.get('fewShot_accuracy', 0):.4f}  ({diff_symbol} {abs(sc.get('accuracy_diff', 0)):.4f})")
                    print(f"    F1        : {sc.get('zeroShot_f1', 0):.4f} -> {sc.get('fewShot_f1', 0):.4f}  ({f1_symbol} {abs(sc.get('f1_diff', 0)):.4f})")
                    print(f"    Token +%  : {sc.get('token_overhead_pct', 0):.1f}%")

        print(f"\n{'='*70}\n")
