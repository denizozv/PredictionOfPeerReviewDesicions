# PredictionOfPeerReviewDesicions

LLM-based peer review decision (accept/reject) prediction system.  
Article and review texts are extracted, cleaned, and classified using zero-shot LLM inference.

---

## Pipeline workflow:

1. Extracts article + review texts from peer review data files  
2. Normalizes and cleans the text  
3. Assigns an academic editor role to the LLM to generate a decision  
4. Stores the prediction results  

---

## Output Location

| `extracted_data/` | Model prediction outputs |
| `failed_ids.json` | Records that failed during processing |

---

## Model Output Schema

```json
{
  "model": " ",
  "decision": {
    "rejection": "true/false",
    "confidence": "0.0-1.0",
    "primary_reason": " "
  },
  "token": {
    "prompt_tokens": " ",
    "completion_tokens": " ",
    "total_tokens": " "
  },
  "time": " "
}
