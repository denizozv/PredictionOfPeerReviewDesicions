import json

from repo_cloner import RepoCloner
from review_extractor import ReviewExtractor
from article_extractor import ArticleExtractor
from text_normalizer import TextNormalizer
from openai_classifier import OpenAiClassifier
from gemini_classifier import GeminiClassifier
from anthropic_classifier import AnthropicClassifier

BASE_DIR = "peerread/data/iclr_2017"
SPLITS = ["train", "test"]


if __name__ == '__main__':
    # 0. Load prompts
    with open("prompts/prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # 1. Clone the repo if it doesn't exist
    cloner = RepoCloner(
        repo_url="https://github.com/allenai/PeerRead.git",
        target_dir="peerread",
    )
    cloner.clone()

    # 2. Extract reviews and enrich with article sections for each split
    for split in SPLITS:
        print(f"\n--- Processing '{split}' split ---")

        review_ext = ReviewExtractor(
            reviews_dir=f"{BASE_DIR}/{split}/reviews",
            output_dir="extracted_data",
        )
        records = review_ext.save()

        article_ext = ArticleExtractor(
            parsed_pdfs_dir=f"{BASE_DIR}/{split}/parsed_pdfs",
            output_dir="extracted_data",
        )
        article_ext.enrich(records)

    # 3. Normalize article texts (clean HTML, LaTeX, special characters)
    normalizer = TextNormalizer(data_dir="extracted_data")
    normalizer.process()

    # 4a. Run OpenAI zero-shot classification
    # classifier = OpenAiClassifier(system_prompt=prompts["zeroShot"], approach="zeroShot", data_dir="extracted_data", model="gpt-4o-mini", limit=1)  # DEBUG: limit=1
    # classifier.run()

    # 4b. Run Gemini zero-shot classification
    # gemini_classifier = GeminiClassifier(system_prompt=prompts["zeroShot"], approach="zeroShot", data_dir="extracted_data", model="gemini-2.5-flash-lite", limit=0)  # DEBUG: limit=1
    # gemini_classifier.run()

    # 4c. Run Anthropic Claude zero-shot classification
    # anthropic_classifier = AnthropicClassifier(system_prompt=prompts["zeroShot"], approach="zeroShot", data_dir="extracted_data", model="claude-haiku-4-5", limit=1,  dry_run=True)  # DEBUG: limit=1
    # anthropic_classifier.run()



    # 5a. Run OpenAI zero-shot classification
    # classifier = OpenAiClassifier(system_prompt=prompts["fewShot"], approach="fewShot", data_dir="extracted_data", model="gpt-4o-mini", limit=1)  # DEBUG: limit=1 # DEBUG: limit=1
    # classifier.run()

    # 5b. Run Gemini zero-shot classification
    # gemini_classifier = GeminiClassifier(system_prompt=prompts["fewShot"], approach="fewShot", data_dir="extracted_data", model="gemini-2.5-flash-lite", limit=0)  # DEBUG: limit=1
    # gemini_classifier.run()

    # 5c. Run Anthropic Claude zero-shot classification
    anthropic_classifier = AnthropicClassifier(system_prompt=prompts["fewShot"], approach="fewShot", data_dir="extracted_data", model="claude-haiku-4-5", limit=0)  # DEBUG: dry_run=True, limit=1
    anthropic_classifier.run()
