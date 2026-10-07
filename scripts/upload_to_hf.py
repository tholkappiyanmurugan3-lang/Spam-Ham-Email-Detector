r"""
scripts/upload_to_hf.py
=======================
Uploads the local fine-tuned DistilBERT model weights (from models/distilbert_spam)
to Hugging Face Model Hub so it can be automatically loaded by Streamlit Community Cloud.

Usage:
    python scripts/upload_to_hf.py --repo-id <your-hf-username>/spamshield-distilbert
    # Or run interactively:
    python scripts/upload_to_hf.py
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi, login

def main():
    parser = argparse.ArgumentParser(description="Upload SpamShield model to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, help="Target Hugging Face repo ID (e.g. username/spamshield-distilbert)")
    parser.add_argument("--token", type=str, help="Hugging Face write token (optional if already logged in)")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    model_dir = Path("models/distilbert_spam")
    if not (model_dir / "model.safetensors").exists() and not (model_dir / "pytorch_model.bin").exists():
        logger.error(f"Model weights not found in {model_dir.resolve()}. Please train or place model weights there first.")
        sys.exit(1)

    repo_id = args.repo_id
    if not repo_id:
        print("\n" + "="*60)
        print("  Hugging Face Model Hub Uploader for SpamShield")
        print("="*60)
        print("To deploy on Streamlit Community Cloud, your model weights can be")
        print("hosted on Hugging Face Model Hub (free, fast, unlimited bandwidth).")
        print("\nGet your free Hugging Face token at: https://huggingface.co/settings/tokens")
        print("="*60 + "\n")
        repo_id = input("Enter your desired HF repo ID (e.g. tholkappiyan/spamshield-distilbert): ").strip()
        if not repo_id:
            logger.error("No repo ID entered. Aborting.")
            sys.exit(1)

    token = args.token
    if not token:
        import os
        token = os.getenv("HF_TOKEN")
        if not token:
            token = input("Enter your Hugging Face write token: ").strip()

    if token:
        logger.info("Logging in to Hugging Face...")
        login(token=token)

    logger.info(f"Loading local model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    logger.info(f"Pushing model & tokenizer to Hugging Face Hub: {repo_id}...")
    tokenizer.push_to_hub(repo_id, private=args.private)
    model.push_to_hub(repo_id, private=args.private)

    logger.success(f"Successfully uploaded model to: https://huggingface.co/{repo_id}")
    print("\n" + "="*60)
    print("SUCCESS! Model is now available on Hugging Face Hub.")
    print(f"URL: https://huggingface.co/{repo_id}")
    print("\nNext step for Streamlit Community Cloud:")
    print(f"1. In Streamlit Cloud App Settings -> Secrets, add:")
    print(f'   HF_MODEL_REPO = "{repo_id}"')
    print("2. Your cloud app will automatically download and run the model!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
