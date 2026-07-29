"""
stage6_eval_harness/hub_upload.py

Uploads the final LoRA adapters to HuggingFace Hub.
Also attaches the model card and evaluation results as Hub artifacts.

Usage:
    python -m stage6_eval_harness.hub_upload \
        --adapter checkpoints/vlm-dpo/dpo_lora_adapters \
        --repo    your-username/identity-doc-vlm-dpo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_config
from utils.logger import get_logger

log = get_logger("stage6.hub_upload")


def upload_to_hub(
    adapter_dir: str,
    repo_id: str,
    model_card_path: str = "outputs/model_card.md",
    results_csv:     str = "outputs/eval_harness/harness_results.csv",
    private:         bool = False,
) -> str:
    """
    Upload LoRA adapters, model card, and eval results to HF Hub.

    Args:
        adapter_dir:     Local directory containing the PEFT adapter files.
        repo_id:         Target HF Hub repo (e.g., "username/model-name").
        model_card_path: Path to the generated model_card.md.
        results_csv:     Path to harness_results.csv.
        private:         If True, creates a private repo.

    Returns:
        URL of the uploaded model.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        log.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return ""

    import os
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN environment variable not set. Cannot upload to Hub.")
        return ""

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", private=private,
                    token=token, exist_ok=True)
        log.info(f"Repository ensured: https://huggingface.co/{repo_id}")
    except Exception as e:
        log.warning(f"create_repo: {e}")

    # Upload adapter files
    adapter_path = Path(adapter_dir)
    if adapter_path.exists():
        log.info(f"Uploading adapter files from {adapter_dir} …")
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload QLoRA LoRA adapters (SFT+DPO)",
        )
        log.info("Adapters uploaded ✓")
    else:
        log.warning(f"Adapter directory not found: {adapter_dir}")

    # Upload model card
    mc_path = Path(model_card_path)
    if mc_path.exists():
        log.info("Uploading model card …")
        api.upload_file(
            path_or_fileobj=str(mc_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
        log.info("Model card uploaded ✓")

    # Upload eval results
    res_path = Path(results_csv)
    if res_path.exists():
        log.info("Uploading evaluation results …")
        api.upload_file(
            path_or_fileobj=str(res_path),
            path_in_repo="eval/harness_results.csv",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add adversarial eval harness results",
        )
        log.info("Eval results uploaded ✓")

    url = f"https://huggingface.co/{repo_id}"
    log.info(f"Upload complete → {url}")
    return url


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 6 — Upload to HF Hub")
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--adapter", required=True,
                        help="Path to LoRA adapter directory to upload")
    parser.add_argument("--repo",    default=None,
                        help="HF Hub repo ID (overrides config)")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_id = args.repo or cfg["stage6"].get("hf_hub_repo", "")
    if not repo_id:
        log.error("No HF Hub repo specified. Set stage6.hf_hub_repo in config or use --repo.")
        sys.exit(1)

    mc_path  = cfg["stage6"].get("model_card_out", "outputs/model_card.md")
    res_csv  = str(Path(cfg["stage6"]["output_dir"]) / "harness_results.csv")

    url = upload_to_hub(
        adapter_dir=args.adapter,
        repo_id=repo_id,
        model_card_path=mc_path,
        results_csv=res_csv,
        private=args.private,
    )
    if url:
        log.info(f"Model available at: {url}")


if __name__ == "__main__":
    main()
