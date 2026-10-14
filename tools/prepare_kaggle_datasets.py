"""Build the two public Kaggle datasets used by ``main.ipynb`` and the training notebooks.

Run once by the author, from the repository root:

    python tools/prepare_kaggle_datasets.py --kaggle-user <your-kaggle-username>

It creates two folders, each with a ``dataset-metadata.json``:

``kaggle_upload/code``
    The Python modules imported by the notebooks, plus a snapshot of the official
    DINOv3 repository (``dinov3/``) so the DINOv3 architecture loads without Internet
    and cannot change if the upstream repository changes.
``kaggle_upload/weights``
    The five evaluated checkpoints, downloaded from W&B at the exact versions used in
    the report (requires a W&B login with access), with the training-only state
    (optimizer, AMP scaler, LR scheduler) removed, under the names the notebook expects.

Upload them as public datasets with the Kaggle CLI (``--dir-mode zip`` keeps ``dinov3/``):

    kaggle datasets create -p kaggle_upload/code --dir-mode zip --public
    kaggle datasets create -p kaggle_upload/weights --public

A third dataset, ``saadqasimabbasi/prw-person-search-results``, holds the saved outputs of the
full four-system run (``metrics.csv``, ``provenance.json``, ``retrieval-*.json``,
``visualization_*``) so ``main.ipynb`` only needs to re-evaluate the systems in
``RERUN_SYSTEMS``. Copy those files from a full run's ``final_prw_comparison`` folder into
``kaggle_upload/results`` (with a ``dataset-metadata.json``) and upload it the same way.
"""
import argparse
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))  # project modules (final_comparison) live in src/
CODE_FILES = (
    "eval_function.py", "official_evaluation.py", "prw_evaluation.py", "final_comparison.py",
    "prw_dataset.py", "retrieval_diagnostics.py", "model.py", "backbone.py", "fpn.py", "rpn.py",
    "roi_heads.py", "swin_reid.py", "dino_reid.py", "dinov3_reid.py", "metric_loss.py", "training.py",
    # Imported only by the training notebooks.
    "detection_training.py", "validation_retrieval.py", "offline_mining.py",
)
DINOV3_ARCHIVE = "https://github.com/facebookresearch/dinov3/archive/{ref}.zip"

WANDB_PROJECT = "models-alma-mater-studio/person-search/"
CHECKPOINTS = [
    dict(weights_file="resnet_baseline.pt", artifact=WANDB_PROJECT + "prw-best-model:v6",
         run_id="jw7ysc43", filename="best.pt"),
    dict(weights_file="swin_single_pass.pt", artifact=WANDB_PROJECT + "swin-reid-crop_to_scene-latest:v4",
         run_id="vdrppx5p", filename="latest.pt"),
    dict(weights_file="swin_two_pass.pt", artifact=WANDB_PROJECT + "swin-reid-crop_to_crop-latest:v4",
         run_id="tv1tvl2u", filename="latest.pt"),
    dict(weights_file="swin_dino_reid.pt",
         artifact=WANDB_PROJECT + "prw-dinov3-last4-p16k4-upper-lower-infonce-256x128-valmap-best-model:v23",
         run_id="xhdye71i", filename="best.pt"),
    dict(weights_file="swin_detector.pt", artifact=WANDB_PROJECT + "swin-detector-512x768-best:v0",
         run_id="77d9acgi", filename="best.pt"),
]
TRAINING_ONLY_KEYS = ("optimizer", "scaler", "scheduler")


def write_metadata(folder, kaggle_user, slug, title):
    metadata = {"title": title, "id": f"{kaggle_user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))


def download_dinov3(destination, ref):
    """Extract the DINOv3 repository at ``ref`` into ``destination`` (contains hubconf.py)."""
    print(f"Downloading DINOv3 code ({ref}) ...")
    with urllib.request.urlopen(DINOV3_ARCHIVE.format(ref=ref)) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    top_level = archive.namelist()[0].split("/")[0]
    archive.extractall(destination.parent)
    if destination.exists():
        shutil.rmtree(destination)
    (destination.parent / top_level).rename(destination)
    if not (destination / "hubconf.py").is_file():
        raise RuntimeError("Downloaded DINOv3 archive has no hubconf.py")


def prepare_code(folder, kaggle_user, dinov3_ref):
    folder.mkdir(parents=True, exist_ok=True)
    for name in CODE_FILES:
        shutil.copy2(SOURCE_DIR / name, folder / name)
    download_dinov3(folder / "dinov3", dinov3_ref)
    write_metadata(folder, kaggle_user, "prw-person-search-code", "PRW person search - code")
    print(f"{folder}: {len(CODE_FILES)} modules + dinov3/")


def strip_training_state(checkpoint):
    """Keep everything needed for inference and provenance; drop optimizer-like state."""
    return {key: value for key, value in checkpoint.items()
            if not any(word in key for word in TRAINING_ONLY_KEYS)}


def prepare_weights(folder, kaggle_user, cache):
    import torch
    from final_comparison import resolve_checkpoint
    folder.mkdir(parents=True, exist_ok=True)
    for spec in CHECKPOINTS:
        destination = folder / spec["weights_file"]
        if destination.is_file():
            print(f"{destination} already exported, skipped")
            continue
        source = resolve_checkpoint(spec, cache)
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        kept = strip_training_state(checkpoint)
        removed = sorted(set(checkpoint) - set(kept))
        kept["source_artifact"] = spec["artifact"]
        partial = destination.with_name(destination.name + ".part")
        torch.save(kept, partial)
        partial.rename(destination)  # never leave a truncated file under the final name
        size_mb = destination.stat().st_size / 2**20
        print(f"{destination}  {size_mb:.0f} MB  (removed: {', '.join(removed) or 'nothing'})")
    write_metadata(folder, kaggle_user, "prw-person-search-weights", "PRW person search - weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kaggle-user", required=True, help="Kaggle username that will own the datasets")
    parser.add_argument("--output", type=Path, default=Path("kaggle_upload"))
    parser.add_argument("--dinov3-ref", default="main", help="DINOv3 git branch/commit to bundle")
    parser.add_argument("--cache", type=Path, default=Path("artifacts/wandb_downloads"))
    parser.add_argument("--skip-weights", action="store_true", help="only rebuild the code dataset")
    args = parser.parse_args()
    prepare_code(args.output / "code", args.kaggle_user, args.dinov3_ref)
    if not args.skip_weights:
        prepare_weights(args.output / "weights", args.kaggle_user, args.cache)


if __name__ == "__main__":
    main()
