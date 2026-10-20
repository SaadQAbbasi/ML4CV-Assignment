"""Pack the PRW test split into one file for the Kaggle dataset used by ``main.ipynb``.

Kaggle serves input datasets file by file, and on some sessions each file takes seconds to open,
so reading the ~14,000 small test files can take hours. One large file reads in seconds, so
``main.ipynb`` unpacks this archive to local disk and reads PRW from there.

Run once by the author, from the repository root (needs the PRW dataset in ``data/``):

    python tools/build_prw_test_archive.py

It writes ``kaggle_upload/prw_test/prw_test_split.bin`` (a tar archive) with a ``dataset-metadata.json``. Upload with

    kaggle datasets create -p kaggle_upload/prw_test
"""
import argparse
import json
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from scipy.io import loadmat  # noqa: E402
from prw_dataset import _mat_strings  # noqa: E402

# A tar archive, but not named .tar: Kaggle unpacks uploaded .tar/.zip files back into separate files.
ARCHIVE_NAME = "prw_test_split.bin"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prw-root", type=Path, default=REPO / "data")
    parser.add_argument("--output", type=Path, default=REPO / "kaggle_upload" / "prw_test")
    parser.add_argument("--kaggle-user", default="saadqasimabbasi")
    args = parser.parse_args()

    root = args.prw_root
    names = _mat_strings(loadmat(root / "frame_test.mat")["img_index_test"])
    members = ["frame_test.mat", "query_info.txt"]
    members += sorted(f"query_box/{path.name}" for path in (root / "query_box").glob("*.jpg"))
    members += [f"frames/{name}.jpg" for name in names]
    members += [f"annotations/{name}.jpg.mat" for name in names]
    missing = [m for m in members if not (root / m).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} PRW files missing, e.g. {missing[:3]}")

    args.output.mkdir(parents=True, exist_ok=True)
    # Uncompressed: the JPEGs do not compress, and an uncompressed tar unpacks fastest.
    with tarfile.open(args.output / ARCHIVE_NAME, "w") as archive:
        for member in members:
            archive.add(root / member, arcname=member)
    (args.output / "dataset-metadata.json").write_text(json.dumps({
        "title": "PRW person search - test split archive",
        "id": f"{args.kaggle_user}/prw-person-search-test-archive",
        "licenses": [{"name": "unknown"}],
    }, indent=2) + "\n")
    size = (args.output / ARCHIVE_NAME).stat().st_size / 2**30
    print(f"Wrote {len(members)} files ({len(names)} test frames), {size:.2f} GiB, to {args.output}")


if __name__ == "__main__":
    main()
