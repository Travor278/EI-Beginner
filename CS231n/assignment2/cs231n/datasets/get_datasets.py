from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


DATASET_ROOT = Path(__file__).resolve().parent


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Found {dest.name}, skipping download.")
        return

    print(f"Downloading {dest.name}...")
    urlretrieve(url, dest)
    print(f"Saved to {dest}.")


def ensure_cifar10() -> None:
    archive = DATASET_ROOT / "cifar-10-python.tar.gz"
    extracted = DATASET_ROOT / "cifar-10-batches-py"

    if extracted.exists():
        print("Found CIFAR-10 dataset, skipping extraction.")
    else:
        download("http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", archive)
        print("Extracting CIFAR-10 archive...")
        with tarfile.open(archive, "r:gz") as tar:
            try:
                tar.extractall(DATASET_ROOT, filter="data")
            except TypeError:
                tar.extractall(DATASET_ROOT)
        archive.unlink(missing_ok=True)
        print("CIFAR-10 is ready.")

    ensure_imagenet_sample()


def ensure_imagenet_sample() -> None:
    sample = DATASET_ROOT / "imagenet_val_25.npz"
    download("http://cs231n.stanford.edu/imagenet_val_25.npz", sample)


def ensure_coco() -> None:
    archive = DATASET_ROOT / "coco_captioning.zip"
    extracted = DATASET_ROOT / "coco_captioning"

    if extracted.exists():
        print("Found COCO captioning dataset, skipping extraction.")
        return

    download("http://cs231n.stanford.edu/coco_captioning.zip", archive)
    print("Extracting COCO captioning archive...")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(DATASET_ROOT)
    archive.unlink(missing_ok=True)
    print("COCO captioning data is ready.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cifar",
        action="store_true",
        help="Ensure CIFAR-10 and imagenet_val_25.npz are available.",
    )
    parser.add_argument(
        "--coco",
        action="store_true",
        help="Ensure the COCO captioning subset is available.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download everything used by assignment2.",
    )
    args = parser.parse_args()

    if not (args.cifar or args.coco or args.all):
        args.cifar = True

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    if args.all or args.cifar:
        ensure_cifar10()
    if args.all or args.coco:
        ensure_coco()


if __name__ == "__main__":
    main()
