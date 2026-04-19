from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as f:
        shutil.copyfileobj(response, f)


def ensure_coco() -> None:
    target_dir = DATASET_DIR / "coco_captioning"
    if target_dir.exists():
        print("COCO captioning dataset already present.")
        return

    archive = DATASET_DIR / "coco_captioning.zip"
    download_file("http://cs231n.stanford.edu/coco_captioning.zip", archive)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(DATASET_DIR)
    archive.unlink()
    print("COCO captioning dataset ready.")


def ensure_imagenet_val() -> None:
    target = DATASET_DIR / "imagenet_val_25.npz"
    if target.exists():
        print("ImageNet validation subset already present.")
        return
    download_file("http://cs231n.stanford.edu/imagenet_val_25.npz", target)
    print("ImageNet validation subset ready.")


def ensure_cifar() -> None:
    target_dir = DATASET_DIR / "cifar-10-batches-py"
    if target_dir.exists():
        print("CIFAR-10 dataset already present.")
        return

    archive = DATASET_DIR / "cifar-10-python.tar.gz"
    download_file("http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", archive)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(DATASET_DIR)
    archive.unlink()
    print("CIFAR-10 dataset ready.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", action="store_true", help="Download COCO captioning data.")
    parser.add_argument("--imagenet-val", action="store_true", help="Download ImageNet validation subset.")
    parser.add_argument("--cifar", action="store_true", help="Download CIFAR-10.")
    parser.add_argument("--all", action="store_true", help="Download all supported datasets.")
    args = parser.parse_args()

    if args.all or not any((args.coco, args.imagenet_val, args.cifar)):
        args.coco = args.imagenet_val = args.cifar = True

    if args.coco:
        ensure_coco()
    if args.imagenet_val:
        ensure_imagenet_val()
    if args.cifar:
        ensure_cifar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
