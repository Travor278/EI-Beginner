from __future__ import annotations

import tarfile
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
        return

    download("http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", archive)
    print("Extracting CIFAR-10 archive...")
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(DATASET_ROOT, filter="data")
        except TypeError:
            tar.extractall(DATASET_ROOT)
    archive.unlink(missing_ok=True)
    print("CIFAR-10 is ready.")


def ensure_imagenet_sample() -> None:
    sample = DATASET_ROOT / "imagenet_val_25.npz"
    download("http://cs231n.stanford.edu/imagenet_val_25.npz", sample)


def main() -> None:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_cifar10()
    ensure_imagenet_sample()


if __name__ == "__main__":
    main()
