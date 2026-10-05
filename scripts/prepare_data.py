"""Safely restore the one known raw CSV from either original archive."""

import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "43c0cdaa2212161a2e9b114e9c2970afe106f2157a025944f0b5b5ec7ff62884"
MEMBERS = ("pe_properties_crude.csv", "DS PRO/pe_properties_crude.csv")


def restore(archive: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest == EXPECTED_SHA256:
            print("Expected raw dataset already present; no changes needed.")
            return
        raise ValueError("Destination already exists with different content; refusing to overwrite it.")
    temporary = destination.with_suffix(".csv.tmp")
    try:
        with zipfile.ZipFile(archive) as source:
            member = next((name for name in MEMBERS if name in source.namelist()), None)
            if member is None:
                raise ValueError("Archive does not contain the expected raw dataset member.")
            with source.open(member) as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        with temporary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != EXPECTED_SHA256:
            raise ValueError("Archive dataset hash does not match the verified historical source.")
        temporary.replace(destination)
        print("Verified raw dataset restored to data/raw/pe_properties_crude.csv.")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT / "data" / "raw" / "pe_properties_crude.csv")
    args = parser.parse_args()
    try:
        restore(args.archive, args.destination)
    except (ValueError, FileNotFoundError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Data preparation failed: {exc}\n")
