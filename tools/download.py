"""Download and verify the assets declared by model.json.

Usage: python3 -I tools/download.py MODEL_DIR
"""
import hashlib
import json
import pathlib
import sys
import tempfile
import urllib.request


def main():
    manifest = json.loads((pathlib.Path(__file__).resolve().parents[1] / "model.json").read_text())
    directory = pathlib.Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    for role in ("model", "data", "tokenizer"):
        pin = manifest[role]
        name, size, digest = pin["file"], pin["size"], pin["sha256"]
        destination = directory / name
        if destination.exists():
            if destination.stat().st_size == size and hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
                print(f"Verified {destination}")
                continue
            raise SystemExit(f"{destination} differs from the pinned asset; select a new directory")
        url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{manifest['directory']}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
            temporary = pathlib.Path(output.name)
            try:
                hasher = hashlib.sha256()
                total = 0
                while chunk := response.read(65536):
                    total += len(chunk)
                    if total > size:
                        raise ValueError(f"{name}: download exceeds pinned size")
                    hasher.update(chunk)
                    output.write(chunk)
                output.close()
                if total != size or hasher.hexdigest() != digest:
                    raise ValueError(f"{name}: download failed integrity verification")
                # Link is exclusive: a concurrent install cannot be overwritten.
                destination.hardlink_to(temporary)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Installed {destination}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 -I tools/download.py MODEL_DIR")
    main()
