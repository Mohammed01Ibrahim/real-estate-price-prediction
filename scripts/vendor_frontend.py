"""Maintain pinned local browser libraries; normal app startup needs no CDN."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "src" / "real_estate" / "static" / "vendor"
ASSETS = {
    "leaflet.js": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
    "leaflet.css": "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
    "leaflet.LICENSE": "https://unpkg.com/leaflet@1.9.4/LICENSE",
    "images/layers.png": "https://unpkg.com/leaflet@1.9.4/dist/images/layers.png",
    "images/layers-2x.png": "https://unpkg.com/leaflet@1.9.4/dist/images/layers-2x.png",
    "images/marker-icon.png": "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    "images/marker-icon-2x.png": "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
    "images/marker-shadow.png": "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
    "leaflet.markercluster.js": "https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js",
    "MarkerCluster.css": "https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css",
    "MarkerCluster.Default.css": "https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css",
    "markercluster.LICENSE": "https://unpkg.com/leaflet.markercluster@1.5.3/MIT-LICENCE.txt",
    "plotly.min.js": "https://unpkg.com/plotly.js-dist-min@3.1.0/plotly.min.js",
    "plotly.LICENSE": "https://unpkg.com/plotly.js@3.1.0/LICENSE",
}


def download(item):
    name, url = item
    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read()
    path = DESTINATION / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return name, {"url": url, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify vendored bytes without network access")
    args = parser.parse_args()
    manifest = DESTINATION / "manifest.json"
    if args.verify:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        for name, expected in recorded.items():
            content = (DESTINATION / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected["sha256"]:
                raise ValueError(f"Vendored file checksum mismatch: {name}")
        print(f"Verified {len(recorded)} pinned browser assets.")
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            result = dict(pool.map(download, ASSETS.items()))
        manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Vendored {len(result)} pinned browser assets with licenses and hashes.")


if __name__ == "__main__":
    main()
