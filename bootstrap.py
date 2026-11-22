"""Windows launcher bootstrap: acquire original benchmark when available, otherwise demo offline.

Optional network is used ONLY to retrieve the public UCI CSV; never for predictions.
"""
from __future__ import annotations
import argparse
import io
from pathlib import Path
import sys
from urllib.request import Request, urlopen
import zipfile
import pandas as pd
from src.constants import OFFICIAL_DATA, SIMULATED_DATA, OFFICIAL_URLS, MODEL_PATH, METRICS_PATH
from src.data import file_sha256
from src.simulation import make_simulated_data
from src.modeling import train_and_save
import joblib
import sklearn


def fetch_official(timeout: int = 6) -> bool:
    if OFFICIAL_DATA.exists():
        return True
    for url in OFFICIAL_URLS:
        try:
            req = Request(url, headers={"User-Agent": "MachineGuard-AI-Student-Project/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read(2_000_000)
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
                    csv_name = next(x for x in zipped.namelist() if x.lower().endswith(".csv"))
                    raw = zipped.read(csv_name)
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
            required = {"Machine failure", "Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"}
            if not required.issubset(df.columns) or len(df) != 10000:
                raise ValueError("Downloaded dataset has unexpected columns or row count")
            if not df["Machine failure"].isin([0,1]).all():
                raise ValueError("Unexpected target values")
            OFFICIAL_DATA.parent.mkdir(exist_ok=True, parents=True)
            OFFICIAL_DATA.write_bytes(raw)
            print("Original UCI AI4I 2020 CSV downloaded and verified.")
            return True
        except Exception as error:
            print(f"Official source unavailable ({type(error).__name__}); trying next URL.")
    return False


def main(argv=None):
    args = argparse.ArgumentParser()
    args.add_argument("--prefer-official", action="store_true", help="Attempt official UCI download once")
    args.add_argument("--refresh-official", action="store_true", help="Retry official UCI download")
    args.add_argument("--force-train", action="store_true", help="Retrain regardless of saved model")
    opts = args.parse_args(argv)
    SIMULATED_DATA.parent.mkdir(parents=True, exist_ok=True)
    if not SIMULATED_DATA.exists():
        make_simulated_data().to_csv(SIMULATED_DATA, index=False)
        print("Created clearly labeled offline demo simulator dataset.")
    marker = SIMULATED_DATA.parent / ".official_download_attempted"
    if opts.refresh_official or (opts.prefer_official and not OFFICIAL_DATA.exists() and not marker.exists()):
        success = fetch_official()
        if success:
            marker.unlink(missing_ok=True)
        else:
            marker.write_text("Offline fallback active. Run python bootstrap.py --refresh-official to retry.\n", encoding="utf-8")
            print("Using included locally simulated data (NOT original UCI data).")
    from src.data import dataset_path
    current = dataset_path()
    signature = file_sha256(current)
    need_train = opts.force_train or not MODEL_PATH.exists() or not METRICS_PATH.exists()
    if not need_train:
        try:
            existing = joblib.load(MODEL_PATH)
            need_train = existing.get("dataset_hash") != signature or existing.get("sklearn_version") != sklearn.__version__
        except Exception:
            need_train = True
    if need_train:
        print("Training and evaluating the ML pipeline...")
        train_and_save(current)
    else:
        print("Saved model and data match; training skipped.")
    from src.data import source_label, dataset_kind
    print("Active dataset:", source_label(dataset_kind(current)))
    print("Bootstrap completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
