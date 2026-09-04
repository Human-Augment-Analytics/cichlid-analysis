# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "marimo==0.24.0",
#     "polars==1.44.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path
    from urllib.request import urlretrieve

    import marimo as mo
    import polars as pl

    DATA_DIR = Path(__file__).parent / "data"
    FPS = 30.0


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Preliminary exploration of pair-level pose data

    This notebook inventories the Parquet inputs, checks their structure and data
    quality, and summarizes pair interactions. The source video is not loaded.
    """)
    return


@app.cell
def _():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the analysis reproducible without automatically downloading the 31 GB
    # source video. Add other Parquet URLs here as they become available.
    _downloads = {
        "0028_vid_pairs.parquet": "https://www.dropbox.com/scl/fo/zdkqd0n1tt6i9hf9wyta9/AL6msxoSPODIp3jfuXxWb6Q/0028_vid_pairs.parquet?rlkey=l0gupg6wkgdd0fq5dbycuqm8v&st=nb7wminh&dl=1",
    }

    for _filename, _url in _downloads.items():
        _destination = DATA_DIR / _filename
        if _destination.exists():
            continue

        _partial = _destination.with_suffix(_destination.suffix + ".part")
        print(f"Downloading {_filename}...")
        try:
            urlretrieve(_url, _partial)
            _partial.replace(_destination)
        except Exception:
            _partial.unlink(missing_ok=True)
            raise

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {DATA_DIR}")
    return


@app.cell
def _():
    table=pl.read_parquet("data/0028_vid_pairs.parquet")
    table
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
