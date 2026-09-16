"""Download VZCrash (real ego-vehicle crashes, 100 Hz IMU) from HuggingFace.

VZCrash is the dataset named in ml/results/REPORT.md as the next validation step: ~190,000 events,
31,090 of them verified real crashes, plus near-misses and kinematic distractors (potholes,
off-road manoeuvres). Unlike our NHTSA-based crash windows, these are recordings of real crashes
as they happened on the road.

It is gated and non-commercial (CC BY-NC 4.0), so you need a free account:

  1. Create a HuggingFace account:  https://huggingface.co/join
  2. Open https://huggingface.co/datasets/vzc-research-chapter/VZCrash and accept the terms
     (approval is automatic).
  3. Create a READ token:           https://huggingface.co/settings/tokens
  4. Log in once:                   hf auth login
     (or export HF_TOKEN=hf_xxx before running this script)

Then, smallest split first (~1.1 GB):

  python ml/scripts/download_vzcrash.py --split validation --shards 1

Full dataset is ~7.7 GB: --split train --split validation --split test  (no --shards limit).
"""
import argparse
import os
import sys
from pathlib import Path

REPO = "vzc-research-chapter/VZCrash"
SHARDS = {"train": 11, "validation": 3, "test": 2}   # shard counts in the published dataset
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "vzcrash"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", choices=list(SHARDS), default=None,
                    help="split to download (repeatable); default: validation")
    ap.add_argument("--shards", type=int, default=None, help="only the first N shards of each split")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    splits = args.split or ["validation"]

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, LocalTokenNotFoundError
    from huggingface_hub.utils import HfHubHTTPError

    patterns = ["README.md"]
    for split in splits:
        n = SHARDS[split] if args.shards is None else min(args.shards, SHARDS[split])
        patterns += [f"data/{split}-{i:05d}-of-{SHARDS[split]:05d}.parquet" for i in range(n)]

    print(f"downloading {len(patterns) - 1} file(s) from {REPO} into {args.out}")
    try:
        snapshot_download(REPO, repo_type="dataset", local_dir=args.out, allow_patterns=patterns,
                          token=os.environ.get("HF_TOKEN"))
    except (GatedRepoError, LocalTokenNotFoundError, HfHubHTTPError) as e:
        print(f"\nDownload refused: {type(e).__name__}\n{e}\n\n"
              "This dataset is gated. Accept the terms at\n"
              f"  https://huggingface.co/datasets/{REPO}\n"
              "then log in with `hf auth login` (or set HF_TOKEN) and run this again.",
              file=sys.stderr)
        return 1

    files = sorted(args.out.glob("data/*.parquet"))
    total = sum(f.stat().st_size for f in files) / 1e9
    print(f"\n{len(files)} parquet file(s), {total:.1f} GB in {args.out}")
    if files:
        import pyarrow.parquet as pq
        t = pq.read_table(files[0], columns=["event_id", "label", "vehiclesize", "gyro_is_hd"])
        counts = t.column("label").to_pandas().value_counts().sort_index()
        print(f"\nfirst shard: {t.num_rows} events")
        for code, n in counts.items():   # 0=crash, 1=near_miss, 2=normal_driving
            print(f"  {['crash', 'near_miss', 'normal_driving'][code]:16s} {n}")
        print("\nEach event: 16 s of 100 Hz tri-axial accelerometer (g) + gyroscope (deg/s), 1 Hz GPS speed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
