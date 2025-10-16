# scripts/filter_pushshift_to_parquet.py
import argparse
import io
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from tqdm import tqdm

MAP = {
    "subreddit": "subreddit",
    "link_id": "post_id",
    "id": "comment_id",
    "body": "text",
    "score": "comment_ups",
    "created_utc": "created_utc",
    "permalink": "permalink",
}


def normalize(o: dict) -> dict:
    return {
        "subreddit": o.get("subreddit", ""),
        "post_id": o.get("link_id", ""),
        "comment_id": o.get("id", ""),
        "text": o.get("body", ""),
        "comment_ups": o.get("score", 0),
        "created_utc": o.get("created_utc", 0),
        "permalink": o.get("permalink", ""),
    }


def main(
    infile: str,
    subs_list: list[str],
    outfile: str,
    batch_size: int,
    max_lines: int | None,
    max_matches: int | None,
):
    in_path, out_path = Path(infile), Path(outfile)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subs = {s.lower() for s in subs_list}

    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    total_lines = 0
    matched = 0
    writer = None
    batch = []

    t0 = time.time()
    with in_path.open("rb") as fh, dctx.stream_reader(fh) as r:
        text_stream = io.TextIOWrapper(r, encoding="utf-8")
        pbar = tqdm(unit=" lines", mininterval=1.0, desc="Scanning")
        for line in text_stream:
            total_lines += 1
            if max_lines and total_lines > max_lines:
                break
            try:
                o = json.loads(line)
            except Exception:
                continue
            sub = (o.get("subreddit") or "").lower()
            if sub in subs:
                batch.append(normalize(o))
                matched += 1
                if max_matches and matched >= max_matches:
                    # flush and stop
                    if batch:
                        table = pa.Table.from_pandas(pd.DataFrame(batch))
                        if writer is None:
                            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
                        writer.write_table(table)
                        batch.clear()
                    break

            if len(batch) >= batch_size:
                table = pa.Table.from_pandas(pd.DataFrame(batch))
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
                writer.write_table(table)
                batch.clear()

            if total_lines % 10000 == 0:
                pbar.set_postfix(matched=matched, outfile=out_path.name)
                pbar.update(10000)

        # progress for last partial chunk
        rem = total_lines % 10000
        if rem:
            pbar.update(rem)
        pbar.close()

    if batch:
        table = pa.Table.from_pandas(pd.DataFrame(batch))
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)
        batch.clear()

    if writer:
        writer.close()

    dt = time.time() - t0
    if matched == 0:
        raise SystemExit("No matching rows written. Check month and subreddit names.")
    print(f"Wrote {matched:,} rows → {out_path}  (scanned {total_lines:,} lines in {dt:.1f}s)")
    print("Tip: pass --max-lines or --max-matches first to test quickly.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--subs", nargs="+", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument("--batch-size", type=int, default=50_000)
    ap.add_argument(
        "--max-lines", type=int, default=None, help="stop after reading this many lines"
    )
    ap.add_argument(
        "--max-matches", type=int, default=None, help="stop after writing this many matches"
    )
    a = ap.parse_args()
    main(a.infile, a.subs, a.outfile, a.batch_size, a.max_lines, a.max_matches)
