"""Parallel range downloader for HuggingFace files (bypasses xet, uses plain HTTP ranges).

Usage: python download_parallel.py <url> <out_path> [threads]
"""
import os
import sys
import threading
import time

import requests

def resolve(url: str) -> tuple[str, int]:
    r = requests.head(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    size = int(r.headers["Content-Length"])
    return r.url, size


def download(url: str, out: str, threads: int = 16):
    final_url, size = resolve(url)
    print(f"[dl] {os.path.basename(out)}: {size/1e9:.2f} GB, {threads} threads", flush=True)
    url_lock = threading.Lock()

    def fresh_url():
        nonlocal final_url
        with url_lock:
            final_url, _ = resolve(url)   # signed URLs expire; re-resolve
            return final_url

    part_dir = out + ".parts"
    os.makedirs(part_dir, exist_ok=True)
    n_parts = threads * 8
    chunk = (size + n_parts - 1) // n_parts
    ranges = [(i, i * chunk, min((i + 1) * chunk, size) - 1) for i in range(n_parts) if i * chunk < size]

    done = [0] * len(ranges)
    lock = threading.Lock()
    t0 = time.time()

    def work(idx: int, beg: int, end: int):
        part = os.path.join(part_dir, f"{idx:05d}")
        have = os.path.getsize(part) if os.path.exists(part) else 0
        want = end - beg + 1
        if have == want:
            with lock:
                done[idx] = want
            return
        for attempt in range(8):
            try:
                u = final_url if attempt == 0 else fresh_url()
                with requests.get(u, headers={"Range": f"bytes={beg + have}-{end}"},
                                  stream=True, timeout=(20, 60)) as r:
                    r.raise_for_status()
                    with open(part, "ab") as f:
                        for data in r.iter_content(1 << 20):
                            f.write(data)
                            have += len(data)
                            with lock:
                                done[idx] = have
                if have == want:
                    return
            except Exception as e:
                print(f"[dl] part {idx} retry {attempt}: {e}", flush=True)
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"part {idx} failed: {have}/{want}")

    pool = [threading.Thread(target=work, args=a) for a in ranges]
    for t in pool:
        t.start()
    while any(t.is_alive() for t in pool):
        time.sleep(5)
        got = sum(done)
        el = time.time() - t0
        print(f"[dl] {got/1e9:.2f}/{size/1e9:.2f} GB  {got/el/1e6:.1f} MB/s  eta {(size-got)/max(got/el,1)/60:.0f} min", flush=True)
    for t in pool:
        t.join()

    # verify BEFORE merge (thread exceptions don't propagate on their own)
    missing = []
    for idx, beg, end in ranges:
        part = os.path.join(part_dir, f"{idx:05d}")
        want = end - beg + 1
        if not os.path.exists(part) or os.path.getsize(part) != want:
            missing.append(idx)
    if missing:
        raise RuntimeError(f"incomplete parts: {missing}")

    print("[dl] merging parts...", flush=True)
    with open(out, "wb") as f:
        for idx, _, _ in ranges:
            part = os.path.join(part_dir, f"{idx:05d}")
            with open(part, "rb") as p:
                while True:
                    b = p.read(1 << 24)
                    if not b:
                        break
                    f.write(b)
    assert os.path.getsize(out) == size, "size mismatch after merge"
    for idx, _, _ in ranges:
        os.remove(os.path.join(part_dir, f"{idx:05d}"))
    os.rmdir(part_dir)
    print(f"[dl] DONE {out} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    url, out = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    for attempt in range(6):
        try:
            download(url, out, threads)
            break
        except RuntimeError as e:
            print(f"[dl] pass {attempt} incomplete: {e}, resuming...", flush=True)
            time.sleep(3)
    else:
        raise SystemExit("download failed after retries")
