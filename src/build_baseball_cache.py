from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import unidecode


def build_cache(cache_dir: Path, max_year: int, lahman_zip_path: Path | None = None, lahman_url: str | None = None) -> None:
    import pybaseball as pb
    import pybaseball.lahman as lahman

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Teams
    frames = []
    for year in range(1876, max_year + 1):
        frames.append(pb.team_ids(year)[["yearID", "teamID", "franchID"]])
    teams = pd.concat(frames, ignore_index=True).drop_duplicates()
    teams.to_csv(cache_dir / "teams.csv", index=False)

    # Players (Chadwick register)
    players = pb.chadwick_register()
    players["name_last"] = players["name_last"].apply(lambda x: unidecode.unidecode(str(x)))
    players["name_first"] = players["name_first"].apply(lambda x: unidecode.unidecode(str(x)))
    players.to_csv(cache_dir / "players.csv", index=False)

    # Appearances
    lahman_zip_candidates = [
        "https://github.com/chadwickbureau/baseballdatabank/archive/refs/heads/main.zip",
        "https://github.com/chadwickbureau/baseballdatabank/archive/refs/heads/master.zip",
        "https://github.com/chadwickbureau/baseballdatabank/archive/master.zip",
        "https://github.com/chadwickbureau/baseballdatabank/releases/latest/download/baseballdatabank.zip",
        "https://seanlahman.com/files/database/baseballdatabank-master.zip",
    ]
    default_lahman_zip = cache_dir / "lahman.zip"
    if lahman_zip_path is None and default_lahman_zip.exists():
        lahman_zip_path = default_lahman_zip

    lahman_zip_url = lahman_url
    try:
        import requests
        from io import BytesIO
        import zipfile

        if lahman_zip_url is None:
            for candidate in lahman_zip_candidates:
                try:
                    resp = requests.get(candidate, timeout=30)
                    status = resp.status_code
                    ok = status == 200 and zipfile.is_zipfile(BytesIO(resp.content))
                    print(
                        f"Lahman probe: {candidate} | status={status} | zip={ok}",
                        flush=True,
                    )
                    if ok:
                        lahman_zip_url = candidate
                        break
                except Exception as exc:
                    print(f"Lahman probe failed: {candidate} | {exc}", flush=True)
    except Exception:
        lahman_zip_url = None
    try:
        original_get_lahman_zip = lahman.get_lahman_zip

        def _patched_get_lahman_zip():
            import requests
            from zipfile import ZipFile
            from io import BytesIO

            if lahman_zip_path:
                if not lahman_zip_path.exists():
                    raise RuntimeError(f"Lahman ZIP not found at {lahman_zip_path}")
                return ZipFile(lahman_zip_path)
            if not lahman_zip_url:
                raise RuntimeError("No reachable Lahman ZIP URL found.")
            resp = requests.get(lahman_zip_url, timeout=30)
            resp.raise_for_status()
            return ZipFile(BytesIO(resp.content))

        lahman.get_lahman_zip = _patched_get_lahman_zip
    except Exception:
        original_get_lahman_zip = None

    try:
        import pybaseball.lahman as lahman
        import requests
        import zipfile
        import hashlib
        import inspect
        from io import BytesIO

        def _probe_url(url: str) -> None:
            resp = requests.get(url, timeout=30, stream=True, allow_redirects=True)
            body = resp.raw.read(4096)
            content_type = (resp.headers.get("Content-Type") or "").lower()
            header_bytes = body[:64]
            magic_hex = header_bytes.hex()
            print(f"Lahman URL: {url}", flush=True)
            if resp.url != url:
                print(f"Redirected to: {resp.url}", flush=True)
            print(f"Status: {resp.status_code} | content-type: {content_type}", flush=True)
            print(f"Header sample size: {len(body)} bytes | sha256: {hashlib.sha256(body).hexdigest()[:16]}...", flush=True)
            print(f"Header magic (hex): {magic_hex}", flush=True)
            if body.startswith(b'PK\x03\x04') or body.startswith(b'PK\x05\x06'):
                print("File signature: ZIP (by magic bytes)", flush=True)
            elif body.startswith(b'<!DOCTYPE html') or body.startswith(b'<html'):
                print("File signature: HTML document", flush=True)
            elif body.startswith(b'%PDF-'):
                print("File signature: PDF document", flush=True)
            else:
                print("File signature: unknown", flush=True)

            if "text/html" in content_type or body.startswith(b'<!DOCTYPE html') or body.startswith(b'<html'):
                preview = body[:200].decode("utf-8", errors="replace").replace("\n", " ")
                print(f"Body preview: {preview}", flush=True)

        # Discover URL from module constants or function source
        candidates = []
        for _, value in vars(lahman).items():
            if isinstance(value, str) and value.startswith("http"):
                candidates.append(value)
        try:
            source = inspect.getsource(lahman.get_lahman_zip)
            for token in source.split():
                if token.startswith("http"):
                    candidates.append(token.strip("\"'()"))
        except Exception:
            pass
        try:
            consts = lahman.get_lahman_zip.__code__.co_consts or []
            for value in consts:
                if isinstance(value, str) and value.startswith("http"):
                    candidates.append(value)
        except Exception:
            pass

        # Filter likely Lahman zip URLs
        deduped = []
        for url in candidates:
            if url not in deduped:
                deduped.append(url)
        lahman_urls = [u for u in deduped if "lahman" in u.lower() or u.lower().endswith(".zip")]

        if not lahman_urls:
            print("Lahman zip URL not found in pybaseball.lahman (no http URL constants).", flush=True)
            print("Probing known Lahman sources...", flush=True)
            lahman_urls = [
                "https://github.com/chadwickbureau/baseballdatabank/archive/refs/heads/master.zip",
                "https://github.com/chadwickbureau/baseballdatabank/archive/master.zip",
                "https://github.com/chadwickbureau/baseballdatabank/releases/latest/download/baseballdatabank.zip",
                "https://seanlahman.com/files/database/baseballdatabank-master.zip",
            ]

        for url in lahman_urls:
            _probe_url(url)

        # Monkeypatch requests used by pybaseball.lahman to capture actual URL + headers
        try:
            original_get = lahman.requests.get

            def _logging_get(url, *args, **kwargs):
                resp = original_get(url, *args, **kwargs)
                ct = (resp.headers.get("Content-Type") or "").lower()
                head = resp.content[:64]
                signature = "unknown"
                if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
                    signature = "zip"
                elif head.startswith(b"<!DOCTYPE html") or head.startswith(b"<html"):
                    signature = "html"
                print(
                    f"pybaseball.lahman requests.get -> {url} | status={resp.status_code} | "
                    f"content-type={ct} | signature={signature}",
                    flush=True,
                )
                return resp

            lahman.requests.get = _logging_get
            try:
                lahman.get_lahman_zip()
            except Exception as exc:
                print(f"lahman.get_lahman_zip() failed: {exc}", flush=True)
            finally:
                lahman.requests.get = original_get
        except Exception as exc:
            print(f"Monkeypatch diagnostic failed: {exc}", flush=True)
    except Exception as exc:
        print(f"Lahman download diagnostic failed: {exc}", flush=True)
    appearances = pb.lahman.appearances()
    if original_get_lahman_zip is not None:
        lahman.get_lahman_zip = original_get_lahman_zip
    appearances.to_csv(cache_dir / "appearances.csv", index=False)

    meta = pd.DataFrame(
        [
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "max_year": max_year,
                "teams_rows": len(teams),
                "players_rows": len(players),
                "appearances_rows": len(appearances),
            }
        ]
    )
    meta.to_csv(cache_dir / "cache_meta.csv", index=False)

    print(f"Cache written to: {cache_dir}")
    print(meta.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local baseball cache from pybaseball.")
    parser.add_argument("--cache-dir", default="bin/baseball_cache", help="Output directory for cached CSVs.")
    parser.add_argument("--max-year", type=int, default=2023, help="Max year for team IDs.")
    parser.add_argument("--lahman-zip", default=None, help="Path to a local Lahman zip file.")
    parser.add_argument("--lahman-url", default=None, help="Override URL for Lahman zip download.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    zip_path = Path(args.lahman_zip).expanduser().resolve() if args.lahman_zip else None
    build_cache(cache_dir, max_year=args.max_year, lahman_zip_path=zip_path, lahman_url=args.lahman_url)


if __name__ == "__main__":
    main()
