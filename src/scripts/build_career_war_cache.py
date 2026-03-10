from __future__ import annotations

import argparse
import shutil
import ssl
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable

import certifi
import pandas as pd
import requests

from data.io.mlb_reference import clean_name

BASEBALL_REFERENCE_WAR_URL = "https://www.baseball-reference.com/data/"
BASEBALL_REFERENCE_WAR_BAT_URL = "https://www.baseball-reference.com/data/war_daily_bat.txt"
BASEBALL_REFERENCE_WAR_PITCH_URL = "https://www.baseball-reference.com/data/war_daily_pitch.txt"
USER_AGENT = "Mozilla/5.0 (ImmaculateGridCache/1.0)"
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.baseball-reference.com/",
}
ProgressCb = Callable[[str], None]


def _emit(msg: str, progress_cb: ProgressCb | None = None) -> None:
    print(msg, flush=True)
    if progress_cb is not None:
        progress_cb(msg)


def _is_cert_verify_failure(exc: Exception) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, ssl.SSLCertVerificationError):
        return True
    return "CERTIFICATE_VERIFY_FAILED" in str(exc)


def _safe_urlopen(req: urllib.request.Request, timeout: int = 60):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except Exception as exc:
        if not _is_cert_verify_failure(exc):
            raise
        ctx = ssl.create_default_context(cafile=certifi.where())
        try:
            return urllib.request.urlopen(req, timeout=timeout, context=ctx)
        except Exception as fallback_exc:
            raise RuntimeError(
                "SSL certificate verification failed while downloading WAR data. "
                f"Original error: {exc}. Fallback with certifi also failed: {fallback_exc}"
            ) from fallback_exc


def _safe_requests_get(url: str, timeout: int = 60) -> requests.Response:
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp
    except requests.exceptions.SSLError:
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout, verify=certifi.where())
            resp.raise_for_status()
            return resp
        except requests.exceptions.SSLError:
            # Last resort for environments with SSL interception where local trust is broken.
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout, verify=False)
            resp.raise_for_status()
            return resp


class _WarLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return

        attr_dict = dict(attrs)
        href = attr_dict.get("href")

        if not href or not href.lower().endswith(".zip") or "war" not in href.lower():
            return

        file_name = Path(urllib.parse.unquote(href)).name
        if not file_name:
            return

        self.links.append((file_name, href))


def _download_url(url: str, dest: Path):
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    try:
        with _safe_urlopen(req, timeout=60) as response, dest.open("wb") as out:
            shutil.copyfileobj(response, out)
        return
    except Exception:
        # Fallback path for urllib SSL/cert/proxy edge cases.
        resp = _safe_requests_get(url, timeout=60)
        dest.write_bytes(resp.content)


def _fetch_latest_war_zip(dest_dir: Path, progress_cb: ProgressCb | None = None) -> Path:
    _emit(f"[WAR 1/3] Fetching WAR index from {BASEBALL_REFERENCE_WAR_URL}", progress_cb)

    req = urllib.request.Request(BASEBALL_REFERENCE_WAR_URL, headers=REQUEST_HEADERS)
    try:
        with _safe_urlopen(req, timeout=60) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            page = response.read().decode(encoding, errors="replace")
    except Exception:
        # Fallback path for urllib SSL/cert/proxy edge cases.
        response = _safe_requests_get(BASEBALL_REFERENCE_WAR_URL, timeout=60)
        encoding = response.encoding or "utf-8"
        page = response.content.decode(encoding, errors="replace")

    parser = _WarLinkParser()
    parser.feed(page)

    if not parser.links:
        raise RuntimeError("Could not find any WAR ZIP links on Baseball-Reference data page.")

    parser.links.sort(key=lambda entry: entry[0].lower(), reverse=True)

    file_name, href = parser.links[0]
    absolute_url = urllib.parse.urljoin(BASEBALL_REFERENCE_WAR_URL, href)

    _emit(f"[WAR 2/3] Attempting to download WAR ZIP {file_name} from {absolute_url}", progress_cb)

    zip_path = dest_dir / file_name
    _download_url(absolute_url, zip_path)

    return zip_path


def _fetch_war_text_fallback(dest_dir: Path, progress_cb: ProgressCb | None = None) -> tuple[Path | None, Path | None]:
    _emit("[WAR fallback] Attempting direct WAR text downloads (war_daily_bat/pitch)...", progress_cb)

    bat_path = dest_dir / "war_daily_bat.txt"
    pitch_path = dest_dir / "war_daily_pitch.txt"

    bat_ok = False
    pitch_ok = False

    def _curl_download(url: str, dest: Path) -> bool:
        curl_bin = shutil.which("curl")
        if not curl_bin:
            return False
        try:
            subprocess.run(
                [
                    curl_bin,
                    "-fL",
                    "-A",
                    USER_AGENT,
                    "--connect-timeout",
                    "30",
                    "--retry",
                    "2",
                    "--retry-delay",
                    "1",
                    url,
                    "-o",
                    str(dest),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return dest.exists() and dest.stat().st_size > 0
        except Exception:
            return False

    try:
        _download_url(BASEBALL_REFERENCE_WAR_BAT_URL, bat_path)
        bat_ok = bat_path.exists() and bat_path.stat().st_size > 0
    except Exception as exc:
        _emit(f"[WAR fallback] Could not download war_daily_bat.txt via Python HTTP clients: {exc}", progress_cb)
        bat_ok = _curl_download(BASEBALL_REFERENCE_WAR_BAT_URL, bat_path)
        if bat_ok:
            _emit("[WAR fallback] Downloaded war_daily_bat.txt via curl fallback.", progress_cb)

    try:
        _download_url(BASEBALL_REFERENCE_WAR_PITCH_URL, pitch_path)
        pitch_ok = pitch_path.exists() and pitch_path.stat().st_size > 0
    except Exception as exc:
        _emit(f"[WAR fallback] Could not download war_daily_pitch.txt via Python HTTP clients: {exc}", progress_cb)
        pitch_ok = _curl_download(BASEBALL_REFERENCE_WAR_PITCH_URL, pitch_path)
        if pitch_ok:
            _emit("[WAR fallback] Downloaded war_daily_pitch.txt via curl fallback.", progress_cb)

    if not bat_ok and not pitch_ok:
        raise RuntimeError("Direct WAR text fallback failed: neither war_daily_bat.txt nor war_daily_pitch.txt could be downloaded.")

    _emit(
        f"[WAR fallback] Direct WAR text downloads complete (bat={'yes' if bat_ok else 'no'}, pitch={'yes' if pitch_ok else 'no'}).",
        progress_cb,
    )
    return (bat_path if bat_ok else None), (pitch_path if pitch_ok else None)


def _extract_war_texts(zip_path: Path, extract_dir: Path) -> dict[str, Path]:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    mapping: dict[str, Path] = {}

    for txt in extract_dir.rglob("*.txt"):
        lowered = txt.name.lower()

        if "war_daily_bat" in lowered and "bat" not in mapping:
            mapping["bat"] = txt

        if "war_daily_pitch" in lowered and "pitch" not in mapping:
            mapping["pitch"] = txt

    return mapping


def _try_read_war_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"WAR file not found: {path}")

    for sep in ["\t", ",", ";", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", dtype=str, comment="#")
        except Exception:
            continue

        columns_lower = {col.lower(): col for col in df.columns}

        if "player_id" not in columns_lower or "war" not in columns_lower:
            continue

        player_col = columns_lower["player_id"]
        war_col = columns_lower["war"]

        df = df[[player_col, war_col]].copy()
        df.columns = ["player_ID", "WAR"]

        return df

    raise RuntimeError(f"Unable to parse WAR file: {path}")


def _load_war_sources(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []

    for path in paths:
        if not path:
            continue

        frames.append(_try_read_war_file(path))

    if not frames:
        raise RuntimeError("No WAR files were provided.")

    return pd.concat(frames, ignore_index=True)


def build_career_war_cache(
    bat_path: Path | None,
    pitch_path: Path | None,
    cache_dir: Path,
    auto_download: bool = False,
    progress_cb: ProgressCb | None = None,
) -> pd.DataFrame:

    print("*" * 60)
    _emit("Starting career WAR cache build...", progress_cb)

    def _build_from_paths(
        bat_input: Path | None,
        pitch_input: Path | None,
    ) -> pd.DataFrame:

        war_sources = [Path(p).expanduser().resolve() for p in (bat_input, pitch_input) if p]

        if not war_sources:
            raise RuntimeError("At least one WAR source file must be supplied (--bat-file or --pitch-file).")

        return _build_with_sources(war_sources)

    def _build_with_sources(war_sources: list[Path]) -> pd.DataFrame:

        war_df = _load_war_sources(war_sources)

        war_df["player_ID"] = war_df["player_ID"].astype(str).str.strip().str.lower()
        war_df["WAR"] = pd.to_numeric(war_df["WAR"], errors="coerce")

        war_df = war_df.dropna(subset=["player_ID", "WAR"])

        if war_df.empty:
            raise RuntimeError("WAR files did not contain any usable WAR rows.")

        career_war = (
            war_df.groupby("player_ID", as_index=False)["WAR"]
            .sum()
            .rename(columns={"WAR": "career_war"})
        )

        career_war["career_war"] = career_war["career_war"].round(2)

        people_path = cache_dir / "People.csv"

        if not people_path.exists():
            raise FileNotFoundError(f"People.csv not found in cache dir: {cache_dir}")

        people = pd.read_csv(
            people_path,
            dtype=str,
            usecols=["playerID", "bbrefID", "nameFirst", "nameLast"],
        )

        people["bbrefID"] = people["bbrefID"].astype(str).str.strip().str.lower()

        people = people.dropna(subset=["bbrefID"])
        # People.csv can contain duplicate bbrefID rows; keep first to preserve 1:1 WAR merge.
        people = people.drop_duplicates(subset=["bbrefID"], keep="first")

        merged = people.merge(
            career_war,
            left_on="bbrefID",
            right_on="player_ID",
            how="inner",
            validate="1:1",
        )

        merged["full_name"] = (
            merged["nameFirst"].fillna("").astype(str).str.strip()
            + " "
            + merged["nameLast"].fillna("").astype(str).str.strip()
        ).str.strip()

        merged["player_norm"] = merged["full_name"].map(clean_name).map(
            lambda v: v.title() if isinstance(v, str) else v
        )

        merged = merged[merged["player_norm"].notna()]

        output = merged[
            [
                "playerID",
                "bbrefID",
                "full_name",
                "player_norm",
                "career_war",
            ]
        ].copy()

        output_path = cache_dir / "war.csv"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        output.to_csv(output_path, index=False)

        return output

    if auto_download and not bat_path and not pitch_path:
        with tempfile.TemporaryDirectory() as tmpdir:

            temp_path = Path(tmpdir)
            bat_file: Path | None = None
            pitch_file: Path | None = None

            try:
                zip_path = _fetch_latest_war_zip(temp_path, progress_cb=progress_cb)
                extracted = _extract_war_texts(zip_path, temp_path)
                _emit("[WAR 3/3] Extracted WAR ZIP and locating WAR text files...", progress_cb)
                bat_file = extracted.get("bat")
                pitch_file = extracted.get("pitch")
            except Exception as exc:
                _emit(f"[WAR fallback] ZIP discovery/download failed: {exc}", progress_cb)
                bat_file, pitch_file = _fetch_war_text_fallback(temp_path, progress_cb=progress_cb)

            if not bat_file and not pitch_file:
                raise RuntimeError("Downloaded WAR sources but no WAR text files were discovered.")

            output = _build_from_paths(bat_file, pitch_file)
            _emit(f"Career WAR cache build complete ({len(output):,} rows).", progress_cb)
            return output

    output = _build_from_paths(bat_path, pitch_path)
    _emit(f"Career WAR cache build complete ({len(output):,} rows).", progress_cb)
    return output


def main() -> None:

    parser = argparse.ArgumentParser(description="Build a career WAR cache from Baseball-Reference WAR files.")

    parser.add_argument("--bat-file", help="Path to `war_daily_bat.txt` (or equivalent).")
    parser.add_argument("--pitch-file", help="Path to `war_daily_pitch.txt` (or equivalent).")

    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Automatically download the latest WAR ZIP from Baseball-Reference and extract the WAR files.",
    )

    parser.add_argument(
        "--cache-dir",
        default="bin/baseball_cache",
        help="Directory where the local Lahman cache lives.",
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()

    output = build_career_war_cache(
        args.bat_file,
        args.pitch_file,
        cache_dir,
        auto_download=args.auto_download,
    )

    print(f"Wrote {len(output):,} career WAR rows -> {cache_dir / 'war.csv'}")


if __name__ == "__main__":
    main()
