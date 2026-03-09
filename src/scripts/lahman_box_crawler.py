from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import requests


APPEARANCES_REQUIRED_COLUMNS = {"yearID", "teamID", "playerID", "G_all"}
PEOPLE_REQUIRED_COLUMNS = {"playerID", "nameFirst", "nameLast"}
TEAMS_REQUIRED_COLUMNS = {"yearID", "teamID", "franchID"}

DEFAULT_DISCOVERY_PAGES = [
    "https://sabr.org/lahman-database/",
    "https://sabr.app.box.com/s/rsry2en86bimvybwsorumfsxmf91002a?page=1",
    "https://sabr.app.box.com/s/rsry2en86bimvybwsorumfsxmf91002a?page=2",
    "https://sabr.app.box.com/s/y1prhc795jk8zvmelfd3jq7tl389y6cd?page=1",
    "https://sabr.app.box.com/s/y1prhc795jk8zvmelfd3jq7tl389y6cd?page=2",
]


def _peek_csv_header_bytes(content: bytes) -> list[str]:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = content.decode(encoding, errors="strict")
            reader = csv.reader(text.splitlines())
            header = next(reader, [])
            return [str(col).strip() for col in header]
        except Exception:
            continue
    return []


def _peek_csv_header_fileobj(file_obj) -> list[str]:
    try:
        sample = file_obj.read(128 * 1024)
    except Exception:
        return []
    return _peek_csv_header_bytes(sample)


def _normalize_candidate_url(url: str, base_url: str = "https://sabr.org") -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if u.startswith("//"):
        return f"https:{u}"
    if u.startswith("/"):
        return f"{base_url}{u}"
    return u


def _extract_candidate_urls_from_html(html: str, base_url: str = "https://sabr.org") -> list[str]:
    urls = re.findall(r"""href=["']([^"']+)["']""", html or "", flags=re.IGNORECASE)
    candidates: list[str] = []
    for raw in urls:
        u = _normalize_candidate_url(raw, base_url=base_url)
        if not u.lower().startswith("http"):
            continue
        low = u.lower()
        if ".zip" in low or ".csv" in low or "download" in low or "box.com/s/" in low:
            candidates.append(u)
            # For Box shared links, keep the canonical /s/<token> URL (adding download=1
            # creates duplicate candidates and doesn't help folder discovery).
            if "box.com/s/" not in low and "box.com" in low and "download=1" not in low:
                sep = "&" if "?" in u else "?"
                candidates.append(f"{u}{sep}download=1")
    deduped: list[str] = []
    for u in candidates:
        if u not in deduped:
            deduped.append(u)
    return deduped


def _extract_comma_delimited_urls(html: str, base_url: str = "https://sabr.org") -> list[str]:
    text = html or ""
    low = text.lower()
    anchor = "comma-delimited version"
    pos = low.find(anchor)
    if pos < 0:
        return []

    # Focus extraction around the relevant section first.
    window = text[max(0, pos - 2500) : pos + 15000]
    candidates = _extract_candidate_urls_from_html(window, base_url=base_url)
    prioritized: list[str] = []
    for u in candidates:
        lu = u.lower()
        if "box.com/s/" in lu or ".zip" in lu or ".csv" in lu:
            prioritized.append(u)
    deduped: list[str] = []
    for u in prioritized:
        if u not in deduped:
            deduped.append(u)
    return deduped


class LahmanBoxCrawler:
    def __init__(self, progress_cb: Callable[[str], None] | None = None, timeout: int = 60):
        self.progress_cb = progress_cb
        self.timeout = timeout
        self.csv_inventory: list[dict[str, str]] = []

    def _emit(self, msg: str) -> None:
        print(msg, flush=True)
        if self.progress_cb is not None:
            self.progress_cb(msg)

    def _discover_urls(self, discovery_pages: list[str]) -> list[str]:
        self._emit(f"[crawler] Starting discovery across {len(discovery_pages)} page(s)")
        discovered: list[str] = []
        for page in discovery_pages:
            self._emit(f"[crawler] Opening discovery page: {page}")
            try:
                resp = requests.get(page, timeout=self.timeout)
                resp.raise_for_status()
                if "sabr.org/lahman-database" in page.lower():
                    comma_urls = _extract_comma_delimited_urls(resp.text)
                    urls = comma_urls + _extract_candidate_urls_from_html(resp.text)
                    self._emit(
                        f"[crawler] Found {len(comma_urls)} comma-delimited candidate link(s), "
                        f"{len(urls)} total candidate link(s)"
                    )
                else:
                    urls = _extract_candidate_urls_from_html(resp.text)
                    self._emit(f"[crawler] Found {len(urls)} candidate link(s) on page")
                for idx, url in enumerate(urls[:20], start=1):
                    self._emit(f"[crawler]   candidate[{idx}]: {url}")
                discovered.extend(urls)
            except Exception as exc:
                self._emit(f"[crawler] Discovery failed for {page}: {exc}")
        deduped: list[str] = []
        for u in discovered:
            if u not in deduped:
                deduped.append(u)
        self._emit(f"[crawler] Discovery produced {len(deduped)} unique candidate URL(s)")
        return deduped

    def _required_index(self, required_files: dict[str, set[str]]) -> dict[str, tuple[str, set[str]]]:
        idx: dict[str, tuple[str, set[str]]] = {}
        for file_name, cols in required_files.items():
            idx[file_name.lower()] = (file_name, set(cols or set()))
        return idx

    def _match_required_file(
        self,
        required_idx: dict[str, tuple[str, set[str]]],
        file_name: str,
        header: list[str],
        missing: set[str],
    ) -> str | None:
        base = Path(file_name).name.lower()
        header_set = set(header)

        direct = required_idx.get(base)
        if direct is not None:
            canonical_name, required_cols = direct
            if canonical_name.lower() in missing and required_cols.issubset(header_set):
                return canonical_name

        for canonical_name, required_cols in required_idx.values():
            if canonical_name.lower() in missing and required_cols.issubset(header_set):
                return canonical_name
        return None

    def _save_bytes(self, output_dir: Path, canonical_name: str, content: bytes) -> Path:
        out_path = output_dir / canonical_name
        out_path.write_bytes(content)
        return out_path

    def _try_zip_content(
        self,
        output_dir: Path,
        required_idx: dict[str, tuple[str, set[str]]],
        missing: set[str],
        zip_bytes: bytes,
        source_label: str,
    ) -> dict[str, Path]:
        found: dict[str, Path] = {}
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            self._emit(f"[crawler] Opened ZIP from {source_label}, entries={len(zf.namelist())}")
            for name in zf.namelist():
                if not missing:
                    break
                if not name.lower().endswith(".csv"):
                    continue
                self._emit(f"[crawler] Inspecting ZIP CSV header: {name}")
                file_size = ""
                try:
                    file_size = str(zf.getinfo(name).file_size)
                except Exception:
                    file_size = ""
                with zf.open(name) as f:
                    header = _peek_csv_header_fileobj(f)
                match_any = any(cols.issubset(set(header)) for _, cols in required_idx.values())
                self._record_csv_candidate(
                    source=source_label,
                    name=name,
                    size_bytes=file_size,
                    header=header,
                    match=match_any,
                )
                canonical = self._match_required_file(required_idx, name, header, missing)
                if canonical is None:
                    continue
                with zf.open(name) as f:
                    content = f.read()
                out = self._save_bytes(output_dir, canonical, content)
                missing.remove(canonical.lower())
                found[canonical] = out
                self._emit(f"[crawler] Matched and saved {canonical} from ZIP member {name}")
        return found

    def _try_csv_content(
        self,
        output_dir: Path,
        required_idx: dict[str, tuple[str, set[str]]],
        missing: set[str],
        content: bytes,
        file_name: str,
        source_label: str,
    ) -> dict[str, Path]:
        found: dict[str, Path] = {}
        header = _peek_csv_header_bytes(content)
        self._emit(f"[crawler] Header peek for {source_label}: {file_name} -> {header[:8]}")
        match_any = any(cols.issubset(set(header)) for _, cols in required_idx.values())
        self._record_csv_candidate(
            source=source_label,
            name=file_name,
            size_bytes=str(len(content)),
            header=header,
            match=match_any,
        )
        canonical = self._match_required_file(required_idx, file_name, header, missing)
        if canonical is None:
            return found
        out = self._save_bytes(output_dir, canonical, content)
        missing.remove(canonical.lower())
        found[canonical] = out
        self._emit(f"[crawler] Matched and saved {canonical} from CSV {file_name}")
        return found

    def _download_box_file_content(self, shared_link: str, file_id: str) -> tuple[bytes, str]:
        headers = {"BoxApi": f"shared_link={shared_link}"}
        info_resp = requests.get(
            f"https://api.box.com/2.0/files/{file_id}",
            headers=headers,
            params={"fields": "id,name,type"},
            timeout=self.timeout,
        )
        info_resp.raise_for_status()
        file_name = str((info_resp.json() or {}).get("name") or "")

        dl_resp = requests.get(
            f"https://api.box.com/2.0/files/{file_id}/content",
            headers=headers,
            timeout=self.timeout,
            allow_redirects=True,
        )
        dl_resp.raise_for_status()
        return dl_resp.content, file_name

    def _download_box_file_via_direct_link(self, file_url: str) -> bytes:
        direct_url = file_url if "download=1" in file_url else f"{file_url}{'&' if '?' in file_url else '?'}download=1"
        self._emit(f"[crawler] Attempting direct Box download: {direct_url}")
        resp = requests.get(direct_url, timeout=self.timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp.content

    def _looks_like_html_shell(self, content: bytes) -> bool:
        head = (content or b"")[:512].lower()
        return b"<!doctype html" in head or b"<html" in head or b"data-resin-client" in head

    def _extract_request_token_from_html(self, html: str) -> str:
        match = re.search(r'"requestToken"\s*:\s*"([a-fA-F0-9]+)"', html or "")
        return match.group(1) if match else ""

    def _download_box_file_by_id(
        self,
        session: requests.Session,
        host: str,
        token: str,
        file_id: str,
        request_token: str = "",
    ) -> bytes:
        candidate_urls = []
        if request_token:
            candidate_urls.append(
                f"https://{host}/index.php?rm=box_download_shared_file&shared_name={token}&file_id=f_{file_id}&request_token={request_token}"
            )
            candidate_urls.append(
                f"https://app.box.com/index.php?rm=box_download_shared_file&shared_name={token}&file_id=f_{file_id}&request_token={request_token}"
            )
        candidate_urls.extend(
            [
                f"https://{host}/index.php?rm=box_download_shared_file&shared_name={token}&file_id=f_{file_id}",
                f"https://app.box.com/index.php?rm=box_download_shared_file&shared_name={token}&file_id=f_{file_id}",
                f"https://{host}/s/{token}/file/{file_id}?download=1",
            ]
        )
        last_exc: Exception | None = None
        for url in candidate_urls:
            try:
                self._emit(f"[crawler] Attempting Box file_id download: {url}")
                resp = session.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True,
                    headers={
                        "Referer": f"https://{host}/s/{token}",
                        "User-Agent": "Mozilla/5.0",
                    },
                )
                resp.raise_for_status()
                content = resp.content
                if self._looks_like_html_shell(content):
                    raise RuntimeError("received HTML shell instead of file bytes")
                return content
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"Failed to download Box file_id={file_id}: {last_exc}")

    def _extract_box_file_refs_from_html(self, html: str, host: str, token: str) -> list[dict[str, str]]:
        text = html or ""
        refs: list[dict[str, str]] = []

        # Direct URL patterns
        path_hits = re.findall(rf"/s/{re.escape(token)}/file/(\d+)", text)
        full_hits = re.findall(rf"https://{re.escape(host)}/s/{re.escape(token)}/file/(\d+)", text)
        for file_id in path_hits + full_hits:
            refs.append({"id": file_id, "name": "", "url": f"https://{host}/s/{token}/file/{file_id}"})

        # Embedded JSON payload patterns
        patterns = [
            r'"type":"file","id":"(\d+)","name":"([^"]+)"',
            r'"id":"(\d+)","type":"file","name":"([^"]+)"',
            r'"typedID":"f_(\d+)".{0,120}?"name":"([^"]+)"',
            r'"typedID":"f_(\d+)".{0,2000}?"name":"([^"]+)"',
        ]
        for pat in patterns:
            for file_id, name in re.findall(pat, text, flags=re.DOTALL):
                refs.append({"id": file_id, "name": name, "url": f"https://{host}/s/{token}/file/{file_id}"})

        # Parse Box.postStreamData JSON, which contains shared-folder items list.
        post_stream_match = re.search(r"Box\.postStreamData\s*=\s*(\{.*?\});\s*</script>", text, flags=re.DOTALL)
        if post_stream_match:
            raw_json = post_stream_match.group(1)
            try:
                payload = json.loads(raw_json)
                shared_folder = payload.get("/app-api/enduserapp/shared-folder") or {}
                items = shared_folder.get("items") or []
                for item in items:
                    if str(item.get("type") or "").lower() != "file":
                        continue
                    file_id = str(item.get("id") or "")
                    name = str(item.get("name") or "")
                    if not file_id:
                        continue
                    refs.append({"id": file_id, "name": name, "url": f"https://{host}/s/{token}/file/{file_id}"})
            except Exception as exc:
                self._emit(f"[crawler] Could not parse Box.postStreamData JSON: {exc}")

        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for ref in refs:
            key = str(ref.get("id") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(ref)
        return deduped

    def _record_csv_candidate(self, source: str, name: str, size_bytes: str, header: list[str], match: bool) -> None:
        row = {
            "source": source[:40],
            "name": (name or "")[:34],
            "size": size_bytes or "",
            "cols": str(len(header)),
            "match": "yes" if match else "no",
            "header": ", ".join(header[:6])[:72],
        }
        self.csv_inventory.append(row)

    def _emit_csv_inventory(self) -> None:
        if not self.csv_inventory:
            self._emit("[crawler] CSV inventory: no CSV headers inspected.")
            return
        headers = ["name", "size", "cols", "match", "header", "source"]
        widths = {h: len(h) for h in headers}
        for row in self.csv_inventory:
            for h in headers:
                widths[h] = max(widths[h], len(str(row.get(h, ""))))
        line = " | ".join(h.ljust(widths[h]) for h in headers)
        sep = "-+-".join("-" * widths[h] for h in headers)
        self._emit("[crawler] CSV inventory:")
        self._emit(line)
        self._emit(sep)
        for row in self.csv_inventory[-80:]:
            self._emit(" | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))

    def _crawl_box_shared_via_html(
        self,
        output_dir: Path,
        required_idx: dict[str, tuple[str, set[str]]],
        missing: set[str],
        host: str,
        token: str,
    ) -> dict[str, Path]:
        found: dict[str, Path] = {}
        seen_file_ids: set[str] = set()
        max_pages = 6
        session = requests.Session()
        request_token = ""
        required_names = {name.lower() for name in required_idx.keys()}

        for page_num in range(1, max_pages + 1):
            if not missing:
                break
            page_url = f"https://{host}/s/{token}?page={page_num}"
            self._emit(f"[crawler] HTML fallback opening Box page: {page_url}")
            try:
                resp = session.get(
                    page_url,
                    timeout=self.timeout,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()
                if not request_token:
                    request_token = self._extract_request_token_from_html(resp.text)
                    if request_token:
                        self._emit("[crawler] Captured Box request token from HTML payload")
            except Exception as exc:
                self._emit(f"[crawler] HTML fallback failed to open page {page_num}: {exc}")
                break

            file_refs = self._extract_box_file_refs_from_html(resp.text, host=host, token=token)
            new_refs = [r for r in file_refs if str(r.get("id") or "") not in seen_file_ids]
            self._emit(f"[crawler] HTML fallback found {len(new_refs)} new Box file reference(s) on page {page_num}")
            if not new_refs and page_num > 2:
                break

            for ref in new_refs:
                if not missing:
                    break
                file_id = str(ref.get("id") or "")
                file_name = str(ref.get("name") or f"{file_id}.csv")
                file_link = str(ref.get("url") or f"https://{host}/s/{token}/file/{file_id}")
                if not file_id:
                    continue
                # If Box payload gives us a concrete filename, skip non-required files.
                if file_name and file_name.lower() not in required_names:
                    continue
                seen_file_ids.add(file_id)
                self._emit(f"[crawler] HTML fallback inspecting file: id={file_id} name={file_name}")
                try:
                    content = self._download_box_file_by_id(
                        session=session,
                        host=host,
                        token=token,
                        file_id=file_id,
                        request_token=request_token,
                    )
                    if zipfile.is_zipfile(BytesIO(content)):
                        found.update(
                            self._try_zip_content(
                                output_dir, required_idx, missing, content, source_label=f"box-html:{file_id}"
                            )
                        )
                    else:
                        found.update(
                            self._try_csv_content(
                                output_dir,
                                required_idx,
                                missing,
                                content,
                                file_name=file_name,
                                source_label=f"box-html:{file_id}",
                            )
                        )
                except Exception as exc:
                    self._emit(f"[crawler] HTML fallback could not download {file_link}: {exc}")

        return found

    def _crawl_box_shared(
        self,
        output_dir: Path,
        required_idx: dict[str, tuple[str, set[str]]],
        missing: set[str],
        url: str,
    ) -> dict[str, Path]:
        found: dict[str, Path] = {}
        parsed = urlparse(url)
        token_match = re.search(r"/s/([A-Za-z0-9]+)", parsed.path or "")
        if not token_match:
            return found

        shared_token = token_match.group(1)
        host = parsed.netloc or "sabr.app.box.com"
        if ".app.box.com" not in host and host.endswith(".box.com"):
            host = host.replace(".box.com", ".app.box.com")
        # Box API is most reliable with app.box.com shared_link format, even when
        # links are branded under sabr.app.box.com or sabr.box.com.
        shared_link = f"https://app.box.com/s/{shared_token}"
        file_id_match = re.search(r"/file/(\d+)", parsed.path or "")
        explicit_file_id = file_id_match.group(1) if file_id_match else None
        headers = {"BoxApi": f"shared_link={shared_link}"}

        if explicit_file_id:
            self._emit(f"[crawler] Opening Box file link: {url}")
            try:
                content, file_name = self._download_box_file_content(shared_link, explicit_file_id)
                if zipfile.is_zipfile(BytesIO(content)):
                    found.update(
                        self._try_zip_content(output_dir, required_idx, missing, content, source_label=f"box-file:{file_name}")
                    )
                else:
                    found.update(
                        self._try_csv_content(
                            output_dir, required_idx, missing, content, file_name=file_name, source_label=f"box-file:{file_name}"
                        )
                    )
            except Exception as exc:
                self._emit(f"[crawler] Box file API fetch failed: {exc}")
                try:
                    content = self._download_box_file_via_direct_link(url)
                    synthetic_name = f"{explicit_file_id}.csv"
                    if zipfile.is_zipfile(BytesIO(content)):
                        found.update(
                            self._try_zip_content(
                                output_dir, required_idx, missing, content, source_label=f"box-file-direct:{explicit_file_id}"
                            )
                        )
                    else:
                        found.update(
                            self._try_csv_content(
                                output_dir,
                                required_idx,
                                missing,
                                content,
                                file_name=synthetic_name,
                                source_label=f"box-file-direct:{explicit_file_id}",
                            )
                        )
                except Exception as direct_exc:
                    self._emit(f"[crawler] Box file direct download failed: {direct_exc}")
            return found

        self._emit(f"[crawler] Opening Box shared link: {shared_link}")
        try:
            meta_resp = requests.get("https://api.box.com/2.0/shared_items", headers=headers, timeout=self.timeout)
            meta_resp.raise_for_status()
            item = meta_resp.json() or {}
        except Exception as exc:
            self._emit(f"[crawler] Box shared metadata lookup failed: {exc}")
            self._emit("[crawler] Falling back to HTML-based Box crawl")
            return self._crawl_box_shared_via_html(
                output_dir=output_dir,
                required_idx=required_idx,
                missing=missing,
                host=host,
                token=shared_token,
            )

        item_type = str(item.get("type") or "").lower()
        if item_type == "file":
            file_id = str(item.get("id") or "")
            file_name = str(item.get("name") or "")
            if not file_id:
                return found
            self._emit(f"[crawler] Shared link points to file: {file_name}")
            try:
                content, _ = self._download_box_file_content(shared_link, file_id)
                if zipfile.is_zipfile(BytesIO(content)):
                    found.update(
                        self._try_zip_content(output_dir, required_idx, missing, content, source_label=f"box-file:{file_name}")
                    )
                else:
                    found.update(
                        self._try_csv_content(
                            output_dir, required_idx, missing, content, file_name=file_name, source_label=f"box-file:{file_name}"
                        )
                    )
            except Exception as exc:
                self._emit(f"[crawler] Box shared file fetch failed: {exc}")
            return found

        if item_type != "folder":
            return found

        folder_id = str(item.get("id") or "")
        if not folder_id:
            return found
        self._emit(f"[crawler] Opening Box folder: {item.get('name') or folder_id}")

        offset = 0
        limit = 1000
        while missing:
            try:
                list_resp = requests.get(
                    f"https://api.box.com/2.0/folders/{folder_id}/items",
                    headers=headers,
                    params={"limit": limit, "offset": offset, "fields": "id,name,type"},
                    timeout=self.timeout,
                )
                list_resp.raise_for_status()
                payload = list_resp.json() or {}
                entries = payload.get("entries") or []
                total_count = int(payload.get("total_count") or len(entries))
                self._emit(f"[crawler] Crawling Box files offset={offset}, fetched={len(entries)}")
            except Exception as exc:
                self._emit(f"[crawler] Box folder crawl failed: {exc}")
                break

            if not entries:
                break

            for entry in entries:
                if not missing:
                    break
                if str(entry.get("type") or "").lower() != "file":
                    continue
                file_id = str(entry.get("id") or "")
                file_name = str(entry.get("name") or "")
                low_name = file_name.lower()
                if not file_id:
                    continue
                if not (low_name.endswith(".csv") or low_name.endswith(".zip")):
                    continue

                self._emit(f"[crawler] Inspecting Box file: {file_name}")
                try:
                    content, _ = self._download_box_file_content(shared_link, file_id)
                    if zipfile.is_zipfile(BytesIO(content)):
                        found.update(
                            self._try_zip_content(
                                output_dir, required_idx, missing, content, source_label=f"box-folder:{file_name}"
                            )
                        )
                    else:
                        found.update(
                            self._try_csv_content(
                                output_dir,
                                required_idx,
                                missing,
                                content,
                                file_name=file_name,
                                source_label=f"box-folder:{file_name}",
                            )
                        )
                except Exception as exc:
                    self._emit(f"[crawler] Failed to inspect {file_name}: {exc}")

            offset += len(entries)
            if offset >= total_count:
                break

        return found

    def _crawl_http_url(
        self,
        output_dir: Path,
        required_idx: dict[str, tuple[str, set[str]]],
        missing: set[str],
        url: str,
    ) -> dict[str, Path]:
        found: dict[str, Path] = {}
        self._emit(f"[crawler] Downloading URL: {url}")
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.content
        except Exception as exc:
            self._emit(f"[crawler] URL download failed: {exc}")
            return found

        path_name = Path(urlparse(url).path).name or "download.csv"
        if zipfile.is_zipfile(BytesIO(content)):
            found.update(self._try_zip_content(output_dir, required_idx, missing, content, source_label=url))
        elif path_name.lower().endswith(".csv"):
            found.update(
                self._try_csv_content(output_dir, required_idx, missing, content, file_name=path_name, source_label=url)
            )
        return found

    def _crawl_local_zip(
        self,
        output_dir: Path,
        required_idx: dict[str, tuple[str, set[str]]],
        missing: set[str],
        zip_path: Path,
    ) -> dict[str, Path]:
        found: dict[str, Path] = {}
        self._emit(f"[crawler] Opening local ZIP: {zip_path}")
        if not zip_path.exists():
            self._emit(f"[crawler] Local ZIP not found: {zip_path}")
            return found
        try:
            content = zip_path.read_bytes()
            if not zipfile.is_zipfile(BytesIO(content)):
                self._emit(f"[crawler] Not a ZIP file: {zip_path}")
                return found
            found.update(
                self._try_zip_content(output_dir, required_idx, missing, content, source_label=f"local-zip:{zip_path.name}")
            )
        except Exception as exc:
            self._emit(f"[crawler] Failed to read local ZIP: {exc}")
        return found

    def download_required_csvs(
        self,
        output_dir: Path,
        required_files: dict[str, set[str]],
        candidate_urls: list[str] | None = None,
        discovery_pages: list[str] | None = None,
        local_zip_paths: list[Path] | None = None,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        required_idx = self._required_index(required_files)
        missing = {name.lower() for name in required_files.keys()}
        found: dict[str, Path] = {}

        self._emit(f"[crawler] Required files: {', '.join(required_files.keys())}")

        for zip_path in (local_zip_paths or []):
            if not missing:
                break
            found.update(self._crawl_local_zip(output_dir, required_idx, missing, Path(zip_path)))

        sources: list[str] = []
        if candidate_urls:
            sources.extend([u for u in candidate_urls if u])
        sources.extend(self._discover_urls(discovery_pages or DEFAULT_DISCOVERY_PAGES))

        deduped_sources: list[str] = []
        seen_box_tokens: set[str] = set()
        for url in sources:
            low = url.lower()
            if "box.com/s/" in low:
                m = re.search(r"/s/([A-Za-z0-9]+)", url)
                if m:
                    token = m.group(1).lower()
                    if token in seen_box_tokens:
                        continue
                    seen_box_tokens.add(token)
            if url not in deduped_sources:
                deduped_sources.append(url)

        for url in deduped_sources:
            if not missing:
                break
            low = url.lower()
            if "box.com/s/" in low:
                found.update(self._crawl_box_shared(output_dir, required_idx, missing, url))
            else:
                found.update(self._crawl_http_url(output_dir, required_idx, missing, url))

        self._emit_csv_inventory()

        if missing:
            missing_labels = ", ".join(sorted(missing))
            raise RuntimeError(f"Lahman crawler could not find required file(s): {missing_labels}")

        self._emit("[crawler] All required files downloaded.")
        return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl SABR/Box Lahman sources and download required CSVs.")
    parser.add_argument("--output-dir", default="bin/baseball_cache", help="Directory to save matched CSV files.")
    parser.add_argument("--url", action="append", default=[], help="Explicit candidate URL (repeatable).")
    parser.add_argument("--discovery-page", action="append", default=[], help="Discovery page URL (repeatable).")
    parser.add_argument("--local-zip", action="append", default=[], help="Local ZIP path to inspect first (repeatable).")
    parser.add_argument(
        "--require",
        action="append",
        default=["appearances"],
        choices=["appearances", "people", "teams"],
        help="Required table(s) to fetch (repeatable).",
    )
    args = parser.parse_args()

    required_files: dict[str, set[str]] = {}
    if "appearances" in args.require:
        required_files["Appearances.csv"] = APPEARANCES_REQUIRED_COLUMNS
    if "people" in args.require:
        required_files["People.csv"] = PEOPLE_REQUIRED_COLUMNS
    if "teams" in args.require:
        required_files["Teams.csv"] = TEAMS_REQUIRED_COLUMNS

    crawler = LahmanBoxCrawler()
    crawler.download_required_csvs(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        required_files=required_files,
        candidate_urls=args.url,
        discovery_pages=args.discovery_page or DEFAULT_DISCOVERY_PAGES,
        local_zip_paths=[Path(p).expanduser().resolve() for p in args.local_zip],
    )


if __name__ == "__main__":
    main()
