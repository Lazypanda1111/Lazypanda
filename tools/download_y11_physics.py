#!/usr/bin/env python3
"""Download a small, curated set of recent NSW Year 11 Physics papers.

The papers are publicly listed by THSC Online. The downloader is deliberately
serial and validates every response as a genuine PDF before saving it.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

APPS = "https://script.google.com/macros/s/AKfycbx69GPoJtf9sSevsUbWtPr46vpa01u4oNkHjFmkkWxmj62AZ0q-/exec"
BASE = "6518"
OUT = Path("bundle")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36"}
PLACEHOLDER_MD5 = "a15ea5c8538acccf0c83ce0314390524"

# Recent publicly listed papers from Sydney selective high schools.
TARGETS = [
    ("Girraween 2024 w. sol", "2024_Girraween_Year11_Physics_with_solutions.pdf"),
    ("Fort St 2023 w. sol", "2023_Fort_Street_Year11_Physics_with_solutions.pdf"),
    ("Girraween 2023 w. sol", "2023_Girraween_Year11_Physics_with_solutions.pdf"),
    ("Penrith 2023", "2023_Penrith_Year11_Physics.pdf"),
    ("Fort St 2022 w. sol", "2022_Fort_Street_Year11_Physics_with_solutions.pdf"),
    ("Hornsby Girls 2022 w. sol", "2022_Hornsby_Girls_Year11_Physics_with_solutions.pdf"),
    ("Hurlstone 2022", "2022_Hurlstone_Year11_Physics.pdf"),
    ("Penrith 2022 w. sol", "2022_Penrith_Year11_Physics_with_solutions.pdf"),
    ("Sydney Boys 2022", "2022_Sydney_Boys_Year11_Physics.pdf"),
]


def fetch(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def resolve(title: str) -> tuple[str, bytes | str]:
    digest = hashlib.sha256(BASE.encode()).hexdigest()
    url = (
        f"{APPS}?export=data&field={urllib.parse.quote(title)}"
        f"&base={BASE}&hash={digest}"
    )
    body = fetch(url).decode("utf-8", "replace")
    match = re.search(r"downloadfile\((\{.*\})\)", body, re.S)
    if not match:
        return "error", "The source returned no downloadable object"
    payload = json.loads(match.group(1))
    encoded = payload.get("data")
    if not encoded:
        return "error", "The source returned an empty file"
    pdf = base64.b64decode(encoded)
    if hashlib.md5(pdf).hexdigest() == PLACEHOLDER_MD5:
        return "throttled", "The source temporarily rate-limited this request"
    if not pdf.startswith(b"%PDF"):
        return "error", "The returned object was not a PDF"
    if len(pdf) < 10_000:
        return "error", f"The returned PDF was unexpectedly small ({len(pdf)} bytes)"
    return "ok", pdf


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    completed = 0

    for index, (title, filename) in enumerate(TARGETS, start=1):
        destination = OUT / filename
        print(f"[{index}/{len(TARGETS)}] {title}", flush=True)
        last_reason = "unknown error"
        for attempt in range(1, 5):
            try:
                status, value = resolve(title)
            except Exception as exc:  # network errors are retriable
                status, value = "error", f"{type(exc).__name__}: {exc}"

            if status == "ok":
                assert isinstance(value, bytes)
                destination.write_bytes(value)
                print(f"  saved {destination} ({len(value) / 1024:.0f} KiB)", flush=True)
                completed += 1
                break

            last_reason = str(value)
            print(f"  attempt {attempt} failed: {last_reason}", flush=True)
            if attempt < 4:
                # Be gentle with the shared free service. Throttling gets a longer rest.
                time.sleep(150 if status == "throttled" else 35)
        else:
            failures.append((title, last_reason))

        # Avoid back-to-back requests to the same shared source base.
        if index != len(TARGETS):
            time.sleep(35)

    readme = OUT / "README.txt"
    lines = [
        "Sydney selective high schools - recent Year 11 Physics papers",
        "",
        f"Successfully downloaded: {completed}/{len(TARGETS)}",
        "Source: THSC Online public Year 11 Physics yearly-exam collection.",
        "Files marked 'with_solutions' contain the paper plus supplied solutions/marking guidance.",
        "",
        "Included targets:",
    ]
    lines.extend(f"- {title}" for title, _ in TARGETS)
    if failures:
        lines.extend(["", "Files that could not be retrieved:"])
        lines.extend(f"- {title}: {reason}" for title, reason in failures)
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Completed {completed}/{len(TARGETS)}", flush=True)
    if failures:
        for title, reason in failures:
            print(f"FAILED: {title}: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
