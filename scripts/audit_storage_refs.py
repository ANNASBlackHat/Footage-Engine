#!/usr/bin/env python3
"""Find which storage account holds each ``storage_path`` in the database.

Why this exists
---------------
``save_file()`` returns a backend-owned key, so a row written under one
ImageKit or Cloudinary account means nothing under another. Rotating to a
different free-tier account silently orphans every asset uploaded by the
previous one, and the row itself records nothing about which account wrote
it -- the key shape is identical across accounts.

This script answers that empirically: it probes each distinct stored path
against every candidate account and reports which ones actually serve it.
It is read-only by default -- rewriting rows needs an explicit ``--apply``,
and even then only paths positively confirmed live are touched, with the
original values backed up to CSV first.

The database is sampled rather than dumped: a ``Chunk`` and the Qwen row
backfilled from it share one ``storage_path`` (see scripts/backfill_qwen.py),
so thousands of rows collapse to a few thousand distinct paths.

Usage:
    # 1. Sanity check a handful of rows with full detail -- costs almost nothing.
    python scripts/audit_storage_refs.py --limit 5 --dry-run

    # 2. Learn which account owns which era (~50 paths per shape).
    python scripts/audit_storage_refs.py --sample 50

    # 3. Probe everything and write the report.
    python scripts/audit_storage_refs.py --out storage_audit.csv

    # 4. Rewrite the confirmed paths. Review the report first.
    python scripts/audit_storage_refs.py --apply

DNS hijacking
-------------
Some ISPs resolve imagekit.io to a captive-portal landing page that answers
HTTP 200 for every path, including ones that cannot exist. Probing through
such a resolver would mark every asset live and pin the table to a host
serving zero bytes. Hosts are therefore resolved over DNS-over-HTTPS and the
connection is pinned to the returned IP, with the real hostname kept for SNI
so TLS still validates. A 200 that is not media is rejected outright.

ImageKit also blocks datacenter IPs, so route through a proxy if needed.
requests picks up HTTPS_PROXY/HTTP_PROXY on its own; --proxy takes precedence:

    python scripts/audit_storage_refs.py --sample 50 \\
        --proxy http://user:pass@host:port
"""

import argparse
import csv
import os
import sys
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlsplit

import requests

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sqlalchemy import text

from footage_engine.models.db import get_engine

# Candidate accounts. ImageKit keys look like "/footage_engine/raw/<name>";
# Cloudinary keys like "footage_engine/raw/<name>" and live under a
# cloud-specific delivery host. Probing every candidate for every path is
# what makes the verdict trustworthy, and the sample phase is what keeps the
# request count affordable on a free plan.
IMAGEKIT_ACCOUNTS = ("annassocial", "annasdev", "annasblackhat")
CLOUDINARY_CLOUDS = ("qdfnsveu", "dq2lgdu6b")

IMAGEKIT_ENDPOINT = "https://ik.imagekit.io/{account}"
CLOUDINARY_ENDPOINT = "https://res.cloudinary.com/{cloud}"

# Cloudinary mints a "v<digits>/" segment on upload and never persists it, so
# the URL stored in a row cannot reproduce it. Both forms are tried and
# whichever one serves the file is the one reported.
CLOUDINARY_VERSION_PREFIXES = ("", "v1791627893/")

# Path shapes seen in the wild. Kept explicit so an unrecognised shape is
# reported rather than silently probed against the wrong URL builder.
SHAPE_IMAGEKIT = "imagekit"      # /footage_engine/raw/chunks_<item>_<chunk>.mp4
SHAPE_CLOUDINARY = "cloudinary"  # footage_engine/raw/chunks/<item>_<chunk>.mp4
SHAPE_OTHER = "other"

PROBE_TIMEOUT_SEC = 20
RATE_LIMIT_SEC = 0.25  # courtesy pause between requests, per worker thread


@dataclass
class Candidate:
    """One endpoint to test a path against."""

    name: str
    kind: str        # "imagekit" | "cloudinary"
    url_template: str

    def urls_for(self, path: str) -> list[str]:
        """URLs to try for ``path``, in preference order."""
        clean = path.lstrip("/")
        if self.kind == "imagekit":
            return [self.url_template.format(account=self.name) + "/" + clean]
        return [
            self.url_template.format(cloud=self.name) + "/" + prefix + clean
            for prefix in CLOUDINARY_VERSION_PREFIXES
        ]


@dataclass
class Verdict:
    """What probing found for a single stored path."""

    path: str
    shape: str
    item_id: str
    matches: list[str] = field(default_factory=list)
    chosen: Optional[str] = None       # first match in probe order
    chosen_url: Optional[str] = None
    attempts: list[tuple[str, int, str]] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)  # accounts we could not test

    @property
    def status(self) -> str:
        if self.matches:
            return "ambiguous" if len(self.matches) > 1 else "live"
        # A transport failure (blocked IP, TLS error, timeout) proves nothing
        # about whether the asset exists. Calling that "dead" would invite a
        # pointless re-upload of files that are perfectly intact, so an
        # untestable account makes the whole verdict inconclusive.
        if self.unreachable:
            return "unreachable"
        return "dead"


@dataclass
class StoredPath:
    """A distinct storage_path plus where it came from."""

    path: str
    shape: str
    item_id: str
    row_count: int = 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sample", type=int, default=None,
                    help="Probe this many paths per shape to learn the account mapping.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Probe at most this many paths overall (cheapest check).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print per-path detail and write no CSV.")
    ap.add_argument("--out", default="storage_audit.csv",
                    help="CSV report path (default: storage_audit.csv).")
    ap.add_argument("--proxy", default=None,
                    help="Proxy URL; overrides HTTPS_PROXY/HTTP_PROXY for this run.")
    ap.add_argument("--imagekit-accounts", default=None,
                    help=f"Comma-separated ImageKit accounts (default: {','.join(IMAGEKIT_ACCOUNTS)}).")
    ap.add_argument("--cloudinary-clouds", default=None,
                    help=f"Comma-separated Cloudinary clouds (default: {','.join(CLOUDINARY_CLOUDS)}).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent probe threads (default: 4). Keep low on free plans.")
    ap.add_argument("--timeout", type=float, default=PROBE_TIMEOUT_SEC,
                    help=f"Per-request timeout in seconds (default: {int(PROBE_TIMEOUT_SEC)}).")
    ap.add_argument("--retries", type=int, default=2,
                    help="Retries for a rate-limited or transient failure (default: 2).")
    ap.add_argument("--only-shape", choices=(SHAPE_IMAGEKIT, SHAPE_CLOUDINARY, SHAPE_OTHER),
                    default=None, help="Probe only one path shape.")
    ap.add_argument("--pin-dns", action=argparse.BooleanOptionalAction, default=True,
                    help="Resolve hosts over DoH and pin the IP (default: on). Needed when "
                         "your ISP hijacks imagekit.io to a landing page.")
    ap.add_argument("--apply", action="store_true",
                    help="REWRITE storage_path in the database for every path confirmed live. "
                         "Without this flag the script never writes to Postgres.")
    ap.add_argument("--apply-ambiguous", action="store_true",
                    help="With --apply, also rewrite paths that matched more than one account "
                         "(first match in probe order wins).")
    ap.add_argument("--backup", default="storage_audit_backup.csv",
                    help="Where --apply records the original values (default: storage_audit_backup.csv).")
    args, _unknown = ap.parse_known_args()  # tolerate kernel argv (e.g. Colab's -f flag)
    return args


# -- database ------------------------------------------------------------------

def classify_shape(path: str) -> str:
    """Which URL builder a stored path needs."""
    clean = path.lstrip("/")
    if clean.startswith("footage_engine/raw/chunks/"):
        return SHAPE_CLOUDINARY
    if clean.startswith("footage_engine/raw/"):
        return SHAPE_IMAGEKIT
    return SHAPE_OTHER


def load_paths(database_url: str, only_shape: Optional[str]) -> list[StoredPath]:
    """Every distinct storage_path that a backend key, with row provenance."""
    engine = get_engine(database_url)
    query = text("""
        SELECT storage_path, count(*) AS rows, min(media_item_id) AS item_id
        FROM chunks
        WHERE storage_path IS NOT NULL AND storage_path <> ''
        GROUP BY storage_path
        UNION ALL
        SELECT storage_path, count(*), min(id)
        FROM media_items
        WHERE storage_path LIKE '/footage_engine/%'
        GROUP BY storage_path
    """)
    with engine.connect() as conn:
        results = conn.execute(query).fetchall()

    paths: list[StoredPath] = []
    for storage_path, rows, item_id in results:
        shape = classify_shape(storage_path)
        if only_shape and shape != only_shape:
            continue
        paths.append(StoredPath(
            path=storage_path,
            shape=shape,
            item_id=str(item_id or "")[:8],
            row_count=int(rows),
        ))
    return paths


# -- DNS ------------------------------------------------------------------------
# Some ISPs (and captive portals) hijack imagekit.io to a landing page instead
# of the CDN. The hijack answers HTTP 200 for *every* path, including ones that
# cannot exist, so probing through a poisoned resolver would mark every asset
# live and pin the whole table to a domain that serves zero bytes.
#
# Resolution therefore goes over DNS-over-HTTPS, and the connection is pinned to
# the returned IP with the real hostname kept for SNI and certificate
# validation. TLS verification is never disabled -- a hostname mismatch means
# something is intercepting, which is a result to report, not to bypass.

DOH_RESOLVER = "https://cloudflare-dns.com/dns-query"
DOH_FALLBACKS = (
    "https://cloudflare-dns.com/dns-query",
    "https://dns.google/resolve",
)

# Media the probe will accept as proof the asset exists. A landing page or
# captive portal answers text/html, so this doubles as a hijack check.
ACCEPTED_CONTENT_TYPES = ("video/", "image/", "audio/", "application/octet-stream")


def resolve_via_doh(host: str, timeout: float) -> list[str]:
    """A/AAAA addresses for ``host`` from a DoH resolver, bypassing local DNS.

    Returns an empty list when every resolver fails, which callers treat as
    "unreachable" rather than "does not exist".
    """
    for resolver in DOH_FALLBACKS:
        try:
            resp = requests.get(
                resolver,
                params={"name": host, "type": "A"},
                headers={"accept": "application/dns-json"},
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError):
            continue

        addresses = [
            str(answer.get("data", "")).strip()
            for answer in payload.get("Answer", []) or []
            if int(answer.get("type", 0)) == 1 and answer.get("data")
        ]
        # Reject obvious interstitial hosts even if a resolver hands them back:
        # a CDN answer is an IP, not a name.
        addresses = [a for a in addresses if _is_ipv4(a)]
        if addresses:
            return addresses
    return []


def _is_ipv4(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


class DnsPinner:
    """Resolves candidate hosts once over DoH, then reuses the addresses.

    SNI and Host both carry the real hostname, so TLS validates against the
    genuine certificate even though the socket is aimed at a fixed IP. Cached
    per host: one DoH lookup covers thousands of probes.

    Implemented with curl's ``--resolve`` rather than requests, because
    requests cannot "dial this IP but verify this hostname" on its own, and
    the pinned form must still pass certificate validation.
    """

    def __init__(self, timeout: float, enabled: bool):
        self.timeout = timeout
        self.enabled = enabled
        self._cache: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def addresses_for(self, host: str) -> list[str]:
        if not self.enabled:
            return []
        with self._lock:
            if host not in self._cache:
                self._cache[host] = resolve_via_doh(host, self.timeout)
            return self._cache[host]

    def resolve_args(self, url: str) -> list[str]:
        """curl ``--resolve`` args pinning this URL's host to each DoH address."""
        if not self.enabled:
            return []
        parts = urlsplit(url)
        host = parts.hostname
        if not host:
            return []
        port = parts.port or (443 if parts.scheme == "https" else 80)
        args: list[str] = []
        for address in self.addresses_for(host):
            args += ["--resolve", f"{host}:{port}:{address}"]
        return args


# -- probing -------------------------------------------------------------------

class Prober:
    """Rate-limited, retrying HTTP client shared across worker threads.

    Uses curl with ``--resolve`` when DNS pinning is on (see DnsPinner), and
    requests otherwise. Both paths verify TLS and both reject a 200 that is not
    actually media, so a hijacked or captive-portal endpoint cannot register as
    a successful hit.
    """

    def __init__(self, proxy: Optional[str], timeout: float, retries: int,
                 pinner: Optional[DnsPinner] = None):
        self.timeout = timeout
        self.retries = retries
        self.pinner = pinner
        self._lock = threading.Lock()
        self._last_request = 0.0
        self.session = requests.Session()
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})

    def _throttle(self) -> None:
        with self._lock:
            wait = RATE_LIMIT_SEC - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    @staticmethod
    def _is_media(content_type: str) -> bool:
        ctype = (content_type or "").lower()
        return any(ctype.startswith(prefix) for prefix in ACCEPTED_CONTENT_TYPES)

    def _probe_pinned(self, url: str) -> tuple[bool, int, str]:
        """curl with a DoH-pinned IP; TLS still validates the real hostname."""
        resolve_args = self.pinner.resolve_args(url)
        if not resolve_args:
            return False, 0, "DoH returned no address for this host"
        cmd = [
            "curl", "-sS", "-I", "--max-time", str(self.timeout),
            "-o", "/dev/null", "-w", "%{http_code}|%{content_type}",
            *resolve_args, url,
        ]
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout + 5)
        except subprocess.TimeoutExpired:
            return False, 0, "curl timed out"
        except FileNotFoundError:
            return False, 0, "curl not installed"

        if done.returncode != 0:
            return False, 0, f"curl exit {done.returncode}: {done.stderr.strip()[:160]}"

        code_str, _, ctype = done.stdout.partition("|")
        try:
            code = int(code_str.strip())
        except ValueError:
            return False, 0, f"unparseable curl output: {done.stdout[:80]!r}"

        if code != 200:
            return False, code, f"{code} {ctype}"
        if not self._is_media(ctype):
            # 200 but not media: an interstitial, not the asset.
            return False, 200, f"200 but content-type={ctype!r} (not media)"
        return True, 200, url

    def probe(self, url: str) -> tuple[bool, int, str]:
        """Returns (ok, status_code, detail). Retries transient failures."""
        pinned = bool(self.pinner and self.pinner.enabled)
        for attempt in range(self.retries + 1):
            self._throttle()
            if pinned:
                ok, code, detail = self._probe_pinned(url)
                if code == 0 and attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                return ok, code, detail
            try:
                resp = self.session.head(url, timeout=self.timeout, allow_redirects=True)
                if resp.status_code == 200:
                    if not self._is_media(resp.headers.get("content-type", "")):
                        return False, 200, (
                            f"200 but content-type={resp.headers.get('content-type')!r} (not media)"
                        )
                    return True, 200, url
                # 429/5xx are worth another go; 403/404 are the real answer.
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                return False, resp.status_code, url
            except requests.RequestException as exc:
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                return False, 0, f"{type(exc).__name__}: {exc}"
        return False, 0, "exhausted retries"


def probe_path(stored: StoredPath, candidates: Iterable[Candidate], prober: Prober) -> Verdict:
    """Test one path against every candidate, keeping those that serve it."""
    verdict = Verdict(path=stored.path, shape=stored.shape, item_id=stored.item_id)
    for candidate in candidates:
        for url in candidate.urls_for(stored.path):
            ok, code, detail = prober.probe(url)
            if code == 0:
                # Network/TLS failure, not a verdict about the asset. Stop
                # probing this account and remember it was untestable.
                verdict.unreachable.append(candidate.name)
                verdict.attempts.append((candidate.name, code, detail))
                break
            verdict.attempts.append((candidate.name, code, detail))
            if ok:
                verdict.matches.append(candidate.name)
                if verdict.chosen is None:  # first match in probe order wins
                    verdict.chosen = candidate.name
                    verdict.chosen_url = detail
                break  # one serving URL per account is enough
    return verdict


def build_candidates(imagekit_accounts: list[str], cloudinary_clouds: list[str]) -> list[Candidate]:
    candidates = [
        Candidate(name=acct, kind="imagekit", url_template=IMAGEKIT_ENDPOINT)
        for acct in imagekit_accounts
    ]
    candidates += [
        Candidate(name=cloud, kind="cloudinary", url_template=CLOUDINARY_ENDPOINT)
        for cloud in cloudinary_clouds
    ]
    return candidates


# -- reporting -----------------------------------------------------------------

def write_csv(path: str, verdicts: list[Verdict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "path", "shape", "item_id", "status",
            "chosen_account", "chosen_url", "all_matches", "unreachable", "attempts",
        ])
        for v in verdicts:
            writer.writerow([
                v.path, v.shape, v.item_id, v.status,
                v.chosen or "", v.chosen_url or "", "|".join(v.matches),
                "|".join(v.unreachable),
                ";".join(f"{name}:{code}" for name, code, _ in v.attempts),
            ])


def print_summary(verdicts: list[Verdict], rows_by_path: dict[str, int]) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    by_status = defaultdict(lambda: [0, 0])
    for v in verdicts:
        by_status[v.status][0] += 1
        by_status[v.status][1] += rows_by_path.get(v.path, 1)
    for status in ("live", "ambiguous", "dead", "unreachable"):
        n_paths, n_rows = by_status.get(status, [0, 0])
        print(f"  {status:11} {n_paths:6d} paths  ({n_rows} db rows)")

    # A "dead" verdict only holds if every candidate was actually reachable,
    # so make the distinction impossible to miss.
    blocked = {name for v in verdicts for name in v.unreachable}
    if blocked:
        print(
            f"\n  WARNING: {len(blocked)} account(s) were unreachable "
            f"({', '.join(sorted(blocked))}). Paths reported 'unreachable' or "
            "'dead' may still exist -- re-run through a proxy before acting on "
            "them. ImageKit blocks datacenter IPs."
        )

    print("\n  shape x account (paths where the account served a hit):")
    matrix = defaultdict(lambda: defaultdict(int))
    for v in verdicts:
        for name in v.matches:
            matrix[v.shape][name] += 1
    for shape in sorted(matrix):
        print(f"    {shape}:")
        for name, count in sorted(matrix[shape].items(), key=lambda kv: -kv[1]):
            print(f"      {name:16} {count}")

    print("\n  dead paths by shape and item (confirmed 404 on every account):")
    dead = defaultdict(lambda: defaultdict(int))
    for v in verdicts:
        if v.status == "dead":
            dead[v.shape][v.item_id or "?"] += 1
    for shape in sorted(dead):
        print(f"    {shape}:")
        for item_id, count in sorted(dead[shape].items(), key=lambda kv: -kv[1]):
            print(f"      item {item_id:10} {count} path(s)")


def apply_verdicts(database_url: str, verdicts: list[Verdict],
                   allow_ambiguous: bool, backup_path: Optional[str]) -> int:
    """Rewrite ``storage_path`` to the confirmed URL for every live path.

    Only paths the probe positively confirmed are touched. Anything dead or
    unreachable keeps its existing key, which still resolves through the
    configured backend -- an unverified rewrite is worse than no rewrite.

    The original values are written to a backup file first so a bad run can be
    reversed, and the whole update runs in one transaction.
    """
    updates: dict[str, str] = {}
    for v in verdicts:
        if not v.chosen_url:
            continue
        if v.status == "ambiguous" and not allow_ambiguous:
            continue
        if v.status not in ("live", "ambiguous"):
            continue
        # Never rewrite something that is already absolute: it would only
        # replace a good URL with an equivalent one.
        if v.path.startswith(("http://", "https://", "file://")):
            continue
        updates[v.path] = v.chosen_url

    if not updates:
        print("\nNothing to apply: no path was confirmed live.")
        return 0

    if backup_path:
        with open(backup_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["old_storage_path", "new_storage_path"])
            for old, new in sorted(updates.items()):
                writer.writerow([old, new])
        print(f"Backup of {len(updates)} original value(s) -> {backup_path}")

    engine = get_engine(database_url)
    total = 0
    with engine.begin() as conn:
        for old, new in sorted(updates.items()):
            # chunks first, then media_items. Both carry keys the audit
            # recorded, so a match on the exact string is safe in both tables.
            for table in ("chunks", "media_items"):
                result = conn.execute(
                    text(f"UPDATE {table} SET storage_path = :new WHERE storage_path = :old"),
                    {"new": new, "old": old},
                )
                total += result.rowcount or 0
    print(f"Updated {total} row(s) across {len(updates)} path(s).")
    return total


def main() -> int:
    args = parse_args()

    if args.proxy:
        os.environ["HTTPS_PROXY"] = args.proxy
        os.environ["HTTP_PROXY"] = args.proxy

    from footage_engine.config import get_settings

    cfg = get_settings()

    imagekit_accounts = (
        args.imagekit_accounts.split(",") if args.imagekit_accounts else list(IMAGEKIT_ACCOUNTS)
    )
    imagekit_accounts = [a.strip() for a in imagekit_accounts if a.strip()]
    cloudinary_clouds = (
        args.cloudinary_clouds.split(",") if args.cloudinary_clouds else list(CLOUDINARY_CLOUDS)
    )
    cloudinary_clouds = [c.strip() for c in cloudinary_clouds if c.strip()]

    candidates = build_candidates(imagekit_accounts, cloudinary_clouds)

    print("Loading distinct storage paths from the database...")
    stored_paths = load_paths(cfg.DATABASE_URL, args.only_shape)
    total_rows = sum(s.row_count for s in stored_paths)

    shapes = defaultdict(int)
    for s in stored_paths:
        shapes[s.shape] += 1
    print(f"  {len(stored_paths)} distinct paths across {total_rows} database rows")
    for shape, count in sorted(shapes.items()):
        print(f"    {shape:11} {count}")

    unrecognised = shapes.get(SHAPE_OTHER, 0)
    if unrecognised:
        print(
            f"\n  WARNING: {unrecognised} path(s) match no known shape and will be "
            "probed as imagekit; review them in the CSV.",
            flush=True,
        )

    # --sample takes a slice of every shape; --limit then caps the whole set.
    selected = list(stored_paths)
    if args.sample:
        per_shape: dict[str, int] = defaultdict(int)
        sampled: list[StoredPath] = []
        for s in selected:
            if per_shape[s.shape] < args.sample:
                sampled.append(s)
                per_shape[s.shape] += 1
        selected = sampled
    if args.limit is not None:
        selected = selected[: args.limit]

    print(f"\nProbing {len(selected)} path(s) against {len(candidates)} candidate(s):")
    for c in candidates:
        print(f"    {c.name:16} {c.url_template}")

    pinner = DnsPinner(args.timeout, args.pin_dns)
    if args.pin_dns:
        hosts = sorted({urlsplit(c.url_template.format(account=c.name, cloud=c.name)).hostname
                        for c in candidates})
        print("\nResolving hosts over DoH (bypasses hijacking resolvers):")
        for host in hosts:
            addresses = pinner.addresses_for(host)
            print(f"    {host:24} {', '.join(addresses) if addresses else 'FAILED'}")

    prober = Prober(args.proxy, args.timeout, args.retries, pinner)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        verdicts = list(pool.map(
            lambda s: probe_path(s, candidates, prober), selected
        ))

    if args.dry_run:
        print("\n" + "=" * 72)
        print("DRY RUN -- per-path detail (no CSV written)")
        print("=" * 72)
        for v in verdicts:
            print(f"\n  {v.status.upper():9} {v.path}")
            print(f"    shape={v.shape} item={v.item_id} rows={next((s.row_count for s in selected if s.path == v.path), 1)}")
            for name, code, detail in v.attempts:
                mark = "HIT " if name in v.matches else "    "
                print(f"    {mark}{name:16} HTTP {code:<4} {detail[:70]}")
            if v.chosen_url:
                print(f"    -> would store: {v.chosen_url}")
        print_summary(verdicts, {s.path: s.row_count for s in selected})
        return 0

    write_csv(args.out, verdicts)
    print(f"\nWrote {args.out}")
    print_summary(verdicts, {s.path: s.row_count for s in selected})

    if not args.apply:
        print("\nRead-only run: no rows were modified. "
              "Re-run with --apply to rewrite the confirmed paths.")
        return 0

    skipped_ambiguous = sum(1 for v in verdicts if v.status == "ambiguous")
    if skipped_ambiguous and not args.apply_ambiguous:
        print(f"\nNote: {skipped_ambiguous} ambiguous path(s) matched more than one account "
              "and were left untouched. Add --apply-ambiguous to include them.")
    if args.dry_run:
        print("\n--dry-run was set, so no rows were modified.")
        return 0

    print()
    apply_verdicts(cfg.DATABASE_URL, verdicts, args.apply_ambiguous, args.backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())