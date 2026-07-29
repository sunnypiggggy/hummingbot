from __future__ import annotations

import html
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .policy import Event

FED_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BLS_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"
BLS_PAGE_URL = "https://www.bls.gov/schedule/news_release/empsit.htm"
NEW_YORK = ZoneInfo("America/New_York")

MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        )
    )
    if name
}


def _strip_tags(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def parse_fomc_calendar_html(raw: str, years: set[int] | None = None) -> list[Event]:
    events: list[Event] = []
    divs = [
        match
        for match in re.finditer(
            r'<div[^>]+class="([^"]*)"[^>]*>', raw, flags=re.IGNORECASE
        )
        if "fomc-meeting" in match.group(1).split()
    ]
    for index, match in enumerate(divs):
        end = divs[index + 1].start() if index + 1 < len(divs) else len(raw)
        part = raw[match.end() : end]
        year_matches = re.findall(r"\b(20\d{2})\b", raw[: match.start()])
        year = int(year_matches[-1]) if year_matches else None
        if year is None or (years and year not in years):
            continue
        text = _strip_tags(part[:5000])
        if "notation vote" in text.lower():
            continue
        month_match = re.search(
            r"\b(" + "|".join(MONTHS) + r")\b", text, re.IGNORECASE
        )
        date_match = re.search(r"\b(\d{1,2})(?:-(\d{1,2}))?\b", text)
        if not month_match or not date_match:
            continue
        month = MONTHS[month_match.group(1).title()]
        day = int(date_match.group(2) or date_match.group(1))
        starts_at = datetime(year, month, day, 14, 0, tzinfo=NEW_YORK)
        event_id = f"fomc-{starts_at.date().isoformat()}"
        events.append(
            Event(event_id, "fomc", starts_at, "FOMC rate decision", FED_URL)
        )
    return _deduplicate(events)


def parse_bls_ics(raw: str, years: set[int] | None = None) -> list[Event]:
    events: list[Event] = []
    for block in raw.replace("\r\n", "\n").split("BEGIN:VEVENT"):
        if not re.search(r"^SUMMARY:Employment Situation$", block, re.MULTILINE):
            continue
        match = re.search(
            r"^DTSTART(?:;[^:]*)?:(\d{8})T?(\d{0,6})(Z?)",
            block,
            re.MULTILINE,
        )
        if not match:
            continue
        date_value = match.group(1)
        year = int(date_value[:4])
        if years and year not in years:
            continue
        hour = int(match.group(2)[:2] or "08")
        minute = int(match.group(2)[2:4] or "30")
        starts_at = datetime(
            year,
            int(date_value[4:6]),
            int(date_value[6:8]),
            hour,
            minute,
            tzinfo=timezone.utc if match.group(3) else NEW_YORK,
        )
        starts_at = starts_at.astimezone(NEW_YORK)
        events.append(
            Event(
                f"nfp-{starts_at.date().isoformat()}",
                "nfp",
                starts_at,
                "Employment Situation",
                BLS_ICS_URL,
            )
        )
    return _deduplicate(events)


def parse_bls_calendar_html(raw: str, years: set[int] | None = None) -> list[Event]:
    text = _strip_tags(raw)
    month_numbers = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    pattern = re.compile(
        r"\b(" + "|".join(month_numbers) + r")\.?\s+(\d{1,2}),\s+"
        r"(20\d{2})\s+(\d{1,2}):(\d{2})\s+(AM|PM)\b",
        re.IGNORECASE,
    )
    events = []
    for match in pattern.finditer(text):
        year = int(match.group(3))
        if years and year not in years:
            continue
        hour = int(match.group(4))
        if match.group(6).upper() == "PM" and hour != 12:
            hour += 12
        elif match.group(6).upper() == "AM" and hour == 12:
            hour = 0
        starts_at = datetime(
            year,
            month_numbers[match.group(1).title()],
            int(match.group(2)),
            hour,
            int(match.group(5)),
            tzinfo=NEW_YORK,
        )
        events.append(
            Event(
                f"nfp-{starts_at.date().isoformat()}",
                "nfp",
                starts_at,
                "Employment Situation",
                BLS_PAGE_URL,
            )
        )
    return _deduplicate(events)


def _deduplicate(events: list[Event]) -> list[Event]:
    return sorted({event.event_id: event for event in events}.values(), key=lambda x: x.starts_at)


def fetch_text(url: str, timeout: int = 20) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
            ),
            "Accept": "text/html,text/calendar;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def sync_official_calendar(output: Path, years: set[int]) -> list[Event]:
    events = parse_fomc_calendar_html(fetch_text(FED_URL), years)
    bls_source = "official_ics"
    try:
        bls_events = parse_bls_ics(fetch_text(BLS_ICS_URL), years)
    except Exception:
        try:
            bls_events = parse_bls_calendar_html(fetch_text(BLS_PAGE_URL), years)
            bls_source = "official_html"
        except Exception:
            seed_path = (
                Path(__file__).resolve().parents[1]
                / "data"
                / "hermes"
                / "bls_employment_situation_official_seed.json"
            )
            if not seed_path.exists():
                raise
            bls_events = [
                event for event in load_calendar(seed_path) if event.starts_at.year in years
            ]
            bls_source = "official_snapshot_fallback"
    if not bls_events:
        raise ValueError("official BLS calendar returned no Employment Situation events")
    events.extend(bls_events)
    events = _deduplicate(events)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "event_id": event.event_id,
            "kind": event.kind,
            "starts_at": event.starts_at.isoformat(),
            "title": event.title,
            "source_url": event.source_url,
        }
        for event in events
    ]
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    status = {
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "fed_source": "official_html",
        "bls_source": bls_source,
        "requested_years": sorted(years),
        "event_count": len(events),
    }
    output.with_suffix(".sync.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    return events


def load_calendar(path: Path) -> list[Event]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Event(
            item["event_id"],
            item["kind"],
            datetime.fromisoformat(item["starts_at"]),
            item["title"],
            item["source_url"],
        )
        for item in payload
    ]
