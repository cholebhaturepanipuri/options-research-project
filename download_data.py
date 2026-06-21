"""
NSE F&O Bhavcopy Downloader
Downloads daily options data from NSE archives.

Usage:
    python3 download_data.py

Files saved to: data/raw/
"""

import os
import time
import datetime
import requests

# ── Config ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

# Date range to download — adjust as needed
START_DATE = datetime.date(2024, 1, 1)
END_DATE   = datetime.date(2024, 6, 30)

# NSE URL patterns (tries new format first, falls back to old)
HEADERS = {
    "User-Agent"      : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36",
    "Accept"          : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language" : "en-US,en;q=0.5",
    "Referer"         : "https://www.nseindia.com/",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_session() -> requests.Session:
    """Open a session with NSE to get cookies (required for archive downloads)."""
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=15)
        time.sleep(1)
    except Exception as e:
        print(f"Warning: could not reach NSE homepage: {e}")
    return session


def download_one_day(session: requests.Session, date: datetime.date, out_dir: str) -> bool:
    """
    Try to download bhavcopy for a single date.
    Tries the new archive format, then the legacy format.
    Returns True on success.
    """
    date_str  = date.strftime("%Y%m%d")   # 20240102
    date_str2 = date.strftime("%d%b%Y").upper()  # 02JAN2024

    # New format (2023+)
    url_new = (
        f"https://nsearchives.nseindia.com/content/fo/"
        f"BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv"
    )
    # Legacy format
    url_old = (
        f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
        f"{date.year}/{date.strftime('%b').upper()}/fo{date_str2}bhav.csv.zip"
    )

    filename = os.path.join(out_dir, f"fo_bhavcopy_{date_str}.csv")

    if os.path.exists(filename):
        return True  # already downloaded

    for url in [url_new, url_old]:
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200 and len(resp.content) > 500:
                # Handle zip
                if url.endswith(".zip"):
                    import zipfile, io
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                        csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                        if csv_names:
                            with z.open(csv_names[0]) as f:
                                with open(filename, "wb") as out:
                                    out.write(f.read())
                            return True
                else:
                    with open(filename, "wb") as f:
                        f.write(resp.content)
                    return True
        except Exception:
            continue

    return False


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build list of trading days (weekdays only — NSE closed weekends)
    all_dates = []
    d = START_DATE
    while d <= END_DATE:
        if d.weekday() < 5:   # Mon–Fri
            all_dates.append(d)
        d += datetime.timedelta(days=1)

    print(f"Dates to download : {len(all_dates)}")
    print(f"Output folder     : {OUTPUT_DIR}")
    print()

    session = get_session()
    print("NSE session established. Starting downloads...\n")

    success = 0
    failed  = []

    for i, date in enumerate(all_dates):
        ok = download_one_day(session, date, OUTPUT_DIR)
        status = "OK" if ok else "SKIP"
        if ok:
            success += 1
        else:
            failed.append(date)

        print(f"  [{i+1:3d}/{len(all_dates)}] {date}  {status}")

        # Refresh session every 20 requests to avoid cookie expiry
        if (i + 1) % 20 == 0:
            time.sleep(2)
            session = get_session()

        time.sleep(0.5)   # be polite to NSE servers

    print()
    print(f"Done. Downloaded: {success}/{len(all_dates)}")
    if failed:
        print(f"Failed dates ({len(failed)}): {[str(d) for d in failed[:5]]}{'...' if len(failed)>5 else ''}")
        print("Failed dates are typically NSE holidays — check the NSE holiday calendar.")

    print()
    print("Next step: open Notebook 03 and re-run the bhavcopy pipeline cell.")


if __name__ == "__main__":
    main()
