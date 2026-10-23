"""Clean the raw JSONL and build data/processed/reports.csv (one row per report)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data_loader, preprocessing  # noqa: E402


def main() -> int:
    if not config.RAW_FILE.exists():
        print(f"Raw file not found: {config.RAW_FILE}\nRun: python scripts/download_data.py")
        return 1

    df, stats = preprocessing.build_reports_table(data_loader.iter_raw_records(config.RAW_FILE))
    if df.empty:
        print("No report mentions a study drug. Check SEDATIVE_DRUGS in src/config.py.")
        return 1

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.REPORTS_FILE, index=False)

    stats["missing_summary"] = preprocessing.missing_value_summary(df).to_dict(orient="records")
    stats["date_range"] = [str(df["receive_date"].min().date()), str(df["receive_date"].max().date())]
    stats["data_is_sample"] = data_loader.is_sample_data()
    data_loader.write_json(config.PREPROCESS_REPORT_FILE, stats)

    print(f"Raw records: {stats['n_raw_records']}")
    print(f"  without a study drug match: {stats['n_no_study_drug_match']}")
    print(f"  removed (same report id, older version): {stats['removed_same_report_id']}")
    print(f"  removed (same case id, older report):    {stats['removed_same_case_id']}")
    print(f"Clean reports: {len(df)} -> {config.REPORTS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
