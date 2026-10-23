"""Generate a SYNTHETIC openFDA-shaped sample file for UI testing only.

!!! The records produced here are random. They are NOT FAERS data, contain NO
real pharmacovigilance signal, and must never be presented as an analysis. The
generator exists so the dashboard and the pipeline can be tested offline.

Usage:  python scripts/make_sample_data.py [--n 1500] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

CONCOMITANT = ["ACETAMINOPHEN", "OMEPRAZOLE", "METFORMIN", "SERTRALINE", "LISINOPRIL", "ATORVASTATIN"]
GENERIC_REACTIONS = ["SOMNOLENCE", "DIZZINESS", "FALL", "CONFUSIONAL STATE", "NAUSEA", "HEADACHE",
                     "FATIGUE", "VOMITING", "ANXIETY", "INSOMNIA", "RASH", "TREMOR", "AMNESIA",
                     "DRUG INEFFECTIVE", "OFF LABEL USE", "DEATH", "HOSPITALISATION"]
COUNTRIES = ["US"] * 12 + ["GB", "DE", "FR", "JP", "CA", "BR", "IT", "ES", "AU", "EG"]


def generate(path: Path, n: int = 1500, seed: int = 7, start: str = "2024-01-01", end: str = "2026-06-30") -> int:
    rng = random.Random(seed)
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    span = (d1 - d0).days
    records = []
    for i in range(n):
        n_rx = rng.randint(1, 6)
        reactions = rng.sample(GENERIC_REACTIONS, n_rx)
        # mild synthetic structure: more listed reactions -> slightly more often "serious"
        p_serious = 0.35 + 0.08 * n_rx
        serious = "1" if rng.random() < p_serious else "2"
        if rng.random() < 0.02:
            serious = ""  # missing label
        drug = rng.choice(config.SEDATIVE_DRUGS)
        drugs = [{"medicinalproduct": drug, "drugcharacterization": rng.choice(["1", "1", "2"]),
                  "openfda": {"generic_name": [drug], "substance_name": [drug]}}]
        if rng.random() < 0.12:
            other = rng.choice([x for x in config.SEDATIVE_DRUGS if x != drug])
            drugs.append({"medicinalproduct": other, "drugcharacterization": "2",
                          "openfda": {"generic_name": [other], "substance_name": [other]}})
        for c in rng.sample(CONCOMITANT, rng.randint(0, 4)):
            drugs.append({"medicinalproduct": c, "drugcharacterization": "2",
                          "openfda": {"generic_name": [c], "substance_name": [c]}})

        roll = rng.random()
        if roll < 0.12:
            age, unit = None, None
        elif roll < 0.14:
            age, unit = str(rng.choice([150, 999, -3])), "801"  # invalid -> must become NaN
        elif roll < 0.18:
            age, unit = str(rng.randint(5, 9)), "800"  # decades
        else:
            age, unit = str(rng.randint(1, 95)), "801"
        rec = {
            "safetyreportid": f"SAMPLE-{i:06d}",
            "safetyreportversion": "1",
            "companynumb": f"SYN-CASE-{i:06d}",
            "receivedate": (d0 + timedelta(days=rng.randint(0, span))).strftime("%Y%m%d"),
            "serious": serious,
            "occurcountry": rng.choice(COUNTRIES) if rng.random() > 0.08 else None,
            "patient": {
                "patientonsetage": age, "patientonsetageunit": unit,
                "patientsex": rng.choice(["1", "2", "2", "0", None]),
                "reaction": [{"reactionmeddrapt": r} for r in reactions],
                "drug": drugs,
            },
            "_synthetic": True,
        }
        if serious == "1":
            rec[rng.choice(["seriousnesshospitalization", "seriousnessother", "seriousnessdeath"])] = "1"
        records.append(rec)

    # exercise de-duplication: follow-up versions and a second report of the same case
    for rec in rng.sample(records, k=max(1, n // 40)):
        follow = json.loads(json.dumps(rec))
        follow["safetyreportversion"] = "2"
        records.append(follow)
    for rec in rng.sample(records[:n], k=max(1, n // 60)):
        dup = json.loads(json.dumps(rec))
        dup["safetyreportid"] = rec["safetyreportid"] + "-B"
        dup["receivedate"] = (date.fromisoformat(rec["receivedate"][:4] + "-" + rec["receivedate"][4:6] + "-" + rec["receivedate"][6:]) + timedelta(days=20)).strftime("%Y%m%d")
        records.append(dup)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return len(records)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    count = generate(config.SAMPLE_FILE, args.n, args.seed)
    print(f"Wrote {count} SYNTHETIC records to {config.SAMPLE_FILE}")
    print("These are random test records, not FAERS data.")
