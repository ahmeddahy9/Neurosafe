"""Train the educational Serious / Non-serious classifier and store metrics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402

from src import config, data_loader, modeling  # noqa: E402


def main() -> int:
    reports = data_loader.load_reports()
    if reports is None or reports.empty:
        print("Processed data not found. Run scripts/preprocess_data.py first.")
        return 1

    metrics, model = modeling.train_and_evaluate(reports, data_loader.data_source_info())
    data_loader.write_json(config.METRICS_FILE, metrics)

    if metrics["status"] != "ok":
        print(f"ML skipped honestly: {metrics['reason']}.\n"
              "The dashboard stays fully usable as an EDA tool; the ML page will explain this.")
        if config.MODEL_FILE.exists():
            config.MODEL_FILE.unlink()
        return 0

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "name": metrics["selected_model"]}, config.MODEL_FILE)

    print(f"Split: {metrics['split_strategy']}  train={metrics['n_train']} val={metrics['n_val']} test={metrics['n_test']}")
    print(f"Selected on validation PR-AUC: {metrics['selected_model']}")
    print(f"{'model':<22}{'acc':>7}{'prec':>7}{'rec':>7}{'f1':>7}{'roc':>7}{'pr':>7}   (TEST)")
    for name, m in metrics["test"].items():
        print(f"{name:<22}{m['accuracy']:>7.3f}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}"
              f"{(m['roc_auc'] or 0):>7.3f}{(m['pr_auc'] or 0):>7.3f}")
    print("Beats the dummy baseline:", metrics["beats_baseline"])
    if metrics["data_is_sample"]:
        print("NOTE: trained on SYNTHETIC sample data - these numbers mean nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
