"""Calibrate duplicate detection thresholds from labeled data."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn import metrics


@dataclass
class Sample:
    label: int
    ssim: float
    orb: float
    cosine: float


@dataclass
class ThresholdSuggestion:
    metric: str
    threshold: float
    score: float


@dataclass
class CalibrationResult:
    auc: float
    average_precision: float
    suggestions: list[ThresholdSuggestion]


def _load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = int(row.get("label", 0))
            ssim = float(row.get("ssim", 0.0))
            orb = float(row.get("orb", 0.0))
            cosine = float(row.get("cosine", 1.0))
            samples.append(Sample(label=label, ssim=ssim, orb=orb, cosine=cosine))
    return samples


def _best_threshold(
    metric: str, scores: np.ndarray, labels: np.ndarray, *, greater_is_better: bool
) -> ThresholdSuggestion:
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    threshold = thresholds[idx]
    if not greater_is_better:
        threshold = -threshold
    return ThresholdSuggestion(metric=metric, threshold=float(threshold), score=float(youden[idx]))


def _optimise_thresholds(samples: list[Sample]) -> list[ThresholdSuggestion]:
    y_true = np.asarray([sample.label for sample in samples], dtype=np.int32)
    if np.unique(y_true).size < 2:
        return []

    suggestions: list[ThresholdSuggestion] = []
    ssim_scores = np.asarray([sample.ssim for sample in samples], dtype=np.float32)
    suggestions.append(_best_threshold("ssim", ssim_scores, y_true, greater_is_better=True))

    orb_scores = np.asarray([sample.orb for sample in samples], dtype=np.float32)
    suggestions.append(_best_threshold("orb", orb_scores, y_true, greater_is_better=True))

    cosine_scores = -np.asarray([sample.cosine for sample in samples], dtype=np.float32)
    cosine_suggestion = _best_threshold("cosine", cosine_scores, y_true, greater_is_better=True)
    suggestions.append(
        ThresholdSuggestion(
            metric="cosine",
            threshold=-cosine_suggestion.threshold,
            score=cosine_suggestion.score,
        )
    )
    return suggestions


def calibrate(samples: list[Sample]) -> CalibrationResult:
    y_true = np.asarray([sample.label for sample in samples], dtype=np.int32)
    if np.unique(y_true).size < 2:
        raise ValueError("Need at least one positive and one negative sample")

    ssim_scores = np.asarray([sample.ssim for sample in samples], dtype=np.float32)
    orb_scores = np.asarray([sample.orb for sample in samples], dtype=np.float32)
    cosine_scores = -np.asarray([sample.cosine for sample in samples], dtype=np.float32)
    combined = (ssim_scores + orb_scores + cosine_scores) / 3.0

    auc = float(metrics.roc_auc_score(y_true, combined))
    ap = float(metrics.average_precision_score(y_true, combined))
    suggestions = _optimise_thresholds(samples)
    return CalibrationResult(auc=auc, average_precision=ap, suggestions=suggestions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV file containing label, ssim, orb, cosine columns")
    parser.add_argument("--json", type=Path, help="Optional path to dump calibration result as JSON")
    args = parser.parse_args()

    samples = _load_samples(args.csv)
    if not samples:
        raise SystemExit("No samples loaded from CSV")

    result = calibrate(samples)

    print(f"Samples: {len(samples)}")
    print(f"ROC AUC: {result.auc:.4f}")
    print(f"Average precision: {result.average_precision:.4f}")
    print("\nSuggested thresholds (maximize Youden's J):")
    for suggestion in result.suggestions:
        print(f"  {suggestion.metric:6s}: {suggestion.threshold:.4f} (J={suggestion.score:.3f})")

    if args.json:
        payload = {
            "samples": len(samples),
            "roc_auc": result.auc,
            "average_precision": result.average_precision,
            "suggestions": [
                {"metric": s.metric, "threshold": s.threshold, "score": s.score} for s in result.suggestions
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved calibration result to {args.json}")


if __name__ == "__main__":
    main()
