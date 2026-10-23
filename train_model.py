"""Train a binary classifier artifact for the model-serving API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from model_core import MODEL_FORMAT_VERSION, build_pipeline, prepare_features


def train(args: argparse.Namespace) -> int:
    output_paths = [args.model, args.metrics, args.predictions]
    resolved_outputs = [path.resolve() for path in output_paths]
    if args.data.resolve() in resolved_outputs:
        raise ValueError("An output must not overwrite the labeled input CSV.")
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("Model, metrics, and prediction output paths must differ.")

    if not args.data.is_file():
        raise ValueError(f"Labeled CSV does not exist: {args.data}")
    frame = pd.read_csv(args.data)
    if frame.empty:
        raise ValueError("Labeled CSV contains no data rows.")
    if args.target not in frame.columns:
        raise ValueError(f"Target column '{args.target}' is missing.")
    labels = pd.to_numeric(frame[args.target], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Target must contain only numeric binary labels 0 and 1.")
    labels = labels.astype("int8")
    counts = labels.value_counts()
    if set(counts.index.tolist()) != {0, 1} or int(counts.min()) < 2:
        raise ValueError("Training needs at least two rows from each class.")
    if args.id_column == args.target:
        raise ValueError("The ID column and target column must be different.")
    if args.id_column and args.id_column not in frame.columns:
        raise ValueError(f"ID column '{args.id_column}' is missing.")

    excluded = [args.target] + ([args.id_column] if args.id_column else [])
    features = frame.drop(columns=excluded).dropna(axis=1, how="all")
    if features.empty:
        raise ValueError("No usable feature columns remain.")
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    features = prepare_features(features, numeric_columns)

    try:
        x_train, x_test, y_train, y_test, _train_rows, test_rows = train_test_split(
            features,
            labels,
            frame.index,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels,
        )
    except ValueError as exc:
        raise ValueError(f"Could not create the stratified holdout: {exc}") from exc

    usable = [column for column in x_train.columns if x_train[column].notna().any()]
    if not usable:
        raise ValueError("No feature has values in the training split.")
    x_train = x_train.loc[:, usable]
    x_test = x_test.loc[:, usable]
    features = features.loc[:, usable]
    numeric_columns = [column for column in numeric_columns if column in usable]
    categorical_columns = [column for column in usable if column not in numeric_columns]

    model = build_pipeline(numeric_columns, categorical_columns)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics: dict[str, Any] = {
        "model": "class-weighted logistic regression",
        "target_column": args.target,
        "rows_total": int(len(frame)),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "positive_rate": float(labels.mean()),
        "seed": int(args.seed),
        "metrics": {
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "brier_score": float(brier_score_loss(y_test, probabilities)),
        },
        "feature_columns": features.columns.tolist(),
    }
    predictions = pd.DataFrame(
        {
            "source_row": np.asarray(test_rows) + 2,
            "actual_label": y_test.to_numpy(),
            "positive_probability": probabilities,
        }
    )
    if args.id_column:
        predictions.insert(
            1, args.id_column, frame.loc[test_rows, args.id_column].to_numpy()
        )

    bundle = {
        "format_version": MODEL_FORMAT_VERSION,
        "pipeline": model,
        "feature_columns": features.columns.tolist(),
        "numeric_columns": numeric_columns,
        "target_column": args.target,
        "id_column": args.id_column,
        "metrics": metrics["metrics"],
    }
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.sort_values("source_row").to_csv(args.predictions, index=False)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    joblib.dump(bundle, args.model)
    print(f"Holdout ROC AUC: {metrics['metrics']['roc_auc']:.4f}")
    print(f"Saved model to {args.model}")
    print(f"Saved metrics to {args.metrics}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a binary model for the serving API.")
    parser.add_argument("--data", required=True, type=Path, help="Labeled CSV containing the target and features.")
    parser.add_argument("--target", required=True, help="Binary 0/1 target column.")
    parser.add_argument("--id-column", help="Optional ID excluded from training and included in holdout predictions.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Stratified holdout fraction (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed (default: 42).")
    parser.add_argument("--model", type=Path, default=Path("model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("training_metrics.json"))
    parser.add_argument("--predictions", type=Path, default=Path("holdout_predictions.csv"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return train(args)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
