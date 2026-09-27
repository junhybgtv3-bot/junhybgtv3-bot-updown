"""Sweep selective-trading thresholds: python -m server.backtest."""

from pathlib import Path
import pickle

import numpy as np

from server.train import prepare_data


MIN_TRADES = 200
THRESHOLDS = tuple(value / 100 for value in range(50, 71, 2))
REFERENCE_THRESHOLD = 0.62


def trade_metrics(labels, probabilities, threshold):
    """Summarize selected bets; at p=t=0.5 the LONG rule takes precedence.

    Mean probability is confidence in the selected direction: p for LONG,
    1-p for SHORT. Empty selections have undefined hit rate and confidence.
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    long = probabilities >= threshold
    selected = long | (probabilities <= 1 - threshold)
    n_trades = int(selected.sum())
    confidence = np.where(long, probabilities, 1 - probabilities)
    return (
        n_trades,
        n_trades / len(labels) if len(labels) else 0.0,
        float((long[selected] == labels[selected]).mean()) if n_trades else float("nan"),
        float(confidence[selected].mean()) if n_trades else float("nan"),
        float(probabilities[selected].mean()) if n_trades else float("nan"),
    )


def print_row(name, threshold, metrics, *, rule=None):
    n_trades, fraction, hit_rate, mean_probability, mean_up = metrics
    prefix = f"{rule:<10} " if rule is not None else ""
    suffix = f" {mean_up:12.6f} {mean_probability:14.6f}" if rule is not None else ""
    print(f"{prefix}{name:<8} {threshold:5.2f} {n_trades:9d} "
          f"{fraction:14.6f} {hit_rate:10.6f}{suffix}", flush=True)


def main():
    _, (X_val, y_val), (X_test, y_test) = prepare_data()
    model_dir = Path(__file__).resolve().parent.parent / "models"
    models = {}
    for name in ("logreg", "lgbm"):
        with (model_dir / f"{name}.pkl").open("rb") as handle:
            models[name] = pickle.load(handle)

    print(f"\nValidation sweep (MIN_TRADES={MIN_TRADES}; ties choose lower t)")
    print("model        t  n_trades trade_fraction   hit_rate")
    thresholds = {}
    for name, model in models.items():
        probabilities = model.predict_proba(X_val)[:, 1]
        eligible = []
        for threshold in THRESHOLDS:
            metrics = trade_metrics(y_val, probabilities, threshold)
            print_row(name, threshold, metrics)
            if metrics[0] >= MIN_TRADES:
                eligible.append((threshold, metrics[2]))
        thresholds[name] = max(eligible, key=lambda item: item[1])[0] if eligible else None

    print("\nSelected thresholds")
    for name, threshold in thresholds.items():
        if threshold is None:
            print(f"{name}: no validation threshold meets MIN_TRADES={MIN_TRADES}")
        else:
            print(f"{name}: t*={threshold:.2f}")

    print("\nTest results (mean_bet_prob = mean p for LONG, 1-p for SHORT)")
    print("rule       model        t  n_trades trade_fraction   hit_rate mean_up_prob  mean_bet_prob")
    for name, model in models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        threshold = thresholds[name]
        if threshold is not None:
            print_row(name, threshold, trade_metrics(y_test, probabilities, threshold),
                      rule="selected")
        print_row(name, REFERENCE_THRESHOLD,
                  trade_metrics(y_test, probabilities, REFERENCE_THRESHOLD),
                  rule="reference")


if __name__ == "__main__":
    main()
