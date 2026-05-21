"""Logging utilities for training metrics."""

import csv
import os


def save_eval_to_csv(exp_name: str, mod_name: str, global_step: int, eval_stats: dict,
                     eval_dir: str | None = None):
    """Append evaluation results to CSV file."""
    if eval_dir is None:
        eval_dir = os.path.join("experiments", os.environ.get("RAE_USER", "jas"), "evals", "stage1")
    os.makedirs(eval_dir, exist_ok=True)
    csv_path = os.path.join(eval_dir, f"{exp_name}_{mod_name}.csv")

    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        fieldnames = ['step'] + list(eval_stats.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {'step': global_step, **eval_stats}
        writer.writerow(row)


__all__ = ["save_eval_to_csv"]
