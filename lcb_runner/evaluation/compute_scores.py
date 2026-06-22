import json
import argparse
import numpy as np
from datetime import datetime

from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.evaluation.pass_k_utils import (
    estimate_pass_at_k,
    compute_metrics_from_results,
)
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.utils.path_utils import get_eval_all_output_path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0301",
        help="Name of the model to use matching `lm_styles.py`",
    )
    parser.add_argument(
        "--scenario",
        type=Scenario,
        default=Scenario.codegeneration,
        help="Type of scenario to run",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for sampling"
    )

    parser.add_argument(
        "--eval_all_file",
        type=str,
        default=None,
        help="Alternative way to provide the evaluation file",
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )

    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Platform to filter the evaluation file",
    )

    args = parser.parse_args()

    if args.eval_all_file is None:
        model = LanguageModelStore[args.model]
        args.eval_all_file = get_eval_all_output_path(model, args)

    return args


def compute_scores(args):
    # Capture the human-readable filter strings before they are parsed to
    # datetime objects below (used only for the table title).
    start_str, end_str = args.start_date, args.end_date

    with open(args.eval_all_file, "r") as f:
        results = json.load(f)

    for res in results:
        res["contest_date"] = datetime.fromisoformat(res["contest_date"])

    if args.start_date is not None:
        args.start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        results = [
            result for result in results if args.start_date <= result["contest_date"]
        ]

    if args.end_date is not None:
        args.end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        results = [
            result for result in results if result["contest_date"] <= args.end_date
        ]

    if args.platform is not None:
        results = [result for result in results if result["platform"] == args.platform]

    from lcb_runner.evaluation.pretty import render_grid

    totals = [len(x["graded_list"]) for x in results]
    corrects = [sum(x["graded_list"]) for x in results]

    splits = {
        "Overall": results,
        "Easy": [x for x in results if x["difficulty"] == "easy"],
        "Medium": [x for x in results if x["difficulty"] == "medium"],
        "Hard": [x for x in results if x["difficulty"] == "hard"],
    }
    # Only show k that is meaningful: at most the smallest sample count present
    # (pass@k for k > n is not well-defined).
    max_k = min(totals) if totals else 1
    k_values = [k for k in [1, 5, 10, 25, 50, 100, 150, 200] if k <= max_k]
    if 1 not in k_values:
        k_values = [1] + k_values

    headers = ["Split", "N"] + [f"pass@{k}" for k in k_values]
    rows = []
    for name, subset in splits.items():
        if not subset:
            continue
        t = [len(x["graded_list"]) for x in subset]
        c = [sum(x["graded_list"]) for x in subset]
        cells = [name, str(len(subset))]
        for k in k_values:
            cells.append(f"{estimate_pass_at_k(t, c, k).mean() * 100:6.2f}%")
        rows.append(cells)

    title = f"{len(results)} problems"
    if start_str or end_str:
        title += f"  ({start_str or '...'} -> {end_str or '...'})"
    if args.platform:
        title += f"  [{args.platform}]"
    print(render_grid(headers, rows, title=title))


if __name__ == "__main__":
    compute_scores(get_parser())
