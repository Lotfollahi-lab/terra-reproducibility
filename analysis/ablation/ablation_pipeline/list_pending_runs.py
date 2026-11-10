"""
This script lists the runs of an ablation campaign that still need work, for the submit scripts
(`submit_ablation_array.sh`, `submit_evaluation_array.sh`), so that a job array asks for GPUs only for those runs. For
each manifest row of the given indices it runs the checks of the job arrays, in one process on the login node:
`evaluate_ablation_run.py --check_only` with the arguments of the jobs (skip if the evaluation is up to date; not with
--force_evaluation), then `package_run.py --check_only` (train if training has not finished, else evaluate). Both
checks read the checkpoint (a few seconds each for 750 MB).

It prints each row's decision and its reason to stderr, and to stdout three tab-separated lines with the array indices
(e.g. `2,5-7`, empty if none) of the rows to train (`train`), to evaluate only (`evaluate`) and both (`work`). It exits
with code 1 without these lines if a row could not be checked (e.g. an unreadable checkpoint), which the jobs would
also stop at.

Usage:
```bash
python list_pending_runs.py --manifest_file_path [MANIFEST] [--indices 1-4] --norm_factor_file_path [NORM_FACTORS] \
    --dataset_name [DATASET_NAME] --label_file_paths [LABEL_FILE ...] --label_keys [KEY ...] \
    --unassigned_label [UNASSIGNED_LABEL] --eval_gene_set [EVAL_GENE_SET] [--fp32_embeddings] [--force_evaluation] \
    [--evaluation_only]
```
"""

import argparse
import contextlib
import csv
import io
import runpy
import sys
from pathlib import Path

from ablation_utils import EVAL_GENE_SETS

# Exit code of both --check_only checks for "needs work" (NEEDS_EVALUATION_EXIT_CODE, NOT_FINISHED_EXIT_CODE)
NEEDS_WORK_EXIT_CODE = 3


def parse_indices(indices: str) -> list[int]:
    """Indices of an LSF array specification: comma-separated indices and ranges, e.g. '1-4' or '5,7,9-12'."""
    parsed = []
    for item in indices.split(","):
        bounds = item.split("-")
        if len(bounds) > 2 or not all(bound.isdigit() for bound in bounds) or int(bounds[0]) > int(bounds[-1]):
            raise ValueError(f"Cannot parse the indices '{indices}' (e.g. 1-4 or 5,7,9-12).")
        parsed.extend(range(int(bounds[0]), int(bounds[-1]) + 1))
    return sorted(set(parsed))


def format_indices(indices: list[int]) -> str:
    """LSF array specification of sorted indices, with ranges of consecutive ones (e.g. [2, 5, 6, 7] -> '2,5-7')."""
    ranges = []
    for index in indices:
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    return ",".join(str(first) if first == last else f"{first}-{last}" for first, last in ranges)


def run_check(script_file_name: str, argv: list) -> tuple[int, list]:
    """
    Runs a script of this folder with the command line arguments `argv` in this process (torch, terra and scanpy are
    imported once for all rows): its exit code (any exception: 1) and the lines it printed.
    """
    output = io.StringIO()
    sys.argv = [script_file_name] + argv
    try:
        with contextlib.redirect_stdout(output):
            runpy.run_path(str(Path(__file__).resolve().parent / script_file_name), run_name="__main__")
        status = 0
    except SystemExit as error:
        status = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
    except Exception as error:  # noqa: BLE001 (reported per row)
        status = 1
        output.write(f"{type(error).__name__}: {error}\n")
    return status, [line.strip() for line in output.getvalue().splitlines() if line.strip()]


def main(args):
    with open(args.manifest_file_path) as file:
        rows = {int(row["index"]): row for row in csv.DictReader(file, delimiter="\t")}
    try:
        indices = parse_indices(args.indices) if args.indices else sorted(rows)
    except ValueError as error:
        sys.exit(f"ERROR: {error}")
    missing_indices = [index for index in indices if index not in rows]
    if missing_indices:
        sys.exit(f"ERROR: {args.manifest_file_path} has no rows with the indices {format_indices(missing_indices)}.")

    evaluation_args = ["--dataset_name", args.dataset_name, "--label_file_paths", *args.label_file_paths,
                       "--label_keys", *args.label_keys, "--unassigned_label", args.unassigned_label,
                       "--eval_gene_set", args.eval_gene_set]
    if args.fp32_embeddings:
        evaluation_args.append("--fp32_embeddings")
    actions = {}
    print(f"Checking {len(indices)} runs (rows {format_indices(indices)}) ...", file=sys.stderr)
    for index in indices:
        run_folder_path = rows[index]["run_folder_path"]
        eval_folder_path = rows[index]["eval_folder_path"]
        if args.force_evaluation:
            evaluation_reason = "FORCE_EVAL=1"
        else:
            status, lines = run_check("evaluate_ablation_run.py",
                                      ["--model_folder_path", run_folder_path, "--output_folder_path",
                                       eval_folder_path, *evaluation_args, "--check_only"])
            if status == 0:
                actions[index] = ("skip", "trained and evaluated (evaluation up to date)")
            elif status != NEEDS_WORK_EXIT_CODE:
                actions[index] = ("error", f"evaluation check failed (exit code {status}): {' '.join(lines[-3:])}")
            # 'Evaluation needed (<marker>):' followed by the reasons, or 'Evaluation needed: <reason>'
            first = next((i for i, line in enumerate(lines) if line.startswith("Evaluation needed")), len(lines) - 1)
            reasons = lines[first + 1:] or [lines[first].removeprefix("Evaluation needed: ")] if lines else []
            evaluation_reason = "; ".join(reasons).replace(eval_folder_path, "<eval>")
        if index not in actions:
            status, lines = run_check("package_run.py", ["--run_folder_path", run_folder_path,
                                                         "--norm_factor_file_path", args.norm_factor_file_path,
                                                         "--check_only"])
            training_reason = (lines[-1] if lines else "").replace(run_folder_path, "the run folder")
            if status == NEEDS_WORK_EXIT_CODE:
                actions[index] = (("skip", f"not trained, submit it with submit_ablation_array.sh ({training_reason})")
                                  if args.evaluation_only else ("train", training_reason))
            elif status == 0:
                actions[index] = ("evaluate", f"trained; {evaluation_reason}")
            else:
                actions[index] = ("error", f"training check failed (exit code {status}): {' '.join(lines[-3:])}")
        action, reason = actions[index]
        print(f"  {index:>3} {action:<8} {rows[index]['run_id']}: {reason}", file=sys.stderr)

    error_indices = [index for index, (action, _) in actions.items() if action == "error"]
    if error_indices:
        print(f"ERROR: rows {format_indices(error_indices)} could not be checked (see above); the jobs would stop at "
              f"the same check. Fix or inspect them, or leave them out of the indices.", file=sys.stderr)
        sys.exit(1)
    for name, names in [("train", ["train"]), ("evaluate", ["evaluate"]), ("work", ["train", "evaluate"])]:
        print(f"{name}\t{format_indices([index for index, (action, _) in actions.items() if action in names])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--manifest_file_path',
                        type=str,
                        required=True,
                        help='manifest.tsv of the campaign (generate_ablation_runs.py).')
    parser.add_argument('--indices',
                        type=str,
                        default="",
                        help='Rows to check, as LSF array indices (e.g. 1-4 or 5,7,9-12; default: all rows).')
    parser.add_argument('--norm_factor_file_path',
                        type=str,
                        required=True,
                        help='Normalization factors of the dataset (package_run.py).')
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        help='Dataset name of the evaluation outputs (evaluate_ablation_run.py).')
    parser.add_argument('--label_file_paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Label files of the evaluation (evaluate_ablation_run.py).')
    parser.add_argument('--label_keys',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Label keys of the evaluation (evaluate_ablation_run.py).')
    parser.add_argument('--unassigned_label',
                        type=str,
                        required=True,
                        help='Unassigned label(s) of the evaluation (evaluate_ablation_run.py).')
    parser.add_argument('--eval_gene_set',
                        type=str,
                        choices=EVAL_GENE_SETS,
                        required=True,
                        help='Gene set of the evaluation data, EVAL_GENE_SET of dataset.env '
                             '(evaluate_ablation_run.py).')
    parser.add_argument('--fp32_embeddings',
                        action='store_true',
                        help='The evaluation embeds in fp32 (EVAL_FP32=1).')
    parser.add_argument('--force_evaluation',
                        action='store_true',
                        help='Evaluate every trained run again, whatever evaluation_complete records (FORCE_EVAL=1).')
    parser.add_argument('--evaluation_only',
                        action='store_true',
                        help='For the evaluation array: skip runs that have not finished training.')
    args = parser.parse_args()

    main(args)
