"""Choose one existing MPI executable before starting any ranks."""

import argparse
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]


def choose_algorithm(processes, count, experimental_tree):
    if not experimental_tree:
        return "alltoall", "default baseline"

    from train_selector import load_dataset, new_tree

    X, y = load_dataset()
    matching = X.loc[X.processes.eq(processes) & X["count"].eq(count)]
    if matching.empty:
        return "alltoall", "configuration outside the benchmark grid"

    prediction = new_tree().fit(X, y).predict(matching)[0]
    program = "pairwise_alltoall" if prediction == 1 else "alltoall"
    return program, "experimental full-data tree"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processes", type=int)
    parser.add_argument("count", type=int, help="integers sent to each rank")
    parser.add_argument("--experimental-tree", action="store_true")
    parser.add_argument(
        "--local-unbound", action="store_true",
        help="use explicit localhost slots and disable core binding (Open MPI)",
    )
    args = parser.parse_args()
    if args.processes < 1 or args.count < 1:
        parser.error("processes and count must be positive")
    # The original C programs encode rank and offsets in signed int values.
    if args.processes ** 2 * args.count > 2**31 - 1:
        parser.error("configuration exceeds the original programs' integer range")

    program, reason = choose_algorithm(args.processes, args.count, args.experimental_tree)
    executable = PROJECT / "bin" / program
    print(f"Selected {program} ({reason})", flush=True)
    launch_options = []
    if args.local_unbound:
        launch_options = [
            "--host", f"localhost:{args.processes}", "--map-by", "slot", "--bind-to", "none",
        ]
    try:
        result = subprocess.run(
            ["mpirun", *launch_options, "-np", str(args.processes), str(executable), str(args.count)],
            stdout=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        parser.exit(1, f"Could not start mpirun: {error}\n")

    print(result.stdout, end="")
    if result.returncode:
        parser.exit(1, f"MPI launcher exited with status {result.returncode}\n")
    passed = result.stdout.count("Verification PASSED")
    if passed != args.processes or "Verification FAILED" in result.stdout:
        parser.exit(1, f"Expected {args.processes} passing ranks, found {passed}\n")


if __name__ == "__main__":
    main()
