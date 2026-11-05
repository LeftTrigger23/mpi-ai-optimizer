"""Open a local benchmark, selector, and message-routing visualizer."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import mkdtemp
import webbrowser

from train_selector import FEATURES, PROJECT, load_dataset, new_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="save HTML to this path instead of opening it")
    args = parser.parse_args()

    X, y = load_dataset()
    predictions = new_tree().fit(X, y).predict(X)
    choices = {
        f"{processes},{count}": int(prediction)
        for (processes, count), prediction in zip(X[FEATURES].itertuples(index=False, name=None), predictions)
    }
    with (PROJECT / "repeated_results.csv").open(newline="") as source:
        timings = {
            f"{row['processes']},{row['count']}": [
                float(row["standard_time"]) * 1e6,
                float(row["pairwise_time"]) * 1e6,
            ]
            for row in csv.DictReader(source)
        }

    template = (PROJECT / "mpi_visualizer.html").read_text()
    html = template.replace("__BENCHMARK_DATA__", json.dumps({"timings": timings, "choices": choices}))
    destination = args.output or Path(mkdtemp(prefix="mpi-visualizer-")) / "index.html"
    with destination.open("x", encoding="utf-8") as output:
        output.write(html)
    print(destination.resolve())
    if args.output is None:
        webbrowser.open(destination.resolve().as_uri())


if __name__ == "__main__":
    main()
