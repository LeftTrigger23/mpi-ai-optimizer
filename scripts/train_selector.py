"""Train and evaluate a small all-to-all algorithm selector."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text

PROJECT = Path(__file__).resolve().parents[1]
FEATURES = ["processes", "count"]


def load_dataset():
    data = pd.read_csv(PROJECT / "data" / "training_data.csv", dtype=int)
    if data.duplicated(FEATURES).any():
        raise ValueError("Expected one row per benchmark configuration")
    return data[FEATURES], data["label"]


def new_tree():
    # A depth of two keeps the rules inspectable and can isolate one corner.
    return DecisionTreeClassifier(max_depth=2, random_state=42)


def main():
    X, y = load_dataset()
    print(f"Configurations: {len(X)}; labels: {y.value_counts().sort_index().to_dict()}")
    held_out = cross_val_predict(new_tree(), X, y, cv=LeaveOneOut())
    baseline = [0] * len(y)

    # Timings are joined only for scoring. They never enter the model.
    timings = pd.read_csv(PROJECT / "data" / "repeated_results.csv").set_index(FEATURES)
    report = X.assign(label=y, held_out=held_out).join(timings, on=FEATURES)
    for name, predictions in [("Always standard", baseline), ("Held-out tree", held_out)]:
        matrix = confusion_matrix(y, predictions, labels=[0, 1])
        print(f"{name}: {matrix.trace()}/{len(y)} correct")
        print(f"  Confusion matrix (actual rows, predicted columns): {matrix.tolist()}")
        chosen = report.standard_time.where(pd.Series(predictions, index=report.index).eq(0),
                                            report.pairwise_time)
        regret = chosen - report[["standard_time", "pairwise_time"]].min(axis=1)
        print(f"  Mean measured regret: {regret.mean() * 1e6:.3f} us/configuration")

    tree = new_tree().fit(X, y)
    print("\nFull-data tree (descriptive fit):")
    print(export_text(tree, feature_names=FEATURES))
    print(X.assign(label=y, held_out=held_out, fitted=tree.predict(X)).to_string(index=False))


if __name__ == "__main__":
    main()
