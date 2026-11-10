import pandas as pd
from sklearn.dummy import DummyClassifier
from data import load_folds_paper
from metrics import evaluate, summarise, log_result


def run():
    f = load_folds_paper()
    res = []
    for k in range(5):
        col = f"paper_fold{k}"
        tr = f[f[col] == "train"]
        va = f[f[col] == "val"]
        clf = DummyClassifier(strategy="most_frequent").fit(tr[["Label"]], tr["Label"])
        res.append(evaluate(va["Label"], clf.predict(va[["Label"]])))
    s = summarise(res)
    print("paper", s)
    log_result("majority_class", "paper", {"strategy": "most_frequent"}, s)


if __name__ == "__main__":
    run()