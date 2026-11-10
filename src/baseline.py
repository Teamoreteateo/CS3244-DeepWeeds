import pandas as pd
from sklearn.dummy import DummyClassifier
from data import load_folds, load_folds_paper
from metrics import evaluate, summarise, log_result


def run(scheme):
    paper = scheme == "paper"
    f = load_folds_paper() if paper else load_folds()
    res = []
    for k in range(5):
        if paper:
            col = f"paper_fold{k}"
            tr, va = f[f[col] == "train"], f[f[col] == "val"]
        else:
            pool = f[~f["is_test"]]
            col = f"fold_{scheme}"
            tr, va = pool[pool[col] != k], pool[pool[col] == k]
        clf = DummyClassifier(strategy="most_frequent").fit(tr[["Label"]], tr["Label"])
        res.append(evaluate(va["Label"], clf.predict(va[["Label"]])))
    s = summarise(res)
    print(scheme, s)
    log_result("majority_class", scheme, {"strategy": "most_frequent"}, s)


if __name__ == "__main__":
    run("random")
    run("grouped")
    run("paper")