import pandas as pd
from sklearn.dummy import DummyClassifier
from data import load_folds
from metrics import evaluate, summarise, log_result


def run(scheme):
    f = load_folds()
    pool = f[~f["is_test"]].reset_index(drop=True)
    col = f"fold_{scheme}"
    res = []
    for k in range(5):
        tr, va = pool[pool[col] != k], pool[pool[col] == k]
        clf = DummyClassifier(strategy="most_frequent").fit(tr[["Label"]], tr["Label"])
        res.append(evaluate(va["Label"], clf.predict(va[["Label"]])))
    s = summarise(res)
    print(scheme, s)
    log_result("majority_class", scheme, {"strategy": "most_frequent"}, s)


if __name__ == "__main__":
    run("random")
    run("grouped")