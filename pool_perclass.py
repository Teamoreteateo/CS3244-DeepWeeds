import pandas as pd, numpy as np, glob

SP = ['Chinee apple','Lantana','Parkinsonia','Parthenium','Prickly acacia',
      'Rubber vine','Siam weed','Snake weed','Negative']

pc = pd.concat([pd.read_csv(f, index_col=0) for f in
                sorted(glob.glob('outputs/error_analysis/per_class_*grouped_fold*.csv'))])
support = pc.groupby(level=0)['support'].sum().reindex(SP)

d = pd.concat([pd.read_csv(f) for f in
               sorted(glob.glob('outputs/error_analysis/misclassified_*grouped_fold*.csv'))])

rows = []
for sp in SP:
    n   = support[sp]
    fn  = (d.true == sp).sum()          # misclassified away from sp
    tp  = n - fn
    fp  = (d.pred == sp).sum()          # other classes wrongly called sp
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / n if n else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    rows.append([sp, round(prec,3), round(rec,3), round(f1,3), int(n)])

t = pd.DataFrame(rows, columns=['class','precision','recall','f1','support'])
print(t.to_string(index=False))
print()
print('macro F1 :', round(t.f1.mean(), 4))
print('weed->neg:', round((d.pred == 'Negative').sum() / support.drop('Negative').sum(), 4))
