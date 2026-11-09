import pandas as pd, glob

files = sorted(glob.glob('outputs/error_analysis/misclassified_*grouped_fold*.csv'))
print(len(files), 'folds found')
d = pd.concat([pd.read_csv(f) for f in files])
print(len(d), 'errors total')

pc = pd.concat([pd.read_csv(f, index_col=0) for f in
                sorted(glob.glob('outputs/error_analysis/per_class_*grouped_fold*.csv'))])
support = pc.groupby(level=0)['support'].sum()

pairs = d.groupby(['true','pred']).size().rename('n').reset_index()
pairs['total'] = pairs.true.map(support)
pairs['rate'] = (pairs.n / pairs.total).round(4)
print()
print(pairs.sort_values('rate', ascending=False).head(15).to_string(index=False))
print()
print('support per class:')
print(support.to_string())
