import pandas as pd

f = pd.read_csv('splits/folds.csv', dtype={'session': str})
base = f[['Filename','Label','Species','date','instrument','session']].copy()

for k in range(5):
    tr = pd.read_csv(f'/tmp/dw/labels/train_subset{k}.csv')['Filename']
    va = pd.read_csv(f'/tmp/dw/labels/val_subset{k}.csv')['Filename']
    te = pd.read_csv(f'/tmp/dw/labels/test_subset{k}.csv')['Filename']
    col = pd.Series('unassigned', index=base.index)
    col[base.Filename.isin(tr)] = 'train'
    col[base.Filename.isin(va)] = 'val'
    col[base.Filename.isin(te)] = 'test'
    base[f'paper_fold{k}'] = col
    print(k, col.value_counts().to_dict())

assert base.Filename.tolist() == f.Filename.tolist(), 'ROW ORDER DIFFERS'
base.to_csv('splits/folds_paper.csv', index=False)
print('wrote', base.shape)
