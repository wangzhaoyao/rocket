from sklearn import tree
import pandas as pd

def smbinning_cutpoint(df, Y, x):
    y = df[Y]
    x_test = df[[x, ]]
    mytree = tree.DecisionTreeClassifier(
        max_features=1,
        # min_weight_fraction_leaf=0.05,
        min_samples_split=0.05,
        criterion="entropy",
        max_leaf_nodes=3)
    mytree.fit(x_test, y)
    cutpoint = mytree.tree_.threshold
    print(x)
    print(cutpoint)
    return cutpoint[cutpoint != -2]


def BadRateEncoding(df, col, target):
    '''

    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad*1.0/x.total,axis = 1)
    br_dict = regroup[[col,'bad_rate']].set_index([col]).to_dict(orient='index')
    for k, v in br_dict.items():
        br_dict[k] = v['bad_rate']
    badRateEnconding = df[col].map(lambda x: br_dict[x])
    return {'encoding':badRateEnconding, 'br_rate':br_dict}
