import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from hyperopt import hp, tpe, STATUS_OK, Trials,fmin
import hyperopt
import seaborn as sns
date_chunks = pd.read_csv("train_date.csv", index_col=0, chunksize=100000, dtype=np.float32)
num_chunks = pd.read_csv("train_numeric.csv", index_col=0,
                         usecols=list(range(969)), chunksize=100000, dtype=np.float32)
X = pd.concat([pd.concat([dchunk, nchunk], axis=1).sample(frac=0.05)
               for dchunk, nchunk in zip(date_chunks, num_chunks)])
y = pd.read_csv("train_numeric.csv", index_col=0, usecols=[0,969], dtype=np.float32).loc[X.index].values.ravel()
X = X.values
clf = XGBClassifier(base_score=0.005)
clf.fit(X, y)
#threshold for a manageable number of features
plt.hist(clf.feature_importances_[clf.feature_importances_>0])
important_indices = np.where(clf.feature_importances_>0.005)[0]

# Got important_indices from above code
#important_indices = []
print ("Found important features %s"%important_indices)
# load entire dataset for these features.
# note where the feature indices are split so we can load the correct ones straight from read_csv
n_date_features = 1156
X = np.concatenate([
    pd.read_csv("train_date.csv", index_col=0, dtype=np.float32,
                usecols=np.concatenate([[0], important_indices[important_indices < n_date_features] + 1])).values,
    pd.read_csv("train_numeric.csv", index_col=0, dtype=np.float32,
                usecols=np.concatenate([[0], important_indices[important_indices >= n_date_features] + 1 - 1156])).values
], axis=1)
y = pd.read_csv("train_numeric.csv", index_col=0, dtype=np.float32, usecols=[0,969]).values.ravel()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,stratify=y)

def objective(params):
    params = {'max_depth': int(params['max_depth']),'n_estimators': int(params['n_estimators']),'learning_rate': float(params['learning_rate']),'alpha': float(params['alpha'])}
    clf = XGBClassifier(n_jobs=4, class_weight='balanced', **params)
    model=clf.fit(X_train,y_train)
    p=model.predict(X_test)
    score = matthews_corrcoef( y_test, p)
    print("Accuracy{:.3f} params {}".format(score, params))
    return score

space = {
    'max_depth': hp.quniform('max_depth', 6, 20, 1),
    'n_estimators': hp.quniform('n_estimators', 100, 200, 25),
    'learning_rate': hp.quniform('learning_rate', 0.1, 1, 0.1),
    'alpha': hp.quniform('alpha', 0, 1, 0.1)
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10
            )
