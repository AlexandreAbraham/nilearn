import numpy as np

### Load kamitani data




### Load Kamitani dataset #####################################################
from nilearn import datasets
dataset = datasets.fetch_kamitani()
X_random = dataset.func[12:]
X_figure = dataset.func[:12]
y_random = dataset.label[12:]
y_figure = dataset.label[:12]
y_shape = (10, 10)

### Preprocess data ###########################################################
import numpy as np
from nilearn.io import MultiNiftiMasker

print "Preprocessing data"

# Load and mask fMRI data
masker = MultiNiftiMasker(mask=dataset.mask, detrend=True,
                          standardize=True)
masker.fit()
X_train = masker.transform(X_random)
X_test = masker.transform(X_figure)

# Load target data
y_train = []
for y in y_random:
    y_train.append(np.reshape(np.loadtxt(y, dtype=np.int, delimiter=','),
                              (-1,) + y_shape, order='F'))

y_test = []
for y in y_figure:
    y_test.append(np.reshape(np.loadtxt(y, dtype=np.int, delimiter=','),
                             (-1,) + y_shape, order='F'))
offset = 2
X_train = [x[offset:] for x in X_train]
y_train = [y[:-offset] for y in y_train]
X_test = [x[offset:] for x in X_test]
y_test = [y[:-offset] for y in y_test]


X_train = np.vstack(X_train)
y_train = np.vstack(y_train).astype(np.float)
X_test = np.vstack(X_test)
y_test = np.vstack(y_test).astype(np.float)

n_pixels = y_train.shape[1]
n_features = X_train.shape[1]



### very simple encoding using ridge regression
from sklearn.linear_model import RidgeCV, Ridge

# ridge = RidgeCV(alphas=np.logspace(1, 3, 10), cv=4)

from sklearn.cross_validation import cross_val_score
from sklearn.externals.joblib import Parallel, delayed

# alphas = np.logspace(1, 3, 5)

# scores = Parallel(n_jobs=20)(delayed(cross_val_score)(RidgeCV(alphas, cv=4),
#     y_train.reshape(len(y_train), -1), target, cv=5, n_jobs=5, verbose=100)
#     for target in X_train.T)


# preselect voxels using f_select
# from sklearn.feature_selection import f_classif

# f_scores = np.array([f_classif(X_train, y_train.reshape(-1, 100)[:, i])[0]
#                      for i in range(100)])

# n_voxels = 1000

# voxel_indices = f_scores.max(0).argsort()[::-1][:n_voxels]

estimator = Ridge(alpha=100.)
from sklearn.cross_validation import KFold
import time
t0 = time.time()
print "doing ridge regression"
cv = KFold(len(y_train), 10)
predictions = [Ridge(alpha=100.).fit(y_train.reshape(-1, 100)[train],
                                     X_train[train]).predict(
                                         y_train.reshape(-1, 100)[test])
               for train, test in cv]

print "scoring"
scores = [1. - (((X_train[test] - pred) ** 2).sum(axis=0) /
           ((X_train[test] - X_train[test].mean(axis=0)) ** 2).sum(axis=0))
for pred, (train, test) in zip(predictions, cv)]

mean_scores = np.array(scores).mean(axis=0)

t1 = time.time()
print "Done in %1.2fs" % (t1 - t0)
# scores = Parallel(n_jobs=20)(delayed(cross_val_score)(RidgeCV(alphas, cv=4),
#     y_train.reshape(len(y_train), -1), target, cv=5, n_jobs=1, verbose=100)
#     for target in X_train.T[voxel_indices])

# scores = np.array(scores).T

# n_voxels_for_lasso = 200
# indices_for_lasso = scores.mean(0).argsort()[::-1][:n_voxels_for_lasso]
# from sklearn.linear_model import LassoLarsCV

# lasso = LassoLarsCV(max_iter=100,)

# receptive_fields = []

# y_train_centered = y_train.reshape(-1, 100)
# y_train_centered = (y_train_centered -
#                     y_train_centered.mean(0)) / y_train_centered.std(0)

# for i, index in enumerate(indices_for_lasso):
#     print "%d %d" % (i, index)
#     receptive_fields.append(
#         lasso.fit(y_train.reshape(-1, 100), X_train[:, index]).coef_)
