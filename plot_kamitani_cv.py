"""
The Kamitani paper: reconstruction of visual stimuli
======================================================

"""

### Init ######################################################################

remove_rest_period = True
detrend = True
standardize = False
offset = 2

#generate_video = 'video.mp4'
#generate_gif = 'video.gif'
generate_video = None
generate_gif = None
generate_image = None
pynax = True

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
masker = MultiNiftiMasker(mask=dataset.mask, detrend=detrend,
                          standardize=standardize)
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


def flatten(list_of_2d_array):
    flattened = []
    for array in list_of_2d_array:
        flattened.append(array.ravel())
    return flattened


# Simply flatten the array
y_train = flatten(y_train)

y_test = np.asarray(flatten(y_test))
y_train = np.asarray(y_train)


# Remove rest period
if remove_rest_period:
    X_train = X_train[y_train[:, 0] != -1]
    y_train = y_train[y_train[:, 0] != -1]
    X_test = X_test[y_test[:, 0] != -1]
    y_test = y_test[y_test[:, 0] != -1]


### Prediction function #######################################################

print "Learning"

# OMP
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
clf = OMP(n_nonzero_coefs=20)
clf.fit(X_train, y_train)


from sklearn.utils import check_arrays
from sklearn.metrics.metrics import _is_1d, _check_1d_array


def log_score(y_true, y_pred):
    y_true, y_pred = check_arrays(y_true, y_pred)

    # Handle mix 1d representation
    if _is_1d(y_true):
        y_true, y_pred = _check_1d_array(y_true, y_pred, ravel=True)

    if len(y_true) == 1:
        raise ValueError("r2_score can only be computed given more than one"
                         " sample.")
    numerator = (np.log(1 + (y_true - y_pred))).sum()
    denominator = (np.log(1 + (y_true - y_true.mean(axis=0)))).sum()

    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0
        else:
            # arbitrary set to zero to avoid -inf scores, having a constant
            # y_true is not interesting for scoring a regression anyway
            return 0.0

    return 1 - numerator / denominator


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.feature_selection import f_classif, SelectKBest

from sklearn.pipeline import Pipeline

pipeline = Pipeline([('selection', SelectKBest(f_classif, 500)),
                     ('clf', LogisticRegression(penalty="l1", C=0.01))])
pipeline_OMP = Pipeline([('selection', SelectKBest(f_classif, 500)),
                     ('clf', OrthogonalMatchingPursuit(n_nonzero_coefs=10))])

from sklearn.cross_validation import cross_val_score

from sklearn.externals.joblib import Parallel, delayed

scores_log = Parallel(n_jobs=10)(delayed(cross_val_score)(pipeline, X_train, y,
    score_func=log_score, cv=5, verbose=True) for y in y_train)

scores_omp = Parallel(n_jobs=10)(delayed(cross_val_score)(pipeline_OMP,
    X_train, y, score_func=log_score, cv=5, verbose=True) for y in y_train)

import pylab as pl

pl.figure()
pl.imshow(np.array(scores_log).mean(1).reshape(10, 10),
        interpolation="nearest")
pl.hot()
pl.colorbar()
pl.show()
