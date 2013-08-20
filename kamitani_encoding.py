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

n_voxels_for_lasso = 10
indices_for_lasso = mean_scores.argsort()[::-1][:n_voxels_for_lasso]
from sklearn.linear_model import LassoLarsCV

lasso = LassoLarsCV(max_iter=10,)

receptive_fields = []

y_train_centered = y_train.reshape(-1, 100)
y_train_centered = (y_train_centered -
                    y_train_centered.mean(0)) / y_train_centered.std(0)

for i, index in enumerate(indices_for_lasso):
    print "%d %d" % (i, index)
    receptive_fields.append(
        lasso.fit(y_train.reshape(-1, 100), X_train[:, index]).coef_)


rfs = np.array(receptive_fields).reshape(-1, 10, 10)
grid = np.mgrid[-4.5:5.5, -4.5:5.5]

bary_coords = ((np.abs(rfs) /
           np.abs(rfs).sum(-1).sum(-1)[:, np.newaxis, np.newaxis]
    )[:, np.newaxis, :, :] *
    grid[np.newaxis, :, :, :]).sum(-1).sum(-1)


grid_squared = grid[:, np.newaxis, :, :] * grid[np.newaxis, :, :, :]

coord_mom2 = ((np.abs(rfs) /
           np.abs(rfs).sum(-1).sum(-1)[:, np.newaxis, np.newaxis]
    )[:, np.newaxis, np.newaxis, :, :] *
    grid_squared[np.newaxis, :, :, :]).sum(-1).sum(-1)

coord_var = coord_mom2 - (bary_coords[:, :, np.newaxis] *
                          bary_coords[:, np.newaxis, :])

coord_var_det = coord_var[:, 0, 0] * coord_var[:, 1, 1] - coord_var[:, 0, 1] ** 2

problem = coord_var_det < 0
no_problem = coord_var_det >= 0
coord_std = np.zeros_like(coord_var_det)
coord_std[no_problem] = np.sqrt(coord_var_det[no_problem])


correlations = ((X_train - X_train.mean(0)) / X_train.std(0)).T.dot(
    ((y_train - y_train.mean(0)) / y_train.std(0)).reshape(-1, 100)) / (len(X_train) - 1)


more_problem = np.logical_or(np.isnan(bary_coords).any(axis=1), problem)


angles = np.zeros_like(coord_std)
angles[more_problem == False] = np.arctan2(bary_coords[:, 1], -bary_coords[:, 0])

# angle_brain = mask.





# this goes towards finding the surroundings of one given voxel

brain_with_indices = masker.inverse_transform(
    np.arange(n_features) + 1).get_data() - 1


def ind_2_sub(ind):
    coord = np.array(np.where(brain_with_indices == ind)).ravel()

    if len(coord == 3):
        return coord


def sub_2_ind(sub):
    try:
        ind = brain_with_indices[tuple(sub)]
    except:
        ind = -1

    if ind >= 0:
        return ind


def extract_27_neighbourhood(sub):
    sub = np.array(sub).ravel()
    if len(sub) != 3:
        raise Exception("Don't understand, want subscript for one point")

    coords = sub.reshape(-1, 1, 1, 1) + np.mgrid[[slice(-1, 2)] * 3]

    coords = coords.reshape(3, -1).T

    return coords


def extract_index_neighbourhood(ind):

    sub = ind_2_sub(ind)

    neighbourhoods = extract_27_neighbourhood(sub)

    return [sub_2_ind(sub) for sub in neighbourhoods]


def get_lasso_rfs(indices):

    lrfs = []
    for index in indices:
        try:
            lrfs.append(
                lasso.fit(y_train.reshape(-1, 100),
                        X_train[:, index]).coef_.reshape(10, 10))
        except:
            lrfs.append(None)

    return lrfs



# show 27 voxel around a central one
vind = 1951

index_neighbourhood = extract_index_neighbourhood(vind)
subscript_neighbourhood = extract_27_neighbourhood(ind_2_sub(vind))

neighbourhood_scores = np.array(scores).mean(0)[index_neighbourhood]

n_lasso_rfs = get_lasso_rfs(index_neighbourhood)

vmin = 0
vmax =  np.array(n_lasso_rfs).max()

import pylab as pl
for i in range(3):
    pl.figure()
    for j in range(9):
        k = i * 9 + j
        pl.subplot(3, 3, j + 1)
        pl.imshow(np.array(n_lasso_rfs).reshape(3, 9, 10, 10)[i, j],
                  interpolation="nearest", vmin=vmin, vmax=vmax)
        pl.gray()
        pl.xticks([])
        pl.yticks([])
        pl.xlabel('%d %s %1.3f' %  (index_neighbourhood[k],
                            str(tuple(subscript_neighbourhood[k])),
            neighbourhood_scores[k]))



the_chosen_voxels = [1780, 1951, 2131, 1935]

where_are_they = np.where((np.array(the_chosen_voxels)[:, np.newaxis] ==
                  np.array(index_neighbourhood)).any(0))[0]

chosen_rfs = np.array(n_lasso_rfs)[where_are_they, :, :]

for rf, index in zip(chosen_rfs, where_are_they):
    pl.figure()
    pl.imshow(rf, vmin=0, interpolation="nearest")
    pl.axis('off')
    pl.title('%d, %s' % (index_neighbourhood[index],
                         str(subscript_neighbourhood[index])))

