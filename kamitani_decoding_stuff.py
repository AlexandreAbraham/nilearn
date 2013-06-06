import numpy as np


X_train = np.load("/volatile/cache/kamitani_filtered.npz")['data']

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.feature_selection import f_classif, SelectKBest

from sklearn.pipeline import Pipeline


pipeline = Pipeline([('selection', SelectKBest(f_classif, 500)),
                     ('clf', LogisticRegression(penalty="l1", C=0.01))])
pipeline_OMP = Pipeline([('selection', SelectKBest(f_classif, 500)),
                     ('clf', OrthogonalMatchingPursuit(n_nonzero_coefs=10))])

from sklearn.cross_validation import cross_val_score

design = np.load("/volatile/cache/kamitani_designs.npz")['lasso_design']


from sklearn.externals.joblib import Parallel, delayed

scores_log = Parallel(n_jobs=10)(delayed(cross_val_score)(pipeline, X_train, y, cv=5, verbose=True) for y in design.T)

scores_omp = Parallel(n_jobs=10)(delayed(cross_val_score)(pipeline_OMP, X_train, y, cv=5, verbose=True) for y in design.T)

import pylab as pl

pl.figure()
pl.imshow(np.array(scores_log).mean(1).reshape(10, 10), interpolation="nearest")
pl.hot()
pl.colorbar()
pl.show()

