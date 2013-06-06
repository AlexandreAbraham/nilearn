import numpy as np
import os
from nisl.datasets import fetch_kamitani
from nisl.io import NiftiMultiMasker

dataset = fetch_kamitani()

X_random = dataset.func[12:]
X_figure = dataset.func[:12]
y_random = dataset.label[12:]
y_figure = dataset.label[:12]


masker = NiftiMultiMasker(mask=dataset.mask)
masker.fit()

X_train = masker.transform(X_random)
X_test = masker.transform(X_figure)

y_shape = (10, 10)
y_train = []
for y in y_random:
    y_train.append(np.reshape(np.loadtxt(y, dtype=np.int, delimiter=','),
                              (-1,) + y_shape, order='F'))
y_test = []
for y in y_figure:
    y_test.append(np.reshape(np.loadtxt(y, dtype=np.int, delimiter=','),
                              (-1,) + y_shape, order='F'))


kamitani_filtered_file = "/volatile/cache/kamitani_filtered.npz"
filtered_from_cache = True
if filtered_from_cache and os.path.exists(kamitani_filtered_file):
    X_train_med_filt = np.load(kamitani_filtered_file)['data']
else:
    from scipy.signal import medfilt
    X_train_med_filt = np.vstack([block - medfilt(block, [61, 1])
                              for block in X_train])
    np.savez(kamitani_filtered_file, data=X_train_med_filt)


y_train = np.concatenate(y_train)

design_ = y_train.reshape(y_train.shape[0], -1)

delays_ = [2, 3, 4]


def make_delayed(des, delays):
    design = []

    for delay in delays:
        delayed = np.roll(des, 1, axis=0)
        delayed[:delay] = 0
        design.append(delayed)

    return np.hstack(design)


design = make_delayed(design_, delays_)
design[design == -1] = 0

from ridge import _RidgeGridCV, _multi_corr_score, _multi_r2_score

ridge = _RidgeGridCV(alpha_min=1, alpha_max=1e6,
                      n_grid_points=6, n_grid_refinements=2, logscale=True,
                      score_func=_multi_r2_score, cv=5, solver="eigen")

from sklearn.cross_validation import cross_val_score, KFold

scores = cross_val_score(ridge, design, X_train_med_filt, cv=5, score_func=_multi_corr_score)
np.savez("/volatile/cache/kamitani_scores", scores=scores)

scores = np.load("/volatile/cache/kamitani_scores.npz")['scores']

pynax = False
if pynax:
    from pynax.core import Mark
    from pynax.view import MatshowView, ImshowView, PlotView
    import pylab as pl
    from nipy.labs import viz

    fig=pl.figure(figsize=(16, 16))
    
    def b(a, b, c, d, v=0.01):
        return [a + v, b + v, c - 2 * v, d - 2 * v]

    ax_ax_score = fig.add_axes(b(0., .7, .33, .3))
    ax_sag_score = fig.add_axes(b(.33, .7, .33, .3))
    ax_cor_score = fig.add_axes(b(.66, .7, .33, .3))

    ax_ax_roi = fig.add_axes(b(0., .4, .33, .3))
    ax_sag_roi = fig.add_axes(b(.33, .4, .33, .3))
    ax_cor_roi = fig.add_axes(b(.66, .4, .33, .3))

    ax_plot_roi = fig.add_axes(b(0., 0.2, .99, .1))

    axess = [ax_ax_score, ax_sag_score, ax_cor_score,
             ax_ax_roi, ax_sag_roi, ax_cor_roi]
    
    for ax in axess:
        ax.axis('off')

    m_ax = Mark(20, {'color': 'r'})
    m_sag = Mark(20, {'color': 'g'})
    m_cor = Mark(20, {'color': 'b'})

    m_roi = Mark(12, {'color': 'k'})

    display_options = dict(
        interpolation="nearest",
        cmap=pl.cm.hot,
        vmin=scores.mean(0).min(),
        vmax=scores.mean(0).max())

    roi_display_options = dict(
        interpolation="nearest",
        cmap=pl.cm.gray,
        vmin=0,
        vmax=1
        )

    from nisl.io import NiftiMasker
    masker2 = NiftiMasker(dataset.mask)
    masker2.fit()
    score_data = masker2.inverse_transform(scores.mean(0)).get_data()

    v1 = ImshowView(ax_ax_score, score_data, ['h', '-v', m_ax],
                    display_options)
    v1.add_vmark(m_sag)
    v1.add_hmark(m_cor)
    v2 = ImshowView(ax_sag_score, score_data, ['h', m_sag, '-v'],
                    display_options)
    v2.add_vmark(m_ax)
    v2.add_hmark(m_cor)
    v3 = ImshowView(ax_cor_score, score_data, [m_cor, '-v', 'h'],
                    display_options)
    v3.add_hmark(m_ax)
    v3.add_vmark(m_sag)

    rois = masker2.inverse_transform(
        masker2.transform(dataset.mask_roi)).get_data()

    v4 = ImshowView(ax_ax_roi, rois, ['h', '-v', m_ax, m_roi],
                    roi_display_options)
    v4.add_vmark(m_sag)
    v4.add_hmark(m_cor)
    v5 = ImshowView(ax_sag_roi, rois, ['h', m_sag, '-v', m_roi],
                    roi_display_options)
    v5.add_vmark(m_ax)
    v5.add_hmark(m_cor)
    v6 = ImshowView(ax_cor_roi, rois, [m_cor, '-v', 'h', m_roi],
                    roi_display_options)
    v6.add_hmark(m_ax)
    v6.add_vmark(m_sag)

    v7 = PlotView(ax_plot_roi, rois, [m_sag, m_cor, m_ax, 'v'])
    pl.ylim(0, 2)
    pl.xlim(0, 38)
    v7.add_hmark(m_roi)

    for v in [v1, v2, v3, v4, v5, v6, v7]:
        v.draw()


# num_voxels_to_lasso = 100

# voxel_indices = scores.mean(0).argsort()[::-1][:num_voxels_to_lasso]

# from sklearn.linear_model import LassoLarsCV

# lasso = LassoLarsCV(max_iter=200)

num_voxels_to_lasso = 100

voxel_indices = scores.mean(0).argsort()[::-1][:num_voxels_to_lasso]

from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, alphas=[.01, .1, .5, 1., 2., 10.])

lasso_design = make_delayed(design_, [2]).astype(np.float64)
lasso_design = (lasso_design - lasso_design.mean(0)) / lasso_design.std(0)

# from sklearn.externals.joblib import Parallel, delayed

# scores = Parallel(n_jobs=10)(delayed(cross_val_score)(lasso, design, vox, cv=4)
#                              for vox in X_train_med_filt.T[voxel_indices])

# from sklearn.cross_validation import KFold
# cv = KFold(len(X_train_med_filt), 4)
# train, test = [(tr, ts) for tr, ts in cv][0]

rfs = []
for vox in X_train_med_filt.T[voxel_indices]:
    lasso.fit(lasso_design, vox)
    rfs.append(lasso.coef_)


y_train_dx = (y_train[:, 1:, :-1] - y_train[:, :-1, :-1])
y_train_dy = (y_train[:, :-1, 1:] - y_train[:, :-1, :-1])


d_y_train = np.sqrt(y_train_dx ** 2 + y_train_dy ** 2)

d_design = make_delayed(d_y_train.reshape(d_y_train.shape[0], -1), [2, 3, 4])

d_scores = cross_val_score(ridge, np.hstack([design, d_design]),
                           X_train_med_filt, cv=5, score_func=_multi_corr_score)
