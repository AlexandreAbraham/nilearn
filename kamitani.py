"""
The Kamitani paper: reconstruction of visual stimuli
======================================================

"""

### Options ###################################################################

# Preprocessing
remove_rest_period = [False, True]
foveal_focus_radius = [None, 2., 2.5, 3.]  # 2.5 is a good value if set
multi_scale = [False, True]  # Not compatible with foveal focal radius
detrend = [False, True]
standardize = [False, True]

# Learning
learn_fusion_params = [True, False]  # Learn params with LinearRegression
classifier = [
    'anova_svc',  # f_classif 50 features + linear SVC (C = 1.)
    'ridge',  # ridge regression
    'omp',  # Orthgonal Matching Pursuit (n_nonzero_coefs=20)
    'anova_ridge',  # anova_ridge: f_classif 50 features + ridge regression
    'lassolars',  # Lasso Lars
    'bayesianridge'  # Bayesian Ridge
]

# Output
generate_video = [None, 'video.mp4']
generate_image = [None, 'picture.png']

### Init ######################################################################

remove_rest_period = True
foveal_focus_radius = None
multi_scale = False
detrend = True
standardize = False

learn_fusion_params = False  # Learn fusion params with LinearRegression
classifier = 'omp'

generate_video = None
generate_image = None

### Imports ###################################################################

from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline


### Load Kamitani dataset #####################################################
from nisl import datasets
dataset = datasets.fetch_kamitani()
X_random = dataset.func[12:]
X_figure = dataset.func[:12]
y_random = dataset.label[12:]
y_figure = dataset.label[:12]


### Preprocess data ###########################################################
import numpy as np
from nisl.io import NiftiMultiMasker

print "Preprocessing data"

# Load and mask fMRI data
masker = NiftiMultiMasker(mask=dataset.mask, detrend=detrend,
                          standardize=standardize)
masker.fit()
X_train = masker.transform(X_random)
X_test = masker.transform(X_figure)

# Load target data
y_train = []
for y in y_random:
    y_train.append(np.load(y)['label'])

y_test = []
for y in y_figure:
    y_train.append(np.load(y)['label'])


X_train = np.hstack(X_random).T
y_train = np.hstack(y_random).astype(np.float).T
X_test = np.hstack(X_figure).T
y_test = np.hstack(y_figure).astype(np.float).T

n_pixels = y_train.shape[1]
n_features = X_train.shape[1]


# Remove rest period
if remove_rest_period:
    X_train = X_train[y_train[:, 0] != -1]
    y_train = y_train[y_train[:, 0] != -1]
    X_test = X_test[y_test[:, 0] != -1]
    y_test = y_test[y_test[:, 0] != -1]


# Take only the foveal part (radius is custom)
if foveal_focus_radius:
    from numpy import linalg
    c = (4.5, 4.5)
    radius = foveal_focus_radius
    y_mask = np.ones(100, dtype='bool')
    for i in range(10):
        for j in range(10):
            y_mask[i * 10 + j] = (linalg.norm((c[0] - i, c[1] - j)) <= radius)
    n_pixels = y_mask.sum()
    y_train = y_train[:, y_mask]
    y_test = y_test[:, y_mask]
    # Show the mask
    # plt.imshow(np.reshape(y_mask, [10, 10]), cmap=plt.cm.gray,
    #         interpolation='nearest')
    # plt.show()


def flatten(list_of_2d_array):
    flattened = []
    for array in list_of_2d_array:
        flattened.append(array.ravel())
    return flattened


# Compute scaled images
if multi_scale:
    y_shape = (10, 10)
    y_rows, y_cols = y_shape

    # Height transform :
    #
    # 0.5 *
    #
    # 1 1 0 0 0 0 0 0 0 0
    # 0 1 1 0 0 0 0 0 0 0
    # 0 0 1 1 0 0 0 0 0 0
    # 0 0 0 1 1 0 0 0 0 0
    # 0 0 0 0 1 1 0 0 0 0
    # 0 0 0 0 0 1 1 0 0 0
    # 0 0 0 0 0 0 1 1 0 0
    # 0 0 0 0 0 0 0 1 1 0
    # 0 0 0 0 0 0 0 0 1 1

    height_tf = (np.eye(y_cols) + np.eye(y_cols, k=1))[:y_cols - 1] * .5
    width_tf = (np.eye(y_cols) + np.eye(y_cols, k=-1))[:, :y_cols - 1] * .5

    yt_tall = [np.dot(height_tf, m) for m in y_train]
    yt_large = [np.dot(m, width_tf) for m in y_train]
    yt_big = [np.dot(height_tf, np.dot(m, width_tf)) for m in y_train]

    # Add it to the training set
    y_train = [np.concatenate((y.ravel(), t.ravel(), l.ravel(), b.ravel()),
                              axis=1)
               for y, t, l, b in zip(y_train, yt_tall, yt_large, yt_big)]

else:
    # Simply flatten the array
    y_train = flatten(y_train)

y_test = np.asarray(flatten(y_test))
y_train = np.asarray(y_train)


# Try to learn coefficients to merge images
if multi_scale and learn_fusion_params:
    from sklearn.linear_model import LinearRegression
    clf = LinearRegression()
    clf.fit(X_train, y_train)

    t_pred = clf.predict(X_train)
    t_preds = np.split(t_pred, [n_pixels, 1.5 * n_pixels,
            2 * n_pixels, 2.25 * n_pixels], axis=1)

    t_pred = t_preds[0]
    t_pred_tall = np.array([np.dot(height_tf.T * 2,
        np.reshape(m, (5, 10))).flatten() for m in t_preds[1]])
    t_pred_large = np.array([np.dot(np.reshape(m, (10, 5)),
        width_tf.T * 2).flatten() for m in t_preds[2]])
    t_pred_big = [np.dot(height_tf.T * 2, np.reshape(m, (5, 5)))
        for m in t_preds[3]]
    t_pred_big = np.array([np.dot(np.reshape(m, (10, 5)),
        width_tf.T * 2).flatten() for m in t_pred_big])

    fusions = []
    from sklearn.linear_model import LinearRegression
    for i, t in enumerate(t_pred.T):
        tX = np.column_stack((t_pred[:, i], t_pred_tall[:, i],
            t_pred_large[:, i], t_pred_big[:, i]))
        f = LinearRegression()
        f.fit(tX, y_train[:, i])
        fusions.append(f.coef_)

    fusions = np.array(fusions)
    fusions = (fusions.T / np.sum(fusions, axis=1)).T

# Feature selection analysis
"""
def roi_stat(indices):
    # get ROI names
    names = dataset.roi_name[0:8, 2:4].flatten()
    roi_indices = dataset.roi_volInd[0:8, 2:4].flatten()
    names = dataset.roi_name[:, 0:2].flatten()
    roi_indices = dataset.roi_volInd[:, 0:2].flatten()
    data_indices = []
    for i, roi_ind in enumerate(roi_indices):
        roi_ind = roi_ind.squeeze()
        data_ind = []
        for p in roi_ind:
            data_ind.append(np.where(dataset.volInd == p)[0])
        data_indices.append(np.array([y for x in data_ind for y in x]))

    count = np.zeros(names.shape)
    for ind in indices:
        for i, data_ind in enumerate(data_indices):
            count[i] += (np.where(np.unique(data_ind) == ind)[0].size != 0)
    return (names, count)


from sklearn.svm import SVC
feature_selection = SelectKBest(f_classif, k=100)

feature_selection = RFE(SVC(kernel='linear', C=1.), n_features_to_select=100)

roi_features = []
for i in range(n_features):
    feature_selection.fit(X_train, y_train[i])
    n, c = roi_stat(np.where(feature_selection.get_support())[0])
    roi_features.append(c)

rf = np.array(roi_features)
plt.figure(1)
for i, nn in enumerate(n):
    plt.subplot(6, 4, i + 1)
    plt.axis('off')
    plt.title(nn)
    plt.imshow(np.reshape(rf[:, i], [10, 10]), cmap=plt.cm.hot,
                      interpolation='nearest', vmin=0, vmax=100)

plt.show()

"""

### Prediction function #######################################################

print "Learning"
"""
import copy

def make_pipelines(feature_selection, clf, X_train, y_train, n_features):
    pipelines = []
    pipeline_ref = Pipeline([('fs', feature_selection), ('clf', clf)])
    for i in range(n_features):
        print "Count %d of %d" % ((i + 1), n_features)
        pipeline = copy.deepcopy(pipeline_ref)
        pipeline.fit(X_train, y_train[i, :])
        pipelines.append(pipeline)
    return pipelines


def predict(pipelines, X_test):
    preds = []
    for i, x_test in enumerate(X_test):
        pred = []
        for p in pipelines:
            pred.append(p.predict(x_test))
        pred = np.array(pred)
        preds.append(pred.squeeze())
    return preds
"""

# f_classif + SVC classique : 65%
if classifier == "anova_svc":
    unique_classifier = False

    from sklearn.svm import SVC
    clfs = []

    for i, pixel_time_serie in enumerate(y_train.T):
        print "Count %d of %d" % ((i + 1), y_train.shape[1])
        clf = SVC(kernel='linear', C=1.)
        feature_selection = SelectKBest(f_classif, k=50)
        anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])
        anova_svc.fit(X_train, pixel_time_serie)
        clfs.append(anova_svc)

# Ridge
elif classifier == "ridge":
    unique_classifier = True
    from sklearn.linear_model import RidgeRegressionCV
    clf = RidgeRegressionCV()
    clf.fit(X_train, y_train)

# f_classif + Ridge
elif classifier == 'anova_ridge':
    unique_classifier = False
    from sklearn.linear_model import RidgeRegressionCV
    clfs = []
    for i, pixel_time_serie in enumerate(y_train.T):
        print "Count %d of %d" % ((i + 1), y_train.shape[1])
        clf = RidgeRegressionCV()
        feature_selection = SelectKBest(f_classif, k=50)
        anova_clf = Pipeline([('anova', feature_selection), ('clf', clf)])
        anova_clf.fit(X_train, pixel_time_serie)
        clfs.append(anova_clf)

# OMP
elif classifier == "omp":
    unique_classifier = True
    from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
    clf = OMP(n_nonzero_coefs=20)
    clf.fit(X_train, y_train)

# LassoLars
elif classifier == "lassolars":
    unique_classifier = False
    from sklearn.linear_model import LassoLarsCV

    clfs = []
    for i, pixel_time_serie in enumerate(y_train.T):
        print "Count %d of %d" % ((i + 1), y_train.shape[1])
        clf = LassoLarsCV()
        clf.fit(X_train, pixel_time_serie)
        clfs.append(clf)

# f_classif 100 + sparse SVC
elif classifier == "anova_sparsesvc":
    unique_classifier = False
    from sklearn.svm.sparse import SVC
    clfs = []

    for i, pixel_time_serie in enumerate(y_train.T):
        print "Count %d of %d" % ((i + 1), y_train.shape[1])
        clf = SVC(kernel='linear', C=1.)
        feature_selection = SelectKBest(f_classif, k=50)
        anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])
        anova_svc.fit(X_train, pixel_time_serie)
        clfs.append(anova_svc)

    # Bayesian Ridge
elif classifier == "bayesianridge":
    unique_classifier = False
    from sklearn.linear_model import BayesianRidge
    clfs = []
    for i, pixel_time_serie in enumerate(y_train.T):
        print "Count %d of %d" % ((i + 1), y_train.shape[1])
        clf = BayesianRidge(normalize=False, n_iter=1000)
        feature_selection = SelectKBest(f_classif, k=500)
        anova_svc = Pipeline([('anova', feature_selection), ('svc', clf)])
        anova_svc.fit(X_train, pixel_time_serie)
        clfs.append(anova_svc)

"""
# Learn
pipes = make_pipelines(SelectKBest(f_classif, k=100),
        SVC(kernel='linear', C=1.), X_train, y_train, n_features)

# Predict
y_pred = predict(pipes, X_test)
"""

### Prediction ################################################################

print "Calculating scores and outputs"

# Different processing for algorithms handling multiple outputs and those who
# do not
if unique_classifier:
    y_pred = clf.predict(X_test)
else:
    y_pred = []
    for i, x_test in enumerate(X_test):
        pred = []
        for c in clfs:
            pred.append(c.predict(x_test))
        pred = np.array(pred)
        y_pred.append(pred.squeeze())
    y_pred = np.array(y_pred)


# Revert scaled images if needed
if multi_scale:
    y_preds = np.split(y_pred, [n_pixels, 1.5 * n_pixels,
            2 * n_pixels, 2.25 * n_pixels], axis=1)

    y_pred = y_preds[0]
    y_pred_tall = np.array([np.dot(height_tf.T * 2,
        np.reshape(m, (5, 10))).flatten() for m in y_preds[1]])
    y_pred_large = np.array([np.dot(np.reshape(m, (10, 5)),
        width_tf.T * 2).flatten() for m in y_preds[2]])
    y_pred_big = [np.dot(height_tf.T * 2, np.reshape(m, (5, 5)))
        for m in y_preds[3]]
    y_pred_big = np.array([np.dot(np.reshape(m, (10, 5)),
        width_tf.T * 2).flatten() for m in y_pred_big])

    if learn_fusion_params:
        y_pred = np.array([y_pred.T, y_pred_tall.T, y_pred_large.T,
            y_pred_big.T])
        y_pred = np.sum(y_pred.T * fusions, axis=2)
    else:
        y_pred = (.25 * y_pred + .25 * y_pred_tall + .25 * y_pred_large
            + .25 * y_pred_big)


"""


# Visualize results
y_pred = np.zeros(y_mask.shape)
y_pred[y_mask] = acc
plt.imshow(np.reshape(y_pred, [10, 10]), cmap=plt.cm.gray,
        interpolation='nearest')
plt.show()

print "Result : %d" % np.mean(acc)
"""

"""
Show brains !
"""

if generate_image:
    print 'ok'

if generate_video:
    from matplotlib import animation
    fig = plt.figure()
    sp1 = plt.subplot(131)
    sp1.axis('off')
    sp2 = plt.subplot(132)
    sp2.axis('off')
    sp3 = plt.subplot(133)
    sp3.axis('off')
    ims = []
    for i, t in enumerate(y_pred):
        ims.append((
            sp1.imshow(np.reshape(y_test[i], (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest'),
            sp2.imshow(np.reshape(t, (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest'),
            sp3.imshow(np.reshape(t > 0.5, (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest')))

    im_ani = animation.ArtistAnimation(fig, ims, interval=1000)
    im_ani.save(generate_video)
