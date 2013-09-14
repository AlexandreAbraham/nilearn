"""
The Kamitani paper: reconstruction of visual stimuli
======================================================

"""


def fitta(fuckin_fitta, *the_fuckin_data):
    fuckin_fitta.fit(*the_fuckin_data)
    return fuckin_fitta

### Init ######################################################################

remove_rest_period = True
multi_scale = True
detrend = True
standardize = False
offset = 2
threshold = 0.5

learn_fusion_params = True  # Learn fusion params with LinearRegression

generate_video = 'video.mp4'
#generate_gif = 'video.gif'
#generate_video = None
generate_gif = None
generate_image = None
pynax = False

### Imports ###################################################################

from matplotlib import pyplot as plt

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


# Compute scaled images
if multi_scale:
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
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed

# Create as many OMP as voxels to predict
clfs = []
for i in range(y_train.shape[1]):
    print('Learning %d/%d' % (i, y_train.shape[1]))
    clf = Pipeline([('selection', SelectKBest(f_classif, 500)),
                    ('clf',
                        # OMP(n_nonzero_coefs=4)
                        LR(penalty='l1', C=0.01)
                    )])
    clfs.append(clf)

clfs = Parallel(n_jobs=-2)(delayed(fitta)(clf, X_train, y_train[:, i])
                    for i, clf in enumerate(clfs))
### Prediction ################################################################
print "Calculating scores and outputs"

y_pred = []
for clf in clfs:
    y_pred.append(clf.predict(X_test))
y_pred = np.asarray(y_pred).T


### Multi scale ###############################################################
def _split_multi_scale(y, y_shape):
    """ Split data into 4 original multi_scale images
    """
    yw, yh = y_shape

    # Index of original image
    split_index = [yw * yh]
    # Index of large image
    split_index.append(split_index[-1] + (yw - 1) * yh)
    # Index of tall image
    split_index.append(split_index[-1] + yw * (yh - 1))
    # Index of big image
    split_index.append(split_index[-1] + (yw - 1) * (yh - 1))

    # We split according to computed indices
    y_preds = np.split(y, split_index, axis=1)

    # y_pred is the original image
    y_pred = y_preds[0]

    # y_pred_tall is the image with 1x2 patch application. We have to make
    # some calculus to get it back in original shape
    height_tf_i = (np.eye(y_cols) + np.eye(y_cols, k=-1))[:, :y_cols - 1] * .5
    height_tf_i.flat[0] = 1
    height_tf_i.flat[-1] = 1
    y_pred_tall = [np.dot(height_tf_i, np.reshape(m, (yw - 1, yh))).flatten()
                   for m in y_preds[1]]
    y_pred_tall = np.asarray(y_pred_tall)

    # y_pred_large is the image with 2x1 patch application. We have to make
    # some calculus to get it back in original shape
    width_tf_i = (np.eye(y_cols) + np.eye(y_cols, k=1))[:y_cols - 1] * .5
    width_tf_i.flat[0] = 1
    width_tf_i.flat[-1] = 1
    y_pred_large = [np.dot(np.reshape(m, (yw, yh - 1)), width_tf_i).flatten()
                   for m in y_preds[2]]
    y_pred_large = np.asarray(y_pred_large)

    # y_pred_big is the image with 2x2 patch application. We use previous
    # matrices to get it back in original shape
    y_pred_big = [np.dot(np.reshape(m, (yw - 1, yh - 1)), width_tf_i)
                  for m in y_preds[3]]
    y_pred_big = [np.dot(height_tf_i, np.reshape(m, (yw - 1, yh))).flatten()
                  for m in y_pred_big]
    y_pred_big = np.asarray(y_pred_big)

    return (y_pred, y_pred_tall, y_pred_large, y_pred_big)


### Learn fusion params ######################################################

# If fusion parameters must be learn, we learn them on the prediction of
# X_train. 4 parameters are computed for each pixel of the image

coefs = [clf.steps[1][1].coef_ for clf in clfs]
coefs = np.asarray(coefs).T
if multi_scale:
    y_pred, y_pred_tall, y_pred_large, y_pred_big = \
            _split_multi_scale(y_pred, y_shape)

    # yc, yc_tall, yc_large, yc_big = _split_multi_scale(coefs, y_shape)

    """
    if learn_fusion_params:

        t_preds = []
        for clf in clfs:
            t_preds.append(clf.predict(X_train))
        t_preds = np.asarray(t_preds).T
        t_pred, t_pred_tall, t_pred_large, t_pred_big = \
            _split_multi_scale(t_preds, y_shape)

        yw, yh = y_shape
        y_train_original = y_train[:yw * yh]

        fusions = []
        from sklearn.linear_model import LinearRegression
        for i in range(yw * yh):
            tX = np.column_stack((t_pred[:, i], t_pred_tall[:, i],
                t_pred_large[:, i], t_pred_big[:, i]))
            f = LinearRegression()
            f.fit(tX, y_train[:, i])
            fusions.append(f.coef_)

        fusions = np.array(fusions)
        fusions = (fusions.T / np.sum(fusions, axis=1)).T

        y_pred = np.array([y_pred.T, y_pred_tall.T, y_pred_large.T,
                y_pred_big.T])
        y_pred = np.sum(y_pred.T * fusions, axis=2)
        y_coef = np.array([yc.T, yc_tall.T, yc_large.T, yc_big.T])
        y_coef = np.sum(y_coef.T * fusions, axis=2)

    else:
        y_pred = (.25 * y_pred + .25 * y_pred_tall + .25 * y_pred_large
            + .25 * y_pred_big)
        y_coef = (.25 * yc + .25 * yc_tall + .25 * yc_large + .25 * yc_big)
        """
else:
    y_coef = coefs

'''
y_coef = y_coef.T
y_coef_ = []
for clf, c in zip(clfs, y_coef):
    y_coef_.append(clf.steps[0][1].inverse_transform(c))
y_coef = np.vstack(y_coef_)
'''
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score

all_black = np.zeros_like(y_test)
all_white = np.ones_like(y_test)

print "Scores"
print "------"
print "  - Percentage: %f (%f-%f)" % (
        accuracy_score(y_test, y_pred > threshold),
        accuracy_score(y_test, all_black),
        accuracy_score(y_test, all_white)
)
print "  - Precision: %f (%f-%f)" % (
        precision_score(y_test, y_pred > threshold, pos_label=None),
        precision_score(y_test, all_black, pos_label=None),
        precision_score(y_test, all_white, pos_label=None)
)
print "  - Recall: %f (%f-%f)" % (
        recall_score(y_test, y_pred > threshold, pos_label=None),
        recall_score(y_test, all_black, pos_label=None),
        recall_score(y_test, all_white, pos_label=None)
)
print "  - F1-score: %f (%f-%f)" % (
        f1_score(y_test, y_pred > threshold, pos_label=None),
        f1_score(y_test, all_black, pos_label=None),
        f1_score(y_test, all_white, pos_label=None)
)

# F1 score of coefs
'''
coef_scores = np.zeros(100)
for p in range(100):
    # coef_scores[p] = f1_score(y_test[:, p], y_pred[:, p] > threshold,
    #         pos_label=None)
    coef_scores[p] = accuracy_score(y_test[:, p], y_pred[:, p] > threshold)
    # coef_scores[p] = np.log(np.abs(y_test[:, p] - y_pred[:, p]) + 1)
'''

"""
Show brains !
"""

if generate_image:
    print 'generate_image: Not implemented'

if generate_video:
    from matplotlib import animation
    fig = plt.figure()
    sp1 = plt.subplot(131)
    sp1.axis('off')
    plt.title('Stimulus')
    sp2 = plt.subplot(132)
    sp2.axis('off')
    plt.title('Reconstruction')
    sp3 = plt.subplot(133)
    sp3.axis('off')
    plt.title('Thresholded')
    ims = []
    for i, t in enumerate(y_pred):
        ims.append((
            sp1.imshow(np.reshape(y_test[i], (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest'),
            sp2.imshow(np.reshape(t, (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest'),
            sp3.imshow(np.reshape(t > threshold, (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest')))

    im_ani = animation.ArtistAnimation(fig, ims, interval=1000)
    im_ani.save(generate_video)


def fig2data(fig):
    """ Convert a Matplotlib figure to a 3D numpy array with RGB channel
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGB buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf


if generate_gif:
    ims = []
    for i, t in enumerate(y_pred[:50]):
        fig = plt.figure()
        sp1 = plt.subplot(131)
        sp1.axis('off')
        plt.title('Stimulus')
        sp2 = plt.subplot(132)
        sp2.axis('off')
        plt.title('Reconstruction')
        sp3 = plt.subplot(133)
        sp3.axis('off')
        plt.title('Thresholded')
        sp1.imshow(np.reshape(y_test[i], (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest')
        sp2.imshow(np.reshape(t, (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest')
        sp3.imshow(np.reshape(t > threshold, (10, 10)), cmap=plt.cm.gray,
            interpolation='nearest')
        ims.append(fig2data(fig))
    from nilearn.external.visvis.images2gif import writeGif
    writeGif(generate_gif, np.asarray(ims), duration=0.5, repeat=True)

if pynax:
    from pynax.core import Mark
    from pynax.view import MatshowView, ImshowView
    import pylab as pl
    from nipy.labs import viz
    # Get all regressors scores
    coef_scores = np.reshape(coef_scores, (10, 10))
    # Unmask the coefs
    coefs = masker.inverse_transform(y_coef).get_data()
    coefs = np.rollaxis(coefs, 3, 0)
    coefs = np.reshape(coefs, (10, 10, 64, 64, 30))
    coefs = np.ma.masked_equal(coefs, 0.)
    bg = masker.inverse_transform(X_train.mean(axis=0) + 5.).get_data()

    def b(a, b, c, d, v=0.01):
        return [a + v, b + v, c - 2 * v, d - 2 * v]

    fig = pl.figure(figsize=(16, 8))
    ax_coef = fig.add_axes(b(0., 0., 0.25, 1.))
    ax_s1 = fig.add_axes(b(0.25, 0., 0.25, 1.))
    ax_f1 = fig.add_axes(b(.5, 0., 0.25, 1.))
    ax_t1 = fig.add_axes(b(.75, 0., 0.25, 1.))

    ax_coef.axis('off')
    ax_t1.axis('off')
    ax_f1.axis('off')
    ax_s1.axis('off')

    # Marks
    coef_x = Mark(0, {'color': 'r'})
    coef_y = Mark(0, {'color': 'r'})
    mx1 = Mark(20, {'color': 'b'})
    my1 = Mark(20, {'color': 'b'})
    mz1 = Mark(20, {'color': 'b'})

    display_options = {}
    display_options['interpolation'] = 'nearest'
    display_options['cmap'] = pl.cm.gray

    #dist_mat = - (np.mgrid[-4.5:5.5, -4.5:5.5] ** 2).sum(axis=0) ** .5
    dist_mat = np.arctan2(*np.mgrid[-4.5:5.5, -4.5:5.5]) + 5
    all_coefs = np.zeros(coefs.shape[2:])

    for x, y in np.ndindex(coefs.shape[:2]):
        all_coefs += (coefs[x, y] != 0) * dist_mat[x, y]

    all_coefs = np.ma.masked_equal(all_coefs, 0.)

    ac_display_options = {}
    ac_display_options['interpolation'] = 'nearest'
    ac_display_options['cmap'] = pl.cm.hsv
    ac_display_options['vmin'] = dist_mat.min()
    ac_display_options['vmax'] = dist_mat.max()

    vx1 = ImshowView(ax_s1, bg, [mx1, 'h', '-v'], display_options)
    vx1.add_layer(all_coefs, [mx1, 'h', '-v'], ac_display_options)
    vx1.add_hmark(my1)
    vx1.add_vmark(mz1)

    vy1 = ImshowView(ax_f1, bg, ['h', my1, '-v'], display_options)
    vy1.add_layer(all_coefs, ['h', my1, '-v'], ac_display_options)
    vy1.add_hmark(mx1)
    vy1.add_vmark(mz1)

    vz1 = ImshowView(ax_t1, bg, ['h', '-v', mz1], display_options)
    vz1.add_layer(all_coefs, ['h', '-v', mz1], ac_display_options)
    vz1.add_hmark(mx1)
    vz1.add_vmark(my1)

    coefs_display_options = {}
    coefs_display_options['interpolation'] = 'nearest'
    
    coefs_display_options['vmax'] = 1.
    coefs_display_options['vmin'] = 0.
    coefs_display_options['cmap'] = pl.cm.hot
    vcoefs = MatshowView(ax_coef, coef_scores, ['v', 'h'],
                         coefs_display_options)

    # coefs_display_options['cmap'] = pl.cm.hsv
    # vcoefs = MatshowView(ax_coef, dist_mat, ['v', 'h'],
    #                      coefs_display_options)

    vx1.draw()
    vy1.draw()
    vz1.draw()
    vcoefs.draw()
    vcoefs.colorbar()

    pl.show()
