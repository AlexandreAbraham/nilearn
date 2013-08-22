import os
import shutil
from os.path import join
import nibabel
from nipype.interfaces.fsl import MELODIC
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Memory
import tempfile
import glob
import re


def _find_fsl_exe(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    # FSL executables are most of the time prefixed with FSL version. We search
    # the path for the latest version.

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
        # Program is not found, we make a deeper search
        programs = []
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            programs += glob.glob(join(path, 'fsl*-%s' % program))
        pattern = 'fsl(.*)-%s' % program
        max_version = None
        cmd = None
        for p in programs:
            if not is_exe(p):
                continue
            r = re.search(pattern, p)
            version = float(r.groups()[0])
            if max_version is None or version > max_version:
                cmd = p
                max_version = version
        if max_version is None:
            raise ValueError('FSL program %d not found' % program)
        print 'Found %s version %f' % (program, max_version)
        return cmd
    return None


class MelodicICA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=20, smoothing_fwhm=None, mask=None,
                 whiten=True, approach='tica',
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
                 ):
        self.n_components = n_components
        self.whiten = whiten
        self.approach = approach
        self.mask = mask

    def fit(self, X, y=None):
        melodic = MELODIC()
        melodic.inputs.approach = approach
        melodic.inputs.num_ICs = self.n_components

        # Temporary directory
        tmp = tempfile.mkdtemp()
        print('Temp dir: %s' % tmp)

        in_files = []
        # For first file: we save or copy it into the temp directory because
        # this is where melodic stores ica results.
        if isinstance(X[0], basestring):
            shutil.copy(X[0], join(tmp, 'ica.nii'))
        else:
            nibabel.save(X[0], join(tmp, 'ica.nii'))
        in_files.append(join(tmp, 'ica.nii'))

        for i, x in enumerate(X[1:]):
            if isinstance(x, basestring):
                in_files.append(x)
            else:
                in_file = join(tmp, 'in_file%d.nii' % i)
                nibabel.save(x, in_file)
                in_files.append(in_file)
        melodic.inputs.in_files = in_files

        # If a mask is given, we use it, otherwise we use bet.
        if self.mask is not None:
            if not isinstance(self.mask, basestring):
                mask_file = join(tmp, 'mask.nii')
                nibabel.save(self.mask, mask_file)
            else:
                mask_file = self.mask
            melodic.inputs.no_bet = True
            melodic.inputs.mask = mask_file

        melodic.inputs.out_white = True
        melodic.inputs.out_unmix = True
        melodic.inputs.out_dir = tmp
        melodic._cmd = _find_fsl_exe(melodic._cmd)
        melodic.run()

        # Load results
        self.maps_img_ = nibabel.load(
                join(tmp, 'ica.ica', 'melodic_IC.nii.gz'))
        # Load maps in memory
        self.maps_img_.get_data()

        # XXX Identify and load unmixing and whitening matrices
        shutil.rmtree(tmp)
        return self

    def transform(self, X, y=None):
        """Apply un-mixing matrix "W" to X to recover the sources

        S = X * W.T
        """
        # X = array2d(X)
        # return np.dot(X, self.components_.T)
        pass

    def get_mixing_matrix(self):
        """Compute the mixing matrix
        """
        # return linalg.pinv(self.components_)
        pass
