import numpy as np
import nibabel as nib
from time import time
from dipy.data import get_data, get_sphere
from dipy.reconst.dsi import DiffusionSpectrumModel, project_hemisph_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dsi import half_to_full_qspace
from dipy.reconst.gqi import GeneralizedQSamplingModel

if __name__ == '__main__':

    fname = '/home/eleftherios/Data/trento_processed/subj_01/101_32/raw'
    img = nib.load(fname + 'bet.nii.gz')
    data = img.get_data()
    affine = img.get_affine()
    bvals = np.loadtxt(fname + '.bval')
    bvecs = np.loadtxt(fname + '.bvec').T

    gtab = gradient_table(bvals, bvecs)
    sphere = get_sphere('symmetric724')

    """
    w=10
    data_roi = data[:, :, 32]
    #data_roi = data[48-w:48+w, 48-w:48+w, 32] #27-w:27+w,:]
    data_roi = data_roi[:,:,None,:]
    data = data_roi.astype('f8')
    """

    S0 = data[..., 0]
    mask = S0 > 50

    gq = GeneralizedQSamplingModel(gtab)
    gq.direction_finder.config(sphere=sphere, 
                                min_separation_angle=25, 
                                relative_peak_threshold=.35)
    gqfit = gq.fit(data, mask)
    gfa = gqfit.gfa
    peak_values = gqfit.peak_values
    peak_indices = gqfit.peak_indices
    qa = gqfit.qa
    #ODF = gqfit.odf(sphere)


    """
    data, gtab = half_to_full_qspace(data, gtab)
    ds = DiffusionSpectrumModel(gtab)
    ds.direction_finder.config(sphere=sphere, 
                                min_separation_angle=25, 
                                relative_peak_threshold=.35)

    dsfit = ds.fit(data, mask)
    #directions = dsfit.directions
    gfa = dsfit.gfa
    peak_values = dsfit.peak_values
    peak_indices = dsfit.peak_indices
    #ODF = dsfit.odf(sphere)
    #PDF = dsfit.pdf()

    #from dipy.viz._show_odfs import show_odfs
    #show_odfs(ODF, (sphere.vertices, sphere.faces))
    """

