import numpy as np
import nibabel as nib
from time import time
from dipy.data import get_data, get_sphere
from dipy.reconst.dsi import DiffusionSpectrumModel, project_hemisph_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dsi import half_to_full_qspace


if __name__ == '__main__':

    path = '/home/eg309/Data/PROC_MR10032/subj_03/101_32/1312211075232351192010121313490254679236085ep2dadvdiffDSI10125x25x25STs004a001'    
   
    img = nib.load(path+'.nii')
    data = img.get_data()
    affine = img.get_affine()
    bvals = np.loadtxt(path+'.bval')
    bvecs = np.loadtxt(path+'.bvec').T
    
    gtab = gradient_table(bvals, bvecs)


    sphere = get_sphere('symmetric724')

    #data_roi = data[48-20:48+20,48-20:48+20,35,:]
    #data_roi = data_roi[:,:,None,:]     
    #data_roi = data_roi.astype('f8')
    data, gtab = half_to_full_qspace(data, gtab)
        
    ds = DiffusionSpectrumModel(gtab)
    ds.direction_finder.config(sphere=sphere, 
                                min_separation_angle=25, 
                                relative_peak_threshold=.35)
    dsfit = ds.fit(data)
    ODF = dsfit.odf(sphere)
    #PDF = dsfit.pdf()

    #from dipy.viz._show_odfs import show_odfs
    #show_odfs(ODF, (sphere.vertices, sphere.faces))


