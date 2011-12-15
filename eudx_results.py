import os
import numpy as np
import nibabel as nib
from dipy.reconst.dti import Tensor
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.tracking.propagation import EuDX
from dipy.io.dpy import Dpy
from dipy.external.fsl import create_displacements, warp_displacements, warp_displacements_tracks
from dipy.viz import fvtk
from dipy.external.fsl import flirt2aff

def transform_tracks(tracks,affine):
        return [(np.dot(affine[:3,:3],t.T).T + affine[:3,3]) for t in tracks]

no_seeds=10**6
visualize = False
dirname = "data/"

for root, dirs, files in os.walk(dirname):
    if root.endswith('101_32'):
        
        base_dir = root+'/'
        filename = 'raw'
        base_filename = base_dir + filename
        nii_filename = base_filename + 'bet.nii.gz'
        bvec_filename = base_filename + '.bvec'
        bval_filename = base_filename + '.bval'
        flirt_mat = base_dir + 'DTI/flirt.mat'    
        fa_filename = base_dir + 'DTI/fa.nii.gz'
        fsl_ref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'

        print bvec_filename
        
        img = nib.load(nii_filename)
        data = img.get_data()
        affine = img.get_affine()
        bvals = np.loadtxt(bval_filename)
        gradients = np.loadtxt(bvec_filename).T # this is the unitary direction of the gradient
        
        tensors = Tensor(data, bvals, gradients, thresh=50)
        FA = tensors.fa()
        euler = EuDX(a=FA, ind=tensors.ind(), seeds=no_seeds, a_low=.2)
        tensor_tracks = [track for track in euler]

        dpy_filename = base_dir + 'DTI/res_tracks_dti.dpy'
        dpw = Dpy(dpy_filename, 'w')
        dpw.write_tracks(tensor_tracks)
        dpw.close()
        
        #gqs=GeneralizedQSampling(data,bvals,gradients)
        #euler=EuDX(a=gqs.qa(),ind=gqs.ind(),seeds=no_seeds,a_low=.0239)
        #gqs_tracks = [track for track in euler]
        
        #dpy_filename = base_dir + 'DTI/res_tracks_gqi.dpy'
        #dpw = Dpy(dpy_filename, 'w')
        #dpw.write_tracks(gqs_tracks)
        #dpw.close()
        
        img_fa =nib.load(fa_filename)
        img_ref =nib.load(fsl_ref)
        mat=flirt2aff(np.loadtxt(flirt_mat),img_fa,img_ref)
        del img_fa
        del img_ref       
        
        tensor_linear = transform_tracks(tensor_tracks,mat)
        #gqi_linear = transform_tracks(gqs_tracks,mat)        
    
        if visualize:
            renderer = fvtk.ren()
            fvtk.add(renderer, fvtk.line(tensor_tracks, fvtk.red, opacity=1.0))
            fvtk.show(renderer)
        
        #save tracks_warped_linear
        dpy_filename = base_dir + 'DTI/tensor_linear.dpy'
        print dpy_filename
        dpr_linear = Dpy(dpy_filename, 'w')
        dpr_linear.write_tracks(tensor_linear)
        dpr_linear.close()        
        
        #save tracks_warped_linear
        #dpy_filename = base_dir + 'gqi_linear.dpy'
        #print dpy_filename
        #dpr_linear = Dpy(dpy_filename, 'w')
        #dpr_linear.write_tracks(gqi_linear)
        #dpr_linear.close()
        
        break
       

