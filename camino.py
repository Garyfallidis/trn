"""
Nipype camino tractography algorithm comparison
"""

import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.camino as camino
import nipype.interfaces.fsl as fsl
import nipype.interfaces.camino2trackvis as cam2trk
import nipype.algorithms.misc as misc
import os                                    # system functions

#os.chdir('/home/jdg45/git_repos/DWI-nipype-interfaces')
#import jg_nipype_interfaces
import jg_camino_interfaces
#wbd = '/work/imaging5/DTI/CBU_DTI/nipype_camino_tractography_algorithm_comparison/64D_2A'  
wbd='/home/eg309/Data/orbital_phantoms/dwi_dir'
os.chdir(wbd)

def get_vox_dims(volume):
    import nibabel as nb
    if isinstance(volume, list):
        volume = volume[0]
    nii = nb.load(volume)
    hdr = nii.get_header()
    voxdims = hdr.get_zooms()
    return [float(voxdims[0]), float(voxdims[1]), float(voxdims[2])]

def get_data_dims(volume):
    import nibabel as nb
    if isinstance(volume, list):
        volume = volume[0]
    nii = nb.load(volume)
    hdr = nii.get_header()
    datadims = hdr.get_data_shape()
    return [int(datadims[0]), int(datadims[1]), int(datadims[2])]

def get_affine(volume):
    import nibabel as nb
    nii = nb.load(volume)
    return nii.get_affine()

subject_list = ['subject1']#, 'CBU070414', 'CBU070415', 'CBU070416', 'CBU070417', 'CBU070421', 'CBU070422']#['subj1'] * JG MOD
fsl.FSLCommand.set_default_output_type('NIFTI')
info = dict(dwi=[['subject_id', 'data']],
            mask=[['subject_id', 'mask']],
            bvecs=[['subject_id','bvecs']],
            bvals=[['subject_id','bvals']])

# Node: infosource
infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),name="infosource")
infosource.iterables = ('subject_id', subject_list)

# Node: datasource
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],outfields=info.keys()), name = 'datasource')
datasource.inputs.template = "%s/%s"

# This needs to point to the fdt folder you can find after extracting
# http://www.fmrib.ox.ac.uk/fslcourse/fsl_course_data2.tar.gz
datasource.inputs.base_directory = wbd #os.path.abspath('fsl_course_data/fdt/')

datasource.inputs.field_template = dict(dwi='%s/%s.nii.gz', mask='%s/%s.nii.gz') #.gz') JG MOD
datasource.inputs.template_args = info

# Node: inputnode
inputnode = pe.Node(interface=util.IdentityInterface(fields=["dwi", "mask","bvecs", "bvals"]), name="inputnode")

"""
Setup for Diffusion Tensor Computation
--------------------------------------
In this section we create the nodes necessary for diffusion analysis.
First, the diffusion image is converted to voxel order.
"""

# Node: image2voxel
image2voxel = pe.Node(interface=camino.Image2Voxel(), name="image2voxel")

# Node: fsl2scheme
fsl2scheme = pe.Node(interface=camino.FSL2Scheme(), name="fsl2scheme")
fsl2scheme.inputs.usegradmod = True

"""
An FSL BET node creates a brain mask is generated from the diffusion image for seeding the PICo tractography.
"""

"""!!!!!!
bet = pe.Node(interface=fsl.BET(), name="bet", frac=0.2) #
bet.inputs.mask = True
"""

# Node: modelfit_oneten
modelfit_oneten = pe.Node(interface=camino.ModelFit(),name="modelfit_oneten")
#modelfit_oneten.inputs.args = ' -inversion 2 '  # inversion 2 = nonlinear fit of 1 tensor
modelfit_oneten.inputs.model = 'nldt_pos' # nonlinear fit, positive definite


# Node: modelfit_twoten
modelfit_twoten = pe.Node(interface=camino.ModelFit(),name="modelfit_twoten")
modelfit_twoten.inputs.model = 'cylcyl nldt_pos'
#modelfit_twoten.inputs.args = ' -inversion 12 '# inversion 12 = nonlinear fit of 2 tensors

"""
EXAMPLES

Generate a LUT for the Bingham distribution, over the range L1 / L3 = 1 to 15. This is a sensible default for single tensor PICo. We use the default sample size (2000) and inversion (1).
 
dtlutgen -lrange 1 15 -step 0.2 -snr 16 -schemefile A.scheme > Ascheme_bingham_snr16_inv1.Bdouble
 
"""

# Node: dtlutgen_oneten
dtlutgen_oneten = pe.Node(interface=camino.DTLUTGen(),name="dtlutgen_oneten")
dtlutgen_oneten.inputs.lrange=[1, 15]
#dtlutgen_oneten_a.inputs.step=0.2
dtlutgen_oneten.inputs.snr=60
 
"""
Two tensor version:
	
Generate a two-Bingham LUT over a range of anisotropy, from FA = 0.3 to FA = 0.9. This is a sensible range for SNR 16, tensors with FA less than about 0.3 are not resolved reliably.
 
dtlutgen -frange 0.3 0.9 -step 0.05 -samples 2000 -snr 16 -schemefile A.scheme -inversion 22 -cross 90 > twoBingham_Ascheme_snr16_inv22_90.Bdouble
 
"""

# Node: dtlutgen_twoten
dtlutgen_twoten = pe.Node(interface=camino.DTLUTGen(),name="dtlutgen_twoten")
dtlutgen_twoten.inputs.frange=[0.3,0.9]
#dtlutgen_twoten.inputs.step=0.05
dtlutgen_twoten.inputs.samples=2000
dtlutgen_twoten.inputs.snr=60#!!!!!16
dtlutgen_twoten.inputs.inversion=22
dtlutgen_twoten.inputs.args = ' -cross 90 '

"""
PICo tractography requires an estimate of the fibre direction and a model of its
uncertainty in each voxel; this is produced using the following node.
"""

# Node: picopdfs_oneten
picopdfs_oneten  = pe.Node(interface=camino.PicoPDFs(), name="picopdfs_oneten")
picopdfs_oneten.inputs.inputmodel='dt'

# Node: picopdfs_twoten
#picopdfs_twoten  = pe.Node(camino.PicoPDFs(), name="picopdfs_twoten") # JG: am using my modified version of this interface for the two-tensor pico for now (allows to specify 2 luts )
picopdfs_twoten = pe.Node(interface=jg_camino_interfaces.PicoPDFs2Fib(),name='picopdfs_twoten')
picopdfs_twoten.inputs.inputmodel='multitensor'

# Node: track_picoprob_twoten
track_picoprob_twoten  = pe.Node(interface=camino.Track(),name="track_picoprob_twoten")
track_picoprob_twoten.inputs.inputmodel = 'pico'
track_picoprob_twoten.inputs.args = ' -iterations 3 -numpds 2 '

#track_picoprob_threeten_Bing_farng0_3_0_9_step0_05_samp2000  = pe.Node(interface=camino.Track(),name="track_picoprob_threeten_Bing_farng0_3_0_9_step0_05_samp2000",modelfit = 'pico', numpds=3)
#track_ballstick = pe.Node(interface=camino.TrackBallStick(), name='track_ballstick', iterations=1)

# Node: cam2trk_pico_twoten
cam2trk_pico_twoten = pe.Node(interface=cam2trk.Camino2Trackvis(), name="cam2trk_pico_twoten") 
cam2trk_pico_twoten.inputs.min_length = 30
cam2trk_pico_twoten.inputs.voxel_order = 'LAS'

tractography = pe.Workflow(name='tractography')
#tractography.connect([(inputnode, bet,[("dwi","in_file")])])

# image2voxel inputs:
tractography.connect([(inputnode, image2voxel, [("dwi", "in_file")])])

# fsl2scheme inputs:
tractography.connect([(inputnode, fsl2scheme, [("bvecs", "bvec_file"),
                                               ("bvals", "bval_file")])])

# modelfit_twoten inputs:
tractography.connect([(image2voxel, modelfit_twoten,[("voxel_order", "in_file")])]) 
tractography.connect([(fsl2scheme, modelfit_twoten,[("scheme", "scheme_file")])]) 
tractography.connect([(inputnode, modelfit_twoten,[("mask","bgmask")])])

# modelfit_oneten inputs:
tractography.connect([(inputnode, modelfit_oneten,[("mask","bgmask")])])
tractography.connect([(image2voxel, modelfit_oneten,[("voxel_order", "in_file")])]) 
tractography.connect([(fsl2scheme, modelfit_oneten,[("scheme", "scheme_file")])]) 

# dtlutgen_oneten inputs:
tractography.connect([(fsl2scheme,dtlutgen_oneten,[("scheme","scheme_file")])])

#dtlutgen_twoten inputs:
tractography.connect([(fsl2scheme,dtlutgen_twoten,[("scheme","scheme_file")])])

# picopdfs_oneten inputs:
tractography.connect([(dtlutgen_oneten,picopdfs_oneten,[("dtLUT","luts")])])
tractography.connect([(modelfit_oneten,picopdfs_oneten,[("fitted_data","in_file")])])

# picopdfs_twoten inputs:
tractography.connect([(dtlutgen_oneten,picopdfs_twoten,[("dtLUT","lut1")])]) # JG: 'lut1' and 'lut2' are what I added in my modified version of swederik's interface for picopdfs
tractography.connect([(dtlutgen_twoten,picopdfs_twoten,[("dtLUT","lut2")])]) 
tractography.connect([(modelfit_twoten,picopdfs_twoten,[("fitted_data","in_file")])])

# track_picoprob_twoten inputs:
tractography.connect([(picopdfs_twoten, track_picoprob_twoten,[("pdfs","in_file")])])
tractography.connect([(inputnode, track_picoprob_twoten,[("mask","seed_file")])])

# cam2trk_pico_twoten inputs:
tractography.connect([(track_picoprob_twoten, cam2trk_pico_twoten, [('tracked','in_file')])]) 
tractography.connect([(inputnode, cam2trk_pico_twoten,[(('dwi', get_vox_dims), 'voxel_dims'),
                                                (('dwi', get_data_dims), 'data_dims')])])




workflow = pe.Workflow(name="workflow")
workflow.base_dir=wbd  #os.path.abspath('camino_dti_tutorial') 
workflow.connect([(infosource,datasource,[('subject_id', 'subject_id')]),
                  (datasource,tractography,[('dwi','inputnode.dwi'),
                                            ('bvals','inputnode.bvals'),
                                            ('bvecs','inputnode.bvecs'),
                                            ('mask', 'inputnode.mask'),
                                           ])
                 ])

workflow.run()
#workflow.write_graph()




























"""
Three models of the PICo fibre-orientation PDF are supported. The Watson PDF has circular contours on the sphere and is a good model of noise-based uncertainty for cylindrically symmetric diffusion tensors. The Bingham PDF has elliptical contours on the sphere, and is more suitable for non cylindrically-symmetric tensors. The Angular Central Gaussian PDF also has elliptical contours, and IS SUITABLE FOR VERY NOISY DATA with low concentration. We recommend the Bingham model for general use. This is the default if no PDF is specified. 
"""




# BayesDirac
"""
BOOTSTRAP TRACKING

 Bootstrap tracking requires the raw DWI data and a reconstruction algorithm. The principal direction or directions in each voxel are determined independently for each bootstrap sample of the data.
 
Currently, diffusion tensor is the only model supported. Both repetition and wild bootstrapping are available. Please see datasynth(1) for more information on the bootstrap techniques.
 
Using the repetition bootstrap, one and two-tensor models may be fitted to the bootstrap data. The reconstruction parameters [see modelfit(1) should be passed to track along with the other
  parameters. For example, given 4 repeats of a scan SubjectA_[1,2,3,4].Bfloat, (in voxel order),  we can track using repetition bootstrapping and DTI:
 

  track  -inputmodel bootstrap -bsdatafiles SubjectA_1.Bfloat SubjectA_2.Bfloat  SubjectA_3.Bfloat 
   SubjectA_4.Bfloat -schemefile A.scheme -inversion 1 -bgmask A_BrainMask.hdr  
  -iterations 1000 -seedfile ROI.hdr -bsmodel dt > A_bs.Bfloat 

To use a two-tensor model, we must pass the voxel classification from voxelclassify.
 

  track  -inputmodel bootstrap -bsdatafiles SubjectA_1.Bfloat SubjectA_2.Bfloat SubjectA_3.Bfloat 
   SubjectA_4.Bfloat -schemefile A.scheme -inversion 21 -voxclassmap A_vc.Bint 
  -iterations 1000 -seedfile ROI.hdr -bsmodel multitensor > A_bs.Bfloat 

The voxel classifications are fixed; they are not re-determined dynamically. 

Note that you may pass either -voxclassmap or -bgmask, but not both. If you are using a voxel classification map, the brain / background mask should be passed to voxelclassify. You may always restrict tracking to any volume of the brain by using the -anisfile and -anisthresh options.
 
Wild bootstrapping requires a single DWI data set. Multi-tensor reconstruction is not supported. The scheme file is required.
 
track -inputfile SubjectA_1.Bfloat -inputmodel bootstrap -schemefile A.scheme -bgmask A_BrainMask.hdr
   -iterations 1000 -seedfile ROI.hdr -wildbsmodel dt > A_wildbs.Bfloat 

"""



  

"""
BAYESIAN TRACKING WITH DIRAC PRIORS

 This method was presented by Friman et al, IEEE-TMI 25:965-978 (2006). In each voxel, we compute the likelihood of the fibre orientation being the axis X, given the data and the model of the data. We wish to sample from
 

   P(X | data) = P(data | X) P(X) / P(data) 

in each voxel. We first fit a model to the data (like the diffusion tensor), the model yields m_i, the predicted measurement i given a principal direction X. The observed data y_i is a noisy estimate of m_i. The noise is modelled on the log data as
 

  ln(y_i) = ln(m_i) + epsilon, 

where epsilon is Gaussian distributed as N(0, sigma^2 / m_i^2), where sigma^2 is the variance of the noise in the complex MR data. Therefore,
 

  P(data | X) = P(y_1 | m_1)P(y_2 | m_2)...P(y_N | m_N)  

where there are N measurements. The prior distribution for all parameters except X is a dirac delta function, so P(data) is the integral of P(data | X) over the sphere. In the case of the diffusion tensor, for example, the priors of S(0) and the tensor eigenvalues L1, and L2 = L3 are fixed around the maximum-likelihood estimate (MLE). The function P(data | X) is then evaluated by setting the tensor principal direction to X and computing the likelihood of the observed data.
 
The prior on X, P(X), may be set to favor low tract curvature. With the -curvepriork option, the user may set a Watson concentration parameter k. Given a previous tract orientation T, P(X) = W(X, T, k), where k >= 0. The default is k = 0, which is a uniform distribution. Higher values of k increase the sharpness of P(X) around its peak axis T. Suggested values of k are in the range of 0 to 5. You may also use -curvepriorg to implement Friman's curvature prior. Note that a curvature prior does not directly impose a curvature threshold, which may be imposed separately.
 
An external prior may also be added, in the form of a PICo PDF O(X) defined for each voxel in the image. The full prior is then W(X, T, k)O(X). Pass a PICo image with -extpriorfile.
 
Example: 

track -inputfile SubjectA_1.Bfloat -inputmodel bayesdirac -schemefile A.scheme -bgmask A_BrainMask.hdr
   -iterations 1000 -seedfile ROI.hdr > A_bd.Bfloat 


"""



