import os
from glob import glob
import numpy as np
from numpy.linalg import inv
import dipy as dp
import nibabel as ni
from nibabel import trackvis as tv
import resources
from time import time
from subprocess import Popen,PIPE
from dipy.core.track_propagation import EuDX
from dipy.io.dpy import Dpy 
from dipy.core.track_performance import track_roi_intersection_check
from dipy.core.track_performance import point_track_sq_distance_check, approx_polygon_track
from dipy.core.track_metrics import length
from dipy.io.fsl import flirt2aff_files,warp_displacements
from scipy.ndimage import map_coordinates as mc
from scipy.ndimage.measurements import center_of_mass 
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, distance_transform_cdt
from scipy.ndimage import affine_transform
#dname='/home/eg309/Data/TEST_MR10032/'#only subj_10

dname='/home/eg309/Data/PROC_MR10032/'
#dname='/home/eg01/Data/dipy_data/PROC_MR10032/'



fnames_64=['subj_01/64/1312211075232351192010091419353265314888418CBUDTI64InLea2x2x2s006a001',
           'subj_02/64/1312211075232351192010091419555679522912256CBUDTI64InLea2x2x2s003a001',
           'subj_03/64/1312211075232351192010092217391520794510817CBUDTI64InLea2x2x2s005a001',
           'subj_04/64/1312211075232351192010092218281312098795926CBUDTI64InLea2x2x2s005a001',
           'subj_05/64/1312211075232351192010092219022010941552754CBUDTI64InLea2x2x2s004a001',
           'subj_06/64/13122110752323511930000010092916083910900000330CBUDTI64InLea2x2x2s005a001',
           'subj_07/64/13122110752323511920100929100151894659315CBUDTI64InLea2x2x2s006a001',
           'subj_08/64/131221107523235119201009291022305957983305CBUDTI64InLea2x2x2s004a001',
           'subj_09/64/1312211075232351192010092911245187367496695CBUDTI64InLea2x2x2s004a001',
           'subj_10/64/1312211075232351192010092911593893074453675CBUDTI64InLea2x2x2s004a001',
           'subj_11/64/1312211075232351192010092913182323664700207CBUDTI64InLea2x2x2s006a001',
           'subj_12/64/1312211075232351192010092915530245676665668CBUDTI64InLea2x2x2s006a001']

fnames_118=['subj_01/118/1312211075232351192010091419172483436954733ep2dadvdiffDTI25x25x25STEAMs004a001',
            'subj_02/118/1312211075232351192010091420053056798634446ep2dadvdiffDTI25x25x25STEAMs004a001',
            'subj_03/118/1312211075232351192010092217101585124049480ep2dadvdiffDTI25x25x25STEAMs003a001',
            'subj_04/118/1312211075232351192010092218133653027362870ep2dadvdiffDTI25x25x25STEAMs004a001',
            'subj_05/118/1312211075232351192010092218475052554919698ep2dadvdiffDTI25x25x25STEAMs003a001',
            'subj_06/118/13122110752323511930000010092916083910900000396ep2dadvdiffDTI25x25x25STEAMs006a001',
            'subj_07/118/1312211075232351192010092909323894392897978ep2dadvdiffDTI25x25x25STEAMs004a001',
            'subj_08/118/1312211075232351192010092910321120007005495ep2dadvdiffDTI25x25x25STEAMs005a001',
            'subj_09/118/1312211075232351192010092911343136998518885ep2dadvdiffDTI25x25x25STEAMs005a001',
            'subj_10/118/1312211075232351192010092912234822181304146ep2dadvdiffDTI25x25x25STEAMs006a001',
            'subj_11/118/1312211075232351192010092913034854325067151ep2dadvdiffDTI25x25x25STEAMs005a001',
            'subj_12/118/1312211075232351192010092915382717771632614ep2dadvdiffDTI25x25x25STEAMs005a001']

fnames_101=['subj_01/101/1312211075232351192010091419011391228126452ep2dadvdiffDSI25x25x25b4000s003a001',
            'subj_02/101/1312211075232351192010091708112071055601107ep2dadvdiffDSI10125x25x25STs002a001',
            'subj_03/101/1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001',
            'subj_04/101/1312211075232351192010092217591167666934589ep2dadvdiffDSI10125x25x25STs003a001',
            'subj_05/101/1312211075232351192010092219115865299674944ep2dadvdiffDSI10125x25x25STs005a001',
            'subj_06/101/13122110752323511930000010092916083910900000227ep2dadvdiffDSI10125x25x25STs004a001',          
            'subj_07/101/1312211075232351192010092909471494688630968ep2dadvdiffDSI10125x25x25STs005a001',
            'subj_08/101/1312211075232351192010092910464296793738485ep2dadvdiffDSI10125x25x25STs006a001',
            'subj_09/101/1312211075232351192010092911101398147468348ep2dadvdiffDSI10125x25x25STs003a001',
            'subj_10/101/1312211075232351192010092912092080924175865ep2dadvdiffDSI10125x25x25STs005a001',
            'subj_11/101/1312211075232351192010092912491582105638870ep2dadvdiffDSI10125x25x25STs004a001',
            'subj_12/101/1312211075232351192010092915235687194704333ep2dadvdiffDSI10125x25x25STs004a001']

                          

def save_volumes_as_mosaic(fname,volume_list):
    import Image
    vols=[]
    for vol in volume_list:            
        vol=np.rollaxis(vol,2,1)
        sh=vol.shape
        arr=vol.reshape(sh[0],sh[1]*sh[2])
        arr=np.interp(arr,[arr.min(),arr.max()],[0,255])
        arr=arr.astype('ubyte')
        print 'arr.shape',arr.shape
        vols.append(arr)
    mosaic=np.concatenate(vols)
    Image.fromarray(mosaic).save(fname)

def pipe(cmd):    
    p = Popen(cmd, shell=True,stdout=PIPE,stderr=PIPE)
    sto=p.stdout.readlines()
    ste=p.stderr.readlines()
    print(sto)
    print(ste)

def extract_S0(in_nii,out_nii,options=' 0 1'):
    cmd='fslroi ' + in_nii + ' ' + out_nii + options
    print(cmd)
    pipe(cmd)    

def bet(in_nii,out_nii,options=' -F -f .2 -g 0'):
    cmd='bet '+in_nii+' '+ out_nii + options
    print(cmd)
    pipe(cmd)

def dcm2nii(dname,outdir,filt='*.dcm',options='-d n -g n -i n -o'):
    cmd='dcm2nii '+options +' ' + outdir +' ' + dname + '/' + filt
    print(cmd)
    pipe(cmd)

def flirt(in_nii, ref_nii,out_nii,transf_mat):
    cmd='flirt -in ' + in_nii + ' -ref ' + ref_nii + ' -out ' \
        + out_nii +' -dof 12 -omat ' + transf_mat
    print(cmd)
    pipe(cmd)
    
def flirt_apply_transform(in_nii, target_nii, out_nii, transf_mat):
    cmd='flirt -in ' + in_nii + ' -ref ' + target_nii + ' -out ' \
        + out_nii +' -init ' + transf_mat +' -applyxfm'
    print(cmd)
    pipe(cmd)
    
def invert_transform(transf_matr, inv_transf_mat):
    cmd='convert_xfm -omat '+inv_transf_matr+' -inverse ' + transf_mat
    print(cmd)
    pipe(cmd)

def dtifit(in_nii,out_nii,mask_nii,bvalsf,bvecsf):
    cmd='dtifit -k ' + in_nii +' -o ' + out_nii +' -m ' \
        + mask_nii +' -r '+ bvecsf +' -b '+ bvalsf
    print(cmd)
    pipe(cmd)

def fa_to_fmrib58(in_nii,affine_mat,nonlin_nii,out_nii):    
    cmd1='flirt -ref ${FSLDIR}/data/standard/FMRIB58_FA_1mm -in '+in_nii+' -omat ' + affine_mat
    cmd2='fnirt --in='+in_nii+' --aff='+affine_mat+' --cout='+nonlin_nii+' --config=FA_2_FMRIB58_1mm'
    cmd3='applywarp --ref=${FSLDIR}/data/standard/FMRIB58_FA_1mm --in='+in_nii+' --warp='+nonlin_nii+' --out='+out_nii
    print(cmd1)
    pipe(cmd1)
    print(cmd2)
    pipe(cmd2)
    print(cmd3)
    pipe(cmd3)

def apply_warp(in_nii,affine_mat,nonlin_nii,out_nii):
    cmd='applywarp --ref=${FSLDIR}/data/standard/FMRIB58_FA_1mm --in='+in_nii+' --warp='+nonlin_nii+' --out='+out_nii
    print(cmd)
    pipe(cmd)

def invwarp(in_nii,nonlin_nii,out_nii):
    #invwarp --ref=my_struct --warp=warps_into_MNI_space --out=warps_into_my_struct_space
    cmd='invwarp --ref='+in_nii+' --warp='+nonlin_nii+' --out='+out_nii
    print(cmd)
    pipe(cmd)

def displacements(in_nii,out_nii):
    cmd='fnirtfileutils --in='+in_nii+' --ref=${FSLDIR}/data/standard/FMRIB58_FA_1mm --out='+out_nii
    print(cmd)
    pipe(cmd)    
    
def displacements_with_aff(in_nii,out_nii):
    cmd='fnirtfileutils --in='+in_nii+' --ref=${FSLDIR}/data/standard/FMRIB58_FA_1mm --out='+out_nii + ' --withaff'
    print(cmd)
    pipe(cmd)    
    
    
def check_qa_norms():
    for fname in [fnames_64,fnames_101,fnames_118]:
        #print(fname)
        for subj in range(0,12):       
            f =dname+fname[subj]+'_bet_QA_normalization.txt'
            fobj=open(f,'r')
            print(subj,fobj.readlines())
            fobj.close()

def create_native_tracks(subj,fnames,data_type='qa'):
    #initial raw data
    fname =dname+fnames[subj]+'.nii.gz'
    #bvecs 
    fbvecs=dname+fnames[subj]+'.bvec'
    #bvals
    fbvals=dname+fnames[subj]+'.bval'
    #beted initial data 
    fbet  =dname+fnames[subj]+'_bet.nii.gz'
    #fa
    ffa   =dname+fnames[subj]+'_bet_FA.nii.gz'
    #qa
    fqa  =dname+fnames[subj]+'_bet_QA.nii.gz'
    #normalization factor for qa
    fqan =dname+fnames[subj]+'_bet_QA_normalization.txt'
    #qa warperd in fmrib58
    fqaw =dname+fnames[subj]+'_bet_QA_warp.nii.gz'
    #mask after bet was applied
    fmask =dname+fnames[subj]+'_bet_mask.nii.gz'
    #mean diffusivity
    fmd   =dname+fnames[subj]+'_bet_MD.nii.gz'
    #b0 volume
    fs0   =dname+fnames[subj]+'_bet_S0.nii.gz'
    #fa registerd in fmri58
    ffareg=dname+fnames[subj]+'_bet_FA_reg.nii.gz'    
    #affine transformation matrix (from flirt before nonlinear registration)
    faff  =dname+fnames[subj]+'_affine_transf.mat'
    #image after transforming(warping) (splines)
    fnon  =dname+fnames[subj]+'_nonlin_transf.nii.gz'#look at fa_to_fmrib58
    #nonlinear displacements
    fdis  =dname+fnames[subj]+'_nonlin_displacements.nii.gz'#look at fa_to_fmrib58
    #warped md 
    fmdw  =dname+fnames[subj]+'_bet_MD_warp.nii.gz'
    #warped s0
    fs0w  =dname+fnames[subj]+'_bet_S0_warp.nii.gz'        
    #warped fa tracks
    ffa_warp_dpy  =dname+fnames[subj]+'_FA_warp.dpy'
    #warped qa tracks
    #fqa_warp_dpy  =dname+fnames[subj]+'_QA_warp.dpy'
    fqa_warp_dpy  =dname+fnames[subj]+'_QA_warp2.dpy'
    #native fa tracks
    ffa_native_dpy =dname+fnames[subj]+'_FA_native.dpy'
    #native qa tracks
    fqa_native_dpy =dname+fnames[subj]+'_QA_native.dpy'
            
    #remove scalp
    #bet(fname,fbet)
        
    #read skull stripped image and get affine
    img=ni.load(fbet)
    data=img.get_data()
    affine=img.get_affine()
    bvals=np.loadtxt(fbvals)
    gradients=np.loadtxt(fbvecs).T

    #load mask
    mask_img=ni.load(fmask)
    mask=mask_img.get_data()
    mask_affine=mask_img.get_affine()

    print 'mask.shape',mask.shape
    #calculate Tensor
    if data_type=='fa':
        ten=dp.Tensor(data,bvals,gradients,thresh=50)        
        #uncomment if you want to save FA,MD,S0 unwarped/warped
        #imgFA=ni.Nifti1Image(ten.fa(),affine)
        #imgMD=ni.Nifti1Image(ten.md(),affine)
        #imgS0=ni.Nifti1Image(data[:,:,:,0].astype(np.uint16),affine)        
        #ni.save(imgFA,ffa)
        #ni.save(imgMD,fmd)
        #ni.save(imgS0,fs0)
        #fa_to_fmrib58(ffa,faff,fnon,ffareg)    
        #apply_warp(fmd,faff,fnon,fmdw)
        #apply_warp(fs0,faff,fnon,fs0w)
        
    #calculate GQI
    if data_type=='qa':
        gqs=dp.GeneralizedQSampling(data,bvals,gradients,mask=mask)        
        print('save normalization factor')        
        fobjn=open(fqan,'w')
        fobjn.write(str(gqs.glob_norm_param))
        fobjn.close()        
        print('save qa image')
        imgQA=ni.Nifti1Image(gqs.QA,affine)
        ni.save(imgQA,fqa)
        print('warp qa to fmrib58')
        apply_warp(fqa,faff,fnon,fqaw)
    
    if data_type=='fa':
        #eudx=EuDX(ten.FA,ten.IN,seed_list=img_pts,qa_thr=.2)
        eudx=EuDX(ten.FA,ten.IN,seed_no=5*10**6,qa_thr=.2)
        fnative_dpy=ffa_native_dpy        
        fwarp_dpy = ffa_warp_dpy
    if data_type=='qa':
        #eudx=EuDX(gqs.QA,gqs.IN,seed_list=img_pts,qa_thr=0.0239)
        eudx=EuDX(gqs.QA,gqs.IN,seed_no=5*10**6,qa_thr=0.0239)
        fnative_dpy=fqa_native_dpy
        fwarp_dpy = fqa_warp_dpy
       
    exi=iter(eudx)    
    print(fnative_dpy)   
    dpw=Dpy(fnative_dpy,'w')   
    #print(fwarp_dpy) 
    #dpww=Dpy(fwarp_dpy,'w')
    
    for (i,track) in enumerate(exi):       
        dpw.write_track(track.astype(np.float32))
    dpw.close()    

def create_all_native_tracks(data_type='fa'):
    t1=time()
    for subj in range(0,12):#starts from 0  
        print 'subj number ',subj, 'starting from 0'
        if data_type=='qa':
            create_native_tracks(subj,fnames_64,data_type=data_type)
            create_native_tracks(subj,fnames_101,data_type=data_type)
            create_native_tracks(subj,fnames_118,data_type=data_type)
        if data_type=='fa':
            create_native_tracks(subj,fnames_64,data_type=data_type)
            create_native_tracks(subj,fnames_101,data_type=data_type)
            create_native_tracks(subj,fnames_118,data_type=data_type)
    print('Done in %f secs' % (time()-t1))
    
def tracks_to_fmrib58(subj,fnames,data_type='fa'):    
    #affine transformation matrix (from flirt before nonlinear registration)
    faff  =dname+fnames[subj]+'_affine_transf.mat'
    #nonlinear displacements
    fdis  =dname+fnames[subj]+'_nonlin_displacements.nii.gz'#look at fa_to_fmrib58
    
    if data_type=='qa':
        #fqa_warp_dpy 
        fqa_warp  =dname+fnames[subj]+'_QA_warp.dpy'
        fwarp=fqa_warp
    if data_type=='fa':
        ffa_warp  =dname+fnames[subj]+'_FA_warp.dpy'
        fwarp=ffa_warp
    #fa
    ffa   =dname+fnames[subj]+'_bet_FA.nii.gz'
    ref_fname = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
            
    print faff
    print ffa
    print fdis
    
    im2im = flirt2aff_files(faff, ffa, ref_fname)
    dimg=ni.load(fdis)
    daff=dimg.get_affine()
    ddata=dimg.get_data()    
    
    di=ddata[:,:,:,0]#copy i
    dj=ddata[:,:,:,1]#copy j
    dk=ddata[:,:,:,2]#copy k 
    
    #WARP TRACKS IN BLOCKS
    print fwarp
    dprw=Dpy(fwarp,'r+')
    rows=len(dprw.f.root.streamlines.tracks)   
    blocks=np.round(np.linspace(0,rows,20)).astype(int)#lets work in blocks
    print rows
    for i in range(len(blocks)-1):        
        print blocks[i],blocks[i+1]        
        caboodle=dprw.f.root.streamlines.tracks[blocks[i]:blocks[i+1]]       
        ntrack=np.dot(caboodle,im2im[:3,:3].T)+im2im[:3,3] #from image vox space to mni image vox
        mci=mc(di,ntrack.T,order=1) #mapping for i
        mcj=mc(dj,ntrack.T,order=1) #mapping for j
        mck=mc(dk,ntrack.T,order=1) #mapping for k
        wtrack=ntrack+np.vstack((mci,mcj,mck)).T
        caboodlew=np.dot(wtrack,daff[:3,:3].T)+daff[:3,3]
        dprw.f.root.streamlines.tracks[blocks[i]:blocks[i+1]]=caboodlew.astype('f4')        
    
    dprw.close()

def warp_all_tracks(data_type='fa___'):
    #!!!!!!!!!!!! be very careful with this as it replaces data
    t1=time()
    for subj in range(0,12):#starts from 0  
        print 'subj number ',subj, 'starting from 0'
        if data_type=='qa':
            tracks_to_fmrib58(subj,fnames_64,data_type=data_type)
            tracks_to_fmrib58(subj,fnames_101,data_type=data_type)
            tracks_to_fmrib58(subj,fnames_118,data_type=data_type)        
        if data_type=='fa':
            tracks_to_fmrib58(subj,fnames_64,data_type=data_type)
            tracks_to_fmrib58(subj,fnames_101,data_type=data_type)
            tracks_to_fmrib58(subj,fnames_118,data_type=data_type)
      
    print('Done in %f secs' % (time()-t1))    
    dpr.close()   
        
def copy_all_native_tracks():
    #!!!!!!!!!!!! will overwrite if previously created _warp.dpy
    import os
    import shutil    
    print dname
    for root, dirs, files in os.walk(dname):
        for file in files:
            if file.endswith('_FA_native.dpy'):
                fname=os.path.join(root,file)
                print fname                
                fname2=fname.split('_FA_native.dpy')[0]+'_FA_warp.dpy'
                print fname2
                shutil.copyfile(fname, fname2)
     
def get_roi(froi,no,erosion_level=1):
    imgroi=ni.load(froi)    
    roidata=imgroi.get_data()
    roiaff=imgroi.get_affine()            
    cross=generate_binary_structure(3,1)
    roidata2=roidata.copy()
    roidata2[roidata2!=no]=0    
    if erosion_level>0:
        roidata2=binary_erosion(roidata2,cross,erosion_level)                 
        I=np.array(np.where(roidata2==True)).T
    else:
        I=np.array(np.where(roidata2==no)).T    
    wI=np.dot(roiaff[:3,:3],I.T).T+roiaff[:3,3]
    wI=wI.astype('f4')
    return wI

def get_center_roi(froi,no,erosion_level=1):
    imgroi=ni.load(froi)    
    roidata=imgroi.get_data()
    roiaff=imgroi.get_affine()            
    cross=generate_binary_structure(3,1)
    roidata2=roidata.copy()
    roidata2[roidata2!=no]=0
    if erosion_level>0:
        roidata2=binary_erosion(roidata2,cross,erosion_level)                 
    I=np.array(np.where(roidata2==True)).T    
    wI=np.dot(roiaff[:3,:3],I.T).T+roiaff[:3,3]
    wI=wI.astype('f4')
    return wI

def sums_length(dname,type='64'):
        for root, dirs, files in os.walk(dname):
            if root.endswith(type):                    
                for file in files:
                    if file.endswith('_warp.dpy'):       
                        fname=os.path.join(root,file)                    
                        dpr=Dpy(fname,'r')
                        sum=0
                        for i in range(dpr.track_no):
                            sum+=length(dpr.read_track())
                        dpr.close()
                        print fname, sum               

def generate_lengths():
    for root, dirs, files in os.walk(dname):          
                for file in files:
                    if file.endswith('_warp.dpy'):       
                        fname=os.path.join(root,file)                    
                        dpr=Dpy(fname,'r')
                        lengths=np.zeros((dpr.track_no,))
                        for i in range(dpr.track_no):
                            lengths[i]=length(dpr.read_track())
                        dpr.close()
                        fname2=fname.split('_warp.dpy')[0]+'_warp_lengths.npy'
                        print fname2
                        np.save(fname2,lengths)

def generate_histograms():    
    for root, dirs, files in os.walk(dname):          
        for file in files:
            if file.endswith('_warp_lengths.npy'):    
                fname=os.path.join(root,file)
                lengths=np.load(fname)
                binss=np.round(np.linspace(0, 400, 50)).astype(np.int)                        
                n, bins, patches = plt.hist(lengths, bins=binss, normed=True, facecolor='green', alpha=0.75)
                plt.ylim( (0, 0.025) )
                plt.xlabel('length(mm)')
                plt.ylabel('counts')
                #plt.title(r'Length distribution histogram')
                plt.title(root)
                #plt.axis([40, 160, 0, 0.03])
                plt.grid(True)     
                fname2=fname.split('_warp_lengths.npy')[0]+'_warp_lengths.png'
                print fname2
                plt.savefig(fname2)
                plt.clf()

def generate_cumulatives():
    subjs=['subj_01','subj_02','subj_03','subj_04','subj_05','subj_06','subj_07','subj_08','subj_09','subj_10','subj_11','subj_12']
    bins=np.round(np.linspace(0, 400, 50)).astype(np.int)
    
    for sub in subjs:
        lengthsFA={}
        lengthsQA={}    
        for root, dirs, files in os.walk(dname+sub):
            #print root
            if root.endswith('64'):
                print root
                for file in files:
                    if file.endswith('_FA_warp_lengths.npy'):
                        fname=os.path.join(root,file)
                        lengthsFA['64']=np.load(fname)
                    if file.endswith('_QA_warp_lengths.npy'):
                        fname=os.path.join(root,file)
                        lengthsQA['64']=np.load(fname)                       
            
            if root.endswith('101'):
                print root
                for file in files:
                    if file.endswith('_FA_warp_lengths.npy'):
                        fname=os.path.join(root,file)
                        lengthsFA['101']=np.load(fname)
                    if file.endswith('_QA_warp_lengths.npy'):
                        fname=os.path.join(root,file)
                        lengthsQA['101']=np.load(fname)

            if root.endswith('118'):
                print root
                for file in files:
                    if file.endswith('_FA_warp_lengths.npy'):
                        fname=os.path.join(root,file)
                        lengthsFA['118']=np.load(fname)
                    if file.endswith('_QA_warp_lengths.npy'):
                        fname=os.path.join(root,file)
                        lengthsQA['118']=np.load(fname)                
        
        dcols=[['64','red'],['101','blue'],['118','green']]
        for d in dcols:
            n, bins, patches = plt.hist(lengthsFA[d[0]], bins=bins, normed=True, cumulative=True, facecolor='none', edgecolor=d[1], alpha=0.75,label=d[0])
        
        plt.ylim( (0, 1) )
        plt.xlabel('length(mm)')
        plt.ylabel('cumulative')
        plt.legend()
        plt.title(sub)        
        plt.grid(True)
        fname2=dname+'sumFA/'+sub+'_FA_cumul.png'
        print fname2
        plt.savefig(fname2)
        plt.clf()
        
        dcols=[['64','red'],['101','blue'],['118','green']]
        for d in dcols:
            n, bins, patches = plt.hist(lengthsQA[d[0]], bins=bins, normed=True, cumulative=True, facecolor='none', edgecolor=d[1], alpha=0.75,label=d[0])
        
        plt.ylim( (0, 1) )
        plt.xlabel('length(mm)')
        plt.ylabel('cumulative')
        plt.legend()
        plt.title(sub)        
        plt.grid(True)
        fname2=dname+'sumQA/'+sub+'_QA_cumul.png'
        print fname2
        plt.savefig(fname2)
        plt.clf()
        
def show_histograms():
    for root, dirs, files in os.walk(dname):          
        if root.endswith('64'):
            for file in files:
                if file.endswith('_FA_warp_lengths.png'):
                    fname=os.path.join(root,file)
                    print fname,

    print
    print
    for root, dirs, files in os.walk(dname):          
        if root.endswith('101'):
            for file in files:
                if file.endswith('_FA_warp_lengths.png'):
                    fname=os.path.join(root,file)
                    print fname,
    print
    print
    for root, dirs, files in os.walk(dname):          
        if root.endswith('118'):
            for file in files:
                if file.endswith('_FA_warp_lengths.png'):
                    fname=os.path.join(root,file)
                    print fname,

def tracks_in_roi():
    
    froi='/home/eg309/Data/ICBM_Wmpm/ICBM_WMPM.nii'
    wI=get_roi(froi,35,1) #4 is genu    
    fname='/home/eg309/Data/PROC_MR10032/subj_03/101/1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_QA_warp.dpy'
    dpr=Dpy(fname,'r')    
    T=dpr.read_indexed(range(2*10**4))
    print len(T)
    Troi=[]
    for t in T:        
        if track_roi_intersection_check(t,wI,.5):
            Troi.append(t)
    print(len(Troi))
    dpr.close()
    
    from dipy.viz import fos
    r=fos.ren()
    fos.add(r,fos.line(Troi,fos.red))
    fos.add(r,fos.point(wI,fos.green))
    fos.show(r)

def tracks_in_cm_roi():
    
    froi='/home/eg309/Data/ICBM_Wmpm/ICBM_WMPM.nii'
    wI=get_roi(froi,35,1) #4 is genu    
    #c=get_cm_roi2(froi,[35])   
    fname='/home/eg309/Data/PROC_MR10032/subj_03/101/1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_QA_warp.dpy'
    dpr=Dpy(fname,'r')    
    T=dpr.read_indexed(range(2*10**4))
    print len(T)
    Troi=[]
    for t in T:        
        #if track_roi_intersection_check(t,wI,.5):        
        if point_track_sq_distance_check(t,c,4.): 
            Troi.append(t)
    print(len(Troi))
    dpr.close()
    
    from dipy.viz import fos
    r=fos.ren()
    fos.add(r,fos.line(Troi,fos.red))
    fos.add(r,fos.point(wI,fos.green))
    fos.add(r,fos.sphere(c,2.))
    fos.show(r)
    
def correct_icbm():    
    froi ='/home/eg309/Data/ICBM_Wmpm/ICBM_WMPM.nii'    
    fref = '/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
    froi2 ='/home/eg309/Data/ICBM_Wmpm/ICBM_WMPM_eleftherios_padded.nii'
    
    roiimg=ni.load(froi)
    roidata=roiimg.get_data()
    roiaff=roiimg.get_affine()
        
    refimg=ni.load(fref)
    refdata=refimg.get_data()
    refaff=refimg.get_affine()
    
    print roiaff
    print refaff
    
    print roidata.shape, roidata.dtype
    print refdata.shape, refdata.dtype
    
    roidata2=np.zeros(refdata.shape,dtype=roidata.dtype)    
    rois=roidata.shape
    
    roidata2[:rois[0],:rois[1],:rois[2]]=roidata
    
    roimg2=ni.Nifti1Image(roidata2,roiaff)
    ni.save(roimg2,froi2)
    
def skeletonize():
    
    froi='/home/eg309/Data/ICBM_Wmpm/ICBM_WMPM.nii'
    wI=get_roi(froi,35,1) #4 is genu    
    #fname='/home/eg309/Data/PROC_MR10032/subj_03/101/1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_QA_warp.dpy'
    #fname='/home/eg309/Data/PROC_MR10032/subj_03/101/1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_QA_native.dpy'
    #fname='/home/eg309/Data/PROC_MR10032/subj_06/101/13122110752323511930000010092916083910900000227ep2dadvdiffDSI10125x25x25STs004a001_QA_native.dpy'
    fname='/home/eg309/Data/PROC_MR10032/subj_06/101/13122110752323511930000010092916083910900000227ep2dadvdiffDSI10125x25x25STs004a001_QA_warp.dpy'
    
    dpr=Dpy(fname,'r')    
    T=dpr.read_indexed(range(2*10**4))
    dpr.close()
    
    print len(T)    
    from dipy.core.track_metrics import downsample
    from dipy.core.track_performance import local_skeleton_clustering, most_similar_track_mam
    Td=[downsample(t,3) for t in T]
    C=local_skeleton_clustering(Td,d_thr=20.)
    
    #Tobject=np.array(T,dtype=np.object)
    
    from dipy.viz import fos
    r=fos.ren()
    
    #skeleton=[]
    
    for c in C:
        color=np.random.rand(3)
        if C[c]['N']>0:
            Ttmp=[]
            for i in C[c]['indices']:
                Ttmp.append(T[i])
            si,s=most_similar_track_mam(Ttmp,'avg')
            print si,C[c]['N']                
            fos.add(r,fos.line(Ttmp[si],color))
            
    #print len(skeleton)
    #fos.add(r,fos.line(skeleton,color))    
    #fos.add(r,fos.line(T,fos.red))    
    fos.show(r)
    
def skeletonize_both():
    from dipy.viz import fos
    from dipy.core.track_metrics import downsample
    from dipy.core.track_performance import local_skeleton_clustering, most_similar_track_mam
    
    froi='/home/eg309/Data/ICBM_Wmpm/ICBM_WMPM.nii'
    wI=get_roi(froi,9,0) #4 is genu    
    fname='/home/eg309/Data/PROC_MR10032/subj_03/101/1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_QA_warp.dpy'
    #fname='/home/eg309/Data/PROC_MR10032/subj_03/101/1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_QA_native.dpy'
    #fname='/home/eg309/Data/PROC_MR10032/subj_06/101/13122110752323511930000010092916083910900000227ep2dadvdiffDSI10125x25x25STs004a001_QA_native.dpy'
    fname2='/home/eg309/Data/PROC_MR10032/subj_06/101/13122110752323511930000010092916083910900000227ep2dadvdiffDSI10125x25x25STs004a001_QA_warp.dpy'
    r=fos.ren()
    #'''
    dpr=Dpy(fname,'r')    
    T=dpr.read_indexed(range(2*10**4))
    dpr.close()    
    print len(T)    
    Td=[downsample(t,3) for t in T if length(t)>40]
    C=local_skeleton_clustering(Td,d_thr=20.)
    
    for c in C:
        #color=np.random.rand(3)
        color=fos.red
        if C[c]['N']>0:
            Ttmp=[]
            for i in C[c]['indices']:
                Ttmp.append(T[i])
            si,s=most_similar_track_mam(Ttmp,'avg')
            print si,C[c]['N']                
            fos.add(r,fos.line(Ttmp[si],color))                

    dpr=Dpy(fname2,'r')    
    T=dpr.read_indexed(range(2*10**4))
    dpr.close()    
    print len(T)    
    Td=[downsample(t,3) for t in T if length(t)>40]
    C=local_skeleton_clustering(Td,d_thr=20.)
    #r=fos.ren()
    for c in C:
        #color=np.random.rand(3)
        color=fos.yellow
        if C[c]['N']>0:
            Ttmp=[]
            for i in C[c]['indices']:
                Ttmp.append(T[i])
            si,s=most_similar_track_mam(Ttmp,'avg')
            print si,C[c]['N']                
            fos.add(r,fos.line(Ttmp[si],color))            
    #'''
    fos.add(r,fos.point(wI,fos.green))
    fos.show(r)

def centre_of_tracks(T):
    centre = np.zeros((3,))
    pass

def warp_image():
    dn='/home/eg309/Data/TEST_MR10032/subj_03/101/'
    ffa=dn+'1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_bet_FA.nii.gz'
    flaff=dn+'1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_affine_transf.mat'
    fdis =dn+'1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_nonlin_displacements.nii.gz'
    fref ='/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
    ffaw = '/tmp/w.nii.gz'
    #fw=dn+'1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_dipywarp_bet_FA.nii.gz'
    warp_displacements(ffa,flaff,fdis,fref,ffaw)


"""
1       MCP             Middle cerebellar peduncle^M
2       PCT             Pontine crossing tract (a part of MCP)^M
3       GCC             Genu of corpus callosum^M
4       BCC             Body of corpus callosum^M
5       SCC             Splenium of corpus callosum^M
6       FX              Fornix (column and body of fornix)^M
7       CST-R           Corticospinal tract right^M
8       CST-L           Corticospinal tract left^M
9       ML-R            Medial lemniscus right^M
10      ML-L            Medial lemniscus left^M
11      ICP-R           Inferior cerebellar peduncle right^M
12      ICP-L           Inferior cerebellar peduncle left^M
13      SCP-R           Superior cerebellar peduncle right^M
14      SCP-L           Superior cerebellar peduncle left^M
15      CP-R            Cerebral peduncle right^M
16      CP-L            Cerebral peduncle left^M
17      ALIC-R          Anterior limb of internal capsule right^M
18      ALIC-L          Anterior limb of internal capsule left^M
19      PLIC-R          Posterior limb of internal capsule right^M
20      PLIC-L          Posterior limb of internal capsule left^M
21      RLIC-R          Retrolenticular part of internal capsule right^M
22      RLIC-L          Retrolenticular part of internal capsule left^M
23      ACR-R           Anterior corona radiata right^M
24      ACR-L           Anterior corona radiata left^M
25      SCR-R           Superior corona radiata right^M
26      SCR-L           Superior corona radiata left^M
27      PCR-R           Posterior corona radiata right^M
28      PCR-L           Posterior corona radiata left^M
29      PTR-R           Posterior thalamic radiation (include optic radiation) right^M
30      PTR-L           Posterior thalamic radiation (include optic radiation) left^M
31      SS-R            Sagittal stratum (include inferior longitidinal fasciculus and inferior fronto-occipital fasciculus) right^M
32      SS-L            Sagittal stratum (include inferior longitidinal fasciculus and inferior fronto-occipital fasciculus) left^M
33      EC-R            External capsule right^M
34      EC-L            External capsule left^M
35      CGC-R           Cingulum (cingulate gyrus) right^M
36      CGC-L           Cingulum (cingulate gyrus) left^M
37      CGH-R           Cingulum (hippocampus) right^M
38      CGH-L           Cingulum (hippocampus) left^M
39      FX/ST-R         Fornix (cres) / Stria terminalis (can not be resolved with current resolution) right^M
40      FX/ST-L         Fornix (cres) / Stria terminalis (can not be resolved with current resolution) left^M
41      SLF-R           Superior longitudinal fasciculus right^M
42      SLF-L           Superior longitudinal fasciculus left^M
43      SFO-R           Superior fronto-occipital fasciculus (could be a part of anterior internal capsule) right^M
44      SFO-L           Superior fronto-occipital fasciculus (could be a part of anterior internal capsule) left^M
45      IFO-R           Inferior fronto-occipital fasciculus right^M
46      IFO-L           Inferior fronto-occipital fasciculus left^M
47      UNC-R           Uncinate fasciculus right^M
48      UNC-L           Uncinate fasciculus left^M
49      TAP-R           Tapatum right^M
50      TAP-L           Tapatum left^M

"""



def warp_reference_grid(flaff,fdis,fref,ffaw):
    ''' Warp reference space grid using fsl displacements
   
    Parameters
    ----------
    flaff: filename of .mat  (flirt)
    fdis:  filename of displacements (fnirtfileutils)
    fref: filename of reference volume e.g. (FMRIB58_FA_1mm.nii.gz)

    Returns
    -------
    warped grid in reference index space
   
    '''
   
    refaff=nib.load(fref).get_affine()   
    disdata=nib.load(fdis).get_data()
    #from fa index to ref index
    res=flirt2aff_files(flaff,ffa,fref)
    #create the 4d volume which has the indices for the reference image 
    refgrid=np.zeros(disdata.shape)
    #create the grid indices for the reference
    refgrid[...,0] = np.arange(disdata.shape[0])[:,newaxis,newaxis]
    refgrid[...,1] = np.arange(disdata.shape[1])[newaxis,:,newaxis]
    refgrid[...,2] = np.arange(disdata.shape[2])[newaxis,newaxis,:]     
    #hold the displacements' shape reshaping
    di,dj,dk,dl=disdata.shape
    N=di*dj*dk
    warpedgrid=np.zeros(disdata.shape)   
    for l in range(dl):
        warpedgrid[:,:,:,l]=mc(disdata[:,:,:,l].reshape(N,1),refgrid.reshape(N,dl).T,order=1).reshape(di,dj,dk)  

    warpedgrid = refgrid+np.dot(warpedgrid,res[:3,:3].T)+res[:3,3]

    return refgrid, warpedgrid

def warp_trackpoint(p,warpedgrid):
    di,dj,dk,dl=warpedgrid.shape
    N=di*dj*dk
    p2=np.dot(p,res[:3,:3].T)+res[3,:3]
    np.argmin(np.sum((warpedgrid.reshape(N,3)-p2)**2,axis=1))    
    #nearest = np.array(np.unravel(np.argmin(np.sum((warpedgrid.reshape(N,3)-p2)**2,axis=1)),(di,dj,dk)),type=np.float)
    return nearest


def warp_tracks():
    dn='/home/eg309/Data/TEST_MR10032/subj_03/101/'
    ffa=dn+'1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_bet_FA.nii.gz'    
    finvw=dn+'1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_warps_in_bet_FA.nii.gz'    
    fqadpy=dn+'1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_QA_native.dpy'
    flaff=dn+'1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_affine_transf.mat'
    fref ='/usr/share/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'    
    fdis =dn+'1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_nonlin_displacements.nii.gz'
    fdis2 =dn+'1312211075232351192010092217244332311282470ep2dadvdiffDSI10125x25x25STs004a001_nonlin_displacements_withaff.nii.gz'
    #read some tracks
    dpr=Dpy(fqadpy,'r')
    T=dpr.read_indexed(range(150))
    dpr.close()
    
    #from fa index to ref index
    res=flirt2aff_files(flaff,ffa,fref)
    
    #load the reference img    
    imgref=ni.load(fref)
    refaff=imgref.get_affine()
    
    #load the invwarp displacements
    imginvw=ni.load(finvw)
    invwdata=imginvw.get_data()
    invwaff = imginvw.get_affine()
    
    #load the forward displacements
    imgdis=ni.load(fdis)
    disdata=imgdis.get_data()
    #load the forward displacements + affine
    imgdis2=ni.load(fdis2)
    disdata2=imgdis2.get_data()
    #from their difference create the affine
    disaff=imgdis2.get_data()-disdata  
    
    shift=np.array([disaff[...,0].mean(),disaff[...,1].mean(),disaff[...,2].mean()])
    
    shape=ni.load(ffa).get_data().shape
    
    disaff0=affine_transform(disaff[...,0],res[:3,:3],res[:3,3],shape,order=1)
    disaff1=affine_transform(disaff[...,1],res[:3,:3],res[:3,3],shape,order=1)
    disaff2=affine_transform(disaff[...,2],res[:3,:3],res[:3,3],shape,order=1)
    
    disdata0=affine_transform(disdata[...,0],res[:3,:3],res[:3,3],shape,order=1)
    disdata1=affine_transform(disdata[...,1],res[:3,:3],res[:3,3],shape,order=1)
    disdata2=affine_transform(disdata[...,2],res[:3,:3],res[:3,3],shape,order=1)
    
    #print disgrad0.shape,disgrad1.shape,disgrad2.shape
    #disdiff=np.empty(invwdata.shape)
    #disdiff[...,0]=disgrad0
    #disdiff[...,1]=disgrad1
    #disdiff[...,2]=disgrad2
    #ni.save(ni.Nifti1Image(disdiff,invwaff),'/tmp/disdiff.nii.gz')
    
    di=disdata0
    dj=disdata1
    dk=disdata2
    
    d2i=invwdata[:,:,:,0] + disaff0
    d2j=invwdata[:,:,:,1] + disaff1
    d2k=invwdata[:,:,:,2] + disaff2
    
    #di=disgrad0
    #dj=disgrad1
    #dk=disgrad2
    
    imgfa=ni.load(ffa)
    fadata=imgfa.get_data()
    faaff =imgfa.get_affine()
    
    Tw=[]
    Tw2=[]
    Tw3=[]
    
    froi='/home/eg309/Data/ICBM_Wmpm/ICBM_WMPM.nii'    
    
    roiI=get_roi(froi,3,1) #3 is GCC     
    roiI2=get_roi(froi,4,1) #4 is BCC
    roiI3=get_roi(froi,5,1) #4 is SCC
    roiI=np.vstack((roiI,roiI2,roiI3))  
  
    for t in T:
        if np.min(t[:,2])>=0:#to be removed
            mci=mc(di,t.T,order=1) #interpolations for i displacement
            mcj=mc(dj,t.T,order=1) #interpolations for j displacement
            mck=mc(dk,t.T,order=1) #interpolations for k displacement            
            D=np.vstack((mci,mcj,mck)).T                        
            WI=np.dot(t,res[:3,:3].T)+res[:3,3]+D#+ shift
            W=np.dot(WI,refaff[:3,:3].T)+refaff[:3,3]
            
            mc2i=mc(d2i,t.T,order=1) #interpolations for i displacement
            mc2j=mc(d2j,t.T,order=1) #interpolations for j displacement
            mc2k=mc(d2k,t.T,order=1) #interpolations for k displacement            
            D2=np.vstack((mc2i,mc2j,mc2k)).T                        
            WI2=np.dot(t,res[:3,:3].T)+res[:3,3]+D2 #+ shift
            W2=np.dot(WI2,refaff[:3,:3].T)+refaff[:3,3]
                        
            WI3=np.dot(t,res[:3,:3].T)+res[:3,3]
            W3=np.dot(WI3,refaff[:3,:3].T)+refaff[:3,3]
            
            Tw.append(W)
            Tw2.append(W2)
            Tw3.append(W3)
    

    from dipy.viz import fos
    r=fos.ren()
    fos.add(r,fos.line(Tw,fos.red))
    fos.add(r,fos.line(Tw2,fos.green))    
    fos.add(r,fos.line(Tw3,fos.yellow))
    fos.add(r,fos.sphere((0,0,0),10,color=fos.blue))
    fos.add(r,fos.point(roiI,fos.blue))
    fos.show(r)

    

    
    
        
if __name__ == "__main__":
    pass
    #generate_cumulatives()    
    #generate_histograms()
    #show_histograms()
    #tracks_in_cm_roi()
    #skeletonize()
    #correct_icbm()
    #skeletonize_both()
    #warp_image()
    warp_tracks()
    #nodamnaffine()
    
    


    
    
    
    
    
    



    
    
