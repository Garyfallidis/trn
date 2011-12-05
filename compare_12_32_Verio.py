#needs compare_steam.py


def compare_12_with_32_Verio():

    tmp_dir='/tmp/compare_12_with_32_Verio_directly/'
    try:
        os.mkdir(tmp_dir)        
    except:
        pass

    img12='DSI 3000 12channels'
    img32='DSI 3000 32channels'

    print(img12)
    
    img12nii=resources.get_paths(img12)[2]
    img12bvals=resources.get_paths(img12)[5]
    img12bvecs=resources.get_paths(img12)[8]    
   
    img12betnii=os.path.join(tmp_dir,os.path.basename(img12nii)+'_bet')    
    bet(img12nii,img12betnii)
    img12dtifit=os.path.join(tmp_dir,os.path.basename(img12nii))  
    dtifit(img12nii,img12dtifit,img12betnii+'_mask.nii.gz',img12bvals,img12bvecs)

    print(img32)
    
    img32nii=resources.get_paths(img32)[2]
    img32bvals=resources.get_paths(img32)[5]
    img32bvecs=resources.get_paths(img32)[8]

    img32betnii=os.path.join(tmp_dir,os.path.basename(img32nii)+'_bet')    
    bet(img32nii,img32betnii)
    img32dtifit=os.path.join(tmp_dir,os.path.basename(img32nii))  
    dtifit(img32nii,img32dtifit,img32betnii+'_mask.nii.gz',img32bvals,img32bvecs)
    

def compare_steam(source,target,results_dir,dicom=(1,1)):

    print('Goal is to compare FA of target versus source acquisitions using STEAM')
    print('find filenames for target and source data')    

    #dname_grid=resources.get_paths('DSI STEAM 101 Trio')[2]
    #dname_shell=resources.get_paths('DTI STEAM 114 Trio')[2]
    dname_target=resources.get_paths(target)[2]    
    dname_source=resources.get_paths(source)[2]
    
    #print('find filenames for T1')
    #fname_T1=resources.get_paths('MPRAGE nifti Trio')[2]
    #dcm2nii(dname_grid,'/tmp/compare_steam')
    #tmp_dir='/tmp/compare_steam'
    
    tmp_dir=results_dir

    if dicom==(1,1):
        print('load dicom data')
        dname_target=resources.get_paths(target)[2]    
        dname_source=resources.get_paths(source)[2]
        data_so,affine_so,bvals_so,gradients_so=dp.load_dcm_dir(dname_source)
        data_ta,affine_ta,bvals_ta,gradients_ta=dp.load_dcm_dir(dname_target)        
        
    if dicom==(1,0):
        print('load dicom for source and nifti for target')        
        dname_source=resources.get_paths(source)[2]
        data_so,affine_so,bvals_so,gradients_so=dp.load_dcm_dir(dname_source)
        img_ta=ni.load(resources.get_paths(target)[2])
        data_ta=img_ta.get_data()
        affine_ta=img_ta.get_affine()
        bvals_ta=np.loadtxt(resources.get_paths(target)[5])
        gradients_ta=np.loadtxt(resources.get_paths(target)[8]).T

    if dicom==(0,1):
        print('load nifti for source and dicom for target')
        dname_target=resources.get_paths(target)[2]   
        data_ta,affine_ta,bvals_ta,gradients_ta=dp.load_dcm_dir(dname_target)
        img_so=ni.load(resources.get_paths(source)[2])
        data_so=img_so.get_data()                                
        affine_so=img_so.get_affine()
        bvals_so=np.loadtxt(resources.get_paths(source)[5])
        gradients_so=np.loadtxt(resources.get_paths(source)[8]).T
        print 'SHAPES', bvals_so.shape,gradients_so.shape

    if dicom==(0,0):
        print('load nifti data')
        img_ta=ni.load(resources.get_paths(target)[2])
        data_ta=img_ta.get_data()
        affine_ta=img_ta.get_affine()
        bvals_ta=np.loadtxt(resources.get_paths(target)[5])
        gradients_ta=np.loadtxt(resources.get_paths(target)[8]).T        
        img_so=ni.load(resources.get_paths(source)[2])
        data_so=img_so.get_data()                                
        affine_so=img_so.get_affine()
        bvals_so=np.loadtxt(resources.get_paths(source)[5])
        gradients_so=np.loadtxt(resources.get_paths(source)[8]).T
        
    data_ta=data_ta.astype('uint16')
    data_so=data_so.astype('uint16')

    print('calculate tensors for target and source data')

    ten_target=dp.Tensor(data_ta,bvals_ta,gradients_ta,thresh=50)
    ten_source=dp.Tensor(data_so,bvals_so,gradients_so,thresh=50)
    
    print('save FAs')
    FA_target=ten_target.FA
    FA_source=ten_source.FA    
    FA_target_img=ni.Nifti1Image(FA_target,affine_ta)
    FA_source_img=ni.Nifti1Image(FA_source,affine_so)    
    FA_tmp_target=os.path.join(tmp_dir,os.path.basename(dname_target)+'_FA.nii.gz')
    FA_tmp_source=os.path.join(tmp_dir,os.path.basename(dname_source)+'_FA.nii.gz')
    ni.save(FA_target_img,FA_tmp_target)
    ni.save(FA_source_img,FA_tmp_source)

    print('save MDs')    
    MD_target=ten_target.MD
    MD_source=ten_source.MD
    MD_target_img=ni.Nifti1Image(MD_target,affine_ta)
    MD_source_img=ni.Nifti1Image(MD_source,affine_so)    
    MD_tmp_target=os.path.join(tmp_dir,os.path.basename(dname_target)+'_MD.nii.gz')
    MD_tmp_source=os.path.join(tmp_dir,os.path.basename(dname_source)+'_MD.nii.gz')
    ni.save(MD_target_img,MD_tmp_target)
    ni.save(MD_source_img,MD_tmp_source)
        
    
    print('save DWI reference as nifti')
    tmp_target=os.path.join(tmp_dir,os.path.basename(dname_target)+'_ref.nii.gz')
    tmp_source=os.path.join(tmp_dir,os.path.basename(dname_source)+'_ref.nii.gz')    
    ni.save(ni.Nifti1Image(data_ta[...,0],affine_ta),tmp_target)    
    ni.save(ni.Nifti1Image(data_so[...,0],affine_so),tmp_source)

    print('remove the scalp using bet')
    tmp_target_bet=os.path.join(tmp_dir,os.path.basename(dname_target)+'_ref_bet.nii.gz')
    tmp_source_bet=os.path.join(tmp_dir,os.path.basename(dname_source)+'_ref_bet.nii.gz')  
    bet(tmp_target,tmp_target_bet)
    bet(tmp_source,tmp_source_bet)
    
    print('register with reference')
    tmp_source_bet_reg=os.path.join(tmp_dir,os.path.basename(dname_source)\
                                       +'_ref_bet_reg.nii.gz')
    tmp_source_T=os.path.join(tmp_dir,os.path.basename(dname_source)+'_T.mat')
    flirt(tmp_source_bet,tmp_target_bet,tmp_source_bet_reg,tmp_source_T)
    
    print('apply transformation to FA')
    FA_tmp_source_reg=os.path.join(tmp_dir,os.path.basename(dname_source)+'_FA_reg.nii.gz')
    flirt_apply_transform(FA_tmp_source,tmp_target_bet,FA_tmp_source_reg,tmp_source_T)

    print('apply transformation to MD')
    MD_tmp_source_reg=os.path.join(tmp_dir,os.path.basename(dname_source)+'_MD_reg.nii.gz')
    flirt_apply_transform(MD_tmp_source,tmp_target_bet,MD_tmp_source_reg,tmp_source_T)
  
    print('get rois')
    ROIS_target=ni.load('/tmp/DTI114_2_DSI101_Trio/114_FA_ROIs.img')
    tmp_source_T=os.path.join(tmp_dir,os.path.basename(dname_source)+'_invT.mat')
    invert_transform(tmp_source_T, inv_source_T)

    ROIS_target_img=ni.Nifti1Image(ROIS_target.get_data(),ROIS_target.get_affine())
    tmp_ROIS_target=os.path.join(tmp_dir,os.path.basename(dname_target)+'_ROIS.nii.gz')
    ni.save(ROIS_target_img,tmp_ROIS_target)
    
    print('apply inverse transforms to rois TARGET')
    tmp_ROIS_target_final=os.path.join(tmp_dir,os.path.basename(dname_source)+'_ROIS_final.nii.gz')
    flirt_apply_transform(tmp_ROIS_target,FA_tmp_source,tmp_ROIS_target_final,inv_source_T)

    print('Create final masks')
    ni.load(tmp_source_bet)

    
    
   
    


if __name__ == '__main__':

    '''
    source='DTI STEAM 114 Trio'
    target='DSI STEAM 101 Trio'    
    tmp_dir='/tmp/DTI114_2_DSI101_Trio'
    try:
        os.mkdir(tmp_dir)        
    except:
        pass
    compare_steam(source,target,results_dir=tmp_dir,dicom=(1,1))

    '''

    '''
    source='DSI 4000 2.5mm Verio 32channels'
    target='DSI STEAM 101 Trio'
    tmp_dir='/tmp/DSI_32channels_Verio_to_12channels_Trio_both_4000_2.5mm'
    try:
        os.mkdir(tmp_dir)        
    except:
        pass
    compare_steam(source,target,results_dir=tmp_dir,dicom=(0,1))
    '''
    #create_fa_from_nifti('DSI 101 2mm Verio')


    '''
    source='DSI 3000 12channels'
    target='DTI STEAM 114 Trio'
    tmp_dir='/tmp/DSI_compare_12_using114_Trio_Yeah'
    try:
        os.mkdir(tmp_dir)        
    except:
        pass
    compare_steam(source,target,results_dir=tmp_dir,dicom=(0,1))

    
    source='DSI 3000 32channels'
    target='DTI STEAM 114 Trio'
    tmp_dir='/tmp/DSI_compare_32_using114_Trio_Yeah'
    try:
        os.mkdir(tmp_dir)        
    except:
        pass
    compare_steam(source,target,results_dir=tmp_dir,dicom=(0,1))

    '''
    
    #compare_12_with_32_Verio_using114()

    
    compare_12_with_32_Verio()
