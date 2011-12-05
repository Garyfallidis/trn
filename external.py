#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author: Eleftherios Garyfallidis

Description: Script that use External command line tools for analysis of Diffusion Data

This script assumes that you are using a modern unix system Debian or Ubuntu

This script was tested on Ubuntu Intrepid 8.10 64bit

External tools used at the moment diffusion toolkit, fsl, mricron, camino

All external tools must be added to the path after installed

'''

import os
import sys
import platform
from subprocess import Popen,PIPE 

#subfolders for results
fsl_out_dir='fsl_out'
dtk_dti_out_dir='dtk_dti_out'
dtk_hardi_out_dir='dtk_hardi_out'
camino_out_dir='cami_out'
mricron_out_dir='mricron_out'
freesurf_out_dir='freesurf_out_dir'


#Siemens default 64 Gradient directions for diffusion imaging
dtk_gm_fpath='/home/eg01/tract/gradientSiemens64.txt'



def find_files_unix(initial_dir,filt):

  #Example
  #initial_dir='/backup/Data/Alexandra/'
  #filt='*DTI_64'

  cmd='find '+initial_dir+' -name '+ filt

  p = Popen(cmd, shell=True,stdout=PIPE,stderr=PIPE)
  sto=p.stdout.readlines()
  ste=p.stderr.readlines()

  return sto,ste


def Alex_dirs():

  dpath_list=[
    '/backup/Data/Alexandra/Patients/CBU060972/Series_010_CBU_DTI_64',
    '/backup/Data/Alexandra/Patients/CBU071017/Series_011_CBU_DTI_64',
    '/backup/Data/Alexandra/Patients/CBU070012/Series_011_CBU_DTI_64',
    '/backup/Data/Alexandra/Patients/CBU070010/Series_011_CBU_DTI_64',
    '/backup/Data/Alexandra/Patients/CBU070096/Series_011_CBU_DTI_64',
    '/backup/Data/Alexandra/Controls/CBU070013/Series_009_CBU_DTI_64',
    '/backup/Data/Alexandra/Controls/CBU060960/Series_010_CBU_DTI_64',
    '/backup/Data/Alexandra/Controls/CBU070011/Series_010_CBU_DTI_64',
    '/backup/Data/Alexandra/Controls/CBU070002/Series_009_CBU_DTI_64',
    '/backup/Data/Alexandra/Controls/CBU070006/Series_010_CBU_DTI_64',
    '/backup/Data/Alexandra/Controls/CBU060967/Series_010_CBU_DTI_64',
    '/backup/Data/Alexandra/Controls/CBU060966/Series_009_CBU_DTI_64',
    '/backup/Data/Alexandra/Controls/CBU070001/Series_010_CBU_DTI_64',
    '/backup/Data/Alexandra/Controls/CBU070092/Series_009_CBU_DTI_64',
    '/backup/Data/Alexandra/Controls/CBU070018/Series_010_CBU_DTI_64',
    '/backup/Data/Alexandra/Controls/CBU070007/Series_010_CBU_DTI_64']

  return dpath_list


def Elef_dirs():
  
  dpath_list=[
    '/backup/Data/Eleftherios/CBU090134_METHODS/20090227_154122/Series_003_CBU_DTI_64D_iso_1000',
    '/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000']

  return dpath_list

def Fibre_Cup_files():
    
    fpath1_list=[
    '/home/eg01/Data/Fibre_Cup/3x3x3/dwi-b0650.nii',
    '/home/eg01/Data/Fibre_Cup/3x3x3/dwi-b1500.nii',
    '/home/eg01/Data/Fibre_Cup/3x3x3/dwi-b2650.nii',
    ]
    
    fpath2_list=[
    '/home/eg01/Data/Fibre_Cup/6x6x6/dwi-b0650.nii',   
    '/home/eg01/Data/Fibre_Cup/6x6x6/dwi-b1500.nii',
    '/home/eg01/Data/Fibre_Cup/6x6x6/dwi-b2000.nii',
    ]
    
    fdirections1='/home/eg01/Data/Fibre_Cup/3x3x3/diffusion_directions.txt'    
    fdirections2='/home/eg01/Data/Fibre_Cup/6x6x6/diffusion_directions.txt'

    return fpath1_list, fpath2_list, fdirections1, fdirections2

def list_files(dpath,filt):

  cmd='ls ' + dpath + '/' + filt    

  p = Popen(cmd, shell=True,stdout=PIPE,stderr=PIPE) 
  #stdout, stderr = p.communicate()   
  sto=p.stdout.readlines()
  #print sto
  stdout, stderr = p.communicate() 
  return sto

def size_files(dpath,filt):

  cmd='du -h ' + dpath + '/' + filt    

  p = Popen(cmd, shell=True,stdout=PIPE,stderr=PIPE) 
  #stdout, stderr = p.communicate()   
  sto=p.stdout.readlines()
  #print sto
  stdout, stderr = p.communicate() 
  return sto
  
def find_first_dcm(dpath,filt='*.dcm'):

  dicom_file=list_files(dpath,filt)[0]
  
  return dicom_file[:-1]

def dicom2nifti_dtk(dcmfile,outdir):

  cmd='diff_unpack "' + dcmfile + '"' + ' "'+ outdir + '/dwi_all" -ot nii'
  p = Popen(cmd, shell=True,stdout=PIPE,stderr=PIPE) 
  stdout, stderr = p.communicate()

  return outdir + '/dwi_all.nii' 
 
def dti_reconstruct_dtk(dwi_all_fpath,dtk_dti_out_dpath,gm_fpath):

  cmd = 'dti_recon "' + dwi_all_fpath + '"' + ' "' + dtk_dti_out_dpath + '/dti' + '"' + ' -gm "' + gm_fpath + '" -b 1000 -b0 1 -oc -p 3 -sn 1 -ot nii'
    
  #print cmd
  p = Popen(cmd,shell=True,stdout=PIPE,stderr=PIPE) 
  stdout, stderr = p.communicate()

def dti_tracker_dtk(dtk_dti_out_dpath,mask_fpath='Default',tracking_method='FACT'):
 
#dti_tracker "/backup/Data/Alexandra/Controls/CBU060960/dti" "/backup/Data/Alexandra/Controls/CBU060960/dti.trk" -at 35 -m "/backup/Data/Alexandra/Controls/CBU060960/dti_dwi.nii" -it nii
  
  if mask_fpath=='Default':

    mask_fpath= dtk_dti_out_dpath + '/'+ 'dti_dwi.nii'

  if tracking_method=='RK2':
    cmd = 'dti_tracker "' + dtk_dti_out_dpath + '/dti' + '"' + ' "' + dtk_dti_out_dpath + '/dti_' + tracking_method + '.trk' + '" ' + '-rk2 -at 35 -m "' +  mask_fpath +'" -it nii'

  elif tracking_method=='TENSOR_LINE':
    cmd = 'dti_tracker "' + dtk_dti_out_dpath + '/dti' + '"' + ' "' + dtk_dti_out_dpath + '/dti_' + tracking_method + '.trk' + '" ' + '-tl -at 35 -m "' +  mask_fpath + '" -it nii'

  else: #use FACT
    cmd = 'dti_tracker "' + dtk_dti_out_dpath + '/dti' + '"' + ' "' + dtk_dti_out_dpath + '/dti_' + tracking_method + '.trk' + '" ' + '-fact -at 35 -m "' +  mask_fpath + '" -it nii'


  p = Popen(cmd,shell=True,stdout=PIPE,stderr=PIPE) 
  stdout, stderr = p.communicate()
  
def mksubdir(dpath,newsubdir):

  cmd='mkdir ' + dpath + '/' + newsubdir
  p = Popen(cmd, shell=True,stdout=PIPE,stderr=PIPE) 
  stdout, stderr = p.communicate()
  #print(p.stdout.readlines())
  #print(p.stderr.readlines())

  return dpath + '/' + newsubdir

def rmfiles(dpath,filt):

  cmd='rm ' + dpath + '/' + filt

  p = Popen(cmd, shell=True,stdout=PIPE,stderr=PIPE) 
  stdout, stderr = p.communicate()
  #print(p.stdout.readlines())
  #print(p.stderr.readlines())


def tractography_dtk_dti(dtk_dti_out_dpath,dwi_all_fpath):

  #print dwi_all_fpath
        
  dti_reconstruct_dtk(dwi_all_fpath,dtk_dti_out_dpath,dtk_gm_fpath)        
  dti_tracker_dtk(dtk_dti_out_dpath, tracking_method='FACT')
  dti_tracker_dtk(dtk_dti_out_dpath, tracking_method='RK2')

def tractography_dtk_hardi(dtk_hardi_out_dpath,dwi_all_fpath):

  pass


def mri_convert(fname,suffix='.nii',out_dir='cur'):

  '''
  Convert image files to different file formats using freesurfer mri_convert 
  e.g. image.mgz will be converted to image.nii if suffix is .nii 

  Todo check else option for out_dir
  '''

  filenameonly=os.path.splitext(os.path.basename(fname))[0]
  
  if out_dir=='cur':
  #Remove suffix
  
    cmd='mri_convert '+ fname + ' ' + filenameonly + suffix

  else :

    cmd='mri_convert '+ fname + ' ' + os.path.dirname(fname) +'/' + filenameonly + suffix

  p = Popen(cmd, shell=True,stdout=PIPE,stderr=PIPE) 
  stdout, stderr = p.communicate()

  print stdout
  print stderr

def eddy_correct(fname,suffix='_eddy.nii'):
    
    '''
    buggy needs work
    
    '''
    filenameonly=os.path.splitext(os.path.basename(fname))[0]
        
    cmd='eddy_correct '+fname + ' ' + os.path.dirname(fname) + '/'+ filenameonly+ suffix 
        
    print cmd
    
    '''    
    p = Popen(cmd, shell=True,stdout=PIPE,stderr=PIPE) 
    stdout, stderr = p.communicate()

    print stdout
    print stderr
    
    '''
    
def mpi_kmeans(finput,foutput,fassign,k,option='assign'):
    
    '''
    http://mloss.org/software/view/48/
    http://www.kyb.mpg.de/bs/people/pgehler/code/index.html
    '''
    
    fkmeans='/home/eg01/Devel/mpi_kmeans-1.5/mpi_kmeans64'
    fkmeans2='/home/eg01/Devel/mpi_kmeans-1.5/mpi_assign64'

    
    if os.path.isfile(fkmeans)==False:
        
        print('mpi_kmeans64 does not exist in the specified folder')
        return
    
    if option=='assign':
        
        cmd=fkmeans2 + ' --k '+str(k)+' --data '+finput+' --output '+ foutput + ' --assignment ' + fassign
    
    else:
        
        cmd=fkmeans + ' --k '+str(k)+' --data '+finput+' --output '+ foutput
        
    
    p = Popen(cmd, shell=True,stdout=PIPE,stderr=PIPE) 
    stdout, stderr = p.communicate()

    print stdout
    print stderr
    

def Pipeline(dicom_dirs):
   
  logf = open('log.txt','w')

  for folder in dicom_dirs:
    
    dcmfile=find_first_dcm(folder)

    '''
    Make subdirectories and generate nii 
    '''

    dtk_dti_out_dpath=mksubdir(folder,dtk_dti_out_dir)
    
    dwi_all_fpath=dicom2nifti_dtk(dcmfile,dtk_dti_out_dpath)

    '''
    DTK DTI Tractography
    '''
    tractography_dtk_dti(dtk_dti_out_dpath,dwi_all_fpath)

    #res=list_files(dtk_dti_out_dpath,'*.trk')
    res=size_files(dtk_dti_out_dpath,'*.trk')

    '''
    Print trks' paths
    '''
    for r in res:

      print r
    
    logf.writelines(res)

    '''
    DTK HARDI Tractography
    '''


  logf.close()


  

if __name__ == "__main__":    

  Pipeline(Alex_dirs())

