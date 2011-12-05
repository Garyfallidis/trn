#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author: Eleftherios Garyfallidis & Ian Nimmo-Smith
Description: Python library for reading different file formats.

  Supported formats:

    nii
    img/hdr
    dcm http://code.google.com/p/pydicom/
    
    gz 

    bvals/bvec (text files)

    xls Excel 


'''

try:
    import volumeimages as vi    
    volumeloading='vi'
except ImportError:
    print('Volumeimages is not installed.')

try:
    import dicom
    dicomloading='dcm'
except ImportError:
    print('Dicom reading/writing is not installed')
    print('http://code.google.com/p/pydicom/')
try:
  import xlrd
except ImportError:
  print('Xlrd is not installed.')

try:
    import xlwt
except ImportError:
    print('xlwt is not installed.')

try:
    from PIL import Image
except ImportError:
    print('PIL is not installed.')

import scipy as sp
from scipy import linalg as lg
from scipy import io
import numpy as np
import os
import string
import struct
import gzip
from optparse import OptionParser

def load(filename):

  '''
  Manipulate here all different formats and call the different load files.
  '''
  print('Not implemented yet.')
  print(string.rstrip(filename, '.gz'))

  pass

def logit(str,filename='logger.txt'):
    
    output=open(filename,'w+')
    output.write(str)
    output.close()

def list_files(dpath,filt='*'):
  '''
  List files under directory path with filt e.g filt='.nii'
  '''

  dirList=os.listdir(dpath)
  list_files=[]

  if filt=='*':
    for fname in dirList:
        list_files.append(dpath+'/'+fname)
  
  else:
    for fname in dirList:
        if fname.rfind(filt) > -1:
            list_files.append(dpath+'/'+fname)

  return list_files

def zipit(filename, mode='w',compress=9):
  '''
  Saves/Zipps a compressed file to disk
  http://code.activestate.com/recipes/534157/
  Compress level is from 1(fastest) to 9(slower but more compression)
  '''
  r_file = open(filename, 'r')
  w_file = gzip.GzipFile(filename + '.gz', mode, compress)
  w_file.write(r_file.read())
  w_file.flush()
  w_file.close()
  r_file.close()

def unzipit(filename,mode='r'):
  '''
  Unzips a compressed file from disk
  http://code.activestate.com/recipes/534157/  
  '''
  r_file = gzip.GzipFile(filename, mode)
  write_file = string.rstrip(filename, '.gz')
  w_file = open(write_file, 'w')
  w_file.write(r_file.read())
  w_file.close()
  r_file.close()

def readCommandLine():
  #Read from commandline
  #Add the following under if __name__ == "__main__":
  #options = readCommandLine()
  #print(options.File_to_be_run)

  parser = OptionParser()
  #read the options in
  parser.add_option("-f","--Full_file_location", 
    dest="File_to_be_run",
    default=r"tn.txt",
    help="This is the fully qualified path name to the file location")

  parser.add_option("-m","--Mode",
    dest="modeTn",
    default="r",
    help="The mode of zip unzip")
	
  parser.add_option("-c","--Compression",
    dest="compress",
    default=9,
    help="The level of compression")
  
  options, args = parser.parse_args()
  #print options
  return options



def loadvol(filename):
    '''
    
    Load a volumetric array stored in a Nifti or analyze file with a specific filename. 
    It returns  an array with the data (arr), the voxel size (voxsz) and a transformation matrix (affine).

    This function is using volumeimages.
    
    Example:

    arr,voxsz,aff=loadvol('test.nii')
    
    '''

    if os.path.isfile(filename)==False:
        
        print('File does not exist')
        
        return [],[],[]

    img = vi.load(filename)      

    arr = np.array(img.get_data())
    voxsz = img.get_metadata().get_zooms()
    aff = img.get_affine()

    return arr,voxsz,aff

def savevol(filename,data,affine):
    '''
        http://mail.scipy.org/pipermail/nipy-devel/2009-April/001258.html
    '''
    new_image=vi.Nifti1Image(data,affine)    
    vi.save(new_image,filename)

def loadmartatrack(filename):
    '''
    This file load's Marta's tractography files. Usualy with suffix .dat.
    
    Output: is a list of 2d arrays . Where every array is a trajectory where in every row there is a different point.
    
    Example:
        
        fname='/home/eg01/Data/Fibre_Cup/3x3x3/Marta_tracks/seed1.dat'
        L=loadmartatrack(fname)
        
        print L
    More info:
        
    There are two main structures fibredb and fibrepath which Marta used to write the file
    struct fibredb {
         int dblength;
         int dim[3];
         double pixdim[3];
         double acpc[3];
         int num_fibres;
    };

    struct fibrepath {
             int flength;
             int num_points;
             int algorithm;
             int seed_index;
             double length;
             double medFA;
             double minFA;
             double curve;
             double angle;
    };
    
    and then we have the points of each trajectory x_1,y_1,z_1, x_2, y_2, z_2, etc...
       
    '''

    if os.path.isfile(filename)==False:
        
        print('File does not exist')
        
        return
    
    data = open(filename,'r').read()
   
    fmt='4i6di'
    start,stop=0,struct.calcsize(fmt)    
    fibredb= struct.unpack(fmt,data[start:stop]) 

    print 'fibredb',fibredb
    num_fibres=fibredb[-1]
    print 'num_fibres',num_fibres
    
    #header=('dblength',fibredb[0],'dim',fibredb[1:4],'pixdim',fibredb[4:])
    
    fmt='4i5d'
        
    list_traj=[]

    for j in xrange(num_fibres):

        start=stop
        stop=start+struct.calcsize(fmt)
        
        fibrepath= struct.unpack(fmt,data[start:stop])         
        
        print fibrepath
            
        num_points=fibrepath[1]
        print 'num_points',num_points

        fmt='3d'
        list_points=[]
        
        for i in xrange(num_points):
        
            start=stop
            stop=start+24
            list_points.append(struct.unpack(fmt,data[start:stop]))
        
        #print list_points
        list_traj.append(sp.array(list_points))
        
    return list_traj
    
def loadmat(fname):
    #Look for scipy.io.loadmat
    #returns a dictionary
    return io.loadmat(fname)
    
def savemat(fname,dic):
    #Look for scipy.io.savemat
    #io.savemat(fname,{'data':data})
    io.savemat(fname,dic)
    return

def loadtrk(filename):
  '''
  Load a tractography file *.trk usually exported from diffusion toolkit (dtk)
  Information about the format is provided in
  http://www.trackvis.org/docs/?subsect=fileformat

  '''
  from time import clock

  #c1=clock()
  data = open(filename,'r').read() #first 1000 bytes are for header
  #c2=clock()
  #print(c2-c1)
 
  #content = struct.unpack('cccccchhh',data)

  start,stop=0,1000
    
  #header = struct.unpack('6c3h3f3fh200ch200c508c4c4c6f2cBBBBBBiii',data[start:stop])
  
  #print header

  start,stop=6,12
  dim = struct.unpack('3h',data[start:stop]) 

  start,stop=12,24
  voxsz = struct.unpack('3f',data[start:stop])

  start,stop=24,26
  n_s = struct.unpack('h',data[start:stop])
  n_s = n_s[0]

  start,stop=226,228  
  n_p = struct.unpack('h',data[start:stop])
  n_p = n_p[0]
  
  print 'dim:', dim, 'voxsz:', voxsz, 'n_s:', n_s, 'n_p:', n_p

  last=len(data)

  print 'data len:', last

  start,stop=1000,1004
   
  cnt=0

  traj_list=[]

  while stop<last:
  
    m = struct.unpack('i',data[start:stop])
    m=m[0]

    #print 'Track'+str(cnt)+ ' Number of points:',m

    traj=[]
    for i in xrange(m):

        start=stop
        stop=stop+(3+n_s)*4
        no_f=str(3+n_s)
        point=struct.unpack(no_f+'f',data[start:stop])

        traj.append(point)

        #print 'point' + str(i) + ':', point
        if n_p>0:

            start=stop
            stop=stop+(n_p*4)
            no_p=str(n_p)
            prop=struct.unpack(no_p+'i',data[start:stop])
            
        #print 


    traj_list.append(traj)

    start=stop
    stop =start+4

    cnt+=1
  
  #c3=clock()
  #print(c3-c1)

  return traj_list

def savetrk():
    '''
    Not implemented yet
    '''

def loadbinfodcm(filename,spm_converted=1):
    
    '''
    Load B-value and B-vector information from the Dicom Header of a file. This assumes that the scanner is Siemens.
    The needed information is under the CSA Image Information Header in the Dicom header. At the moment only version SV10 is supported.
    
    This was inspired by the work of Williams & Brett using the following matlab script http://imaging.mrc-cbu.cam.ac.uk/svn/utilities/devel/cbu_dti_params.m
    However here we are using pydicom and then read directly from the CSA header.
    
    Input: Dicom filename
    Output: B_value (stored in the dicom), B_vec (B_vector calculated from stored B_matrix), G_direction(gradient direction stored in dicom), 
    B_value_B_matrix (B value calculated from the stored in dcm B_matrix after using eigenvalue decomposition).
    
    Example:
    
    B_value, B_vec, G_direction, B_value_B_matrix =  loadbinfodcm(fname)
    
    '''

    #filename = '/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/1.3.12.2.1107.5.2.32.35119.2009022715012276195181703.dcm'
    #filename = '/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/1.3.12.2.1107.5.2.32.35119.2009022715073976305795724.dcm'
    
    if os.path.isfile(filename)==False:
        
        print('Filename does not exist')
        
        return 
        
    data=dicom.read_file(filename)

    if spm_converted:
        y_flipper=sp.diag([1, -1, 1])
    else:
        y_flipper=sp.eye(3)
           
    #print 'y_flipper',y_flipper
    
    orient=data.ImageOrientationPatient
    orient=sp.transpose(sp.reshape(orient,(2,3)))
       
    v1=sp.array([orient[0,0],orient[1,0],orient[2,0]])
    v2=sp.array([orient[0,1],orient[1,1],orient[2,1]])   
    v3=sp.cross(v1,v2)
    
    #print 'v3',v3
    
    orient=sp.column_stack((v1.transpose(),v2.transpose(),v3.transpose()))      

    if lg.det(orient<0):
        #print('Det orient < 0')
        print 'Negative determinant.'
        orient[:,2]=-orient[:,2]

    #print 'orient',orient

    vox_to_dicom = sp.dot(orient, y_flipper)
    
    #print 'vox_to_dicom',vox_to_dicom
    mat = lg.inv(vox_to_dicom)
    
    #print 'mat',mat    
    #print vox_to_dicom*vox_to_dicom
    
    csainfo=data[0x029,0x1010]
    
    #print csainfo[0:4]
    
    if csainfo[0:4]!='SV10':
        print 'No SV10'
        
        B_vec=sp.array([sp.NaN,sp.NaN,sp.NaN])
        B_value=0
        G_direction=sp.array([0.0,0.0,0.0])
        B_value_B_matrix=0
                
        return B_value, B_vec, G_direction, B_value_B_matrix        
                
    start,stop=8,12

    #print 'CSA Image Info'
    
    n=struct.unpack('I',csainfo[start:stop])
    n=n[0]
    #print 'n:',n
        
    B_value=-1
    B_matrix=[]
    G_direction=[]
        
    #Read B-related Info
    start=16
    for i in xrange(n):
        
        rec=[]
        
        stop=start+64+4+4+4+4+4
        name=struct.unpack('64ci4ciii',csainfo[start:stop])
        nitems=int(name[-2])
        startstore=start
        start =stop       
        
        #print(''.join(name[0:64]))
        #print(''.join(name[0:25]))
        matstart=0
        valstart=0
        diffgradstart=0
        
        if ''.join(name[0:8])=='B_matrix':
            matstart=startstore

        if ''.join(name[0:7])=='B_value':
            valstart=startstore        
        
        if ''.join(name[0:26])== 'DiffusionGradientDirection':
            diffgradstart=startstore
                
        for j in xrange(nitems):
            
            xx=struct.unpack('4i',csainfo[start:start+4*4])
            length=int(xx[1])    
           
            value =struct.unpack(str(length)+'c',csainfo[start+4*4:start+4*4+length])                 
            
                           
            if matstart > 0:
                if len(value)>0:
                    B_matrix.append(float(''.join(value[:-1] )))
                else :
                    B_matrix.append(0.0)
                           
            if valstart > 0 :
                if len(value)>0:
                    B_value=float(''.join(value[:-1] ))

            if diffgradstart > 0 :
                if len(value)>0 :
                    G_direction.append(float(''.join(value[:-1] )))
                        
            stop=start+4*4+length+(4-length%4)%4
            start=stop                 
       
    if B_value >0: 
        
        B_mat=sp.array([[B_matrix[0],B_matrix[1],B_matrix[2]], [B_matrix[1],B_matrix[3],B_matrix[4]], [B_matrix[2],B_matrix[4],B_matrix[5]]])
        [vals, vecs]=lg.eigh(B_mat)       
       
        dbvec = vecs[:,2]
        
        if dbvec[0] < 0:
            dbvec = dbvec * -1
            
        B_vec=sp.transpose(sp.dot(mat, dbvec))         
        B_value_B_matrix=vals.max()
        
    else:
        
        B_vec=sp.array([0.0,0.0,0.0])
        B_value=0
        G_direction=sp.array([0.0,0.0,0.0])
        B_value_B_matrix=0
                
    return B_value, B_vec, G_direction, B_value_B_matrix

#def savebinfo(filename):
    
def loadbinfodir(dirname):
    '''
    Load information about b-values and b-vectors from a directory with multiple dicom files. Returns a list (binfo) were every node holds 
    the following info B_value, B_vec[0], B_vec[1], B_vec[2], G_direction[0], G_direction[1], G_direction[2], B_value_B_matrix.
    
    Example:
    
    dirname='/backup/Data/Eleftherios/CBU090134_METHODS/20090227_154122/Series_003_CBU_DTI_64D_iso_1000'
    binfo=loadbinfodir(dirname)
    savebinfo(binfo=binfo) 
    
    This snippet outputs three files binfo.txt, bvecs.txt, bvals.txt
    '''
    if os.path.isdir(dirname):
        pass
    else:
        print 'No Directory found'
    
    lfiles=list_files(dirname,filt='.dcm')
    lfiles.sort()
    binfo=[]
    row=[]
    
    for fname in lfiles:
        #print fname
        #print loadbinfodcm(fname)
        B_value, B_vec, G_direction, B_value_B_matrix = loadbinfodcm(fname)
        row=[B_value, B_vec[0], B_vec[1], B_vec[2], G_direction[0], G_direction[1], G_direction[2], B_value_B_matrix]
        
        binfo.append(row)

    return binfo

def savebinfo(filename='binfo.txt',binfo=[],fnamebvecs='bvecs.txt',fnamebvals='bvals.txt'):
    '''
    Every row in binfo has the following information
    [B_value, B_vec[0], B_vec[1], B_vec[2], G_direction[0], G_direction[1], G_direction[2], B_value_B_matrix]
    '''
    if binfo==[]:
        print 'Binfo list is needed.'
        return
    
    f = open(filename,'w')
    f2= open(fnamebvecs,'w')    
    f3= open(fnamebvals,'w')
    
    for i in binfo:

        f.write(str(i[0]));  f.write(' ');  f.write(str(i[1])); f.write(' ');  f.write(str(i[2])); f.write(' ');  
        f.write(str(i[3]));  f.write(' ');  f.write(str(i[4])); f.write(' ');  f.write(str(i[5])); f.write(' ');
        f.write(str(i[6]));  f.write(' ');  f.write(str(i[7])); f.write('\n')   
       
        
        f3.write(str(i[7]))
        f3.write('   ')
    
    f3.write('\n')    
    
    f.close()
    f3.close()
    
    Binfo=sp.array(binfo)
    bvecsX=Binfo[:,1]
    bvecsY=Binfo[:,2]
    bvecsZ=Binfo[:,3]
    
    for i in bvecsX:
        f2.write(str(i)); f2.write('   ')
    for i in bvecsY:
        f2.write(str(i)); f2.write('   ')
    for i in bvecsZ:
        f2.write(str(i)); f2.write('   ')
        
    f2.write('\n')
    
    f2.close()
        
    return    

def loadbvals(filename):
    '''
    Loads B-values from a txt file named usually bvals
    '''

    bvals = []
    for line in open(filename, 'rt'):
        bvals =[float(val) for val in line.split()]            
    bvals=sp.array(bvals)
    return bvals
    
def loadbvecs(filename):    
    '''
    Loads B-vecs from a txt file named bvecs
    '''
    
    bvecs = []
    for line in open(filename, 'rt'):
        bvecs.append([float(val) for val in line.split()])
    bvecs=sp.array(bvecs)
    
    return bvecs

def loaddata(filename):    
    '''
    Loads row col data
    '''
    
    vecs = []
    for line in open(filename, 'rt'):
        vecs.append([float(val) for val in line.split()])
    vecs=sp.array(vecs)
    
    return vecs

def loadexcel(filename):
  '''
  Load the full excel spreadsheet book using xlrd
  book has the following important properties 

  book.nsheets
  book.sheet_names()
  sh = book.sheet_by_index(0)  
  sh = book.sheet_by_name(sheetname)
  
  sh.name, sh.nrows, sh.ncols
  print "Cell D30 is", sh.cell_value(rowx=29, colx=3)
  for rx in range(sh.nrows):
    print sh.row(rx)
  
  More about xlrd 
  http://www.lexicon.net/sjmachin/xlrd.html
  And an example of usage here
  http://www.lexicon.net/sjmachin/README.html
  
  '''
  try:
    return xlrd.open_workbook(filename)
  except:
    print('Cannot open file.')
    pass
    
def saveexcel(filename):
    
    '''
    Similar with xlrd but writing this time
    http://pypi.python.org/pypi/xlwt
    '''
    try:
        xlwt.Workbook
        book=Workbook()
        sheet1 = book.add_sheet('Sheet 1')
        sheet1.write(0,0, 'Words')
        book.save(filename)
        
    except:
        print('Cannot create workbook.')
    

def loadpic(filename,option='color2gray'):
    
    '''
    Load a 2d image all formats that are supported by PIL are supported also here.
    '''
    if os.path.isfile(filename)==False:
        
        print('Filename does not exist')
    
        return None
    
    if option=='color2gray':
        
        im = sp.array(Image.open(filename).convert('L'))        
    
    elif option=='default':

        im = sp.array(Image.open(filename))
                
    return im
   

def expertrk(traj_size=0):

    
    print 'Loading Data ...'
    
    filename='/home/eg01/Backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dti_FACT.trk'
    filename2='/home/eg01/Backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dti_RK2.trk'
      
    trajs=loadtrk(filename)
    
    trajs=trajs[1:10000]
    print 'Data Loaded.'
    
    import lights as li
     
    ren=li.vtk.vtkRenderer()
    
    print 'Loading trajectories ...'
    
    act=li.trajectories(trajs)
      
    ren.AddActor(act)

    print 'Lines On!'
        
    print 'Showing window'

    ap=li.AppThread(frame_type=0,ren=ren,width=1024,height=800)


if __name__ == "__main__":    

    #experSami()
    #expertrk()1.0326315229473091
    #dirname='/backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000'

    '''
    dirname='/backup/Data/Eleftherios/CBU090134_METHODS/20090227_154122/Series_003_CBU_DTI_64D_iso_1000'
    lfiles=list_files(dirname,filt='.dcm')
    lfiles.sort()
    for fname in lfiles:
        #print fname
        print loadbinfodcm(fname)
    '''
    
    fname='/home/eg01/Data/Fibre_Cup/3x3x3/Marta_tracks/seed1.dat'
    L=loadmartatrack(fname)
    print L
    '''
    fname='/backup/Data/Eleftherios/CBU090134_METHODS/20090227_154122/Series_003_CBU_DTI_64D_iso_1000/1.3.12.2.1107.5.2.32.35119.2009022715490973107728450.dcm'
    
    print fname
    bvals,bvecs,g_direct,bvalue2= loadbinfodcm(fname)
    print 'bvals',bvals
    print 'bvecs', bvecs
    print 'g_direct',g_direct
    print 'bvalue2',bvalue2
    '''
    
    fname='/home/eg309/Data/Fibre_Cup/3x3x3/Marta_tracks/seed1.dat'

    loadmartatrack(fname)