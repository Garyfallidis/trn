#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author: Eleftherios Garyfallidis
Description: Python library for diffusion MRI

Check testfunctions for examples of usage...

'''



try:
    import nifti as ni

    volumeloading='ni'

except:
    print('Nifti module is  not installed. Volumeimages will be used instead')

try:
    import volumeimages as vi
    
    volumeloading='vi'

except:
    print('Volumeimages is not installed.')


try:
    import wx
except:
    print('WX module is not installed.')

import os
import sys
import platform

import scipy as sc
import numpy as np
from scipy import linalg

import pickle

try:
    import pylab as plt
    import matplotlib.axes3d as plt3
    
except:
    print('Pylab module is not installed.')

try:
    import wxVTKRenderWindowInteractor as wxvi
except:
    print('WX or VTK modules are not installed.')

try:
    import vtk
except:
    print('VTK module is not installed.')

#try:
#    from enthought.tvtk.api import tvtk
#    from enthought.tvtk.array_handler import get_vtk_array_type
#except:
#    print('Enthought.tvtk module is not installed.')

from ctypes import *


class shared:

    project='Tractarian'

    opacity=[]
    opacityprop=np.array([[  0.0, 0.0],
                          [255.0, 0.2]])
    color=[]
    #colorprop=np.array([[  0.0, 0.0, 0.0, 0.0],
    #                    [255.0, 1.0, 1.0, 1.0]])

    colorprop=np.array([[  0.0, 0.0, 0.0, 0.0],
                        [ 64.0, 0.0, 0.0, 1.0],
                        [128.0, 0.0, 1.0, 0.0],
                        [192.0, 1.0, 0.0, 0.0],
                        [255.0, 1.0, 1.0, 1.0]])
    im=[]

    widget=[]
    def __init__(self):    
        
        pass
           
sh=shared()

def loadnifti(fname,info=0):
    '''
    Returns two arguments 
    eg. arr,voxsz=loadnifti(filename)
    filename should be ending in nii or nii.gz
    '''
    
    fname=fname.encode() #change from unicode
    
    if volumeloading=='ni':
    
      img = ni.NiftiImage(fname)    
    
      '''
      problems with C1 from adam's  data 256*256*40*26
      '''
    
      arr = img.asarray()        
      voxsz = img.getVoxDims()
    
      if info==1:
          print('Filename: ',fname)   
          print('Bytes: ',arr.nbytes)
          print('Data type: ',arr.dtype.name)
          print('Dimensions: ',arr.shape)

      return arr,voxsz

    elif volumeloading=='vi':

      img = vi.load(fname)      

      arr = np.array(img.get_data())
      voxsz = img.get_metadata().get_zooms()

      return arr,voxsz

    else:
      return [],[]

def savenifti(fname,arr,voxsz=(1,1,1)):

    if volumeloading=='ni':

      nim=ni.NiftiImage(arr)
      nim.setVoxDims(voxsz)
      nim.save(fname)

    elif volumeloading=='vi':
    
      print('Saving with volumeimages not developed yet')
      #new_image = vi.Nifti1Image(data, affine)
      #vi.save(new_image, 'new_image.nii.gz')

    else:

      print('Nothing to save')


    return

def loadbvals(fname):

    bvals = []
    for line in open(fname, 'rt'):
        bvals =[float(val) for val in line.split()]            
    bvals=sc.array(bvals)
    return bvals
    
def loadbvecs(fname):

    bvecs = []
    for line in open(fname, 'rt'):
        bvecs.append([float(val) for val in line.split()])
    bvecs=sc.array(bvecs)
    return bvecs
       
def filedialog(dlgtype):
    '''
    dlgtype can be wx.OPEN or wx.SAVE
    '''

    path = ''; fname = ''; dirname = '';    
    app = wx.PySimpleApp()
    frame = wx.Frame(None, wx.ID_ANY, "Dark", style=wx.STAY_ON_TOP)    
    #frame.Show(True)         
    dlg = wx.FileDialog(frame, "Choose a file", fname, "", "*", dlgtype) 
    if dlg.ShowModal() == wx.ID_OK:
        #fname=dlg.GetFilename(); #dirname=dlg.GetDirectory()        
        path = dlg.GetPath()
    dlg.Destroy()        
    app.MainLoop()

    return path

def opendialog():

    return filedialog(wx.OPEN)         

def savedialog():

    return filedialog(wx.SAVE)     
    
def msgdialog(title,msg):    
    
    app = wx.PySimpleApp()
    frame = wx.Frame(None, wx.ID_ANY, "Dark",style=wx.STAY_ON_TOP)    
    #frame.Show(True)        
    dlg= wx.MessageDialog( frame, msg, title, wx.OK)       
    dlg.ShowModal()
    dlg.Destroy()
    app.MainLoop()  

def systeminfo():

    arch=platform.architecture()
    release=platform.release()
    uname=platform.uname()
    plat=sys.platform
    pyversion=sys.version.split(' ')[0]

    #print(sys.getwindowsversion())
    
    info='This is a '+arch[0]+' '+uname[0]+' os.' 
    if uname[0]=='Linux':
        info2='Exact os version is ' + uname[2]+ '.'
    else:
        info2='Exact os version is ' + uname[2] +' '+ uname[3]+'.'

    info3='The name of this computer is ' + uname[1]+'.'    
    
    info4='The version of Python is '+pyversion+'.\n'
    

    print(info)
    print(info2)
    print(info3)
    print(info4)


    if uname[0]=='Windows':

        try:
            from ctypes.wintypes import windll
       
            class MEMORYSTATUS(Structure):

                _fields_ = [
                    ('dwLength', DWORD),
                    ('dwMemoryLoad', DWORD),
                    ('dwTotalPhys', DWORD),
                    ('dwAvailPhys', DWORD),
                    ('dwTotalPageFile', DWORD),
                    ('dwAvailPageFile', DWORD),
                    ('dwTotalVirtual', DWORD),
                    ('dwAvailVirtual', DWORD),
                ]
    
            x = MEMORYSTATUS()
            windll.kernel32.GlobalMemoryStatus(byref(x))    
            print('%d MB physical RAM left.' % (x.dwAvailPhys/1024**2))
            print('%d MB physical RAM in total.' % (x.dwTotalPhys/1024**2))
            print('%d MB total virtual memory.' % (x.dwTotalVirtual/1024**2))

        except:
            print('Ctypes.wintypes module is not installed.')

    if uname[0]=='Linux':

        import re
        re_meminfo_parser = re.compile(r'^(?P<key>\S*):\s*(?P<value>\d*)\s*kB')


        result = dict()
        for line in open('/proc/meminfo'):
            match = re_meminfo_parser.match(line)
            if not match:
                continue  # skip lines that don't parse
            key, value = match.groups(['key', 'value'])
            result[key] = int(value)
        
        print('%d MB total memory.' %(result['MemTotal']/1024))
        
        '''
        for cpu usage see /proc/stat
        for loadvg  see /proc/loadavg
        '''

def transp_arr3_vtk(arr):

    vol = arr
    vol = vol/vol.max()
    vol = vol*255
    vol = np.round(vol)
    vol = np.uint8(vol)
    vol = vol.transpose([1,0,2]).copy()
    
    #vol=np.transpose(arr).copy()   
    #vol.shape = vol.shape[::-1]
    
    return vol
    
def updateview():

    sh.opacity.RemoveAllPoints()
    sh.color.RemoveAllPoints()
        
    for i in range(sh.opacityprop.shape[0]):
        sh.opacity.AddPoint(sh.opacityprop[i,0],sh.opacityprop[i,1])
       
    for i in range(sh.colorprop.shape[0]):
        sh.color.AddRGBPoint(sh.colorprop[i,0],sh.colorprop[i,1],sh.colorprop[i,2],sh.colorprop[i,3])

    sh.widget.Render()

def viewvol(arr,voxsz=(1.0,1.0,1.0),maptype=1):

    print(arr.dtype)

    arr = arr/arr.max()
    arr = arr*255    
    
    sh.im = vtk.vtkImageData()
    #sh.im.SetScalarTypeToFloat()
    sh.im.SetScalarTypeToUnsignedChar()
    sh.im.SetDimensions(arr.shape[0],arr.shape[1],arr.shape[2])
    sh.im.SetOrigin(0,0,0)    
    sh.im.SetSpacing(voxsz[2],voxsz[0],voxsz[1])
    sh.im.AllocateScalars()
    
    print(sh.im.GetNumberOfScalarComponents())
    print(arr.shape)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                
                sh.im.SetScalarComponentFromFloat(i,j,k,0,arr[i,j,k])

    sh.opacity = vtk.vtkPiecewiseFunction()
    for i in range(sh.opacityprop.shape[0]):
        sh.opacity.AddPoint(sh.opacityprop[i,0],sh.opacityprop[i,1])

    sh.color = vtk.vtkColorTransferFunction()
    for i in range(sh.colorprop.shape[0]):
        sh.color.AddRGBPoint(sh.colorprop[i,0],sh.colorprop[i,1],sh.colorprop[i,2],sh.colorprop[i,3])


    if(maptype==0): 
        property = vtk.vtkVolumeProperty()
        property.SetColor(sh.color)
        property.SetScalarOpacity(sh.opacity)
        
        mapper = vtk.vtkVolumeTextureMapper2D()
        mapper.SetInput(sh.im)
    
    if (maptype==1):

        property = vtk.vtkVolumeProperty()
        property.SetColor(sh.color)
        property.SetScalarOpacity(sh.opacity)
        property.ShadeOn()
        property.SetInterpolationTypeToLinear()
     
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        mapper = vtk.vtkVolumeRayCastMapper()
        mapper.SetVolumeRayCastFunction(compositeFunction)
        mapper.SetInput(sh.im)
   
    ren=vtk.vtkRenderer()
    
    volume = vtk.vtkVolume()

    volume.SetMapper(mapper)
    volume.SetProperty(property)

    ren.AddVolume(volume)

    #cone = tvtk.ConeSource(resolution=8)#,height=100, radius=50)
    #coneMapper = tvtk.PolyDataMapper(input=cone.get_output())
    #coneActor = tvtk.Actor(mapper=coneMapper)    
    #coneActor.position=sc.array([100.,100.,100.])    
    #ren.add_actor(coneActor)    
    
    #try:
    #simplewx(ren,title=sh.project,width=600,height=400)
    #except:
    simpletk(ren,title=sh.project,width=600,height=400) 


    return ren



def viewcone():       
    #'''    
    cone = vtk.vtkConeSource()
    
    coneMapper = vtk.vtkPolyDataMapper()
    coneMapper.SetInput(cone.GetOutput())
    
    coneActor = vtk.vtkActor()
    coneActor.SetMapper(coneMapper)
    
    ren = vtk.vtkRenderer()
    ren.AddActor(coneActor)
    simpletk(ren,title='Cone Demo',width=1024,height=768)
    
    #'''
    return

def addpoint(position,radius=0.1,thetares=8,phires=8,color=(0,0,1),opacity=1):
    
    
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(thetares)
    sphere.SetPhiResolution(phires)
   
    spherem = vtk.vtkPolyDataMapper()
    spherem.SetInput(sphere.GetOutput())
    spherea = vtk.vtkActor()
    spherea.SetMapper(spherem)
    spherea.SetPosition(position)
    spherea.GetProperty().SetColor(color)
    spherea.GetProperty().SetOpacity(opacity)
        
    return spherea

def addline(pos1,pos2,color=(1,0,0),res=1):
    
    line=vtk.vtkLineSource()
    line.SetResolution(res)
    line.SetPoint1(pos1)
    line.SetPoint2(pos2)
    linem = vtk.vtkPolyDataMapper()
    linem.SetInput(line.GetOutput())
    linea = vtk.vtkActor()
    linea.SetMapper(linem)
    linea.GetProperty().SetColor(color)
    
    return linea
    


def addarrow(pos=(0,0,0),color=(1,0,0),opacity=1):
    
    arrow = vtk.vtkArrowSource()
    arrowm = vtk.vtkPolyDataMapper()
    arrowm.SetInput(arrow.GetOutput())
    
    arrowa= vtk.vtkActor()
    arrowa.SetMapper(arrowm)
    arrowa.GetProperty().SetColor(color)
    arrowa.GetProperty().SetOpacity(opacity)
    
    return arrowa

def addaxes():
    
    arrowx=addarrow(color=(1,0,0))
    arrowy=addarrow(color=(0,1,0))
    arrowz=addarrow(color=(0,0,1))
    arrowy.RotateZ(90)
    arrowz.RotateY(-90)
    
    ass=vtk.vtkPropAssembly()
    ass.AddPart(arrowx)
    ass.AddPart(arrowy)
    ass.AddPart(arrowz)
    
    return ass

def addlabel(ren,text='Origin',pos=(0,0,0),scale=(0.1,0.1,0.1)):
    
    atext=vtk.vtkVectorText()
    atext.SetText(text)
    
    textm=vtk.vtkPolyDataMapper()
    textm.SetInput(atext.GetOutput())
    
    texta=vtk.vtkFollower()
    texta.SetMapper(textm)
    texta.SetScale(scale)    
    texta.SetPosition(pos)
    
    ren.AddActor(texta)
    
    texta.SetCamera(ren.GetActiveCamera())
    
    return texta
    
def  addaxeswithlabels(length=(1,1,1),labelx="x",labely="y",labelz="z"):
    
    axes=vtk.vtkAxesActor()
    #axes.SetShaftTypeToCylinder()
    axes.SetXAxisLabelText(labelx)
    axes.SetYAxisLabelText(labely)
    axes.SetZAxisLabelText(labelz)    
    axes.SetTotalLength(length)
    
    tprop = vtk.vtkTextProperty()
    tprop.ItalicOn()
    tprop.ShadowOn()
    tprop.SetFontFamilyToTimes()
    axes.GetXAxisCaptionActor2D().SetCaptionTextProperty(tprop)
    
    tprop2 = vtk.vtkTextProperty()
    tprop2.ShallowCopy(tprop)
    axes.GetYAxisCaptionActor2D().SetCaptionTextProperty(tprop2)

    tprop3 = vtk.vtkTextProperty()
    tprop3.ShallowCopy(tprop)
    axes.GetZAxisCaptionActor2D().SetCaptionTextProperty(tprop3)
    
    return axes
    
def viewscatter(listofpoints):
           
    ren=vtk.vtkRenderer()      
    #ren.AddActor(addaxeswithlabels(length=(10,10,10)))
    ren.AddActor(addaxes())
        
    ren.AddActor(addline(pos1=(-10,0,0),pos2=(10,0,0),color=(1,0,0)))
    ren.AddActor(addline(pos1=(0,-10,0),pos2=(0,10,0),color=(0,1,0)))
    ren.AddActor(addline(pos1=(0,0,-10),pos2=(0,0,10),color=(0,0,1)))
    
    for var in listofpoints:
        ren.AddActor(addpoint(position=var))
    
    #ren.AddActor(addlabel())
    addlabel(ren)
       
    simpletk(ren,title='Scatter Plot',width=1024,height=768)


   

def simpletk(ren,title='Tractarian',width=600,height=400):

    print('Using tk')

    picker = vtk.vtkCellPicker()   
    
    def annotatePick(object, event):        
        
        if picker.GetCellId() < 0:
            print('No object')
            print(np.round(picker.GetSelectionPoint(), decimals=2))
        else:
            print('Object Found')
            print(np.round(picker.GetSelectionPoint(), decimals=2))
            print(np.round(picker.GetPickPosition(), decimals=2))            
            print(np.round(picker.GetActors().GetLastActor().GetPosition(), decimals=2))
        
    picker.AddObserver("EndPickEvent", annotatePick)
    
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.SetPicker(picker)
    
    renWin.SetSize(width,height)
    renWin.Render()
    renWin.SetWindowName(title)
    
    
    def CheckAbort(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)
 
    renWin.AddObserver("AbortCheckEvent", CheckAbort)

    sh.widget=renWin

    iren.Initialize()
    picker.Pick(0, 0, 0, ren)

    renWin.Render()
    iren.Start()

    return

def simplewx(ren,title='Tractarian',width=600,height=400):
    '''
    Create a simple wx window and render vtk inside
    '''
    print('Using wx')

    app = wx.PySimpleApp()
    frame = wx.Frame(None, -1, title, size=(width,height))  
    sh.widget = wxvi.wxVTKRenderWindowInteractor(frame, -1)    
    sizer = wx.BoxSizer(wx.VERTICAL)
    sizer.Add(sh.widget, 1, wx.EXPAND)
    frame.SetSizer(sizer)
    frame.Layout()
    sh.widget.Enable(1)
    sh.widget.AddObserver("ExitEvent", lambda o,e,f=frame: f.Close())
    sh.widget.GetRenderWindow().AddRenderer(ren)
    frame.Show()
    app.MainLoop()

    return 

def threshold(arr, lower=0, upper=100):

    '''
    Threshold array arr using an upper and lower threshold in % (percentage)
    e.g threshold(arr, 20, 70)
    '''
    if lower<0:
        
        print('lower threshold cannot be smaller than 0')
        return

    if upper>100:

        print('upper threshold cannot be higher than 100')
        return

    if upper<lower:

        print('upper cannot be smaller than lower')
        return

    min=arr.min()
    max=arr.max()

    y=lower

    x=((max-min)/100.0)*y+min

    print('lower:',x)

    arr[arr<x]=0
    
    y=upper

    x=((max-min)/100.0)*y+min
    
    print('upper:',x)

    arr[arr>x]=0

    return


def slider(title='Slider',size=(478,94)):
 
    class MyFrame(wx.Frame):
        def __init__(self, *args, **kwds):
            kwds["style"] = wx.DEFAULT_FRAME_STYLE
            wx.Frame.__init__(self, *args, **kwds)
            self.slider_1 = wx.Slider(self, -1, 0, 0, 10, style=wx.SL_HORIZONTAL|wx.SL_LABELS)
            self.__set_properties()
            self.__do_layout()
            self.Bind(wx.EVT_COMMAND_SCROLL, self.OnScroll, self.slider_1)

        def __set_properties(self):
            self.SetTitle(title)
            self.SetSize(size)
            self.SetBackgroundColour(wx.Colour(0, 0, 0))
            self.slider_1.SetBackgroundColour(wx.Colour(0, 0, 0))
            self.slider_1.SetForegroundColour(wx.Colour(255, 255, 255))
            self.slider_1.SetFont(wx.Font(8, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, ""))

        def __do_layout(self):
            sizer_1 = wx.BoxSizer(wx.VERTICAL)
            sizer_1.Add(self.slider_1, 0, wx.EXPAND, 0)
            self.SetSizer(sizer_1)
            self.Layout()

        def OnScroll(self, event):            
            obj=self.slider_1.GetValue()
            event.Skip()

    app = wx.PySimpleApp(0)
    wx.InitAllImageHandlers()
    frame_1 = MyFrame(None, -1, "")
    app.SetTopWindow(frame_1)
    frame_1.Show()
    app.MainLoop()

    return

def plotarr3(arr,point=(0,0,0)):    

    #if arr.ndim != 3:
    #    print('arr can only be 3d')
    #    return

    print(point[0])
    print(point[1])
    print(point[2])

    axial = arr[point[0]]
    coronal=arr[:,point[1]]
    saggital=arr[:,:,point[2]]

    plotarr2(axial,'Axial')
    plotarr2(coronal,'Coronal')
    plotarr2(saggital,'Saggital')
    
    '''
    doplt = partial(plt.imshow, origin='lower', cmap=plt.jet())
    doplt(arr[42])
    plt.title('Axial slice')
    plt.show()
    plt.figure()
    doplt(arr[:,42])
    plt.title('Coronal slice')
    plt.show()
    plt.figure()
    doplt(arr[:,:,42])
    plt.title('Sagittal slice')
    plt.show()       
    '''
 
    return

def plotarr2(mat,title):

    #if arr.ndim != 2:
    #    print('arr can only be 2d')
    #    return

    plt.figure()
    plt.imshow(mat,origin='lower',cmap=plt.jet())
    plt.title(title)
    plt.show()

def viewgl(list):

    return
     
def main():

    print('In main')

	
    return

def testvol():

    fname=opendialog()
    #fname='FAsimple.nii.gz'
    
    print(fname)
    arr,voxsz=loadnifti(fname)
    print(arr.shape)

    if arr.ndim ==4:
        viewvol(arr[0],voxsz)
    elif arr.ndim==3:
        viewvol(arr,voxsz)
    else:
        print('Please use only 3d or 4d volumes')
    
        
    #df.sh.opacity.add_point(255,0.2)    
    #df.sh.widget.Render()

def testvolsl():

    fname=opendialog()
    arr,voxsz=loadnifti(fname)

    sz=arr.shape

    ker=np.zeros(sz)
    msk=np.ones((sz[1],sz[2]))  
    ker[sz[0]/2]=msk

    arr=arr*ker
    viewvol(arr,voxsz)

def axialfilter(arr,no):

    sz=arr.shape
    ker=np.zeros(sz)
    msk=np.ones((sz[1],sz[2]))  
    ker[no]=msk
    arr=arr*ker
   
    return arr,ker

def coronalfilter(arr,no):

    sz=arr.shape
    ker=np.zeros(sz)
    msk=np.ones((sz[0],sz[2]))  
    ker[:,no]=msk
    arr=arr*ker
   
    return arr,ker

def saggitalfilter(arr,no):

    sz=arr.shape
    ker=np.zeros(sz)
    msk=np.ones((sz[0],sz[1]))  
    ker[:,:,no]=msk
    arr=arr*ker
   
    return arr,ker

def slicerfilter(arr,point=(0,0,0)):

    if arr.ndim != 3:
        print('arr needs to be 3d')    

    arr1,ker1=axialfilter(arr,point[0])
    arr2,ker2=coronalfilter(arr,point[1])
    arr3,ker3=saggitalfilter(arr,point[2])
    
    ker12=np.logical_or(ker1,ker2)
    ker=np.logical_or(ker12,ker3)

    arr=arr*ker

    return arr

def testnifti():
    '''
    Create some simulated data in a numpy array and save them as a niftiimage. 
    Then load them again from the disk in memory and visualize them with viewvol.
    '''
    dims=(60,60,60)	
    x, y, z = sc.ogrid[-5:5:dims[0]*1j,-5:5:dims[1]*1j,-5:5:dims[2]*1j]
    x = x.astype('f')
    y = y.astype('f')
    z = z.astype('f')
    scalars = (sc.sin(x*y*z)/(x*y*z))
    fname='test.nii.gz'
    savenifti(fname,scalars,voxsz=(1,1,1))
    arr,voxsz=loadnifti(fname)
    viewvol(arr,voxsz)

def testplotarr3():

    fname=opendialog()
    arr,voxsz=loadnifti(fname)
    print(arr.ndim)
    print(arr.shape)
    plotarr3(arr,point=(42,42,42))


def tensorfitsimple(arr,bvals,bvecs,mskthr=100.0,averaging=1):
    '''
    Optimized version of tensorfit

    Calculate tensors from a 4d numpy array and return an FA image.
    bvals and bvecs must be provided as well.

    FA calculated from Mori et.al, Neuron 2006
    
    In WindowsXP the memory used can be ~1GB where it is a problem
    with the amount of diffusion data.

    '''
    if arr.ndim!=4:
        print('Please provide a 4d numpy array as arr here')
        return      
 
    B=bvals.astype('float32')
    G=bvecs.astype('float32')

    imshape=arr.shape
    volshape=(imshape[1],imshape[2],imshape[3])
    volnos=arr.shape[0]
 
    arr=arr.astype('float32')

    vollen=volshape[0]*volshape[1]*volshape[2]

    if averaging==1:
        
        arr[0:volnos/2]=0.5*(arr[0:volnos/2]+arr[volnos/2:volnos])
    
    S=arr[0:volnos/2].copy()    
   
    del arr,bvals,bvecs

    A=sc.zeros((volnos/2-1,6))
        
    FA=sc.zeros(volshape,dtype='float32')
    msk=sc.zeros(volshape,dtype='float32')    

    S0=S[0]
    
    msk[S0<mskthr]=1    

    S[S<1.0]=1.0
    S=sc.log(S)
    S=S[0]-S[:]
    S=S[1:]
       
    #try simple tensor fit
      
    for i in sc.arange(volnos/2-1):

        b=B[i+1]
        g=G[:,i+1]
        
        g1=g[0]; g2=g[1]; g3=g[2]
        
        A[i,0]=g1*g1
        A[i,1]=g2*g2
        A[i,2]=g3*g3
        A[i,3]=2*g1*g2
        A[i,4]=2*g1*g3
        A[i,5]=2*g2*g3
    
        S[i]=S[i]/b
         
    S=S.reshape(volnos/2-1,vollen)        
    FA=FA.reshape(vollen)    
    msk=msk.reshape(vollen)       

    d,resids,rank,sing=sc.linalg.lstsq(A,S)
        
       
    cnt=0

    #D=sc.array([[d11,d12,d13],[d12,d22,d23],[d13,d23,d33]])

    D=sc.zeros((3,3))

    for i in sc.arange(vollen):

        if msk[i]== 1:

            FA[i]=0      
        
        else:
                       
                                
            #d11=d[0,i]; d22=d[1,i]; d33=d[2,i]; d12=d[3,i]; d13=d[4,i]; d23=d[5,i]; 
       
            D[0,0]=d[0,i];
            D[1,1]=d[1,i];
            D[2,2]=d[2,i];

            D[0,1]=d[3,i];
            D[0,2]=d[4,i];
            D[1,2]=d[5,i];

            D[1,0]=D[0,1]
            D[2,0]=D[0,2]
            D[2,1]=D[1,2]
            
            evals,evecs=sc.linalg.eig(D)
                
            l1=evals[0];l2=evals[1];l3=evals[2]

            if l1<0 or l2<0 or l3<0:
                cnt+=1

            FA[i]=sc.sqrt( ( (l1-l2)**2 + (l2-l3)**2 + (l3-l1)**2 )/( 2*(l1**2+l2**2+l3**2) )  )                       
    
    print('Not positive definite ',cnt, ' times.')
    
    FA=FA.reshape(volshape[0],volshape[1],volshape[2])
    del S,A,msk
    

    return FA


def residuals_example(p, y, x):
    
    A,k,theta = p
    err = y-A*sc.sin(2*sc.pi*k*x+theta)
    return err

def testnonlinearlsqexample():
    
    x = sc.arange(0,6e-2,6e-2/30)
    A,k,theta = 10, 1.0/3e-2, sc.pi/6
    y_true = A*sc.sin(2*sc.pi*k*x+theta)
    y_meas = y_true + 2*sc.randn(len(x))
    
    p0 = [8, 1/2.3e-2, sc.pi/3]
    print(p0)

    from scipy.optimize import leastsq
    plsq = leastsq(residuals_example, p0, args=(y_meas, x),full_output=1)
    print(plsq)

    #from scipy.optimize import anneal
    #xann = anneal(residuals_example, p0, args=(y_meas,x),schedule='cauchy')
    #print(xann[0])
'''    
def residuals_rosenbrock(x1,x2):
    
    return 100*(x2-x1**2)**2+(1-x1)**2
    
def testoptimizationalgs():
    
    x0=[1,1]    
    from scipy.optimize import leastsq    
    x=leastsq(residuals_rosenbrock,
    
    return 
'''
def residualsf_positive(a,g0,g1,g2,c):
    
    return (a[0]*g0)**2+ (a[1]*g0+a[2]*g1)**2 + (a[3]*g0+a[4]*g1+a[5]*g2)**2 - c            
        
    
def tensorfitpositive(arr,bvals,bvecs,mskthr=100.0,averaging=1):
    '''
    Calculate tensors from a 4d numpy array and return an FA image.
    bvals and bvecs must be provided as well.

    FA calculated from Mori et.al, Neuron 2006

    This function needs to be optimized further for 64bit systems.
    In WindowsXP the memory used can be ~1GB where it is a problem
    with the amount of diffusion data.

    '''
    if arr.ndim!=4:
        print('Please provide a 4d numpy array as arr here')
        return      
 
    B=bvals.astype('float32')
    G=bvecs.astype('float32')

    imshape=arr.shape
    volshape=(imshape[1],imshape[2],imshape[3])
    volnos=arr.shape[0]
 
    arr=arr.astype('float32')

    vollen=volshape[0]*volshape[1]*volshape[2]

    arr=arr.reshape(imshape[0],vollen)

    for i in sc.arange(vollen):

        arr[0:volnos/2,i]=0.5*(arr[0:volnos/2,i]+arr[volnos/2:volnos,i])

    arr=arr.reshape(imshape[0],imshape[1],imshape[2],imshape[3]) 

    S=arr[0:volnos/2].copy()    
   
    del arr,bvals,bvecs

    A=sc.zeros((volnos/2-1,6))
        
    FA=sc.zeros(volshape,dtype='float32')
    msk=sc.zeros(volshape,dtype='float32')    

    S0=S[0]
    
    msk[S0<mskthr]=1    

    S[S<1.0]=1.0
    S=sc.log(S)
    S=S[0]-S[:]
    S=S[1:]
   
    for i in sc.arange(volnos/2-1):

        b=B[i+1]
        g=G[:,i+1]
        
        g1=g[0]; g2=g[1]; g3=g[2]
        
        A[i,0]=g1*g1
        A[i,1]=g2*g2
        A[i,2]=g3*g3
        A[i,3]=2*g1*g2
        A[i,4]=2*g1*g3
        A[i,5]=2*g2*g3
    
        S[i]=S[i]/b
         
    #G2=G[:,1:volnos/2]   
       
    print(G.shape)
    #print(G2.shape)
     
    g0=G[0,1:volnos/2]
    g1=G[1,1:volnos/2]
    g2=G[2,1:volnos/2]
      
    #G2.transpose()
    
    S=S.reshape(volnos/2-1,vollen)        
    FA=FA.reshape(vollen)    
    msk=msk.reshape(vollen)       
  
    print('g0,g1,g2,S')
    print(g0.shape)
    print(g1.shape)
    print(g2.shape)
    print(S.shape)    

    #f = lambda a,g0,g1,g2,c: a[0]*g0**2+ (a[1]*g0+a[2]*g1)**2 + (a[3]*g0+a[4]*g1+a[5]*g2)**2 - c        

    from scipy.optimize import leastsq as nlsq
            
    cnt=0
        
    #for i in sc.arange(vollen):
    for i in sc.arange(190745,190747):

        if msk[i]== 1:

             FA[i]=0      
        
        else:
            
            d,resids,rank,sing=sc.linalg.lstsq(A,S[:,i])       
            
                
            
            #a0=sc.array([1,0,1,0,0,1])         
            #a0=[1,0,1,0,0,1]           
            if(d[2]>0):
                a_5=sc.sqrt(d[2])    
            else:
                a_5=1
                    
            a_4=d[5]/a_5    
            a_3=d[4]/a_5            
            
            if(d[1]-a_4**2)>0:
                a_2=sc.sqrt(d[1]-a_4**2)
            else:
                a_2=1
            
            a_1=(d[3]-a_3*a_4)/a_2
            a_0=d[0]-a_1**2-a_3**2
            
            a0=[a_0,a_1,a_2,a_3,a_4,a_5]
            
            #print('a0')
            #print(a0.shape)
            
            #print(g0.shape)
            #print(g1.shape)
            #print(g2.shape)
            #print(S.shape)
            print('-------')
            print('d=',d)
            print('-------')
            print('a0=',a0)
            res=nlsq(residualsf_positive,a0,args=(g0,g1,g2,S[:,i]),full_output=1)
            #print(res)
            a=res[0]
            print(res)
            #print(a)
            #print(res[1])
            #print(res[2]) 
            #print(a.shape)
            #print('-------')
        
            d11=a[0]**2+a[1]**2+a[3]**2
            d22=a[2]**2+a[4]**2
            d33=a[5]**2
            d12=a[1]*a[2]+a[3]*a[4]
            d13=a[3]*a[5]
            d23=a[4]*a[5]
   
            D=sc.array([[d11,d12,d13],[d12,d22,d23],[d13,d23,d33]])
    
            evals,evecs=sc.linalg.eig(D)
                
            l1=evals[0]; l2=evals[1]; l3=evals[2]

            if l1<0 or l2<0 or l3<0:
                cnt+=1

            FA[i]=sc.sqrt( ( (l1-l2)**2 + (l2-l3)**2 + (l3-l1)**2 )/( 2*(l1**2+l2**2+l3**2) )  )                       
    
    print('Not positive definite ', cnt, ' times.')

    FA=FA.reshape(volshape[0],volshape[1],volshape[2])
    del S,A,msk
    

    return FA




def testtensorfitsimple():
    
    print('Loading bvalues...')

    #bvalsf='/home/eg01/data/bvals'
    bvals=loadbvals(opendialog())

    print('Loading bvecs ...')

    #bvecsf='/home/eg01/data/bvecs'
    bvecs=loadbvecs(opendialog())

    print('Loading data ...')
    arr,voxsz=loadnifti(opendialog())
    
    print('Data loaded.')
    print('Calculating FA...')
    #FA=tensorfitsimple(arr,bvals,bvecs)
    FA=tensorfitsimple(arr,bvals,bvecs)
    
    savenifti('FAsimple.nii.gz',FA,voxsz)
    print('Show Volume')
    viewvol(FA,voxsz)    
    
    return 

def testtensorfitpositive():
    
    print('Loading bvalues...')

    bvalsf='../testdata/bvals'
    bvals=loadbvals(bvalsf)

    print('Loading bvecs ...')

    bvecsf='../testdata/bvecs'
    bvecs=loadbvecs(bvecsf)

    print('Loading data ...')
    dataf='../testdata/data.nii.gz'
    arr,voxsz=loadnifti(dataf)
    
    print(voxsz)
    
    print('Data loaded.')
    print('Calculating FA...')
    
    FA=tensorfitpositive(arr,bvals,bvecs)
    
    #rint('Show Volume')
    #viewvol(FA,voxsz)    
    
    return 
    
def picklesaveFA(fname,FA,voxsz=(1.0,1.0,1.0)):

    try:
        data={'FA':FA,'voxsz':voxsz}
        output = open(fname, 'wb')
        pickle.dump(data,output)
        output.close()
    except MemoryError:
        print('Not enough memory.')

def pickleloadFA(fname):
    
    input=open(fname,'rb')
    data=pickle.load(input)
    input.close()
    FA=data.values()[0]
    voxsz=data.values()[1]

    return FA,voxsz

def picklesavevol(fname,arr,voxsz=(1.0,1.0,1.0)):

    try:
        data={'broken':0,'arr':arr,'voxsz':voxsz,'ba':arr.shape[0]}
        output = open(fname, 'wb')
        pickle.dump(data,output)    
        output.close()

    except MemoryError:
        print('Not enough memory. Breaking the array in parts... ')
        ba=arr.shape[0]
        for i in range(ba):
            data={'broken':1,'arr':arr[i],'voxsz':voxsz ,'ba':ba}
            ending='%(#)05d' %{"#": i}
            output = open(fname+'.'+ending, 'wb')
            pickle.dump(data,output)    
            output.close()
        print('Files ready.')


def pickleloadvol(fname):    

    input=open(fname,'rb')
    data=pickle.load(input)
    input.close()
       
    broken=data.values()[0]
    arr=data.values()[1]
    voxsz=data.values()[2]
    ba=data.values()[3] 

    if broken==1:        
        
        print('More than one files need to be loaded...')
        del data
        
        fname2=fname.rsplit('.',1)[0]
        
        ar=sc.zeros((ba,arr.shape[0],arr.shape[1],arr.shape[2]),dtype='float32')
        del arr
        

        for i in range(ba):

            ending='%(#)05d' %{"#": i}
            input=open(fname2+'.'+ending,'rb')
            datatmp=pickle.load(input)
            input.close()
            ar[i]=datatmp.values()[1]
            del datatmp
            
        print('All files loaded.')
        return ar,voxsz

    else:             
               
        del data    
        return arr,voxsz


def testviewbs():

    print('Select bvals file.')
    #bvalsf='/home/eg309/Devel/tractarian/devel/testdata/bvals'   
    #bvalsf=opendialog()
    bvalsf='../testdata/bvals'
    bvals=loadbvals(bvalsf)

    print('Select bvecs file.')
    #bvecsf='/home/eg309/Devel/tractarian/devel/testdata/bvecs'
    #bvecsf=opendialog()
    bvecsf='../testdata/bvecs'
    bvecs=loadbvecs(bvecsf)
    

    D=sc.diag(bvals)
            
    bvecs=bvecs.transpose()/100
    
    print(D.shape)
    
    bvecs2=sc.dot(D,bvecs)
    
    print(bvals[1:64].min())
    print(bvals.shape)
    print(bvecs.shape)
    viewscatter(bvecs2)    

    
def testviewdiffusionsignal():
    
    print('Select bvals file.')
    #bvalsf='/home/eg309/Devel/tractarian/devel/testdata/bvals'   
    #bvalsf=opendialog()
    bvalsf='../testdata/bvals'
    bvals=loadbvals(bvalsf)

    print('Select bvecs file.')
    #bvecsf='/home/eg309/Devel/tractarian/devel/testdata/bvecs'
    #bvecsf=opendialog()
    bvecsf='../testdata/bvecs'
    bvecs=loadbvecs(bvecsf)
    
    D=sc.diag(bvals)
            
    bvecs=bvecs.transpose()
    print(D.shape)
    
    bvecs2=sc.dot(D,bvecs)
    
    print(bvals[1:64].min())
    print(bvals.shape)
    print(bvecs.shape)
    
    
    arr,voxsz = loadnifti('../testdata/data.nii.gz')
    print(arr.shape)

    #S=arr[:,24,64,64]

    S=arr[:,11,82,26]

    #print(S.shape)
    #print(S.max())
    #print(S.min())

    DS=sc.diag(S)
    
    bvecs3=sc.dot(DS,bvecs)

    bvecs3=bvecs3/float(S.max())
    
    bvecs3=10*bvecs3
    
    viewscatter(bvecs3[1:64,:])
    #viewscatter(bvecs)





if __name__ == "__main__":    
    

    #testtensorfitpositive()    
    #viewcone()
    #viewscatter()
    #testviewbs()
    testvol()
    #testviewdiffusionsignal()
    #opendialog()
