import numpy as np
import matplotlib.pyplot as plt  
from dipy.viz import fvtk
from dipy.core.geometry import sphere2cart
import nipy
from dipy.data import get_data

import nibabel as nib
from nibabel.dicom import dicomreaders as dcm
from dipy.utils.spheremakers import sphere_vf_from

parameters={'101_32':['/home/ian/Data/PROC_MR10032/101_32',
                    '1312211075232351192010121313490254679236085ep2dadvdiffDSI10125x25x25STs004a001','.bval','.bvec','.nii'],
            '118_32':['/home/ian/Data/PROC_MR10032/118_32',\
                      '131221107523235119201012131413348979887031ep2dadvdiffDTI25x25x25STEAMs011a001','.bval','.bvec','.nii'],
            '64_32':['/home/ian/Data/PROC_MR10032/64_32',\
                     '1312211075232351192010121314035338138564502CBUDTI64InLea2x2x2s005a001','.bval','.bvec','.nii']}
       

def get_data(name='101_32'):    
    bvals=np.loadtxt(parameters[name][0]+'/'+parameters[name][1]+parameters[name][2])
    bvecs=np.loadtxt(parameters[name][0]+'/'+parameters[name][1]+parameters[name][3]).T
    img=nib.load(parameters[name][0]+'/'+parameters[name][1]+parameters[name][4])     
    return img.get_data(),bvals,bvecs


#siem64 =  nipy.load_image('/home/ian/Devel/dipy/dipy/core/tests/data/small_64D.gradients.npy')

data102,affine102,bvals102,dsi102=dcm.read_mosaic_dir('/home/ian/Data/Frank_Eleftherios/frank/20100511_m030y_cbu100624/08_ep2d_advdiff_101dir_DSI')

bvals102=bvals102.real
dsi102=dsi102.real

v362,f362 = sphere_vf_from('symmetric362')
v642,f642 = sphere_vf_from('symmetric642')

d = 0.0015
S0 = 100
f = [.33,.33,.33]
#f = [1.,0.]

b = 1200

#needles = np.array([-np.pi/4,np.pi/4])
needles2d = np.array([0,np.pi/4.,np.pi/2.])

angles2d = np.linspace(-np.pi, np.pi,100)

def signal2d(S0,f,b,d,needles2d,angles2d):
    return S0*np.array([(1-f[0]-f[1]-f[2])*np.exp(-b*d) + \
                      f[0]*np.exp(-b*d*(np.cos(needles2d[0]-gradient))**2) + \
                      f[1]*np.exp(-b*d*(np.cos(needles2d[1]-gradient))**2) + \
                      f[2]*np.exp(-b*d*(np.cos(needles2d[2]-gradient))**2) \
                      for gradient in angles2d])


#needles3d = [sphere2cart(1,0,0),sphere2cart(1,np.pi/2.,0),sphere2cart(1,np.pi/2.,np.pi/2.)]
needles3d = np.array([sphere2cart(1,0,0),sphere2cart(1,np.pi/2.,0),sphere2cart(1,np.pi/2.,np.pi/2.)])

angles2d = np.linspace(-np.pi, np.pi,100)

def signal3d(S0,f,b0,d,needles3d,bvals,gradients):
    S = S0*np.array([(1-f[0]-f[1]-f[2])*np.exp(-b0*d) + \
                      f[0]*np.exp(-bvals[i]*d*(np.dot(needles3d[0],gradient))**2) + \
                      f[1]*np.exp(-bvals[i]*d*(np.dot(needles3d[1],gradient))**2) + \
                      f[2]*np.exp(-bvals[i]*d*(np.dot(needles3d[2],gradient))**2) \
                      for (i,gradient) in enumerate(gradients)])
    return S, np.dot(np.diag(S),gradients)

#dsi202 = np.row_stack((dsi102[1:,:],-dsi102[1:,:]))
#print dsi202.shape

#print bvals102.shape

#bvals202 = np.column_stack((bvals102[1:],bvals102[1:])).ravel()
#print bvals202.shape

#bvals101 = bvals102[1:]
#dsi101 =dsi102[1:,:]

'''
bvals = bvals102[1:]
b0 = bvals102[0]
gradients = dsi102[1:]

S1, Sg1 = signal3d(S0,[0,1,0],b0,d,needles3d, bvals, gradients)

#S1 = S1.ravel

print Sg1.shape, S1.shape, bvals.shape
#S1=signal3d(S0,[1,0,0],bvals102[0],d,needles3d, bvals202, dsi202)
#S2=signal3d(S0,[0,0,1],bvals102[0],d,needles3d, bvals202, dsi202)

#print S
r = fvtk.ren()

SS1 = -np.log(S1)/bvals
PS1 = np.dot(np.diag(SS1),gradients)
pd0 = -np.log(S0)/b0 
#for (i,s) in enumerate(S):
fvtk.add(r,fvtk.point(100*PS1/pd0,fvtk.cyan,point_radius=0.05,theta=8,phi=8))
fvtk.add(r,fvtk.point(100*-PS1/pd0,fvtk.cyan,point_radius=0.05,theta=8,phi=8))
fvtk.add(r,fvtk.point([[0.,0.,0.]],fvtk.green,point_radius=0.1,theta=8,phi=8))    
fvtk.add(r,fvtk.axes((1,1,1)))
lines = fvtk.line([1.5*np.row_stack((needles3d[0],-needles3d[0])), \
                   1.5*np.row_stack((needles3d[1],-needles3d[1]))], \
                   colors=np.row_stack((fvtk.golden,fvtk.aquamarine)),linewidth=10)
fvtk.add(r,lines)
fvtk.show(r)
'''

'''
f = [0.,0.,1.]
needles2d = np.array([0,np.pi/4.,np.pi/2.])
plt.polar(angles2d,signal2d(S0,f,b,d,needles2d,angles2d),label='1 fibre at 90')
f = [.5,.0,.5]
needles2d = np.array([0,np.pi/4.,np.pi/2.])
plt.polar(angles2d,signal2d(S0,f,b,d,needles2d,angles2d),label='2 fibres at 0 and 90')
f = [.33,.33,.33]
needles2d = np.array([0,np.pi/4.,np.pi/2.])
plt.polar(angles2d,signal2d(S0,f,b,d,needles2d,angles2d),label='3 fibres at 0, 45 and 90')

plt.legend()
plt.title('Signal strength for 1, 2 and 3 fibres for gradients in the fibre plane')

plt.show()
'''

'''
thetas = np.linspace(0,np.pi/2.,6)
f = [.5,.0,.5]
for theta in thetas:
    needles2d = np.array([0,0.,theta])
    plt.polar(angles2d,signal2d(S0,f,b,d,needles2d,angles2d),label='angle '+str(np.rad2deg(theta)))

plt.legend(loc='lower left')
plt.title('Signal strength for 2 fibres for gradient direections in the fibre plane')

plt.show()
'''

def show_signal(r,S,gradients,offset=np.zeros(3)):
    PS = np.dot(np.diag(S),gradients)
#for (i,s) in enumerate(S):
    fvtk.add(r,fvtk.point(offset+PS/np.max(PS),fvtk.cyan,point_radius=0.05,theta=8,phi=8))
    fvtk.add(r,fvtk.point([offset],fvtk.green,point_radius=0.1,theta=8,phi=8))    
    fvtk.add(r,fvtk.axes((1,1,1)))
    lines = fvtk.line([1.5*np.row_stack((needles3d[0],-needles3d[0])), \
                   1.5*np.row_stack((needles3d[1],-needles3d[1]))], \
                   colors=np.row_stack((fvtk.golden,fvtk.aquamarine)),linewidth=10)
    fvtk.add(r,lines)

data118,bvals118,bvecs118=get_data(name='118_32')

#bvals = np.hstack((bvals102[1:],bvals102[1:]))
#b0 = bvals102[0]
#gradients = np.vstack((dsi102[1:],-dsi102[1:]))

bvals = np.hstack((bvals118[1:],bvals118[1:]))
b0 = bvals102[0]
gradients = np.vstack((bvecs118[1:],-bvecs118[1:]))

Signals = []
Needles = []
#for mainfraction in np.linspace():
for theta in np.linspace(0,np.pi/2.,5):
#for theta in [np.pi/2.]:
    needles3d = np.array([sphere2cart(1,0,0),sphere2cart(1,theta,0),sphere2cart(1,np.pi/2.,np.pi/2.)])
    S, Sg = signal3d(S0,[.5,.5,0],b0,d,needles3d, bvals, gradients)
    Signals += [S]
    #S1, Sg1 = signal3d(S0,[0,1,0],b0,d,needles3d, bvals, gradients)
    #Signals += [S]

'''
'r = fvtk.ren()

G = gradients
L = len(Signals)
gap = 1.
for (l,signal) in enumerate(Signals):
    offset=np.array([gap*(2*l-(2*L-1)/2.),0,0])
    show_signal(r,signal,gradients,offset)
    X = np.dot(np.diag(signal),gradients)
    XX = np.dot(X.T,X)
    u,s,v = np.linalg.svd(X)
    U,S,V = np.linalg.svd(X)

fvtk.show(r)
'''
#print S
'''
r = fvtk.ren()
SS1 = -np.log(S1)/bvals
PS1 = np.dot(np.diag(SS1),gradients)
pd0 = -np.log(S0)/b0 
#for (i,s) in enumerate(S):
fvtk.add(r,fvtk.point(100*PS1/pd0,fvtk.cyan,point_radius=0.05,theta=8,phi=8))
fvtk.add(r,fvtk.point(100*-PS1/pd0,fvtk.cyan,point_radius=0.05,theta=8,phi=8))
fvtk.add(r,fvtk.point([[0.,0.,0.]],fvtk.green,point_radius=0.1,theta=8,phi=8))    
fvtk.add(r,fvtk.axes((1,1,1)))
lines = fvtk.line([1.5*np.row_stack((needles3d[0],-needles3d[0])), \
                   1.5*np.row_stack((needles3d[1],-needles3d[1]))], \
                   colors=np.row_stack((fvtk.golden,fvtk.aquamarine)),linewidth=10)
fvtk.add(r,lines)
fvtk.show(r)
'''

'''
Test = Signals[2]
Model =Signals[4]

#Now fit the model Test = scale*Model+iso
c = np.cov(np.vstack((Model,Test)))
scale = c[0,1]/c[0,0]
isotropic = np.mean(Test)-scale*np.mean(Model)
Fitted = scale*Model+isotropic
R2 = c[0,1]**2/(c[0,0]*c[1,1])
r = np.corrcoef(Test,Fitted)[0,1]
print 'scale',scale,'isotropic',isotropic
print R2, r**2
# check that the R-squared equals the squared correlation of Test signal woith Fitted signal
'''

'''
needles2d = np.array([0.,np.pi/2.,0.])
f = [1.,0.,0.]
s00=signal2d(S0,f,b,d,needles2d,angles2d)
f = [0.,1.,0.]
s90=signal2d(S0,f,b,d,needles2d,angles2d)
plt.polar(angles2d,s00,label='fibre at 0')
plt.polar(angles2d,s90,label='fibre at 90')
plt.polar(angles2d,(s00+s90)/2.,label='mixed 0 and 90')
plt.polar(angles2d,(s00-s90)/2.,label='diff fibres 0 and 90')

plt.legend()
plt.title('Sum and difference of fibres')

plt.show()
'''

def subtract(n0,n1):
    plt.figure()
    needles2d = np.array([n0,n1,0.])
    f = [1.,0.,0.]
    sn0=signal2d(S0,f,b,d,needles2d,angles2d)
    f = [0.,1.,0.]
    sn1=signal2d(S0,f,b,d,needles2d,angles2d)
    #plt.polar(angles2d,s00,label='fibre at 0')
    #plt.polar(angles2d,s90,label='fibre at 90')
    plt.subplot(2,4,1,polar=True)
    plt.polar(angles2d,(sn0+sn1)/2.,label='mix')
    lines, labels = plt.rgrids( (20, 40, 60) )
    #plt.polar(angles2d,(s00-s90)/2.,label='diff fibres 0 and 90')
    for (i,theta) in enumerate(np.linspace(-np.pi/2,np.pi/2,7)):
        plt.subplot(2,4,i+2,polar=True)
        needles2d = np.array([0.,theta,0.])
        f = [0.,1.,0.]
        stheta=signal2d(S0,f,b,d,needles2d,angles2d)
        plt.polar(angles2d,(sn0+sn1)/2.-stheta,label=str(-180*theta/np.pi))
        lines, labels = plt.rgrids( (20, 40, 60) )
        plt.legend(loc='best')
    plt.suptitle('Subtraction from '+str(180*n0/np.pi)+'+'+str(180*n1/np.pi))

subtract(0,1.*np.pi/16.)
subtract(0,2.*np.pi/16.)
subtract(0,3.*np.pi/16.)
subtract(0,4.*np.pi/16.)
subtract(0,5.*np.pi/16.)
subtract(0,6.*np.pi/16.)
subtract(0,7.*np.pi/16.)
subtract(0,8.*np.pi/16.)

plt.show()
