#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Lights is a visualization and gui library written in python
#Authors: Eleftherios Garyfallidis

'''
Starting Remarks

For visualization import lights.py
For the moment just write your functions inside lights.py under ian_???
 So, for plotting things available are:
    a) scatterplot
    b) surfaceplot
    c) gridaxes
   Some primitives are: line, arrow, point, label, plane
General idea
    a) AppThread(ren)
       where ren is a vtk.vtkRenderer()
    b) Add actors to the renderer
       where actors are objects with surface, position, orientation info in the 3D space.
       ren.AddActor(act)
    c) The primitives always return actors or assemblies of actors
    See testline for example
    d) The plot functions like surfaceplot return a new renderer if not given as input or the same renderer
    updated if it given as input
    
Testplot has an example of animation by moving the camera.
'''


try:
    import vtk   
    from vtk.util.colors import *
except:
    print('VTK is not installed.')
    
try:
    import wx
except:
    print('Wx is not installed.')

try:
    import wxVTKRenderWindowInteractor as wxvi
except:
    print('No wxVTKRenderWindowInteractor.')

try:
    import numpy as np    
except:
    print('Numpy is not installed.')
    
try:
    import scipy as sp
except:
    print('Scipy is not installed.')


import platform
import sys
import threading
import time


ID_ABOUT = 101
ID_EXIT  = 102

class shared():
    
    def __init__(self):
        self.widget=None
        self.ren=None
        self.ren_list=[]
        
sh=shared()

class Interactor(wxvi.wxVTKRenderWindowInteractor):
    
    def __init__(self, parent, ID, *args, **kw):
        
        self.parent =parent
        wxvi.wxVTKRenderWindowInteractor.__init__(self,parent,ID, *args,**kw)
        self.picker = vtk.vtkCellPicker()   
        
        self.picker.AddObserver("EndPickEvent", self.annotatePick)    
        self.AddObserver("ExitEvent",self.exitEvent)
    
        self.SetPicker(self.picker)
        
    def annotatePick(object, event):        
        
        if picker.GetCellId() < 0:
            print('No object')
            print(np.round(picker.GetSelectionPoint(), decimals=2))
        else:
            print('Object Found')
            print(np.round(picker.GetSelectionPoint(), decimals=2))
            print(np.round(picker.GetPickPosition(), decimals=2))            
            print(np.round(picker.GetActors().GetLastActor().GetPosition(), decimals=2))
        
        
    def exitEvent(object,event,f):    
        #self.parent.Close()
        pass
        
    
class FrameMenu(wx.Frame):
    
    
    
    
    
    def __init__(self, parent, ID, title,width,height):
                
        wx.Frame.__init__(self, parent, ID, title,
                         wx.DefaultPosition, wx.Size(width, height))
        self.CreateStatusBar()
        self.SetStatusText("This is the statusbar")

        menu = wx.Menu()
        menu.Append(ID_ABOUT, "&About",
                    "More information about this program")
        menu.AppendSeparator()
        menu.Append(ID_EXIT, "E&xit", "Terminate the program")

        menuBar = wx.MenuBar()
        menuBar.Append(menu, "&File");

        self.SetMenuBar(menuBar)

        wx.EVT_MENU(self, ID_ABOUT, self.OnAbout)
        wx.EVT_MENU(self, ID_EXIT,  self.TimeToQuit)

    def OnAbout(self, event):
        dlg = wx.MessageDialog(self, "This sample program shows off\n"
                              "frames, menus, statusbars, and this\n"
                              "message dialog.",
                              "About Me", wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def TimeToQuit(self, event):
        self.Close(1)

def NewKeyPress(obj,event):
            print('Inside NewKeyPress')
            
            key = obj.GetKeySym()
            if key == "o":
                print('o')
                #obj.InvokeEvent("DeleteAllObjects")
                #sys.exit()
            elif key == "s":
                print('s')
                #ren.GetActiveCamera().Azimuth(40)
                #sh.widget.Enable(1)
                
            elif key =="t":
                print('t')
                #ren.GetActiveCamera().Azimuth(-40)
                #sh.widget.Enable(1)

class FrameVtkMinimal(wx.Frame):
    
    def __init__(self, parent, ID,ren, title,width,height):
        
        wx.Frame.__init__(self, parent, ID, title,wx.DefaultPosition, wx.Size(width, height))
        widget = wxvi.wxVTKRenderWindowInteractor(self, -1)    
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(widget, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Layout()
    
        #widget.SetInteractorStyle(None)
        widget.Enable(1)
        widget.AddObserver("ExitEvent", lambda o,e,f=self: f.Close()) 
                        
        #widget._Iren.AddObserver("NewKeyPressEvent",NewKeyPress)
        
        #widget.AddObserver("KeyPressEvent", Keypress)        
        
        #widget.Enable(1)
        
        widget.GetRenderWindow().AddRenderer(ren)

        sh.widget=widget
        
    
                
     
    '''
        wx.Frame.__init__(self, parent, ID, title,wx.DefaultPosition, wx.Size(width, height))
        widget = Interactor(self,-1)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(widget, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Layout()
        widget.Enable(1)
        widget.GetRenderWindow().AddRenderer(ren)
    '''

def FrameSelector(frame_type,parent,ID,ren,title,width,height):
        '''
        frame_type 0: FrameVtkMinimal
        frame_type 1: FrameMenu
        '''
        if frame_type==0:
            frame=FrameVtkMinimal(parent,ID,ren,title,width,height)
            return frame
        else:
            frame=FrameMenu(parent,ID, title,width,height)
            return frame
         

class FrameThread(threading.Thread):
    
    def __init__(self,frame_type,title="Second Frame",width=300,height=300,autoStart=1,ren=None):

        threading.Thread.__init__(self)
        
        #self.setDaemon(1)
        
        self.start_orig = self.start
        self.start = self.start_local
        self.frame = None
        self.frame_type=frame_type
        self.ren = ren
        self.title = title
        self.width=width
        self.height=height
        
        self.lock = threading.Lock()
        self.lock.acquire()
        
        if autoStart:
            self.start()
        
    def run(self):
        
        print('Inside run')
               
        frame=FrameSelector(self.frame_type,None,-1,self.ren,self.title,self.width,self.height)                
        frame.Show(1)    
            
        self.frame = frame
        self.lock.release()           
        
    def start_local(self):
        
        self.start_orig()
        self.lock.acquire()
    
    

class AppThread(threading.Thread):
    
    def __init__(self,frame_type=0,title="Lights",width=300,height=300,autoStart=1,ren=None):

        threading.Thread.__init__(self)
        
        #self.setDaemon(1)
        
        self.start_orig = self.start
        self.start = self.start_local
        self.frame = None
        self.frame_type=frame_type
        self.ren = ren
        self.title = title
        self.width=width
        self.height=height
        
        self.lock = threading.Lock()
        self.lock.acquire()
        
        if autoStart:
            self.start()
        
    def run(self):
        
        print('Inside run')
        
        app = wx.PySimpleApp()                          
        
        frame=FrameSelector(self.frame_type,None,-1,self.ren,self.title,self.width,self.height)    
            
        frame.Show(1)    
            
        self.frame = frame
        self.lock.release()
            
        print('Before Mainloop')
        app.MainLoop()
        print('After Mainloop')
        
    def start_local(self):
        
        self.start_orig()
        self.lock.acquire()
    

class NoAppThread():
        
    def __init__(self,frame_type,title="Lights",width=300,height=300,autoStart=1,ren=None):
        
        self.ren=ren
        self.title=title
        self.width=width
        self.height=height
        self.frame_type=frame_type
                        
        app = wx.PySimpleApp()                          
        
        frame=FrameSelector(self.frame_type,None,-1,self.ren,self.title,self.width,self.height)    
            
        frame.Show(1)    
            
        self.frame = frame                  
               
        
        print('NoAppThread - MainLoop Started...')
        app.MainLoop()
        print('NoAppThread - MainLoop Ended.')
        
        '''
         def runVideo(vidSeq):
            """The video sequence function, vidSeq, is run on a separate
                thread to update the GUI. The vidSeq should have one argument for
            SetData"""
 
        app = wx.PySimpleApp()
        frame = ImageFrame(None)
        frame.SetSize((800, 600))
        frame.Show(True)
 
        myImageIn = ImageIn(frame.window)
        t = threading.Thread(target=vidSeq, args=(myImageIn.SetData,))
        t.setDaemon(1)
        t.start()
 
        app.MainLoop()
        '''
        
        
def renderer():
    
    return vtk.vtkRenderer()        
        
def cone(position=(0,0,0),height=1):
    
    cone = vtk.vtkConeSource() 
    cone.SetHeight(height)
    
    conem = vtk.vtkPolyDataMapper()
    conem.SetInput(cone.GetOutput())    
    conea = vtk.vtkActor()
    conea.SetMapper(conem)
    conea.SetPosition(position)
       
    return conea

def triangulate(points,color=(1,0,0),opacity=1):
    
    '''
    Input: a numpy array points with size Nx3 whith the x,y,z positions of the vertices
    Output: an actor with the 3d triangulation result
    
    '''
    
    print('points.shape',points.shape[0])
    
    if points.shape[0]<5:
        print('Warning:Possibly not enough points are given...')
    
    max=points.max(axis=0)
    min=points.min(axis=0)
        
    print(max)
    print(min)    
      
    math = vtk.vtkMath()
    pointsv = vtk.vtkPoints()

    cnt=0
    # *vec
    #for cnt, vec in enumerate(points)
    #   p.InsertPoint(cnt, *vec)
    
    for vec in points:
        pointsv.InsertPoint(cnt,vec[0],vec[1],vec[2])
        cnt+=1
            
        
    profile = vtk.vtkPolyData()
    profile.SetPoints(pointsv)

    # Delaunay3D is used to triangulate the points. The Tolerance is the
    # distance that nearly coincident points are merged
    # together. (Delaunay does better if points are well spaced.) The
    # alpha value is the radius of circumcircles, circumspheres. Any mesh
    # entity whose circumcircle is smaller than this value is output.
    delny = vtk.vtkDelaunay3D()
    delny.SetInput(profile)
    #below options for vtkDelaunay2D    
    #delny.SetTolerance(0.01)
    #delny.SetAlpha(1)
    #delny.BoundingTriangulationOff()

    # Shrink the result to help see it better.
    '''
    shrink = vtk.vtkShrinkFilter()
    shrink.SetInputConnection(delny.GetOutputPort())
    shrink.SetShrinkFactor(0.9)
    '''
    map = vtk.vtkDataSetMapper()
    
    #map.SetInputConnection(shrink.GetOutputPort())
    map.SetInputConnection(delny.GetOutputPort())

    triangulation = vtk.vtkActor()
    triangulation.SetMapper(map)
    triangulation.GetProperty().SetColor(color)
    triangulation.GetProperty().SetOpacity(opacity)
    
    return triangulation



def torus(position=(0,0,0),scalex=1,scaley=1,scalez=1,toruson=1,thickness=0.01):

    torus = vtk.vtkSuperquadricSource()
    #tube.SetCenter(position)    
    torus.SetThickness(thickness)
    torus.SetScale(scalex,scaley,scalez)

    torus.SetPhiResolution(100)
    torus.SetThetaResolution(100)

    torus.SetPhiRoundness(1)
    torus.SetThetaRoundness(1)

    if toruson:
        torus.ToroidalOn()

    torusm = vtk.vtkPolyDataMapper()
    torusm.SetInput(torus.GetOutput())    
    torusa = vtk.vtkActor()
    torusa.SetMapper(torusm)
    torusa.SetPosition(position)
    
    return torusa



def tube(point1=(0,0,0),point2=(1,0,0),color=(1,0,0),opacity=1,radius=0.1,capson=1,specular=1,sides=8):
    
    '''
    Wrap a tube around a line connecting point1 with point2 with a specific radius
    
    '''

           
    points = vtk.vtkPoints()    
    points.InsertPoint(0,point1[0],point1[1],point1[2])    
    points.InsertPoint(1,point2[0],point2[1],point2[2])
    
    lines=vtk.vtkCellArray()
    lines.InsertNextCell(2)
    
    lines.InsertCellPoint(0)
    lines.InsertCellPoint(1)
    
    profileData=vtk.vtkPolyData()
    profileData.SetPoints(points)
    profileData.SetLines(lines)
    
    # Add thickness to the resulting line.
    profileTubes = vtk.vtkTubeFilter()
    profileTubes.SetNumberOfSides(sides)
    profileTubes.SetInput(profileData)
    profileTubes.SetRadius(radius)

    if capson:
        profileTubes.SetCapping(1)
    else:
        profileTubes.SetCapping(0)

    profileMapper = vtk.vtkPolyDataMapper()
    profileMapper.SetInputConnection(profileTubes.GetOutputPort())

    profile = vtk.vtkActor()
    profile.SetMapper(profileMapper)
    profile.GetProperty().SetDiffuseColor(color)
    profile.GetProperty().SetSpecular(specular)
    profile.GetProperty().SetSpecularPower(30)
    profile.GetProperty().SetOpacity(opacity)

    return profile



def pipeplot(ren,x,y,z,u,v,w,r,colr,colg,colb,opacity,texton=1):
    
    '''
    
    Create a pipe or many pipes connecting [x_i,y_i,z_i] with [u_i,v_i,w_i] and radius r_i,
    
    See testpipeplot for an example.
    '''

    try:
        no_points=len(u)
    except:
        no_points=1
        
        
    ass=vtk.vtkPropAssembly()    
    ass2=vtk.vtkPropAssembly()    
    
    if no_points>1:
    
        maxr=np.max(r)
        
        for i in xrange(no_points):
        
            point1=np.array([x[i],y[i],z[i]])
            point2=np.array([u[i],v[i],w[i]])
            
            color=(colr[i],colg[i],colb[i])
            opac=opacity[i]
            #if r[i]>rthr:
            #
            #if r[i]==maxr:
                    
            ass.AddPart(tube(point1,point2,color=color,opacity=opac,radius=r[i],capson=1) )
            #ass.AddPart(text3d(text=str(r),pos=(point2[0],point2[1],point2[2]),scale=(0.01,0.01,0.01)))

            if texton:
                ass2.AddPart(label(ren=ren,text=str(r[i]),pos=(point2[0],point2[1],point2[2]),scale=(0.05,0.05,0.05)))
                ass2.AddPart(label(ren=ren,text=str(r[i]),pos=(point1[0],point1[1],point1[2]),scale=(0.05,0.05,0.05)))

    else:
    
            point1=np.array([x,y,z])
            point2=np.array([u,v,w])            
                       
            color=(colr,colg,colb)
            opac=opacity
            ass.AddPart(tube(point1,point2,color=color,opacity=opac,radius=r,capson=1) )

            if texton:
                ass2.AddPart(label(ren=ren,text=str(r),pos=(point2[0],point2[1],point2[2]),scale=(0.05,0.05,0.05)))
                ass2.AddPart(label(ren=ren,text=str(r),pos=(point1[0],point1[1],point1[2]),scale=(0.05,0.05,0.05)))
        
    ren.AddActor(ass)
    
    return
    



def cube(position=(0,0,0),X=1,Y=1,Z=1):
    
    cube = vtk.vtkCubeSource() 
    
    cube.SetXLength(X)
    cube.SetYLength(Y)
    cube.SetZLength(Z)

    cubem = vtk.vtkPolyDataMapper()
    
    cubem.SetInput(cube.GetOutput())    
    cubea = vtk.vtkActor()
    cubea.SetMapper(cubem)
    cubea.SetPosition(position)
       
    return cubea

def cube(Xmin=-1,Xmax=1,Ymin=-1,Ymax=1,Zmin=-1,Zmax=1,color=(0,0.2,1),opacity=0.4):
    
    cube = vtk.vtkCubeSource() 
    cube.SetBounds(Xmin,Xmax,Ymin,Ymax,Zmin,Zmax)
    
    #cube.SetXLength(X)
    #cube.SetYLength(Y)
    #cube.SetZLength(Z)

    cubem = vtk.vtkPolyDataMapper()
    
    cubem.SetInput(cube.GetOutput())    
    cubea = vtk.vtkActor()
    cubea.SetMapper(cubem)
    #cubea.SetPosition(position)
    cubea.GetProperty().SetColor(color)   
    cubea.GetProperty().SetOpacity(opacity)   
    return cubea

def point(position=(0,0,0),radius=0.1,thetares=8,phires=8,color=(0,0,1),opacity=1):
    '''
    Adds a spherical point
    '''
        
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

def dots(points,color=(1,0,0),opacity=1):
  '''
  Adds one or more 3d dots(pixels) returns one actor handling all the points
  '''

  points_no=len(points)

  #print points_no

  polyVertexPoints = vtk.vtkPoints()
  polyVertexPoints.SetNumberOfPoints(points_no)
  aPolyVertex = vtk.vtkPolyVertex()
  aPolyVertex.GetPointIds().SetNumberOfIds(points_no)

  cnt=0
  for point in points:
    polyVertexPoints.InsertPoint(cnt, point[0], point[1], point[2])
    aPolyVertex.GetPointIds().SetId(cnt, cnt)
    cnt+=1

  aPolyVertexGrid = vtk.vtkUnstructuredGrid()
  aPolyVertexGrid.Allocate(1, 1)
  aPolyVertexGrid.InsertNextCell(aPolyVertex.GetCellType(), aPolyVertex.GetPointIds())

  aPolyVertexGrid.SetPoints(polyVertexPoints)
  aPolyVertexMapper = vtk.vtkDataSetMapper()
  aPolyVertexMapper.SetInput(aPolyVertexGrid)
  aPolyVertexActor = vtk.vtkActor()
  aPolyVertexActor.SetMapper(aPolyVertexMapper)

  aPolyVertexActor.GetProperty().SetColor(color)
  aPolyVertexActor.GetProperty().SetOpacity(opacity)

  #ren=vtk.vtkRenderer()
  #ren.AddActor(aPixelActor)
  #ren.AddActor(aPolyVertexActor)
  #ap=AppThread(ren=ren)

  #del polyVertexPoints
  #del aPolyVertex
  #del aPolyVertexGrid
  #del aPolyVertexMapper

  return aPolyVertexActor

def sphere(position=(0,0,0),radius=0.5,thetares=8,phires=8,color=(0,0,1),opacity=1,tessel=0):
    
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetLatLongTessellation(tessel)
   
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

def ellipsoid(R=sp.array([[2, 0, 0],[0, 1, 0],[0, 0, 1] ]),position=(0,0,0),thetares=20,phires=20,color=(0,0,1),opacity=1,tessel=0):

       
    
    '''
    Stretch a unit sphere to make it an ellipsoid under a 3x3 translation matrix R 
    
    R=sp.array([[2, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1] ])
    '''
    
    Mat=sp.identity(4)
    Mat[0:3,0:3]=R
       
    '''
    Mat=sp.array([[2, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0,  1]  ])
    '''
    mat=vtk.vtkMatrix4x4()
    
    for i in sp.ndindex(4,4):
        
        mat.SetElement(i[0],i[1],Mat[i])
    
    radius=1
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetLatLongTessellation(tessel)
   
    sphere.SetThetaResolution(thetares)
    sphere.SetPhiResolution(phires)
    
    trans=vtk.vtkTransform()
    
    trans.Identity()
    #trans.Scale(0.3,0.9,0.2)
    trans.SetMatrix(mat)
    trans.Update()
    
    transf=vtk.vtkTransformPolyDataFilter()
    transf.SetTransform(trans)
    transf.SetInput(sphere.GetOutput())
    transf.Update()
    
    spherem = vtk.vtkPolyDataMapper()
    spherem.SetInput(transf.GetOutput())
    
    spherea = vtk.vtkActor()
    spherea.SetMapper(spherem)
    spherea.SetPosition(position)
    spherea.GetProperty().SetColor(color)
    spherea.GetProperty().SetOpacity(opacity)
    #spherea.GetProperty().SetRepresentationToWireframe()
    
    return spherea

def line(point1=(0,0,0),point2=(1,0,0),color=(1,0,0),res=1,opacity=1,return_polydata=0):

    '''
    Deprecated
    '''
    
    line=vtk.vtkLineSource()
    line.SetResolution(res)
    line.SetPoint1(point1)
    line.SetPoint2(point2)
    line.Update()
    
    if return_polydata:
        return line.GetOutput()
    else:
        linem = vtk.vtkPolyDataMapper()
        linem.SetInput(line.GetOutput())
        linea = vtk.vtkActor()
        linea.SetMapper(linem)
        linea.GetProperty().SetColor(color)
        linea.GetProperty().SetOpacity(opacity)
        
        return linea

def polyline(points,color=(1,0,0),opacity=1):

  #'''
  points_no=len(points)
  #print points_no

  polyLinePoints = vtk.vtkPoints()
  polyLinePoints.SetNumberOfPoints(points_no)

  aPolyLine = vtk.vtkPolyLine()
  aPolyLine.GetPointIds().SetNumberOfIds(points_no)
 
  cnt=0

  for point in points:
    #print point
    polyLinePoints.InsertPoint(cnt, point[0], point[1], point[2])
    aPolyLine.GetPointIds().SetId(cnt, cnt)
      
    cnt+=1

  aPolyLineGrid = vtk.vtkUnstructuredGrid()
  aPolyLineGrid.Allocate(1, 1)

  aPolyLineGrid.InsertNextCell(aPolyLine.GetCellType(), aPolyLine.GetPointIds())
  aPolyLineGrid.SetPoints(polyLinePoints)

  aPolyLineMapper = vtk.vtkDataSetMapper()
  aPolyLineMapper.SetInput(aPolyLineGrid)

  aPolyLineActor = vtk.vtkActor()
  aPolyLineActor.SetMapper(aPolyLineMapper)

  aPolyLineActor.GetProperty().SetColor(color)
  aPolyLineActor.GetProperty().SetOpacity(opacity)

  #aPolyLineActor.AddPosition(2, 0, 4)
  #aPolyLineActor.GetProperty().SetDiffuseColor(1, 1, 1)

  return aPolyLineActor



def trajectories(trajs,traj_thr=0,color=(1,0,0),opacity=1,res=1):

    '''
    trajs is a list of list of tuples with 3 elements (x,y,z coordinates) i.e. a list of trajectories
    returns only one actor handling all trajectories
    '''

    trajs_no=len(trajs)
    cnt = 0

    apd=vtk.vtkAppendPolyData()
    

    for traj in trajs:
        traj_no=len(traj)    
        
        if traj_no > traj_thr:
            
            for i in xrange(traj_no-1):
  
                         
                line=vtk.vtkLineSource()    
                line.SetResolution(res)
                line.SetPoint1(traj[i])
                line.SetPoint2(traj[i+1])
                #line.Update()
                apd.AddInput(line.GetOutput())
                #apd.AddInput(line(point1=traj[i],point2=traj[i+1],return_polydata=1))
                del line
            
            cnt+=1
            if cnt % 1000 ==0:
                print round(100*cnt / float(trajs_no)) , '%'   
    
    
    map = vtk.vtkPolyDataMapper()
    map.SetInput(apd.GetOutput())        
    
    trajsActor = vtk.vtkActor()
    trajsActor.SetMapper(map)

    trajsActor.GetProperty().SetColor(color)
    trajsActor.GetProperty().SetOpacity(opacity)

    return trajsActor 



    
def trajectories_fast():
    
    points= vtk.vtkPoints()
    lines=vtk.vtkCellArray()
    linescalars=vtk.vtkFloatArray()
   
    lookuptable=vtk.vtkLookupTable()
    

    scalar=1.0
    curPointID=0
    scalarmin=0.0
    scalarmax=1.0
    
    linewidth=1
    
    #Plot line
    m=(0.0,0.0,0.0)
    n=(1.0,0.0,0.0)
    
    for i in xrange(6000000):
        
        m=sp.rand(3)
        n=sp.rand(3)
        
        scalar=sp.rand(1)
        
        linescalars.SetNumberOfComponents(1)
        points.InsertNextPoint(m)
        linescalars.InsertNextTuple1(scalar)
       
        points.InsertNextPoint(n)
        linescalars.InsertNextTuple1(scalar)
        
        lines.InsertNextCell(2)
        lines.InsertCellPoint(curPointID)
        lines.InsertCellPoint(curPointID+1)
        
        curPointID+=2
  
    
    ##

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetPointData().SetScalars(linescalars)
    
    ##
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(polydata)
    mapper.SetLookupTable(lookuptable)
    
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarRange(scalarmin,scalarmax)
    mapper.SetScalarModeToUsePointData()
    
    actor=vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(linewidth)
    
    return actor

def trajectories_fast(traj):
    
    points= vtk.vtkPoints()
    lines=vtk.vtkCellArray()
    linescalars=vtk.vtkFloatArray()
   
    lookuptable=vtk.vtkLookupTable()
    

    scalar=1.0
    curPointID=0
    scalarmin=0.0
    scalarmax=1.0
    
    linewidth=1
    
    #Plot line
    m=(0.0,0.0,0.0)
    n=(1.0,0.0,0.0)
    
    '''
    for i in xrange(6000000):
        
        m=sp.rand(3)
        n=sp.rand(3)
    '''
    
    mit=iter(traj)
    nit=iter(traj)
    nit.next()
    
    while(True):
        
        try:
            m=mit.next() 
            n=nit.next()
            
            scalar=sp.rand(1)
            
            linescalars.SetNumberOfComponents(1)
            points.InsertNextPoint(m)
            linescalars.InsertNextTuple1(scalar)
           
            points.InsertNextPoint(n)
            linescalars.InsertNextTuple1(scalar)
            
            lines.InsertNextCell(2)
            lines.InsertCellPoint(curPointID)
            lines.InsertCellPoint(curPointID+1)
            
            curPointID+=2
        except StopIteration:
            print 'Done'
            break
  
    
    ##

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetPointData().SetScalars(linescalars)
    
    ##
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(polydata)
    mapper.SetLookupTable(lookuptable)
    
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarRange(scalarmin,scalarmax)
    mapper.SetScalarModeToUsePointData()
    
    actor=vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(linewidth)
    
    return actor
    

def trajectories_fast(list_traj,list_labels=None,opacity=1,remove0labels=True,color=1):
    
    points= vtk.vtkPoints()
    lines=vtk.vtkCellArray()
    linescalars=vtk.vtkFloatArray()
   
    lookuptable=vtk.vtkLookupTable()
    
    scalar=1.0
    curPointID=0
    scalarmin=0.0
    scalarmax=1.0
    
    linewidth=1
    
    #Plot line
    m=(0.0,0.0,0.0)
    n=(1.0,0.0,0.0)
    
    '''
    for i in xrange(6000000):
        
        m=sp.rand(3)
        n=sp.rand(3)
    '''
    if list_labels!=None:
        
        lit=iter(list_labels)
    
    for traj in list_traj:
        inw=True
        mit=iter(traj)
        nit=iter(traj)
        nit.next()
        if list_labels==None:
            #scalar=sp.rand(1)
            scalar=color
        else:
            scalar=lit.next()
            if scalar==0 and remove0labels==True:
                inw=False   
        
        while(inw):
            
            try:
                m=mit.next() 
                n=nit.next()
                
                #scalar=sp.rand(1)
                
                linescalars.SetNumberOfComponents(1)
                points.InsertNextPoint(m)
                linescalars.InsertNextTuple1(scalar)
               
                points.InsertNextPoint(n)
                linescalars.InsertNextTuple1(scalar)
                
                lines.InsertNextCell(2)
                lines.InsertCellPoint(curPointID)
                lines.InsertCellPoint(curPointID+1)
                
                curPointID+=2
            except StopIteration:
                #print 'Done'
                break
      
    print 'Done'

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetPointData().SetScalars(linescalars)
    
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(polydata)
    mapper.SetLookupTable(lookuptable)
    
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarRange(scalarmin,scalarmax)
    mapper.SetScalarModeToUsePointData()
    
    actor=vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(linewidth)
    actor.GetProperty().SetOpacity(opacity)
    
    return actor
    
def label(ren,text='Origin',pos=(0,0,0),scale=(0.2,0.2,0.2),color=(1,1,1)):
    
    atext=vtk.vtkVectorText()
    atext.SetText(text)
    
    textm=vtk.vtkPolyDataMapper()
    textm.SetInput(atext.GetOutput())
    
    texta=vtk.vtkFollower()
    texta.SetMapper(textm)
    texta.SetScale(scale)    

    texta.GetProperty().SetColor(color)
    texta.SetPosition(pos)
    
    ren.AddActor(texta)
    
    texta.SetCamera(ren.GetActiveCamera())
        
    return texta




def arrow(pos=(0,0,0),color=(1,0,0),scale=(1,1,1),opacity=1):
    
    arrow = vtk.vtkArrowSource()
    #arrow.SetTipLength(length)
    
    arrowm = vtk.vtkPolyDataMapper()
    arrowm.SetInput(arrow.GetOutput())
    
    arrowa= vtk.vtkActor()
    arrowa.SetMapper(arrowm)
    
    arrowa.GetProperty().SetColor(color)
    arrowa.GetProperty().SetOpacity(opacity)
    arrowa.SetScale(scale)
    
    return arrowa

def plane(orig=(0,0,0),point1=(1,0,0),point2=(0,1,0),color=(0.6,0.7,0.6),xresolution=10,yresolution=10,opacity=1):
    
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(orig)
    plane.SetPoint1(point1)
    plane.SetPoint2(point2)
    plane.SetXResolution(xresolution)
    plane.SetYResolution(yresolution)
        
    planem = vtk.vtkPolyDataMapper()
    planem.SetInputConnection(plane.GetOutputPort())
    
    planea =vtk.vtkActor()
    planea.SetMapper(planem)
    planea.GetProperty().SetColor(color)
    planea.GetProperty().SetOpacity(opacity)
    
    return planea

def axes(scale=(1,1,1),colorx=(1,0,0),colory=(0,1,0),colorz=(0,0,1),opacity=1):
    
    arrowx=arrow(color=colorx,scale=scale,opacity=opacity)
    arrowy=arrow(color=colory,scale=scale,opacity=opacity)
    arrowz=arrow(color=colorz,scale=scale,opacity=opacity)
    
    arrowy.RotateZ(90)
    arrowz.RotateY(-90)
   
    #ass=vtk.vtkPropAssembly()
    ass=vtk.vtkAssembly()
    ass.AddPart(arrowx)
    ass.AddPart(arrowy)
    ass.AddPart(arrowz)
           
    return ass


def gridaxes(ren=None,min=(0,0,0),max=(1,1,1),axeson=1,labelsdecimals=2,axeslabelsscale=(0.3,0.3,0.3),gridson=1,gridcolor=(0.2,0.2,0.2),gridslabelson=1,gridlabelsscale=(0.3,0.3,0.3),gridlabeldist=(0.5,0.5,0.3),scaleon=1):
    '''
    Draw grid, axes, put labels.    
    '''    
    if ren==None:
        
        ren=vtk.vtkRenderer()         
        
    print(max)
    print(min)    

    xmax=max[0]
    ymax=max[1]
    zmax=max[2]
    
    xmin=min[0]
    ymin=min[1]
    zmin=min[2]
    
    ren.AddActor(label(ren,text=str(np.round([0,0,0], decimals=labelsdecimals)),pos=(0,0,0)))   
    
    ren.AddActor(line(pos1=(xmin,0,0),pos2=(xmax,0,0),color=(1,0,0)))
    ren.AddActor(line(pos1=(0,ymin,0),pos2=(0,ymax,0),color=(0,1,0)))
    ren.AddActor(line(pos1=(0,0,zmin),pos2=(0,0,zmax),color=(0,0,1)))
    
    ren.AddActor(label(ren,text='x(+)',pos=(xmax,0,0),scale=axeslabelsscale,color=(1,0,0)))    
    ren.AddActor(label(ren,text='x(-)',pos=(xmin,0,0),scale=axeslabelsscale,color=(1,0,0)))    
    ren.AddActor(label(ren,text='y(+)',pos=(0,ymax,0),scale=axeslabelsscale,color=(0,1,0)))    
    ren.AddActor(label(ren,text='y(-)',pos=(0,ymin,0),scale=axeslabelsscale,color=(0,1,0)))    
    ren.AddActor(label(ren,text='z(+)',pos=(0,0,zmax),scale=axeslabelsscale,color=(0,0,1)))    
    ren.AddActor(label(ren,text='z(-)',pos=(0,0,zmin),scale=axeslabelsscale,color=(0,0,1)))    
    
    if gridson:
    
        resolutionx=10.0
        resolutiony=10.0
        resolutionz=10.0
    
        deltax=(xmax-xmin)/resolutionx
        deltay=(ymax-ymin)/resolutiony
        deltaz=(zmax-zmin)/resolutionz

        stepperx=xmin    
        steppery=ymin
        stepperz=zmin
                
        gridlabeldistx=gridlabeldist[0]
        gridlabeldisty=gridlabeldist[1]
        gridlabeldistz=gridlabeldist[2]
        
        while stepperx<= xmax:
            
            ren.AddActor(line(pos1=(stepperx,ymin,zmin),pos2=(stepperx,ymax,zmin),color=gridcolor))        
            ren.AddActor(line(pos1=(stepperx,ymin,zmin),pos2=(stepperx,ymin,zmax),color=gridcolor))        
            if gridslabelson:
                ren.AddActor(label(ren,text=str(np.round(stepperx, decimals=labelsdecimals)),pos=(stepperx,ymin,zmax+gridlabeldistx*deltaz),scale=gridlabelsscale))    
                
            stepperx=stepperx+deltax
        
        while steppery <= ymax:
            
            ren.AddActor(line(pos1=(xmin,steppery,zmin),pos2=(xmax,steppery,zmin),color=gridcolor))
            ren.AddActor(line(pos1=(xmin,steppery,zmin),pos2=(xmin,steppery,zmax),color=gridcolor))        
            if gridslabelson:
                ren.AddActor(label(ren,text=str(np.round(steppery, decimals=labelsdecimals)),pos=(xmin-gridlabeldisty*deltax,steppery,zmax),scale=gridlabelsscale))    
                        
            steppery=steppery+deltay
        
        if zmin!=zmax:
        
            while stepperz <= zmax:
            
                ren.AddActor(line(pos1=(xmin,ymin,stepperz),pos2=(xmin,ymax,stepperz),color=gridcolor))
                ren.AddActor(line(pos1=(xmin,ymin,stepperz),pos2=(xmax,ymin,stepperz),color=gridcolor))
            
                if gridslabelson:
                    ren.AddActor(label(ren,text=str(np.round(stepperz, decimals=labelsdecimals)),pos=(xmax+gridlabeldistz*deltax,ymin,stepperz),scale=gridlabelsscale)) 
                stepperz=stepperz+deltaz
        
   
    if axeson:
        ren.AddActor(axes())      
    
        
                    
    return ren    

def parallelplot():
    '''
    http://en.wikipedia.org/wiki/Parallel_coordinates
    http://www.math.tau.ac.il/~aiisreal/
    http://www.wallinfire.net/picviz
    '''
    
    pass
    
def histogramplot():
    
    pass

def scatterplot(points,ren=None,pointsradii=0.2,axeson=0,labelson=0,labelsdecimals=2,axeslabelsscale=(0.3,0.3,0.3),gridson=1,gridcolor=(0.2,0.2,0.2),gridslabelson=1,gridlabelsscale=(0.3,0.3,0.3),gridlabeldist=(0.5,0.5,0.3),scaleon=1,randomcolor=1,color=(0,0,1)):
    '''
    Scatter plot
    points is a list of 3d vectors i.e. a scipy or numpy array of size Nx3
    
        x1,y1,z1
        x2,y2,z2
        ...
        xn,yn,zn
    '''    
    if ren==None:
        
        ren=vtk.vtkRenderer()    
    
    max=points.max(axis=0)
    min=points.min(axis=0)
        
    print(max)
    print(min)    
    
    if axeson:
        gridaxes(ren,min,max,axeson,labelsdecimals, axeslabelsscale,gridson,gridcolor,gridslabelson,gridlabelsscale,gridlabeldist,scaleon)  
    
    if randomcolor==1:
        for vec in points:
            ren.AddActor(point(position=vec,color=sp.rand(3),radius=pointsradii))
    else:
        for vec in points:
            ren.AddActor(point(position=vec,color=color,radius=pointsradii))
            
    if labelson:
        
        for vec in points:
            ren.AddActor(label(ren,text=str(np.round(vec, decimals=labelsdecimals)),pos=vec))
    
                
    return ren    

def spherepoints(N=100,scale=1,half=0,alg=0):
        
    '''
    Calculate the position of evenly distributed points on the sphere
    
    Adapted from
    http://www.math.niu.edu/~rusin/known-math/97/spherefaq
    
    Distributing many points on a sphere, by E.B. Saff and A.B.J. Kuijlaars, Mathematical
    Intelligence 19.1 (1997)
    
    N is the number of points on the sphere
    
    '''
    points = []    
    if alg==0:

      
        if half==1:
            old_phi = 0
            for k in range(1,N+1):
                h = -1 + 2*(k-1)/float(N-1)
                theta = sp.arccos(h)
                if k==1 or k==N:
                    phi = 0
                else:
               
                    if k==N/2:
                        break;
                    
                    phi = (old_phi + 3.6/sp.sqrt(N*(1-h*h))) % (2*sp.pi)

                points.append((sp.sin(phi)*sp.sin(theta), sp.cos(theta), sp.cos(phi)*sp.sin(theta) ))
                old_phi = phi

    else:
        old_phi = 0
        for k in range(1,N+1):
            h = -1 + 2*(k-1)/float(N-1)
            theta = sp.arccos(h)
            if k==1 or k==N:
                phi = 0
            else:
                phi = (old_phi + 3.6/sp.sqrt(N*(1-h*h))) % (2*sp.pi)

            points.append((sp.sin(phi)*sp.sin(theta), sp.cos(theta), sp.cos(phi)*sp.sin(theta) ))
            old_phi = phi
 
    
    points=sp.array(points)    
    return scale*points
    
    if alg==1:

      R=1
      a=1.2
      #N=500
      x=sp.linspace(-a*R,a*R)
      #[X,Y]=sp.meshgrid(x,x,N)
      [X,Y]=sp.meshgrid(x,x)
      Z = X
      #Fp = sp.where(R**2 -(X**2+Y**2) >=0)
      #Fm = sp.where(R**2 -(X**2+Y**2) <0)
      #Z[Fm]=sp.NaN;
      #Z[Fp]=sp.sqrt(R**2 - (X[Fp]**2+Y[Fp]**2))

      #points=sp.hstack((X,Y,Z))

      print '1.shape'
      print X.shape
      print '1.shape'
      print x.shape

      return scale*points


def boundarywireframe(vertices):

	#get the  boundary wireframe grid return an actor

        cnt=0
        points = vtk.vtkPoints()
    
        for vec in vertices:
            points.InsertPoint(cnt,vec[0],vec[1],vec[2])
            cnt+=1   
        
        profile = vtk.vtkPolyData()
        profile.SetPoints(points)
        
        delny = vtk.vtkDelaunay3D()
        delny.SetInput(profile)
        delny.SetTolerance(0.001)
        delny.SetAlpha(0)
        delny.BoundingTriangulationOff()
        delny.Update()
        #delny.BoundingTriangulationOn()
              
        map = vtk.vtkDataSetMapper()
        #map.SetInputConnection(shrink.GetOutputPort())
        map.SetInputConnection(delny.GetOutputPort())
        
        #From Nabble
        #http://www.nabble.com/Delaunay-surface-triangulation-in-3D-td20034644.html
        
        #Store Delaunay output as PolyData
        usg = vtk.vtkUnstructuredGrid()
        usg = delny.GetOutput()
        
        pd = vtk.vtkPolyData()
        pd.SetPoints(usg.GetPoints())
        pd.SetPolys(usg.GetCells())
        
        
        #Get surface triangles with geometry filter
        geom = vtk.vtkGeometryFilter()
        geom.SetInput(pd)
        geom.Update()                
        
        pd2 = vtk.vtkPolyData()
        pd2 = geom.GetOutput()
        pd2.Update()
        
        npts = pd2.GetNumberOfPoints()
        ncells = pd2.GetNumberOfCells()        
        pts = pd2.GetPoints()        
        polys = pd2.GetPolys()
        id = polys.GetData()       
        
        nfc=id.GetValue(0)
                
        cnt =0
        while cnt < ncells:
            
            vid1=id.GetValue(cnt*(nfc+1)+1)
            vid2=id.GetValue(cnt*(nfc+1)+2)
            vid3=id.GetValue(cnt*(nfc+1)+3)
            vid4=id.GetValue(cnt*(nfc+1)+4)
        
            vx1=pts.GetPoint(vid1)
            vx2=pts.GetPoint(vid2)
            vx3=pts.GetPoint(vid3)
            vx4=pts.GetPoint(vid4)
       
            cnt+=1
                
        triangulation = vtk.vtkActor()
        triangulation.SetMapper(map)
        triangulation.GetProperty().SetColor(1, 1, 1)
        triangulation.GetProperty().SetOpacity(0.5)
        triangulation.GetProperty().SetRepresentationToWireframe()
        
        return triangulation
    

    

def sphericalgrid(vertices,ren=None,spheres_on=1, sphere_radius=0.1,sphere_color=(0,0,1),tubes_on=1,tube_radius=0.1,tube_color=(0,1,0.5),surface_on=1,surface_color=(1,1,1)):
    
    if ren==None:
        
        ren=vtk.vtkRenderer() 
    
    max=vertices.max(axis=0)
    min=vertices.min(axis=0)
   
    if spheres_on :
        for pos in vertices:
            ren.AddActor(sphere(position=pos,radius=sphere_radius,color=sphere_color))
       
    if tubes_on:
                
        pass
        
    if surface_on :
        
        cnt=0
        points = vtk.vtkPoints()
    
        for vec in vertices:
            points.InsertPoint(cnt,vec[0],vec[1],vec[2])
            cnt+=1   
        
        profile = vtk.vtkPolyData()
        profile.SetPoints(points)

        # Delaunay3D is used to triangulate the points. The Tolerance is the
        # distance that nearly coincident points are merged
        # together. (Delaunay does better if points are well spaced.) The
        # alpha value is the radius of circumcircles, circumspheres. Any mesh
        # entity whose circumcircle is smaller than this value is output.        
        
        delny = vtk.vtkDelaunay3D()
        delny.SetInput(profile)
        delny.SetTolerance(0.001)
        delny.SetAlpha(0)
        delny.BoundingTriangulationOff()
        delny.Update()
        #delny.BoundingTriangulationOn()
      
        # Shrink the result to help see it better.
        #shrink = vtk.vtkShrinkFilter()
        #shrink.SetInputConnection(delny.GetOutputPort())
        #shrink.SetShrinkFactor(1)

        map = vtk.vtkDataSetMapper()
        #map.SetInputConnection(shrink.GetOutputPort())
        map.SetInputConnection(delny.GetOutputPort())
        
        #From Nabble
        #http://www.nabble.com/Delaunay-surface-triangulation-in-3D-td20034644.html
        
        #Store Delaunay output as PolyData
        usg = vtk.vtkUnstructuredGrid()
        usg = delny.GetOutput()
        
        pd = vtk.vtkPolyData()
        pd.SetPoints(usg.GetPoints())
        pd.SetPolys(usg.GetCells())
        
        
        #Get surface triangles with geometry filter
        geom = vtk.vtkGeometryFilter()
        geom.SetInput(pd)
        geom.Update()
                
        print 'Writing Boundary triangles ...'
        writer = vtk.vtkPolyDataWriter()
        writer.SetInput(geom.GetOutput())        
        writer.SetFileName("testBound.vtk")
        writer.SetFileTypeToASCII()
        writer.Write()
        print 'End writing'
        
        #Get Points & Cells after  
        geom.Update()
        
        pd2 = vtk.vtkPolyData()
        pd2 = geom.GetOutput()
        pd2.Update()
        
        npts = pd2.GetNumberOfPoints()
        ncells = pd2.GetNumberOfCells()        
        pts = pd2.GetPoints()        
        polys = pd2.GetPolys()
        id = polys.GetData()
        
        print 'id'
        print id.GetValue(0)
        
        nfc=id.GetValue(0)
        
        
        cnt =0
        while cnt < ncells:
            
            vid1=id.GetValue(cnt*(nfc+1)+1)
            vid2=id.GetValue(cnt*(nfc+1)+2)
            vid3=id.GetValue(cnt*(nfc+1)+3)
            vid4=id.GetValue(cnt*(nfc+1)+4)
        
            vx1=pts.GetPoint(vid1)
            vx2=pts.GetPoint(vid2)
            vx3=pts.GetPoint(vid3)
            vx4=pts.GetPoint(vid4)
            
            '''
            ren.AddActor(line(pos1=vx1,pos2=vx2))
            ren.AddActor(line(pos1=vx1,pos2=vx3))
            ren.AddActor(line(pos1=vx1,pos2=vx4))
        
            ren.AddActor(line(pos1=vx2,pos2=vx3))
            ren.AddActor(line(pos1=vx2,pos2=vx4))
            ren.AddActor(line(pos1=vx3,pos2=vx4))
            '''
        
            cnt+=1       
        
        '''
        print 'Number of Points'
        print(pd2.GetNumberOfPoints()) 
                
        print 'Number of Cells'
        print(pd2.GetNumberOfCells()) 
        
        print 'Number of Vertices'
        print(pd2.GetNumberOfVerts()) 
               
        print 'Number of Polys'
        print(pd2.GetNumberOfPolys()) 
        
        print 'Number of Lines'
        print(pd2.GetNumberOfLines()) 
        
        print 'Number of Pieces'
        print(pd2.GetNumberOfPieces()) 
        
        print 'Number of Strips'
        print(pd2.GetNumberOfStrips()) 
        '''
                        
        triangulation = vtk.vtkActor()
        triangulation.SetMapper(map)
        triangulation.GetProperty().SetColor(surface_color)
        triangulation.GetProperty().SetOpacity(0.5)
        triangulation.GetProperty().SetRepresentationToWireframe()
        
        ren.AddActor(triangulation)
        
    return ren

def sphericalgridassembly(vertices,ren=None,spheres_on=1, sphere_radius=0.1,sphere_color=(0,0,1),tubes_on=1,tube_radius=0.1,tube_color=(0,1,0.5),surface_on=1):
    
    if ren==None:
        
        ren=vtk.vtkRenderer() 
    
    max=vertices.max(axis=0)
    min=vertices.min(axis=0)
   
    if spheres_on :
        for pos in vertices:
            ren.AddActor(sphere(position=pos,radius=sphere_radius,color=sphere_color))
       
    if tubes_on:
                
        pass
        
    if surface_on :
        
        cnt=0
        points = vtk.vtkPoints()
    
        for vec in vertices:
            points.InsertPoint(cnt,vec[0],vec[1],vec[2])
            cnt+=1   
        
        profile = vtk.vtkPolyData()
        profile.SetPoints(points)

        # Delaunay3D is used to triangulate the points. The Tolerance is the
        # distance that nearly coincident points are merged
        # together. (Delaunay does better if points are well spaced.) The
        # alpha value is the radius of circumcircles, circumspheres. Any mesh
        # entity whose circumcircle is smaller than this value is output.        
        
        delny = vtk.vtkDelaunay3D()
        delny.SetInput(profile)
        delny.SetTolerance(0.001)
        delny.SetAlpha(0)
        delny.BoundingTriangulationOff()
        delny.Update()
        #delny.BoundingTriangulationOn()
      
        # Shrink the result to help see it better.
        #shrink = vtk.vtkShrinkFilter()
        #shrink.SetInputConnection(delny.GetOutputPort())
        #shrink.SetShrinkFactor(1)

        map = vtk.vtkDataSetMapper()
        #map.SetInputConnection(shrink.GetOutputPort())
        map.SetInputConnection(delny.GetOutputPort())
        
        #From Nabble
        #http://www.nabble.com/Delaunay-surface-triangulation-in-3D-td20034644.html
        
        #Store Delaunay output as PolyData
        usg = vtk.vtkUnstructuredGrid()
        usg = delny.GetOutput()
        
        pd = vtk.vtkPolyData()
        pd.SetPoints(usg.GetPoints())
        pd.SetPolys(usg.GetCells())
        
        
        #Get surface triangles with geometry filter
        geom = vtk.vtkGeometryFilter()
        geom.SetInput(pd)
        geom.Update()
                
        print 'Writing Boundary triangles ...'
        writer = vtk.vtkPolyDataWriter()
        writer.SetInput(geom.GetOutput())        
        writer.SetFileName("testBound.vtk")
        writer.SetFileTypeToASCII()
        writer.Write()
        print 'End writing'
        
        #Get Points & Cells after  
        geom.Update()
        
        pd2 = vtk.vtkPolyData()
        pd2 = geom.GetOutput()
        pd2.Update()
        
        npts = pd2.GetNumberOfPoints()
        ncells = pd2.GetNumberOfCells()        
        pts = pd2.GetPoints()        
        polys = pd2.GetPolys()
        id = polys.GetData()
        
        print 'id'
        print id.GetValue(0)
        
        nfc=id.GetValue(0)
        
        
        cnt =0
        while cnt < ncells:
            
            vid1=id.GetValue(cnt*(nfc+1)+1)
            vid2=id.GetValue(cnt*(nfc+1)+2)
            vid3=id.GetValue(cnt*(nfc+1)+3)
            vid4=id.GetValue(cnt*(nfc+1)+4)
        
            vx1=pts.GetPoint(vid1)
            vx2=pts.GetPoint(vid2)
            vx3=pts.GetPoint(vid3)
            vx4=pts.GetPoint(vid4)
            
            '''
            ren.AddActor(line(pos1=vx1,pos2=vx2))
            ren.AddActor(line(pos1=vx1,pos2=vx3))
            ren.AddActor(line(pos1=vx1,pos2=vx4))
        
            ren.AddActor(line(pos1=vx2,pos2=vx3))
            ren.AddActor(line(pos1=vx2,pos2=vx4))
            ren.AddActor(line(pos1=vx3,pos2=vx4))
            '''
        
            cnt+=1
        
        
        
        '''
        print 'Number of Points'
        print(pd2.GetNumberOfPoints()) 
                
        print 'Number of Cells'
        print(pd2.GetNumberOfCells()) 
        
        print 'Number of Vertices'
        print(pd2.GetNumberOfVerts()) 
               
        print 'Number of Polys'
        print(pd2.GetNumberOfPolys()) 
        
        print 'Number of Lines'
        print(pd2.GetNumberOfLines()) 
        
        print 'Number of Pieces'
        print(pd2.GetNumberOfPieces()) 
        
        print 'Number of Strips'
        print(pd2.GetNumberOfStrips()) 
        '''
        
                
        triangulation = vtk.vtkActor()
        triangulation.SetMapper(map)
        triangulation.GetProperty().SetColor(1, 1, 1)
        triangulation.GetProperty().SetOpacity(0.5)
        triangulation.GetProperty().SetRepresentationToWireframe()
        
        ren.AddActor(triangulation)
        
    return ren

    
def surfaceplot(points,ren=None,surface_type='Delaunay',surface_color=(1,0,0),surface_opacity=0.6,pointsradii=0.2,axeson=1,labelson=1,labelsdecimals=2,axeslabelsscale=(0.3,0.3,0.3),gridson=1,gridcolor=(0.2,0.2,0.2),gridslabelson=1,gridlabelsscale=(0.3,0.3,0.3),gridlabeldist=(0.5,0.5,0.3),scaleon=1,colormap=1):
    
    if ren==None:
        
        ren=vtk.vtkRenderer()    
    
    max=points.max(axis=0)
    min=points.min(axis=0)
        
    print(max)
    print(min)    
    
    #gridaxes(ren,min,max,axeson,labelsdecimals, axeslabelsscale,gridson,gridcolor,gridslabelson,gridlabelsscale,gridlabeldist,scaleon)
        
    # The points to be triangulated are generated randomly in the unit
    # cube located at the origin. The points are then associated with a
    # vtkPolyData.
    math = vtk.vtkMath()
    pointsv = vtk.vtkPoints()
    
    #for i in range(0, 25):
    #    pointsv.InsertPoint(i, math.Random(0, 1), math.Random(0, 1),
    #                   math.Random(0, 1))
                    
    cnt=0
    # *vec
    #for cnt, vec in enumerate(points)
    #   p.InsertPoint(cnt, *vec)
    
    for vec in points:
        pointsv.InsertPoint(cnt,vec[0],vec[1],vec[2])
        cnt+=1
            
    
    
    profile = vtk.vtkPolyData()
    profile.SetPoints(pointsv)

    # Delaunay3D is used to triangulate the points. The Tolerance is the
    # distance that nearly coincident points are merged
    # together. (Delaunay does better if points are well spaced.) The
    # alpha value is the radius of circumcircles, circumspheres. Any mesh
    # entity whose circumcircle is smaller than this value is output.
    delny = vtk.vtkDelaunay3D()
    delny.SetInput(profile)
    #below options for vtkDelaunay2D    
    #delny.SetTolerance(0.01)
    #delny.SetAlpha(1)
    #delny.BoundingTriangulationOff()

    # Shrink the result to help see it better.
    '''
    shrink = vtk.vtkShrinkFilter()
    shrink.SetInputConnection(delny.GetOutputPort())
    shrink.SetShrinkFactor(0.9)
    '''
    map = vtk.vtkDataSetMapper()
    
    #map.SetInputConnection(shrink.GetOutputPort())
    map.SetInputConnection(delny.GetOutputPort())

    triangulation = vtk.vtkActor()
    triangulation.SetMapper(map)
    triangulation.GetProperty().SetColor(surface_color)
    triangulation.GetProperty().SetOpacity(surface_opacity)

    # Add the actors to the renderer, set the background
    ren.AddActor(triangulation)
    #ren.SetBackground(1, 1, 1)
    
    return ren



def splineplot(points,ren=None,InputPoints = 10,OutputPoints = 400):
    
    from vtk.util.colors import tomato, banana

    if ren==None:
        
        ren=vtk.vtkRenderer()

    # This will be used later to get random numbers.
    math = vtk.vtkMath()
    
    # One spline for each direction.
    aSplineX = vtk.vtkCardinalSpline()
    aSplineY = vtk.vtkCardinalSpline()
    aSplineZ = vtk.vtkCardinalSpline()

    # Generate random (pivot) points and add the corresponding
    # coordinates to the splines.
    # aSplineX will interpolate the x values of the points
    # aSplineY will interpolate the y values of the points
    # aSplineZ will interpolate the z values of the points
    inputPoints = vtk.vtkPoints()
    
    InputPoints=points.shape[0]
    print(InputPoints)
    '''
    for i in range(0, InputPoints):
        x = math.Random(0, 1)
        y = math.Random(0, 1)
        z = math.Random(0, 1)     
        
        aSplineX.AddPoint(i, x)
        aSplineY.AddPoint(i, y)
        aSplineZ.AddPoint(i, z)
        inputPoints.InsertPoint(i, x, y, z)
    '''
       
    for i, vec in enumerate(points):
        
        #print vec
        x,y,z=vec
        
        aSplineX.AddPoint(i, x)
        aSplineY.AddPoint(i, y)
        aSplineZ.AddPoint(i, z)
        inputPoints.InsertPoint(i, x, y, z)
        
        

    # The following section will create glyphs for the pivot points
    # in order to make the effect of the spline more clear.

    # Create a polydata to be glyphed.
    inputData = vtk.vtkPolyData()
    inputData.SetPoints(inputPoints)

    # Use sphere as glyph source.
    balls = vtk.vtkSphereSource()
    balls.SetRadius(.01)
    balls.SetPhiResolution(10)
    balls.SetThetaResolution(10)

    glyphPoints = vtk.vtkGlyph3D()
    glyphPoints.SetInput(inputData)
    glyphPoints.SetSource(balls.GetOutput())

    glyphMapper = vtk.vtkPolyDataMapper()
    glyphMapper.SetInputConnection(glyphPoints.GetOutputPort())

    glyph = vtk.vtkActor()
    glyph.SetMapper(glyphMapper)
    glyph.GetProperty().SetDiffuseColor(tomato)
    glyph.GetProperty().SetSpecular(.3)
    glyph.GetProperty().SetSpecularPower(30)

    # Generate the polyline for the spline.
    points = vtk.vtkPoints()
    profileData = vtk.vtkPolyData()   

    # Interpolate x, y and z by using the three spline filters and
    # create new points
    for i in range(0, OutputPoints):
        t = (InputPoints-1.0)/(OutputPoints-1.0)*i
        points.InsertPoint(i, aSplineX.Evaluate(t), aSplineY.Evaluate(t),
                           aSplineZ.Evaluate(t))
     

    # Create the polyline.
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(OutputPoints)
    for i in range(0, OutputPoints):
        lines.InsertCellPoint(i)
     
    profileData.SetPoints(points)
    profileData.SetLines(lines)

    # Add thickness to the resulting line.
    profileTubes = vtk.vtkTubeFilter()
    profileTubes.SetNumberOfSides(8)
    profileTubes.SetInput(profileData)
    profileTubes.SetRadius(.005)

    profileMapper = vtk.vtkPolyDataMapper()
    profileMapper.SetInputConnection(profileTubes.GetOutputPort())

    profile = vtk.vtkActor()
    profile.SetMapper(profileMapper)
    profile.GetProperty().SetDiffuseColor(banana)
    profile.GetProperty().SetSpecular(.3)
    profile.GetProperty().SetSpecularPower(30)
    
    # Add the actors
    ren.AddActor(glyph)
    ren.AddActor(profile)

    return ren

def vectorplot():
    
    pass

def densityplot(vol,ren=None,voxsz=(1.0,1.0,1.0),maptype=1,opacitymap=None,colormap=None):
    
    volumeplot(vol,ren=ren,voxsz=voxsz,maptype=maptype,opacitymap=opacitymap,colormap=colormap)
    
    return ren

def volumeplot(vol,ren=None,voxsz=(1.0,1.0,1.0),maptype=1,opacitymap=None,colormap=None):
    
    if ren==None:
        
        ren=vtk.vtkRenderer()        
    
    print(vol.dtype)

    if opacitymap==None:
        
        opacitymap=np.array([[ 0.0, 0.0],
                          [255.0, 0.05]])
        
    if colormap==None:
        '''
        colormap=np.array([[  0.0, 0.0, 0.0, 0.0],
                            [255.0, 0.0, 0.0, 1.0]])
        '''
        colormap=np.array([[  0.0, 0.0, 0.0, 0.0],
                        [ 64.0, 0.0, 0.0, 1.0],
                        [128.0, 0.0, 1.0, 0.0],
                        [192.0, 1.0, 0.0, 0.0],
                        [255.0, 1.0, 1.0, 1.0]])
        
        '''
        colormap=np.array([[0.0, 0.5, 0.0, 0.0],
                                        [64.0, 1.0, 0.5, 0.5],
                                        [128.0, 0.9, 0.2, 0.3],
                                        [196.0, 0.81, 0.27, 0.1],
                                        [255.0, 0.5, 0.5, 0.5]])

        '''
    im = vtk.vtkImageData()
    im.SetScalarTypeToUnsignedChar()
    im.SetDimensions(vol.shape[0],vol.shape[1],vol.shape[2])
    im.SetOrigin(0,0,0)
    im.SetSpacing(voxsz[2],voxsz[0],voxsz[1])
    im.AllocateScalars()        
    
    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in range(vol.shape[2]):
                
                im.SetScalarComponentFromFloat(i,j,k,0,vol[i,j,k])
    
    opacity = vtk.vtkPiecewiseFunction()
    for i in range(opacitymap.shape[0]):
        opacity.AddPoint(opacitymap[i,0],opacitymap[i,1])

    color = vtk.vtkColorTransferFunction()
    for i in range(colormap.shape[0]):
        color.AddRGBPoint(colormap[i,0],colormap[i,1],colormap[i,2],colormap[i,3])
        
    if(maptype==0): 
    
        property = vtk.vtkVolumeProperty()
        property.SetColor(color)
        property.SetScalarOpacity(opacity)
        
        mapper = vtk.vtkVolumeTextureMapper2D()
        mapper.SetInput(im)
    
    if (maptype==1):

        property = vtk.vtkVolumeProperty()
        property.SetColor(color)
        property.SetScalarOpacity(opacity)
        property.ShadeOn()
        property.SetInterpolationTypeToLinear()
     
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        mapper = vtk.vtkVolumeRayCastMapper()
        mapper.SetVolumeRayCastFunction(compositeFunction)
        mapper.SetInput(im)
        
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(property)
    
    ren.AddVolume(volume)
    
    #Return mid position in world space
    
    index=im.FindPoint(vol.shape[0]/2.0,vol.shape[1]/2.0,vol.shape[2]/2.0)
    
    print im.GetPoint(index)
    
    return ren

def volume(vol,voxsz=(1.0,1.0,1.0),affine=None,center_origin=1,final_volume_info=1,maptype=1,iso=0,iso_thr=100,opacitymap=None,colormap=None):
        
    print(vol.dtype)

    if opacitymap==None:
        
        opacitymap=np.array([[ 0.0, 0.0],
                          [50.0, 0.1]])
        
    if colormap==None:
        
    
        colormap=np.array([[0.0, 0.5, 0.0, 0.0],
                                        [64.0, 1.0, 0.5, 0.5],
                                        [128.0, 0.9, 0.2, 0.3],
                                        [196.0, 0.81, 0.27, 0.1],
                                        [255.0, 0.5, 0.5, 0.5]])

    
    im = vtk.vtkImageData()
    im.SetScalarTypeToUnsignedChar()
    im.SetDimensions(vol.shape[0],vol.shape[1],vol.shape[2])
    #im.SetOrigin(0,0,0)
    #im.SetSpacing(voxsz[2],voxsz[0],voxsz[1])
    im.AllocateScalars()        
    
    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in range(vol.shape[2]):
                
                im.SetScalarComponentFromFloat(i,j,k,0,vol[i,j,k])
    
    if affine != None:

        aff = vtk.vtkMatrix4x4()
        aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],affine[0,3],affine[1,0],affine[1,1],affine[1,2],affine[1,3],affine[2,0],affine[2,1],affine[2,2],affine[2,3],affine[3,0],affine[3,1],affine[3,2],affine[3,3]))
        #aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],0,affine[1,0],affine[1,1],affine[1,2],0,affine[2,0],affine[2,1],affine[2,2],0,affine[3,0],affine[3,1],affine[3,2],1))
        #aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],127.5,affine[1,0],affine[1,1],affine[1,2],-127.5,affine[2,0],affine[2,1],affine[2,2],-127.5,affine[3,0],affine[3,1],affine[3,2],1))
        
        reslice = vtk.vtkImageReslice()
        reslice.SetInput(im)
        #reslice.SetOutputDimensionality(2)
        #reslice.SetOutputOrigin(127,-145,147)    
        
        reslice.SetResliceAxes(aff)
        #reslice.SetOutputOrigin(-127,-127,-127)    
        #reslice.SetOutputExtent(-127,128,-127,128,-127,128)
        #reslice.SetResliceAxesOrigin(0,0,0)
        #print 'Get Reslice Axes Origin ', reslice.GetResliceAxesOrigin()
        #reslice.SetOutputSpacing(1.0,1.0,1.0)
        
        reslice.SetInterpolationModeToLinear()    
        #reslice.UpdateWholeExtent()
        
        #print 'reslice GetOutputOrigin', reslice.GetOutputOrigin()
        #print 'reslice GetOutputExtent',reslice.GetOutputExtent()
        #print 'reslice GetOutputSpacing',reslice.GetOutputSpacing()
    
        changeFilter=vtk.vtkImageChangeInformation() 
        changeFilter.SetInput(reslice.GetOutput())
        #changeFilter.SetInput(im)
        if center_origin:
            changeFilter.SetOutputOrigin(-vol.shape[0]/2.0+0.5,-vol.shape[1]/2.0+0.5,-vol.shape[2]/2.0+0.5)
            print 'ChangeFilter ', changeFilter.GetOutputOrigin()
        
    opacity = vtk.vtkPiecewiseFunction()
    for i in range(opacitymap.shape[0]):
        opacity.AddPoint(opacitymap[i,0],opacitymap[i,1])

    color = vtk.vtkColorTransferFunction()
    for i in range(colormap.shape[0]):
        color.AddRGBPoint(colormap[i,0],colormap[i,1],colormap[i,2],colormap[i,3])
        
    if(maptype==0): 
    
        property = vtk.vtkVolumeProperty()
        property.SetColor(color)
        property.SetScalarOpacity(opacity)
        
        mapper = vtk.vtkVolumeTextureMapper2D()
        if affine == None:
            mapper.SetInput(im)
        else:
            #mapper.SetInput(reslice.GetOutput())
            mapper.SetInput(changeFilter.GetOutput())
        
    
    if (maptype==1):

        property = vtk.vtkVolumeProperty()
        property.SetColor(color)
        property.SetScalarOpacity(opacity)
        property.ShadeOn()
        property.SetInterpolationTypeToLinear()

        if iso:
            isofunc=vtk.vtkVolumeRayCastIsosurfaceFunction()
            isofunc.SetIsoValue(iso_thr)
        else:
            compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        
        mapper = vtk.vtkVolumeRayCastMapper()
        if iso:
            mapper.SetVolumeRayCastFunction(isofunc)
        else:
            mapper.SetVolumeRayCastFunction(compositeFunction)   
            #mapper.SetMinimumImageSampleDistance(0.2)
             
        if affine == None:
            mapper.SetInput(im)
        else:
            #mapper.SetInput(reslice.GetOutput())    
            mapper.SetInput(changeFilter.GetOutput())
            #Return mid position in world space    
            #im2=reslice.GetOutput()
            #index=im2.FindPoint(vol.shape[0]/2.0,vol.shape[1]/2.0,vol.shape[2]/2.0)
            #print 'Image Getpoint ' , im2.GetPoint(index)
           
        
    volum = vtk.vtkVolume()
    volum.SetMapper(mapper)
    volum.SetProperty(property)

    if final_volume_info :  
         
        print 'Origin',   volum.GetOrigin()
        print 'Orientation',   volum.GetOrientation()
        print 'OrientationW',    volum.GetOrientationWXYZ()
        print 'Position',    volum.GetPosition()
        print 'Center',    volum.GetCenter()  
        print 'Get XRange', volum.GetXRange()
        print 'Get YRange', volum.GetYRange()
        print 'Get ZRange', volum.GetZRange()  
        
    return volum

def volumeplot2(vol,ren=None,voxsz=(1.0,1.0,1.0),affine=None,center_origin=1,maptype=1,opacitymap=None,colormap=None):
    
    if ren==None:
        
        ren=vtk.vtkRenderer()        
    
    print(vol.dtype)

    if opacitymap==None:
        
        opacitymap=np.array([[ 0.0, 0.0],
                          [255.0, 0.05]])
        
    if colormap==None:
        '''
        colormap=np.array([[  0.0, 0.0, 0.0, 0.0],
                            [255.0, 0.0, 0.0, 1.0]])
        '''
        colormap=np.array([[  0.0, 0.0, 0.0, 0.0],
                        [ 64.0, 0.0, 0.0, 1.0],
                        [128.0, 0.0, 1.0, 0.0],
                        [192.0, 1.0, 0.0, 0.0],
                        [255.0, 1.0, 1.0, 1.0]])
        
        '''
        colormap=np.array([[0.0, 0.5, 0.0, 0.0],
                                        [64.0, 1.0, 0.5, 0.5],
                                        [128.0, 0.9, 0.2, 0.3],
                                        [196.0, 0.81, 0.27, 0.1],
                                        [255.0, 0.5, 0.5, 0.5]])

        '''
    im = vtk.vtkImageData()
    im.SetScalarTypeToUnsignedChar()
    im.SetDimensions(vol.shape[0],vol.shape[1],vol.shape[2])
    #im.SetOrigin(0,0,0)
    #im.SetSpacing(voxsz[2],voxsz[0],voxsz[1])
    im.AllocateScalars()        
    
    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in range(vol.shape[2]):
                
                im.SetScalarComponentFromFloat(i,j,k,0,vol[i,j,k])
    
    if affine != None:

        aff = vtk.vtkMatrix4x4()
        aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],affine[0,3],affine[1,0],affine[1,1],affine[1,2],affine[1,3],affine[2,0],affine[2,1],affine[2,2],affine[2,3],affine[3,0],affine[3,1],affine[3,2],affine[3,3]))
        #aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],0,affine[1,0],affine[1,1],affine[1,2],0,affine[2,0],affine[2,1],affine[2,2],0,affine[3,0],affine[3,1],affine[3,2],1))
        #aff.DeepCopy((affine[0,0],affine[0,1],affine[0,2],127.5,affine[1,0],affine[1,1],affine[1,2],-127.5,affine[2,0],affine[2,1],affine[2,2],-127.5,affine[3,0],affine[3,1],affine[3,2],1))
        
        reslice = vtk.vtkImageReslice()
        reslice.SetInput(im)
        #reslice.SetOutputDimensionality(2)
        #reslice.SetOutputOrigin(127,-145,147)    
        
        reslice.SetResliceAxes(aff)
        #reslice.SetOutputOrigin(-127,-127,-127)    
        #reslice.SetOutputExtent(-127,128,-127,128,-127,128)
        #reslice.SetResliceAxesOrigin(0,0,0)
        #print 'Get Reslice Axes Origin ', reslice.GetResliceAxesOrigin()
        #reslice.SetOutputSpacing(1.0,1.0,1.0)
        
        reslice.SetInterpolationModeToLinear()    
        #reslice.UpdateWholeExtent()
        
        #print 'reslice GetOutputOrigin', reslice.GetOutputOrigin()
        #print 'reslice GetOutputExtent',reslice.GetOutputExtent()
        #print 'reslice GetOutputSpacing',reslice.GetOutputSpacing()
    
        changeFilter=vtk.vtkImageChangeInformation() 
        changeFilter.SetInput(reslice.GetOutput())
        #changeFilter.SetInput(im)
        if center_origin:
            changeFilter.SetOutputOrigin(-vol.shape[0]/2.0+0.5,-vol.shape[1]/2.0+0.5,-vol.shape[2]/2.0+0.5)
            print 'ChangeFilter ', changeFilter.GetOutputOrigin()
        
    opacity = vtk.vtkPiecewiseFunction()
    for i in range(opacitymap.shape[0]):
        opacity.AddPoint(opacitymap[i,0],opacitymap[i,1])

    color = vtk.vtkColorTransferFunction()
    for i in range(colormap.shape[0]):
        color.AddRGBPoint(colormap[i,0],colormap[i,1],colormap[i,2],colormap[i,3])
        
    if(maptype==0): 
    
        property = vtk.vtkVolumeProperty()
        property.SetColor(color)
        property.SetScalarOpacity(opacity)
        
        mapper = vtk.vtkVolumeTextureMapper2D()
        if affine == None:
            mapper.SetInput(im)
        else:
            #mapper.SetInput(reslice.GetOutput())
            mapper.SetInputConnection(changeFilter.GetOutpuPort())
        
    
    if (maptype==1):

        property = vtk.vtkVolumeProperty()
        property.SetColor(color)
        property.SetScalarOpacity(opacity)
        property.ShadeOn()
        property.SetInterpolationTypeToLinear()
     
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        
        mapper = vtk.vtkVolumeRayCastMapper()
        mapper.SetVolumeRayCastFunction(compositeFunction)   
        #mapper.SetMinimumImageSampleDistance(0.2)
             
        if affine == None:
            mapper.SetInput(im)
        else:
            #mapper.SetInput(reslice.GetOutput())    
            mapper.SetInput(changeFilter.GetOutput())
            #Return mid position in world space    
            #im2=reslice.GetOutput()
            #index=im2.FindPoint(vol.shape[0]/2.0,vol.shape[1]/2.0,vol.shape[2]/2.0)
            #print 'Image Getpoint ' , im2.GetPoint(index)
    
        
        
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(property)

    #volume.SetUserMatrix(reslice.GetResliceAxes())
    
    #if center_origin:
    #    volume.SetPosition(-0.5,19.5,0)
    #    volume.SetOrigin(0,0,0)
    
    
    
    ren.AddVolume(volume)
    
    print 'Origin',   volume.GetOrigin()
    print 'Orientation',   volume.GetOrientation()
    print 'OrientationW',    volume.GetOrientationWXYZ()
    print 'Position',    volume.GetPosition()
    print 'Center',    volume.GetCenter()  
    print 'Get XRange', volume.GetXRange()
    print 'Get YRange', volume.GetYRange()
    print 'Get ZRange', volume.GetZRange()  
    
    return ren
    #return volume


def testplot():
    '''
    ps=sp.rand(30).reshape((10,3))
    ps=sp.dot(sp.diag([1,1,1,1,1,-1,-1,-1,-1,-1]),ps)
    ps=ps*10
    
    #ren=scatterplot(ps)  
    '''
    ps=sp.array([[0,0,10],[-10,0,0],[10,0,0],[0,0,-10],[0,10,0]])
    
    def gaussian(height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x,y: height*sp.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

    Xin, Yin = sp.mgrid[-30:31, -30:31]
    data = gaussian(30, 0, 0, 10, 10)(Xin, Yin)
    
    print(data.shape)
    
    Xin=np.ravel(Xin)
    Yin=np.ravel(Yin)
    data=np.ravel(data)
    
    ps=sp.zeros((Xin.size,3))
    for i in sp.arange(Xin.size):
        ps[i,:]=sp.array([Xin[i],Yin[i],data[i]])
    
    #return
    '''
    #[x,y]=sp.mgrid[-10:11,-10:11]
    [x,y]=sp.mgrid[-10:1.02:11,-10:1.02:11]
    r=sp.sqrt(x**2+y**2)
    z=1000*sp.sin(3*r)/(r)
    
    x=np.ravel(x)
    y=np.ravel(y)
    z=np.ravel(z)
    
    ps=sp.zeros((x.size,3))
    for i in sp.arange(x.size):
        ps[i,:]=sp.array([x[i],y[i],z[i]])
    
    #ps=np.vstack([x,y,z])
    #'''  
    print(ps.shape)
    #print(ps)
    #print(x.shape)
    #print(y.shape)
    #print(z.shape)
    
    ren=surfaceplot(ps)
    #return
    
    
    print('Before AppThread')
    ap=AppThread(frame_type=0,ren=ren)
    print('After AppThread')
    
    angle=0
    
    #cam=vtk.vtkCamera()
    
    #ren.SetActiveCamera(cam)
    ren.ResetCamera()
    #cam.Azimuth(45)
    #cam.Elevation(20)
    #ren.GetActiveCamera().Elevation(90)
    ren.GetActiveCamera().Azimuth(45)
    ren.GetActiveCamera().Elevation(20)
    sh.widget.Enable(1)

    sh.ren=ren
    
    #'''
    
    angle=0
    lock = threading.Lock()   
    
    time.sleep(2) 
    
    while angle<360:
        
        print(angle)
        ren.GetActiveCamera().Azimuth(angle)        
        angle+=0.1
        
        #time.sleep(0.1)
        
        lock.acquire()
        sh.widget.Enable(0)
        lock.release()
        
        time.sleep(0.1)        
        
        lock.acquire()
        sh.widget.Enable(1)
        lock.release()
        
        if(angle>6):
            time.sleep(3)
            angle=0
            
    
    #time.sleep(0.2)
    #FrameThread(frame_type=1)
    #'''

def in_a_thread(ren):
    
    ren.GetActiveCamera().SetPosition([0,0,40])
    ren.GetActiveCamera().Elevation(20)
    
    time.sleep(5)
    print('In a thread')
    
    angle=0
    lock = threading.Lock()    
    
    while angle<360:
        
        print(angle)
        ren.GetActiveCamera().Azimuth(angle)        
        angle+=1
        
        #time.sleep(0.1)
        
        lock.acquire()
        sh.widget.Enable(0)
        lock.release()
        
        time.sleep(0.1)        
        
        lock.acquire()
        sh.widget.Enable(1)
        lock.release()
        
        if(angle>360):
            time.sleep(3)
            angle=0
    

def threading_test():
    
    ps=sp.rand(30).reshape((10,3))
    ps=sp.dot(sp.diag([1,1,1,1,1,-1,-1,-1,-1,-1]),ps)
    ps=ps*10
    
    ren=scatterplot(ps)  

    app=wx.PySimpleApp()
    frame=FrameVtkMinimal(None,-1,ren,"Hello",300,300)
    frame.Show(1)   
    
    t = threading.Thread(target=in_a_thread, args=(ren,))
    t.setDaemon(1)
    t.start()

    
    app.MainLoop()


    
def testline():
    
    #print('1#')
    x=sp.arange(-4*sp.pi,4*sp.pi,0.1)
    
    y=10*sp.sinc(x)    
    
    z=x
    
    print(x.size)    
        
    #print('2#')
    
    ren=vtk.vtkRenderer()
    
    ps=sp.zeros((x.size,3))
    for i in sp.arange(x.size):        
        ps[i,:]=sp.array([x[i],y[i],z[i]])
        
        if i<x.size-1:
            ren.AddActor(line(pos1=(x[i],y[i],z[i]),pos2=(x[i+1],y[i+1],z[i+1])))
        
    print('3#')
    #ren=scatterplot(ps,labelson=0)
    #ren=surfaceplot(ps)
    print('4#')
    
    
    print('Before AppThread')
    ap=AppThread(frame_type=0,ren=ren)
    print('After AppThread')


def testvolume():
    
    dims=(128,128,128)
    
    np=dims[0]*dims[1]*dims[2]
    
    x, y, z = sp.ogrid[-5:5:dims[0]*1j,-5:5:dims[1]*1j,-5:5:dims[2]*1j]
    
    x= x.astype('f')
    y= y.astype('f')
    z= z.astype('f')
    
    vol = (sp.sin(x*y*z)/(x*y*z))
    
    #vol = sp.transpose(vol).copy()
    
    vol= vol/vol.max()
    vol=vol*255
    
    ren=volumeplot(vol,maptype=1)

    ren.AddActor(axes(scale=dims))
    
    print('Before AppThread')
    ap=AppThread(frame_type=0,ren=ren)
    print('After AppThread')

def testspline():
    
    '''
    x=sp.arange(-4*sp.pi,4*sp.pi,1)
    
    y=10*sp.sinc(x)    
    
    z=x   
    
    ps=sp.zeros((x.size,3))
    for i in sp.arange(x.size):        
        ps[i,:]=sp.array([x[i],y[i],z[i]])
    '''
    #ps = sp.array([[0,0,0],[1,1,1],[0,1,0]])
    
    ps = sp.array([[0,0,1],[0,0,0],[1,1,0],[1,0,0],[1,1,1]])
    
        
    ren=splineplot(ps,OutputPoints=100)    
    
    for vec in ps:
            ren.AddActor(label(ren,text=str(np.round(vec, decimals=1)),pos=vec,scale=(0.05,0.05,0.05)))
    
    
    
    print('Before AppThread')
    ap=AppThread(frame_type=0,ren=ren,width=600,height=400)
    print('After AppThread')

def testsphere():
    
    ren= vtk.vtkRenderer()
    
    ren.AddActor(sphere(tessel=1))
    
    print('Before AppThread')
    ap=AppThread(frame_type=0,ren=ren,width=600,height=400)
    print('After AppThread')
    

def testplatonicsolid():
    
    plat = vtk.vtkPlatonicSolidSource() 
    #plat.SetSolidTypeToTetrahedron ()
    #plat.SetSolidTypeToCube ()
    #plat.SetSolidTypeToOctahedron ()
    plat.SetSolidTypeToIcosahedron ()
    #plat.SetSolidTypeToDodecahedron ()

    platm = vtk.vtkPolyDataMapper()
    platm.SetInput(plat.GetOutput())    
    
    plata = vtk.vtkActor()
    plata.SetMapper(platm)
    plata.GetProperty().SetColor([0,0,1])
    
    ren = vtk.vtkRenderer()    
    ren.AddActor(plata)
    
    
    print('Before AppThread')
    ap=AppThread(frame_type=0,ren=ren,width=600,height=400)
    print('After AppThread')




#uniform or shell arranged points
def testpointsource():
    
    pts = vtk.vtkPointSource()
    pts.SetNumberOfPoints(10000)
    #pts.SetDistributionToUniform()
    pts.SetDistributionToShell()
    
    ptsm = vtk.vtkPolyDataMapper()
    ptsm.SetInput(pts.GetOutput())    
    
    ptsa = vtk.vtkActor()
    ptsa.SetMapper(ptsm)
    ptsa.GetProperty().SetColor([1,0,1])
    
    ren = vtk.vtkRenderer()    
    ren.AddActor(ptsa)
    
    
    print('Before AppThread')
    ap=AppThread(frame_type=0,ren=ren,width=600,height=400)
    print('After AppThread')


def testcontour():
    
    dims=(64,64,64)
    
    np=dims[0]*dims[1]*dims[2]
    
    x, y, z = sp.ogrid[-5:5:dims[0]*1j,-5:5:dims[1]*1j,-5:5:dims[2]*1j]
    
    x= x.astype('f')
    y= y.astype('f')
    z= z.astype('f')
    
    vol = (sp.sin(x*y*z)/(x*y*z))    
    vol = vol/vol.max()    
    vol = vol*255
    
    voxsz=dims
    
    im = vtk.vtkImageData()
    im.SetScalarTypeToUnsignedChar()
    im.SetDimensions(vol.shape[0],vol.shape[1],vol.shape[2])
    im.SetOrigin(0,0,0)
    im.SetSpacing(voxsz[2],voxsz[0],voxsz[1])
    im.AllocateScalars()        
    
    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in range(vol.shape[2]):
                
                im.SetScalarComponentFromFloat(i,j,k,0,vol[i,j,k])
    
    contour = vtk.vtkContourFilter()
    contour.SetInput(im)
    
    #contour.SetValue(0,180)
    contour.GenerateValues(3,0,255)
    
    cnormals=vtk.vtkPolyDataNormals()
    cnormals.SetInputConnection(contour.GetOutputPort())
    cnormals.SetFeatureAngle(60.0)
    
    cmapper = vtk.vtkPolyDataMapper()
    cmapper.SetInputConnection(cnormals.GetOutputPort())
    #cmapper.ScalarVisibilityOff()
    cmapper.SetScalarRange(0,255)
    
    cactor = vtk.vtkActor()
    cactor.SetMapper(cmapper)

    ren =vtk.vtkRenderer()
    ren.AddActor(cactor)
    
    print('Before AppThread')
    ap=AppThread(frame_type=0,ren=ren)
    print('After AppThread')
    
def testtegmark():

    fname='/home/eg01/Devel/test.dat'
    pts = []
    for line in open(fname, 'rt'):
        pts.append([float(val) for val in line.split()])
    pts=sp.array(pts)
    
    pts=pts[:,2:5]
    print(pts)
    
    pts=10*pts
    ren=scatterplot(pts,pointsradii=0.02,randomcolor=0)
    #ren=surfaceplot(pts,surface_opacity=1)    
    #scatterplot(pts,pointsradii=0.1,randomcolor=0)
    #ren=glyphplot(pts)
    
    ap=AppThread(frame_type=0,ren=ren)
    
    return

def testsphericalgrid():
    
    #vertices=spherepoints(125)
    
    #ren=sphericalgrid(vertices,sphere_radius=0.05,spheres_on=1, tubes_on=0,surface_on=1)

    #ren.AddActor(axes())

    vertices=spherepoints(65,scale=2,alg=1)
    
    ren=sphericalgrid(vertices,sphere_radius=0.05,spheres_on=1,sphere_color=(0,0.5,1), tubes_on=0,surface_on=1)
    
    #ren.AddActor(plane(orig=(-3.5,-3.5,0),point1=(-3.5,3.5,0),point2=(3.5,-3.5,0),color=(0.1,0.2,0.4)))
    
    ren.ResetCamera()
    #ren.GetActiveCamera().Roll(-2)
    ren.GetActiveCamera().Zoom(1.3)
        
    ren.GetActiveCamera().Elevation(-60)
    '''
  
    ren.GetActiveCamera().SetPosition(0, 1, 0)
    ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
    ren.GetActiveCamera().SetViewUp(0, 0, 1)

    # Let the renderer compute a good position and focal point.
    ren.ResetCamera()
    ren.GetActiveCamera().Dolly(1.4)
    ren.ResetCameraClippingRange()
    '''
    #NoAppThread(frame_type=0,ren=ren)
    ap=AppThread(frame_type=0,ren=ren)
    TakePhotoVTK(ren=ren,magnification=10,bgr_color=(0,0,0))
    print 'Check!'
    #TakeVideoVTK(ren=ren,N_Frames=10,size=(600,600))

def testDSIgrid():

    vertices=gridpointscube(-1,1,14)

    ren=vtk.vtkRenderer()

    cnt=0
    vert_sph=[]
    for vec in vertices:
      
        dist=vec[0]**2+vec[1]**2+vec[2]**2

        #print dist

        if  dist< 1.05 :

            ren.AddActor(sphere(2*vec, radius=0.05,color=(0,0.5,1)))
            vert_sph.append(vec)
            cnt+=1

    ren.AddActor(plane(orig=(-3.5,-3.5,0),point1=(-3.5,3.5,0),point2=(3.5,-3.5,0),color=(0.1,0.2,0.4)))
        
    vert_sph=sp.vstack(vert_sph)
    #ren.AddActor(boundarywireframe(2*vert_sph))
  
    act=torus(scalex=4,scaley=4,scalez=4)

    act.RotateX(90)
    ren.AddActor(act)

    ren.ResetCamera()
    ren.GetActiveCamera().Roll(-20)
    ren.GetActiveCamera().Zoom(1.1)        
    ren.GetActiveCamera().Elevation(-60)

    print 'Number of points under semi-sphere : ' + str(cnt/2.0)
    ap=AppThread(frame_type=0,ren=ren)

    TakePhotoVTK(ren=ren,magnification=10,filename='DSI_604_2.png',bgr_color=(0,0,0))
    
def testQBIgrid():

    vertices=spherepoints(515*2)
    
    ren=sphericalgrid(2*vertices,sphere_color=(0,0.5,1),sphere_radius=0.05,spheres_on=1, tubes_on=0,surface_on=1)

    #ren.AddActor(axes())

    #vertices=spherepoints(130,scale=2,alg=0)
    
    #sphericalgrid(vertices,ren=ren,sphere_radius=0.05,spheres_on=1,sphere_color=(0,0.5,1), tubes_on=0,surface_on=1)
    
    ren.AddActor(plane(orig=(-3.5,-3.5,0),point1=(-3.5,3.5,0),point2=(3.5,-3.5,0),color=(0.1,0.2,0.4)))
    #ren.AddActor(plane(orig=(-3.5,-3.5,0),point1=(-3.5,3.5,0),point2=(3.5,-3.5,0),color=(0.0,0.0,0.0)))

    act=torus(scalex=4,scaley=4,scalez=4)

    act.RotateX(90)
    #ren.AddActor(act)

    ren.ResetCamera()
    ren.GetActiveCamera().Roll(-20)
    ren.GetActiveCamera().Zoom(1.1)        
    ren.GetActiveCamera().Elevation(-60)

    ap=AppThread(frame_type=0,ren=ren)
  
    TakePhotoVTK(ren=ren,magnification=10,filename='QBI_515.png',bgr_color=(0,0,0))     
    #TakeVideoVTK(ren=ren,N_Frames=10,size=(600,600))

    return

def testDTIgrid():
    
    #vertices=sp.vstack((sp.rand(3),sp.rand(3),sp.rand(3),sp.rand(3),sp.rand(3),sp.rand(3),sp.rand(3),sp.rand(3)))

    #print vertices

    ren = vtk.vtkRenderer()

    ren.AddActor(plane(orig=(-3.5,-3.5,0),point1=(-3.5,3.5,0),point2=(3.5,-3.5,0),color=(0.1,0.2,0.4)))
    #for i in range(15):
      
    #ren.AddActor(sphere(position=tuple(sp.randn(3)), radius=0.05,color=(0,0.5,1)))
    '''
    
    ren.AddActor(sphere(position=(1,0,0.6), radius=0.09,color=(0,0.5,1)))
    ren.AddActor(sphere(position=(-1,0,0.6), radius=0.09,color=(0,0.5,1)))

    ren.AddActor(sphere(position=(0,1,0.6), radius=0.09,color=(0,0.5,1)))
    ren.AddActor(sphere(position=(0,-1,0.6), radius=0.09,color=(0,0.5,1)))

    ren.AddActor(sphere(position=(0.5,0.5,1.3), radius=0.09,color=(0,0.5,1)))
    ren.AddActor(sphere(position=(-0.5,-0.5,1.3), radius=0.09,color=(0,0.5,1)))

    ren.AddActor(sphere(position=(0.,0,1.4), radius=0.09,color=(0,0.5,1)))
    '''
    #'''
    sph1=sphere(position=(0.707,  0,0.707), radius=0.09,color=(0,0.5,1))
    sph2=sphere(position=(-0.707, 0,0.707), radius=0.09,color=(0,0.5,1))
    #sph1.RotateX(90)
    #sph2.RotateX(90)
    ren.AddActor(sph1)
    ren.AddActor(sph2)
       
    sph3=sphere(position=(0, 0.707, 0.707), radius=0.09,color=(0,0.5,1))
    sph4=sphere(position=(0, 0.707,-0.707), radius=0.09,color=(0,0.5,1))
    #sph3.RotateX(90)
    #sph4.RotateX(90)
    
    ren.AddActor(sph3)
    ren.AddActor(sph4)
    
    sph5=sphere(position=( 0.707, 0.707, 0), radius=0.09,color=(0,0.5,1))
    sph6=sphere(position=(-0.707, 0.707, 0), radius=0.09,color=(0,0.5,1))
    
    #sph5.RotateX(90)
    #sph6.RotateX(90)

    ren.AddActor(sph5)
    ren.AddActor(sph6)
    #'''
    '''
    ren.AddActor(sphere(position=(0.707,  0.707,0), radius=0.09,color=(0,0.5,1)))
    ren.AddActor(sphere(position=(-0.707, 0.707,0), radius=0.09,color=(0,0.5,1)))

    ren.AddActor(sphere(position=(0,  0.707, 0.707), radius=0.09,color=(0,0.5,1)))
    ren.AddActor(sphere(position=(0, -0.707, 0.707), radius=0.09,color=(0,0.5,1)))

    ren.AddActor(sphere(position=( 0.707, 0,0.707), radius=0.09,color=(0,0.5,1)))
    ren.AddActor(sphere(position=(-0.707, 0,0.707), radius=0.09,color=(0,0.5,1)))
    '''

    #ren.AddActor(sphere(position=(0.,0,1.4), radius=0.09,color=(0,0.5,1)))
 
    ren.AddActor(axes())
    
    cub=cube(Xmin=-1,Xmax=1,Ymin=-1,Ymax=1,Zmin=0.1,Zmax=2)
    cub.GetProperty().SetRepresentationToWireframe()
    ren.AddActor(cub)
    
    #ren.AddActor(sphere(position=tuple(sp.array([0,0,0])), radius=0.05,color=(0,0.5,1)))    
   
    ren.ResetCamera()
    
    #ren.GetActiveCamera().SetPosition(0,0,20.6576012)
    #ren.GetActiveCamera().SetFocalPoint(0,0,0)
    ren.GetActiveCamera().Roll(-20)
    ren.GetActiveCamera().Zoom(1.1)        
    ren.GetActiveCamera().Elevation(-60)
  
    print 'Get Camera Position'
    print ren.GetActiveCamera().GetPosition()

    ap=AppThread(frame_type=0,ren=ren)
 
    TakePhotoVTK(ren=ren,magnification=10,filename='DTI_7_.png',bgr_color=(0,0,0))     

    return

def testDTI2gridtrick():

    vertices=spherepoints(130)
    
    ren=sphericalgrid(vertices,sphere_radius=0.05,spheres_on=1, tubes_on=0,surface_on=1)

    #ren.AddActor(axes())

    vertices=spherepoints(130,scale=2,alg=0)
    
    sphericalgrid(vertices,ren=ren,sphere_radius=0.05,spheres_on=1,sphere_color=(0,0.5,1), tubes_on=0,surface_on=1)
    
    ren.AddActor(plane(orig=(-3.5,-3.5,0),point1=(-3.5,3.5,0),point2=(3.5,-3.5,0),color=(0.1,0.2,0.4)))
    #ren.AddActor(plane(orig=(-3.0,-3.0,0),point1=(-3.0,3.0,0),point2=(3.0,-3.0,0),color=(0.1,0.2,0.4)))
    #ren.AddActor(plane(orig=(-3.5,-3.5,0),point1=(-3.5,3.5,0),point2=(3.5,-3.5,0),color=(0.0,0.0,0.0)))

    ren.ResetCamera()    
    ren.GetActiveCamera().Roll(-20)
    ren.GetActiveCamera().Zoom(1.3)        
    ren.GetActiveCamera().Elevation(-60)

    ren.RemoveAllViewProps()

    ren.AddActor(plane(orig=(-3.0,-3.0,0),point1=(-3.0,3.0,0),point2=(3.0,-3.0,0),color=(0.1,0.2,0.4)))
    #for i in range(15):  
      
    #ren.AddActor(sphere(position=tuple(sp.randn(3)), radius=0.05,color=(0,0.5,1)))
    #'''
    ren.AddActor(sphere(position=(1,0,0.6), radius=0.09,color=(0,0.5,1)))
    ren.AddActor(sphere(position=(-1,0,0.6), radius=0.09,color=(0,0.5,1)))

    ren.AddActor(sphere(position=(0,1,0.6), radius=0.09,color=(0,0.5,1)))
    ren.AddActor(sphere(position=(0,-1,0.6), radius=0.09,color=(0,0.5,1)))

    ren.AddActor(sphere(position=(0.5,0.5,1.3), radius=0.09,color=(0,0.5,1)))
    ren.AddActor(sphere(position=(-0.5,-0.5,1.3), radius=0.09,color=(0,0.5,1)))

    ren.AddActor(sphere(position=(0.,0,1.4), radius=0.09,color=(0,0.5,1)))
    #'''

    cub=cube(Xmin=-1,Xmax=1,Ymin=-1,Ymax=1,Zmin=0.1,Zmax=2)
    cub.GetProperty().SetRepresentationToWireframe()
    ren.AddActor(cub)
    
  
    TakePhotoVTK(ren=ren,magnification=10,filename='image.png',bgr_color=(0,0,0))
    print 'Check 1' 

def testHARDIgrid():

    

    #vertices=spherepoints(130)
    
    #ren=sphericalgrid(vertices,sphere_radius=0.05,spheres_on=1, tubes_on=0,surface_on=1)

    #ren.AddActor(axes())

    ren=vtk.vtkRenderer()

    vertices=spherepoints(130,scale=2,alg=0)
    
    sphericalgrid(vertices,ren=ren,sphere_radius=0.05,spheres_on=1,sphere_color=(0,0.5,1), tubes_on=0,surface_on=1,surface_color=(1,1,1))
    
    ren.AddActor(plane(orig=(-3.5,-3.5,0),point1=(-3.5,3.5,0),point2=(3.5,-3.5,0),color=(0.1,0.2,0.4)))
    #ren.AddActor(plane(orig=(-3.5,-3.5,0),point1=(-3.5,3.5,0),point2=(3.5,-3.5,0),color=(1.0,1.0,1.0)))

    act=torus(scalex=4,scaley=4,scalez=4)

    act.RotateX(90)
    ren.AddActor(act)

    ren.ResetCamera()

    print 'Get Camera Position'
    print ren.GetActiveCamera().GetPosition()

    ren.GetActiveCamera().Roll(-20)
    ren.GetActiveCamera().Zoom(1.1)        
    ren.GetActiveCamera().Elevation(-60)

    #plane2=ren.GetActors().GetLastActor()
    #ren.RemoveActor(plane2)
    #ren.AddActor(plane(orig=(-3.0,-3.0,0),point1=(-3.0,3.0,0),point2=(3.0,-3.0,0),color=(0.1,0.2,0.4)))
      
  
    TakePhotoVTK(ren=ren,magnification=10,filename='HARDI_65_2.png',bgr_color=(0,0,0))
    print 'Check 1' 
    #TakeVideoVTK(ren=ren,N_Frames=10,size=(600,600))

def testDTI2grid():

    
    ren=vtk.vtkRenderer()

    vertices=spherepoints(40,scale=2,half=0,alg=0)
    
    sphericalgrid(vertices,ren=ren,sphere_radius=0.05,spheres_on=1,sphere_color=(0,0.5,1), tubes_on=0,surface_on=1,surface_color=(1,1,1))
    
    ren.AddActor(plane(orig=(-3.5,-3.5,0),point1=(-3.5,3.5,0),point2=(3.5,-3.5,0),color=(0.1,0.2,0.4)))
    #ren.AddActor(plane(orig=(-3.5,-3.5,0),point1=(-3.5,3.5,0),point2=(3.5,-3.5,0),color=(1.0,1.0,1.0)))

    act=torus(scalex=4,scaley=4,scalez=4)

    act.RotateX(90)
    ren.AddActor(act)

    #ren.AddActor(axes())

    ren.ResetCamera()

    print 'Get Camera Position'
    print ren.GetActiveCamera().GetPosition()

    ren.GetActiveCamera().Roll(-20)
    ren.GetActiveCamera().Zoom(1.1)        
    ren.GetActiveCamera().Elevation(-60)

    #plane2=ren.GetActors().GetLastActor()
    #ren.RemoveActor(plane2)
    #ren.AddActor(plane(orig=(-3.0,-3.0,0),point1=(-3.0,3.0,0),point2=(3.0,-3.0,0),color=(0.1,0.2,0.4)))
      
    ap=AppThread(frame_type=0,ren=ren)
    TakePhotoVTK(ren=ren,magnification=10,filename='DTI_20.png',bgr_color=(0,0,0))
    print 'Check 1' 
    #TakeVideoVTK(ren=ren,N_Frames=10,size=(600,600))
    
    return ren
    
    
def TakeVideoVTK(ren=None,N_Frames=10,magnification=1,size=(125,125),bgr_color=(0.1,0.2,0.4)):
    
    if ren==None:
        ren = vtk.vtkRenderer()
   
    ren.SetBackground(bgr_color)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(size)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    #ren.GetActiveCamera().Azimuth(180)   

    '''
    # We'll set up the view we want.
    ren.GetActiveCamera().SetPosition(0, 1, 0)
    ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
    ren.GetActiveCamera().SetViewUp(0, 0, 1)

    # Let the renderer compute a good position and focal point.
    ren.ResetCamera()
    ren.GetActiveCamera().Dolly(1.4)
    ren.ResetCameraClippingRange()
    '''

    renderLarge = vtk.vtkRenderLargeImage()
    renderLarge.SetInput(ren)
    renderLarge.SetMagnification(magnification)
    renderLarge.Update()

    # We write out the image which causes the rendering to occur. If you
    # watch your screen you might see the pieces being rendered right
    # after one another.
    '''
    writer = vtk.vtkTIFFWriter()
    writer.SetInputConnection(renderLarge.GetOutputPort())
    writer.SetFileName("/home/eg01/Devel/largeImage.tif")
    writer.Write()
    '''
    
    writer = vtk.vtkPNGWriter()    
    
    ang=0    
    
    for i in range(N_Frames):
        
        ren.GetActiveCamera().Azimuth(ang)
        
        renderLarge = vtk.vtkRenderLargeImage()
        renderLarge.SetInput(ren)
        renderLarge.SetMagnification(magnification)
        renderLarge.Update()
        
        writer.SetInputConnection(renderLarge.GetOutputPort())
        filename='images/'+str(1000000+i)+'.png'
        writer.SetFileName(filename)
        writer.Write()               
        
        ang=+10

    
def TakePhotoVTK(ren=None,size=(125,125),magnification=10,filename="image.png",bgr_color=(0.1,0.2,0.4)):
    
    if ren==None:
        ren = vtk.vtkRenderer()
   
    ren.SetBackground(bgr_color)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(size)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    #ren.GetActiveCamera().Azimuth(180)   

    '''
    importer = vtk.vtk3DSImporter()
    importer.SetRenderWindow(renWin)
    #importer.SetFileName(VTK_DATA_ROOT + "/Data/Viewpoint/iflamigm.3ds")
    importer.SetFileName("/usr/share/VTKData/Data/Viewpoint/iflamigm.3ds")
    importer.ComputeNormalsOn()
    importer.Read()
    '''
    '''
    position=(0,0,0)
    height=1
    
    cone = vtk.vtkConeSource() 
    cone.SetHeight(height)
    
    conem = vtk.vtkPolyDataMapper()
    conem.SetInput(cone.GetOutput())   
 
    conea = vtk.vtkActor()
    conea.SetMapper(conem)
    conea.SetPosition(position)

    ren.AddActor(conea)
    '''

    '''
    # We'll set up the view we want.
    ren.GetActiveCamera().SetPosition(0, 1, 0)
    ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
    ren.GetActiveCamera().SetViewUp(0, 0, 1)

    # Let the renderer compute a good position and focal point.
    ren.ResetCamera()
    ren.GetActiveCamera().Dolly(1.4)
    ren.ResetCameraClippingRange()
    '''

    renderLarge = vtk.vtkRenderLargeImage()
    renderLarge.SetInput(ren)
    renderLarge.SetMagnification(magnification)
    renderLarge.Update()

    # We write out the image which causes the rendering to occur. If you
    # watch your screen you might see the pieces being rendered right
    # after one another.
    '''
    writer = vtk.vtkTIFFWriter()
    writer.SetInputConnection(renderLarge.GetOutputPort())
    writer.SetFileName("/home/eg01/Devel/largeImage.tif")
    writer.Write()
    '''
    
    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(renderLarge.GetOutputPort())
    writer.SetFileName(filename)
    writer.Write()

def gridpointscube(start=-1,end=1, N=3):

    x=sp.linspace(-1,1,N)

    [X,Y]=sp.meshgrid(x,x)
    X=X.ravel()
    Y=Y.ravel()

    lZ=X.size

    print lZ

    pts=[]

    for z in x:

      Z=sp.repeat(z,lZ)
      
      A=sp.vstack((X,Y,Z)).T                  
      pts.append(A)
    
    pts=sp.vstack(pts)
    #print pts
   

    return pts

def loadpolydata(filename,color=(0,0.2,1),opacity=0.5,specular=1,decim_smooth=0,reduction=0.5,topology=0,iterations=50,cut=1,plane_normal=(0,1,0),radius=10):

    '''
    Loads Polydata vtk files and returns an actor
    '''
    
    reader = vtk.vtkPolyDataReader() 

    reader.SetFileName(filename)
    reader.Update()

    if decim_smooth: 
        
        deci=vtk.vtkDecimatePro()
        deci.SetInput(reader.GetOutput())
        deci.SetTargetReduction(reduction)
        
        if topology:
            deci.PreserveTopologyOn()
        else:
            deci.PreserveTopologyOff()
            
        smoother=vtk.vtkSmoothPolyDataFilter()
        smoother.SetInput(deci.GetOutput())
        smoother.SetNumberOfIterations(iterations)
    
        normals=vtk.vtkPolyDataNormals()
        normals.SetInput(smoother.GetOutput())
        normals.FlipNormalsOn()
    
    if cut:
        
        #sph=vtk.vtkSphere()
        #sph.SetRadius(radius)
        sph=vtk.vtkPlane()
        sph.SetOrigin(0,0,0)
        sph.SetNormal(plane_normal)
        
        cutter= vtk.vtkCutter()
        cutter.SetInput(reader.GetOutput())
        cutter.SetCutFunction(sph)
     
        print 'OK1'
        
        
    isom = vtk.vtkPolyDataMapper()
    
    if decim_smooth:
        isom.SetInput(normals.GetOutput())
    else:
        if cut:
            isom.SetInput(cutter.GetOutput())
            #scal=reader.GetOutput().GetPointData().GetScalars()
            print 'GetScalarRange ',reader.GetOutput().GetScalarRange()
            
            isom.SetScalarRange(0,1)
            print 'OK2'
        else:
            isom.SetInputConnection(reader.GetOutputPort())
    
    isoa = vtk.vtkActor()
    isoa.SetMapper(isom)

    isoa.GetProperty().SetColor(color)
    
    isoa.GetProperty().SetOpacity(opacity)
    isoa.GetProperty().SetSpecular(specular)

    return isoa


def ian_3dsinc():
    
    #x=sp.arange(-4*sp.pi,4*sp.pi,0.1)    
    Xin, Yin = sp.mgrid[-4*sp.pi:4*sp.pi:0.1, -4*sp.pi:4*sp.pi:0.1]
    Zin = sp.sinc(sp.sqrt(Xin**2 + Yin**2))
    
    print(Zin.shape)
    
    Xin=np.ravel(Xin)
    Yin=np.ravel(Yin)
    Zin=10*np.ravel(Zin)
    
    ps=sp.zeros((Xin.size,3))
    for i in sp.arange(Xin.size):
        ps[i,:]=sp.array([Xin[i],Yin[i],Zin[i]])

    ren=surfaceplot(ps)
    label(ren,text='Ian and Lights.',pos=(0,0,10),color=(0,1,0),scale=(1,1,1))
    
    
    print('Before AppThread')
    ap=AppThread(frame_type=0,ren=ren)
    print('After AppThread')

    FrameThread(frame_type=1,title="Second Frame",width=300,height=300,autoStart=1,ren=None)

def normalize0255(arr):
    m=arr.min()    
    M=arr.max()
    arr=255/(M-m)*arr
    
    print 'New m ',arr.min()
    print 'New M ', arr.max()
    
    return arr

def testtube():


    point1=np.array([0,0,0])
    point2=np.array([1,1,1])
    
    ren=vtk.vtkRenderer()
    ren.AddActor(tube(point1,point2))
    ren.AddActor(axes())
    
    ap=AppThread(frame_type=0,ren=ren,width=1024,height=800)
    

def testpipeplot():

      
    ren=renderer()
    #''' Works
    x,y,z,u,v,w,r=sp.array([-1,0]),sp.array([0,-1]),sp.array([0,0]),sp.array([1,0]),sp.array([0,1]),sp.array([0,0]),sp.array([0.2,0.5]) 
   
    colr,colg,colb=sp.array([1,0]),sp.array([0,1]),sp.array([0,0])
    
    opacity=sp.array([0.6,0.4])
    #'''

    ''' Works
    x,y,z,u,v,w,r=sp.array([-1]),sp.array([0]),sp.array([0]),sp.array([1]),sp.array([0]),sp.array([0]),sp.array([0.2]) 
   
    colr,colg,colb=sp.array([1]),sp.array([0]),sp.array([0])
    
    opacity=sp.array([0.6])
    '''

    ''' Works
    x,y,z,u,v,w,r=-1,0,0,1,0,0,0.2 
   
    colr,colg,colb=1,0,0
    
    opacity=0.6
    '''
        
    pipeplot(ren,x,y,z,u,v,w,r,colr,colg,colb,opacity)
    
    ren.ResetCamera()
    ap=AppThread(frame_type=0,ren=ren,width=1024,height=800)    
    
def testtensor():
    
    R=sp.array([[2, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1] ])
    
    
    #Mat=sp.identity(4)
    #Mat[0:3,0:3]=R
    
    ren=renderer()
    ren.AddActor(ellipsoid(R))
    ren.AddActor(axes(scale=(5,5,5)))   

    '''
    Other properties
 
    SetAmbient
    SetDiffuse
    SetSpecularPower
    SetInterpolationToGouraud
    SetInterpolationToPhong
    SetInterpolationToFlatq
    SetAmbientColor
    SetDiffuseColor
    SetSpecularColor
    '''
    
    
    ap=AppThread(frame_type=0,ren=ren,width=1024,height=800)
 
def testtriangulate():
    
    ren=renderer()    
    ren.AddActor(triangulate(sp.rand(4,3)))
    ap=AppThread(ren=ren)

def testtrajectories_fast():
    
    ren=renderer()
    #ren.AddActor(trajectories_fast())
    act=trajectories_fast()
    #act.GetProperty().SetOpacity(0.01)
    ren.AddActor(act)
    ap=AppThread(ren=ren)

def experiment_multiple_volumes():
    
    #import diffusion as df
    import form as frm
    
    dname='/backup/Data/Adam/multiple_transp_volumes'

    #fname1=dname + '/' + 'rcglm_clustertest2.nii'
    fname1=dname + '/' + 'ClusterMapNoNaNs.nii'
    fname2=dname + '/' + 'EPI.nii'
    fname3=dname + '/' + 'meanoutput.nii'

    fname4=dname + '/' + 'rmask.nii'
    fname5=dname + '/' + 'rsingle_subj_T1.nii'
    fname6=dname + '/' + 'rsingle_subj_T1_brain_mesh.vtk'
    fname7=dname + '/' + 'rsingle_subj_T1_brain_inskull_mesh.nii'
    #fname8=dname + '/' + 'test_out_mesh.vtk'

    fname8=dname + '/freesurfer_trich/lh.pial.vtk'
    fname9=dname + '/freesurfer_trich/rh.pial.vtk'
    
    fname10=dname + '/freesurfer_trich/lh.dpial.ribbon.nii'
    fname11=dname + '/freesurfer_trich/rh.dpial.ribbon.nii'

    fname12=dname + '/'+'rrT1.nii'

    #fname12=dname + '/'+'rcglm_clustertest.nii'
    fname13=dname + '/freesurfer_trich/brain.finalsurfs.nii'
    fname14=dname + '/rsingle_subj_T1_brain_outskin_mesh.nii'

    '''
    We used spm corregistration to cooregister rT1.nii to T1_free.nii

    The translation matrix was
      2 0  0 35.985
      0 0 -2 222.011
     -0 2 -0 18.018
    '''


    #'''
    print(fname1)
    #arr1,voxsz1 = df.loadnifti(fname1)
    arr1,voxsz1,aff1 = frm.loadvol(fname1)
    print(arr1.shape)
    print(voxsz1)
    print(arr1.max())
    print(aff1)

    print(fname12)
    arr12,voxsz12,aff12 = frm.loadvol(fname12)
    print(arr12.shape)
    print(voxsz12)
    print(arr12.max())
    print(aff12)

    '''
    opacitymap1=np.array([[ 0.0, 0.0],              
                [255.0, 1]])

    colormap1=np.array([[  0.0, 0.0, 0.0, 0.0],
                        [102.0, 1.0, 1.0, 0.0],
                        [255.0, 1.0, 0.0, 0.0]])
    '''

    opacitymap1=np.array([[ 0.0, 0.0],              
                [255.0, 1]])

    colormap1=np.array([[  0.0, 0.0, 0.0, 0.0],
                        [102.0, 1.0, 1.0, 0.0],
                        [152.0, 1.0, 0.0, 0.0],
                        [255.0, 1.0, 0.0, 0.0]])
    '''

    opacitymap2=np.array([[ 0.0, 0.0],
                    [ 28.0, 0.005],
                    [ 129.0, 0.005],	
                    [255.0, 0.005]])

    colormap2=np.array([[  0.0, 0.0, 0.0, 0.0],
                    [28.0, 0.0, 1.0, 1.0], 
                    [ 64.0, 0.0, 0.0, 1.0],
                    #[128.0, 0.0, 1.0, 0.0],
                    #[192.0, 1.0, 0.0, 0.0],
                    [255.0, 0.0, 1.0, 1.0]])

    '''
    
    opacitymap2=np.array([[ 0.0, 0.0],
                    [ 28.0, 0.05],
                    [ 129.0, 0.05],	
                    [255.0, 0.05]])

    colormap2=np.array([[  0.0, 0.0, 0.0, 0.0],
                    [28.0, 0.0, 1.0, 1.0], 
                    [ 64.0, 0.0, 0.0, 1.0],
                    [128.0, 0.0, 1.0, 0.0],
                    [192.0, 1.0, 0.0, 0.0],
                    [255.0, 1.0, 0.5, 0.3]])


    '''
    opacitymap2=np.array([[ 0.0, 0.0],
                    [ 28.0, 0.001],
                    [ 129.0, 0.002],	
                    [255.0, 0.01]])

    colormap2=np.array([[  0.0, 0.0, 0.0, 0.0],
                    [28.0, 0.0, 1.0, 1.0], 
                    [ 64.0, 0.0, 0.0, 1.0],
                    [128.0, 0.0, 1.0, 0.0],
                    [192.0, 1.0, 0.0, 0.0],
                    [255.0, 1.0, 0.2, 0.2]])
    '''

    ren = vtk.vtkRenderer()

    ren.SetBackground((1,1,1))


    arr=arr1
    voxsz=voxsz1
    print 'Max ', arr.max(), ' Min ', arr.min(), ' Mean ', arr.mean()

    arrnext=arr12
    voxsznext=voxsz12
    print 'Max ', arrnext.max(), ' Min ', arrnext.min(), ' Mean ', arrnext.mean()
     
    
    
    #'''
    
    arr=normalize0255(arr)    
    
    hist,bin_edges=sp.histogram(arr,5,new=True)

    print(hist,bin_edges)
    
    hist2,bin_edges2=sp.histogram(arrnext,5,new=True)

    print(hist2,bin_edges2)
    
    arrnext=normalize0255(arrnext)        
    #'''
    vol=volume(arr, voxsz=voxsz, affine=aff1, maptype=1, iso=1, iso_thr=100, opacitymap=opacitymap1, colormap=colormap1)    
    vol.RotateX(-180)
    ren.AddVolume(vol)
    #'''
    
    '''
    vol2=volume(arrnext, voxsz=voxsz, affine=aff12, maptype=1, opacitymap=opacitymap2, colormap=colormap2)    
    vol2.RotateX(-180)
    ren.AddVolume(vol2)
    '''
    
    #volumeplot2(arr*255, ren=ren, voxsz=voxsz, affine=aff1, maptype=1, opacitymap=opacitymap1, colormap=colormap1)

    #volumeplot2(arrnext, ren=ren, voxsz=voxsznext, affine=aff12, maptype=1, opacitymap=opacitymap2, colormap=colormap2)

    #Show internal and external
    
    '''
    lh=loadpolydata(fname8,color=(0,0,0),opacity=1,specular=1,decim_smooth=0,reduction=0.9,topology=0,iterations=1,cut=1,plane_normal=(0,1,0),radius=10)

    lh2=loadpolydata(fname8,color=(0,0,0),opacity=1,specular=1,decim_smooth=0,reduction=0.9,topology=0,iterations=1,cut=1,plane_normal=(-1,1,0),radius=10)
    
    lh3=loadpolydata(fname8,color=(0,0,0),opacity=1,specular=1,decim_smooth=0,reduction=0.9,topology=0,iterations=1,cut=1,plane_normal=(1,1,0),radius=10)
    '''
    
    #rh=loadpolydata(fname9,color=(0,0.2,1),opacity=1,specular=1,decim_smooth=1,reduction=0.9,topology=1,iterations=10,cut=0)
    
    lh=loadpolydata(fname8,color=(0,0.2,1),opacity=1,specular=1,decim_smooth=0,reduction=0.9,topology=1,iterations=10,cut=0)
    rh=loadpolydata(fname9,color=(0,0.2,1),opacity=0.1,specular=1,decim_smooth=1,reduction=0.9,topology=1,iterations=10,cut=0)
    
    #rh=loadpolydata(fname9,color=(0,0.2,1),opacity=0.1,specular=1,decim_smooth=1,reduction=0.9,topology=1,iterations=10,cut=0)
    #rh=loadpolydata(fname9,color=(0,0.2,1),opacity=0.1,specular=1,decim_smooth=1,reduction=0.9,topology=0,iterations=10,cut=0)

    #lh.GetProperty().SetRepresentationToWireframe()
    rh.GetProperty().SetRepresentationToWireframe()

    #    lh.GetProperty().SetLineWidth(2)
    #lh2.GetProperty().SetLineWidth(2)
    #lh3.GetProperty().SetLineWidth(2)

    ren.AddActor(rh)
    ren.AddActor(lh)
    #ren.AddActor(lh2)
    #ren.AddActor(lh3)
    
    #ren.AddActor(axes(scale=(91,109,91)))
    #ren.AddActor(axes(scale=(100,100,100)))
    '''
    Other properties
 
    SetAmbient
    SetDiffuse
    SetSpecularPower
    SetInterpolationToGouraud
    SetInterpolationToPhong
    SetInterpolationToFlatq
    SetAmbientColor
    SetDiffuseColor
    SetSpecularColor
    '''
 
    ap=AppThread(frame_type=0,ren=ren,width=1024,height=800)

    #TakePhotoVTK(ren=ren,bgr_color=(0,0,0))

    return ren

if __name__ == "__main__":    
        
    #t=TestThread()
    #t.start()
    #NoAppThread()    
    
    #testplot()
    #threading_test()
    #testline()
    #testplot()
    #ian_3dsinc()
    #testvolume()
    #testcontour()
    #testspline()
    #testplatonicsolid()
    #testpointsource()
    #testsphere()
    #testtegmark()
    #testline()
    #print(sphereeven())
    
    #testsphericalgrid()
    #testcubicgrid()
    #testtube()
    experiment_multiple_volumes()
    #testtensor()
    #testtriangulate()
    #testtrajectories_fast()
    
    
    
    