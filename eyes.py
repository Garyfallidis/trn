#!/usr/bin/env python   
# -*- coding: utf-8 -*-
  
import wx   
import sys  
from wx import glcanvas   
from OpenGL.GL import *   
from OpenGL.GLUT import *   
from OpenGL.GLU import *

  
class MyCanvasBase(glcanvas.GLCanvas):   



    def __init__(self, parent):   

        print 'Loading Data ...'
        filename='/home/eg01/Backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dti_FACT.trk'
        filename2='/home/eg01/Backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dti_RK2.trk'
        
        import form
        
        self.trajs=form.loadtrk(filename2)

        print 'Data Loaded.'

        glcanvas.GLCanvas.__init__(self, parent, -1)   
        self.init = False  
        # initial mouse position   
        self.lastx = self.x = 30   
        self.lasty = self.y = 30   
        
        
        self.xw=0.0
        self.yw=0.0
        self.zw=0.0

        #self.displayListId=glGenLists(1)

        #self.trajs=[]

        self.size = None   
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)   
        self.Bind(wx.EVT_SIZE, self.OnSize)   
        self.Bind(wx.EVT_PAINT, self.OnPaint)   
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)   
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)   
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)  
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
  
    def OnEraseBackground(self, event):   
        pass # Do nothing, to avoid flashing on MSW.   
  
    def OnSize(self, event):   
        size = self.size = self.GetClientSize()   
        if self.GetContext():   
            self.SetCurrent()   
            glViewport(0, 0, size.width, size.height)   
        event.Skip()   
  
    def OnPaint(self, event):   
        dc = wx.PaintDC(self)   
        self.SetCurrent()   
        if not self.init:   
            self.InitGL()   
            self.init = True  
        self.OnDraw()   
  
    def OnMouseDown(self, evt):   
        self.CaptureMouse()   
        self.x, self.y = self.lastx, self.lasty = evt.GetPosition()   
  
    def OnMouseUp(self, evt):   
        self.ReleaseMouse()   
  
    def OnMouseMotion(self, evt):   
        if evt.Dragging() and evt.LeftIsDown():   
            self.lastx, self.lasty = self.x, self.y   
            self.x, self.y = evt.GetPosition()   
            self.Refresh(False)   

    def OnKeyDown(self,evt):
        if evt.KeyCode()==wx.WXK_UP:
            print 'UP Arrow'
            self.zw-=1
            print (self.xw,self.yw,self.zw)
            self.OnDraw()

        elif evt.KeyCode()==wx.WXK_DOWN:
            print 'DOWN Arrow'
            self.zw+=1
            print (self.xw,self.yw,self.zw)
            self.OnDraw()
        elif evt.KeyCode()==wx.WXK_LEFT:
            print 'Left Arrow'
            self.xw-=1
            print (self.xw,self.yw,self.zw)
            self.OnDraw()
        elif evt.KeyCode()==wx.WXK_RIGHT:
            print 'Right Arrow'
            self.xw+=1
            print (self.xw,self.yw,self.zw)
            self.OnDraw()
        elif evt.KeyCode()==wx.WXK_HOME:
            print 'HOME'
            self.yw-=1
            print (self.xw,self.yw,self.zw)
            self.OnDraw()
        elif evt.KeyCode()==wx.WXK_END:
            print 'END'
            self.yw+=1
            print (self.xw,self.yw,self.zw)
            self.OnDraw()
      
        print evt.GetPosition()


class SceneCanvas(MyCanvasBase):
    
    def InitGL(self):

        glEnable(GL_CULL_FACE);


        #load data

        traj_no=len(self.trajs)

        glClearColor( 1, 1,1, 0 )
        #glViewport( 0, 0,1024, 800 )
        glMatrixMode( GL_PROJECTION )
        glLoadIdentity( )
        gluPerspective( 60.0, float(1024)/float(800), 0.1, 300.0 )
        #glDepthMask(1) 
        glMatrixMode(GL_MODELVIEW)   
        glLoadIdentity()
        #self.xw=-2
        #self.yw=-2
        #self.zw=-94
        self.xw=-12
        self.yw=-10
        self.zw=-17
        
        self.objects_added=0

        
        #glTranslatef(self.xw, self.yw, self.zw)

        glutInit()

    def AddObjects(self):
        
        glNewList(1, GL_COMPILE)
        
        glBegin(GL_LINES)
        
        glColor3f(1.0,0.0,0.0)			# Red
        glVertex3f(0.0, 0.0, 0.0) # origin of the line
        glVertex3f(100.0, 0.0, 0.0) # ending point of the line

        glColor3f(0.0,1.0,0.0)			# Green
        glVertex3f(0.0, 0.0, 0.0) # origin of the line
        glVertex3f(0.0, 100.0, 0.0) # ending point of the line

        glColor3f(0.0,0.0,1.0)			# Blue
        glVertex3f(0.0, 0.0, 0.0) # origin of the line
        glVertex3f(0.0, 0.0, 100.0) # ending point of the line
        
        #glVertex3f(1.0, 1.0, 0.0) # ending point of the line

        glEnd()

        
        glBegin(GL_LINES)

        trajs_no=len(self.trajs)

        cnt=0
        for traj in self.trajs:
        
        #for point in traj[:-1]:
        #glVertex3f(point[0], point[1], point[2]) # origin of the line
        #glVertex3f(100.0, 0.0, ) # ending point of the line

            #print traj
            #print cnt
            trl=len(traj)

            if trl> 80:

                for i in xrange(trl-1):

                    point1=traj[i]
                    point2=traj[i+1]

                    glColor3f(1.0,0.2,0.2)
                    glVertex3f(point1[0]/10.0, point1[1]/10.0, point1[2]/10.0)
                    glVertex3f(point2[0]/10.0, point2[1]/10.0, point2[2]/10.0)
                      
                    cnt+=1
                    #print str(100*cnt/float(trajs_no))+'%'

        glEnd()   
        glEndList() 
        
        self.objects_added=1
        
    def OnDraw(self):

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)   
        glLoadIdentity()

        #glTranslatef(self.xw, self.yw, self.zw)
        #glLoadIdentity()

        #'''
        #glPushMatrix()

        glTranslatef(self.xw, self.yw, self.zw)
        #'''
        if self.objects_added==0:
            
            
            self.AddObjects()
            self.objects_added=1
        else:
            glutWireTeapot(15)
            #glPopMatrix()
            
            print 'Call list'
            glCallList(1);
        
        self.SwapBuffers()


'''
class CubeCanvas(MyCanvasBase):   
    def InitGL(self):   
        # set viewing projection   
        glMatrixMode(GL_PROJECTION)   
        glFrustum(-0.5, 0.5, -0.5, 0.5, 1.0, 3.0)   
  
        # position viewer   
        glMatrixMode(GL_MODELVIEW)   
        glTranslatef(0.0, 0.0, -2.0)   
  
        # position object   
        glRotatef(self.y, 1.0, 0.0, 0.0)   
        glRotatef(self.x, 0.0, 1.0, 0.0)   
  
        glEnable(GL_DEPTH_TEST)   
        glEnable(GL_LIGHTING)   
        glEnable(GL_LIGHT0)   
  
    def OnDraw(self):   
        # clear color and depth buffers   
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)   
  
        # draw six faces of a cube   
        glBegin(GL_QUADS)   
        glNormal3f( 0.0, 0.0, 1.0)   
        glVertex3f( 0.5, 0.5, 0.5)   
        glVertex3f(-0.5, 0.5, 0.5)   
        glVertex3f(-0.5,-0.5, 0.5)   
        glVertex3f( 0.5,-0.5, 0.5)   
  
        glNormal3f( 0.0, 0.0,-1.0)   
        glVertex3f(-0.5,-0.5,-0.5)   
        glVertex3f(-0.5, 0.5,-0.5)   
        glVertex3f( 0.5, 0.5,-0.5)   
        glVertex3f( 0.5,-0.5,-0.5)   
  
        glNormal3f( 0.0, 1.0, 0.0)   
        glVertex3f( 0.5, 0.5, 0.5)   
        glVertex3f( 0.5, 0.5,-0.5)   
        glVertex3f(-0.5, 0.5,-0.5)   
        glVertex3f(-0.5, 0.5, 0.5)   
  
        glNormal3f( 0.0,-1.0, 0.0)   
        glVertex3f(-0.5,-0.5,-0.5)   
        glVertex3f( 0.5,-0.5,-0.5)   
        glVertex3f( 0.5,-0.5, 0.5)   
        glVertex3f(-0.5,-0.5, 0.5)   
  
        glNormal3f( 1.0, 0.0, 0.0)   
        glVertex3f( 0.5, 0.5, 0.5)   
        glVertex3f( 0.5,-0.5, 0.5)   
        glVertex3f( 0.5,-0.5,-0.5)   
        glVertex3f( 0.5, 0.5,-0.5)   
  
        glNormal3f(-1.0, 0.0, 0.0)   
        glVertex3f(-0.5,-0.5,-0.5)   
        glVertex3f(-0.5,-0.5, 0.5)   
        glVertex3f(-0.5, 0.5, 0.5)   
        glVertex3f(-0.5, 0.5,-0.5)   
        glEnd()   
  
        if self.size is None:   
            self.size = self.GetClientSize()   
        w, h = self.size   
        w = max(w, 1.0)   
        h = max(h, 1.0)   
        xScale = 180.0 / w   
        yScale = 180.0 / h   
        glRotatef((self.y - self.lasty) * yScale, 1.0, 0.0, 0.0);   
        glRotatef((self.x - self.lastx) * xScale, 0.0, 1.0, 0.0);   
  
        self.SwapBuffers()   
  
  
class ConeCanvas(MyCanvasBase):   
    def InitGL( self ):   
        glMatrixMode(GL_PROJECTION)   
        # camera frustrum setup   
        glFrustum(-0.5, 0.5, -0.5, 0.5, 1.0, 3.0)   
        glMaterial(GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])   
        glMaterial(GL_FRONT, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])   
        glMaterial(GL_FRONT, GL_SPECULAR, [1.0, 0.0, 1.0, 1.0])   
        glMaterial(GL_FRONT, GL_SHININESS, 50.0)   
        glLight(GL_LIGHT0, GL_AMBIENT, [0.0, 1.0, 0.0, 1.0])   
        glLight(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])   
        glLight(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])   
        glLight(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])   
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.2, 0.2, 0.2, 1.0])   
        glEnable(GL_LIGHTING)   
        glEnable(GL_LIGHT0)   
        glDepthFunc(GL_LESS)   
        glEnable(GL_DEPTH_TEST)   
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)   
        # position viewer   
        glMatrixMode(GL_MODELVIEW)   
        # position viewer   
        glTranslatef(0.0, 0.0, -2.0);   
        #   
        glutInit(sys.argv)   
  
    def OnDraw(self):   
        # clear color and depth buffers   
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)   
        # use a fresh transformation matrix   
        glPushMatrix()   
        # position object   
        #glTranslate(0.0, 0.0, -2.0)   
        glRotate(30.0, 1.0, 0.0, 0.0)   
        glRotate(30.0, 0.0, 1.0, 0.0)   
  
        glTranslate(0, -1, 0)   
        glRotate(250, 1, 0, 0)   
        glutSolidCone(0.5, 1, 30, 5)   
        glPopMatrix()   
        glRotatef((self.y - self.lasty), 0.0, 0.0, 1.0);   
        glRotatef((self.x - self.lastx), 1.0, 0.0, 0.0);   
        # push into visible buffer   
        self.SwapBuffers()   
  
'''  
class MainWindow(wx.Frame):   
    def __init__(self, parent = None, id = -1, title = "Eyes"):   
        # Init   
        wx.Frame.__init__(   
                self, parent, id, title, size = (1024,800),   
                style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE   
        )   
  
        # TextCtrl   
        # self.control = wx.TextCtrl(self, -1, style = wx.TE_MULTILINE)   
           
        #self.control = ConeCanvas(self)   
  
        box = wx.BoxSizer(wx.HORIZONTAL)   
        box.Add(SceneCanvas(self), 1, wx.EXPAND)   
        #box.Add(ConeCanvas(self), 1, wx.EXPAND)   
        #box.Add(CubeCanvas(self), 1, wx.EXPAND)   
  
        self.SetAutoLayout(True)   
        self.SetSizer(box)   
        self.Layout()   
  
        # StatusBar   
        self.CreateStatusBar()   
  
        # Filemenu   
        filemenu = wx.Menu()   
  
        # Filemenu - About   
        menuitem = filemenu.Append(-1, "&About", "Information about this program")   
        self.Bind(wx.EVT_MENU, self.OnAbout, menuitem) # here comes the event-handler   
        # Filemenu - Separator   
        filemenu.AppendSeparator()   
  
        # Filemenu - Exit   
        menuitem = filemenu.Append(-1, "E&xit", "Terminate the program")   
        self.Bind(wx.EVT_MENU, self.OnExit, menuitem) # here comes the event-handler   
  
        # Menubar   
        menubar = wx.MenuBar()   
        menubar.Append(filemenu,"&File")   
        self.SetMenuBar(menubar)   
  
        # Show   
        self.Show(True)   
  
    def OnAbout(self,event):   
        message = "Using PyOpenGL in wxPython"   
        caption = "About PyOpenGL Example"   
        wx.MessageBox(message, caption, wx.OK)   
  
    def OnExit(self,event):   
        self.Close(True)  # Close the frame.   
 
#

filename='/home/eg01/Backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dti_FACT.trk'
filename2='/home/eg01/Backup/Data/Eleftherios/CBU090133_METHODS/20090227_145404/Series_003_CBU_DTI_64D_iso_1000/dtk_dti_out/dti_RK2.trk'
  
#import form

#trajs=form.loadtrk(filename)


 
app = wx.PySimpleApp()   
frame = MainWindow()   
app.MainLoop()   
  
# destroying the objects, so that this script works more than once in IDLEdieses Beispiel   
del frame   
del app   
