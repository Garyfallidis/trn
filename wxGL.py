#!/usr/bin/env python   
# -*- coding: utf-8 -*-
  
import wx   
import sys  
from wx import glcanvas   
from OpenGL.GL import *   
from OpenGL.GLUT import *   
  
class MyCanvasBase(glcanvas.GLCanvas):   
    def __init__(self, parent):   
        glcanvas.GLCanvas.__init__(self, parent, -1)   
        self.init = False  
        # initial mouse position   
        self.lastx = self.x = 30   
        self.lasty = self.y = 30   
        self.size = None   
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)   
        self.Bind(wx.EVT_SIZE, self.OnSize)   
        self.Bind(wx.EVT_PAINT, self.OnPaint)   
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)   
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)   
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)   
  
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
        glTranslatef(0.0, 0.0, -2.0)   
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
  
  
class MainWindow(wx.Frame):   
    def __init__(self, parent = None, id = -1, title = "PyOpenGL Example 1"):   
        # Init   
        wx.Frame.__init__(   
                self, parent, id, title, size = (1024,800),   
                style = wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE   
        )   
  
        # TextCtrl   
        # self.control = wx.TextCtrl(self, -1, style = wx.TE_MULTILINE)   
           
        #self.control = ConeCanvas(self)   
  
        box = wx.BoxSizer(wx.HORIZONTAL)   
        box.Add(ConeCanvas(self), 1, wx.EXPAND)   
        box.Add(CubeCanvas(self), 1, wx.EXPAND)   
  
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
  
app = wx.PySimpleApp()   
frame = MainWindow()   
app.MainLoop()   
  
# destroying the objects, so that this script works more than once in IDLEdieses Beispiel   
del frame   
del app   
