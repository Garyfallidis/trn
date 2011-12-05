def showline(myline):
    from dipy.viz import fos
    r = fos.ren()
    fos.add(r,fos.line(myline,fos.blue,opacity=0.5))
    fos.show(r)