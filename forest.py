import direct.directbase.DirectStart
from pandac.PandaModules import LineSegs




#Load the first environment model
environ = loader.loadModel("models/environment")
environ.reparentTo(render)
environ.setScale(0.25,0.25,0.25)
environ.setPos(-8,42,0)


#Run the tutorial
run()
