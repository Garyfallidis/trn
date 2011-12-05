#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Author: Eleftherios Garyfallidis
    Info:
        Alignment & Registration of 3d volumes at the moment using ITK.
        
        Information on how to install Insight Toolkit(ITK)  and extras for compatibility with scipy can be found here.
            http://paulnovo.org/node/2
            http://paulnovo.org/WrapITKTutorial
    Example:
  
'''

try:
    import itk
except:
    print('Insight Toolkit is not installed')
    print('To install it go here http://paulnovo.org/node/2')
    
    

try:
    import scipy as sp
except:
    print('Scipy is not installed')

if __name__ == "__main__":
    
    image_type = itk.Image[itk.UC, 3]
    '''
    Working with images with ITK is great, but when working in python, you want all the functionality of python. Fortunately, converting ITK images to python arrays is simple with the python-insighttoolkit-extras package.
    '''

    itk_py_converter = itk.PyBuffer[image_type]
    #image_array = itk_py_converter.GetArrayFromImage( reader.GetOutput() )
    '''
    How about converting a 10x10x10 python array to an ITK image.
    '''

    another_image_array = sp.zeros( (10,10,10) )
    itk_image = itk_py_converter.GetImageFromArray( another_image_array )
    