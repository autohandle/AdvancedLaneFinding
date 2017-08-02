import numpy as np
import cv2

fullImagePerspectiveTransforms= {'./test_images/straight_lines1.jpg': 
                                 np.array([
                                     [ -5.31244816e-01,  -1.51726430e+00,   9.84704530e+02],
                                     [ -1.82076576e-14,  -1.63799355e+00,   7.18279930e+02],
                                     [ -3.03576608e-17,  -2.35939109e-03,   1.00000000e+00]]),
                                 './test_images/straight_lines2.jpg': np.array([
                                     [ -5.44776959e-01,  -1.50649420e+00,   9.77714734e+02],
                                     [ -1.19904087e-14,  -1.65418663e+00,   7.20006876e+02],
                                     [ -1.99493200e-17,  -2.38024185e-03,   1.00000000e+00]])
                                }
# TOPOFTRAPEZOID=.4
fullImagePerspectiveTransforms= {'./test_images/straight_lines2.jpg': np.array([[ -5.44706987e-01,  -1.50703683e+00,   9.78066903e+02],
       [ -6.43929354e-15,  -1.76349840e+00,   7.90995625e+02],
       [ -1.25767452e-17,  -2.38013403e-03,   1.00000000e+00]]), './test_images/straight_lines1.jpg': np.array([[ -5.31333757e-01,  -1.51800368e+00,   9.85184385e+02],
       [ -7.99360578e-15,  -1.74486685e+00,   7.87582977e+02],
       [ -1.12757026e-17,  -2.35952813e-03,   1.00000000e+00]])}
# TOPOFTRAPEZOID=.6
fullImagePerspectiveTransforms= {'./test_images/straight_lines1.jpg': 
                                 np.array([
                                     [ -5.31513375e-01,  -1.51949207e+00,   9.86150355e+02],
                                     [ -3.99680289e-15,  -1.85192138e+00,   8.56944796e+02],
                                     [ -6.93889390e-18,  -2.35980489e-03,   1.00000000e+00]]),
                                 './test_images/straight_lines2.jpg': 
                                 np.array([
                                     [ -5.44565668e-01,  -1.50812703e+00,   9.78774441e+02],
                                     [ -3.55271368e-15,  -1.87266770e+00,   8.61938220e+02],
                                     [ -6.50521303e-18,  -2.37991628e-03,   1.00000000e+00]])}
print("fullImagePerspectiveTransforms:", fullImagePerspectiveTransforms, ", fullImagePerspectiveTransforms.keys:", fullImagePerspectiveTransforms.keys())
fullImageInverseTransforms= {'./test_images/straight_lines2.jpg': 
                             np.array([
                                 [  1.75205780e-01,  -8.05336297e-01,   5.22663257e+02],
                                 [ -1.11022302e-16,  -5.33997399e-01,   4.60272863e+02],
                                 [ -2.38524478e-18,  -1.27086937e-03,   1.00000000e+00]]),
                             './test_images/straight_lines1.jpg':
                             np.array([
                                 [  1.73013823e-01,  -8.20494982e-01,   5.32501243e+02],
                                 [  1.22124533e-15,  -5.39979734e-01,   4.62732819e+02],
                                 [  2.81892565e-18,  -1.27424681e-03,   1.00000000e+00]])
                            }
print("fullImageInverseTransforms:", fullImageInverseTransforms, ", fullImageInverseTransforms.keys:", fullImageInverseTransforms.keys())

PERSPECTIVETRANSFORM=fullImagePerspectiveTransforms[list(fullImagePerspectiveTransforms.keys())[1]]
print("PERSPECTIVETRANSFORM:", PERSPECTIVETRANSFORM)
INVERSETRANSFORM=fullImageInverseTransforms[list(fullImageInverseTransforms.keys())[1]]
print("INVERSETRANSFORM:", INVERSETRANSFORM)

SCALEX=1
SCALEY=1

def transform(undistortedRgbImage, perspectiveTransform=PERSPECTIVETRANSFORM):
    targetImageSize=(SCALEX*undistortedRgbImage.shape[1], SCALEY*undistortedRgbImage.shape[0])
    return cv2.warpPerspective(undistortedRgbImage, perspectiveTransform, targetImageSize, flags=cv2.INTER_LINEAR)

def unwarp(distortedRgbImage):
    return transform(distortedRgbImage, INVERSETRANSFORM)

import UndistortImage
def undistortAndTransform(rgbImage):
    return transform(UndistortImage.undistortImage(rgbImage))

import matplotlib.image as mpimage
def testUndistortAndTransform():
    testImageName='./test_images/test1.jpg'
    print("testImageName:", testImageName)
    testRgbImage=mpimage.imread(testImageName)
    warpedImage=undistortAndTransform(testRgbImage)
    print("testImageName: ",testImageName, ", testRgbImage.shape:", testRgbImage.shape, ", type:", testRgbImage.dtype)
    print("testImageName: ",testImageName, ", warpedImage.shape:", warpedImage.shape, ", type:", warpedImage.dtype)
    unwarpedImage=unwarp(testRgbImage)
    print("testImageName: ",testImageName, ", unwarpedImage.shape:", unwarpedImage.shape, ", type:", unwarpedImage.dtype)

print ("dir:", dir(),", dir(UndistortImage):", dir(UndistortImage))
testUndistortAndTransform()