import numpy as np

CAMERAMATRIX = np.array([[  1.15730136e+03,   0.00000000e+00,   6.67042380e+02],
                        [  0.00000000e+00,   1.15270113e+03,   3.90488964e+02],
                        [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
DISTORTIONCOEFFICIENTS = np.array([[ -2.38666546e-01,  -2.98287548e-02,  -5.14437800e-04,  -1.76570650e-04, -4.55111368e-02]])

print("CAMERAMATRIX:", CAMERAMATRIX, ", DISTORTIONCOEFFICIENTS:", DISTORTIONCOEFFICIENTS)

import cv2
import matplotlib.image as mpimage
import matplotlib.pyplot as plt

def undistortImage(rgbImage):
    undistortedImage = cv2.undistort(rgbImage, CAMERAMATRIX, DISTORTIONCOEFFICIENTS, None, CAMERAMATRIX)
    return undistortedImage
