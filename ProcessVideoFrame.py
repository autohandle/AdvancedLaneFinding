import numpy as np
import cv2
import matplotlib.image as mpimage
import matplotlib.pyplot as plt

import AnnotateImage
import EnhanceLaneMarkers
import ProcessImage
import PerspectiveTransform

TRIMLEFT=100 # x
TRIMTOP=0 # y
TRIMBOTTOM=0

MARGIN=100

def processVideoFrame(videoFrame, frameNumber, left_fit, right_fit):
    print("processVideoFrame-frameNumber: ",frameNumber, ", videoFrame.shape:", videoFrame.shape, ", type:", videoFrame.dtype)
    binaryImage,incrementalImages=EnhanceLaneMarkers.enhanceLaneMarkers(videoFrame, trimLeft=TRIMLEFT, trimTop=TRIMTOP, trimBottom=TRIMBOTTOM)
    #print("processVideoFrame-frameNumber: ",frameNumber, ", binaryImage.shape:", binaryImage.shape, ", type:", binaryImage.dtype)
    
    if frameNumber==0:
        [left_fit, right_fit], visualizationImage=ProcessImage.processInitialImage(binaryImage)
    else: # process a new frame by assuming the old 2nd order polynomial still fits the new frame
        [left_fit, right_fit], visualizationImage=ProcessImage.processImage(binaryImage, left_fit, right_fit)
    
    # Generate new x and y values for plotting (the new) 2nd order poly
    ploty = np.linspace(0, binaryImage.shape[0]-1, binaryImage.shape[0] )
 
    lanePolygonImage=AnnotateImage.createLanePolygonImage(visualizationImage.shape, MARGIN, ploty, left_fit, right_fit)
    #print("processVideoFrame-visualizationImage.shape:", visualizationImage.shape, ", type:", visualizationImage.dtype)
    #print("processVideoFrame-lanePolygonImage.shape:", lanePolygonImage.shape, ", type:", lanePolygonImage.dtype)
    annotatedVisualizationImage = cv2.addWeighted(visualizationImage, 1, lanePolygonImage, 0.3, 0) # combine the 2 images
    #print("processVideoFrame-annotatedVisualizationImage.shape:", annotatedVisualizationImage.shape, ", type:", annotatedVisualizationImage.dtype)
    
    transformedImage=incrementalImages['undistortedAndTransformedImage'] # 720,1280
    #print("processVideoFrame-transformedImage.shape:", transformedImage.shape, ", type:", transformedImage.dtype)
   
    filledLaneWarpedImage=AnnotateImage.fillLaneInTransformImage(transformedImage, visualizationImage, (TRIMLEFT,TRIMTOP), ploty, left_fit, right_fit)
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad_P, right_curverad_P=AnnotateImage.calculateRadiusCurveInPixelsSigned(y_eval, left_fit, right_fit)
    #print("processVideoFrame-left_curverad_P:", left_curverad_P, ", right_curverad_P:", right_curverad_P)
    # Example values: 1926.74 1908.48
    
    # For each y position generate random x position within +/-50 pix
    left_curverad, right_curverad=AnnotateImage.calculateRadiusCurveInMeters(y_eval, ploty, left_fit, right_fit)
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    
    carOffset=AnnotateImage.calculateCarOffset(binaryImage, left_fit, right_fit)
    #print ("processVideoFrame-carOffset:", carOffset)
   
    leftCurveAnnotation="left curve: %.2f(m)/%.2f(p)" % (left_curverad,left_curverad_P)
    rightCurveAnnotation="right curve: %.2f(m)/%.2f(p)" % (right_curverad,right_curverad_P)
    carOffsetAnnotation="car offset: %.2f(p)" % (carOffset)
    annotateWarpedImage = lambda axes : axes.text(0.1, 0.9, ("frame: "+str(frameNumber)
                                                           +"\n"+leftCurveAnnotation
                                                           +"\n"+rightCurveAnnotation
                                                           +"\n"+carOffsetAnnotation
                                                          ),
                                                  fontsize=16,
                                                  horizontalalignment='left',
                                                  verticalalignment='top',
                                                  transform = axes.transAxes,
                                                  color='w')
    emptyImage=np.zeros_like(transformedImage)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    filledLaneUnwarpedImage = cv2.warpPerspective(filledLaneWarpedImage, PerspectiveTransform.INVERSETRANSFORM, (filledLaneWarpedImage.shape[1], filledLaneWarpedImage.shape[0])) 
    # Combine the result with the original image
    filledLaneVideoFrame = cv2.addWeighted(videoFrame, 1, filledLaneUnwarpedImage, 0.5, 0)
  
    figure=AnnotateImage.plotFigure(filledLaneVideoFrame, annotatedVisualizationImage, filledLaneWarpedImage, annotateWarpedImage)
    return [left_fit,right_fit], figure

def testProcessVideoFrame():
    left_fit = None # x = left_fit[0] y**2 + left_fit[1] y + left_fit[2]
    right_fit = None
    videoFrameName='./test_images/video_frames/frame0001.jpg'
    videoFrame=mpimage.imread(videoFrameName)
    print("testProcessVideoFrame-videoFrameName: ",videoFrameName, ", videoFrame.shape:", videoFrame.shape, ", type:", videoFrame.dtype)
    [left_fit,right_fit], figure=processVideoFrame(videoFrame, 0, left_fit, right_fit)
    plt.show()
    plt.close(figure)
    [left_fit,right_fit], figure=processVideoFrame(videoFrame, 1, left_fit, right_fit)
    plt.show()
    plt.close(figure)

#testProcessVideoFrame()