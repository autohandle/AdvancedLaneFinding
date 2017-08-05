import numpy as np
import cv2
import matplotlib.pyplot as plt

def invertHistogram(histogram, maxValue):
    maxValueInHistogram=np.max(histogram)
    scale=float(maxValue)/float(maxValueInHistogram)
    invertedHistogram=np.zeros(histogram.shape[0], dtype=histogram.dtype)
    invertedHistogram=maxValue-histogram*scale
    return invertedHistogram

def combineImages(img, initial_img, α=0.8, λ=0.):
    return cv2.addWeighted(initial_img, α, img, 1.-α, λ)


def calculateCarOffset(binaryImage, left_fit, right_fit):
    yAtBottom=binaryImage.shape[0]
    xLeftAtBottom=left_fit[0]*yAtBottom**2 + left_fit[1]*yAtBottom + left_fit[2]
    xRightAtBottom=right_fit[0]*yAtBottom**2 + right_fit[1]*yAtBottom + right_fit[2]
    #print ("calculateCarOffset-yAtBottom:",yAtBottom,", xLeftAtBottom:", xLeftAtBottom, ", xRightAtBottom:", xRightAtBottom)
    imageMidpointX=binaryImage.shape[1]//2
    laneMidpoint=xLeftAtBottom+(xRightAtBottom-xLeftAtBottom)//2
    carOffset=laneMidpoint-imageMidpointX
    #print ("calculateCarOffset-imageMidpointX:", imageMidpointX, ", laneMidpoint:", laneMidpoint, ", carOffset:", carOffset)
    return carOffset

# transformedRgbImage: 720x 1180 x3
def fillLaneInVisualizationImage(visualizationRgbImage, ploty, left_fitx, right_fitx, fillColor=(0,255,0)): # green
    #print("fillLaneInVisualizationImage-visualizationRgbImage.shape:", visualizationRgbImage.shape, ", type:", visualizationRgbImage.dtype)
    #print("fillLaneInVisualizationImage-ploty.shape:",ploty.shape,", left_fitx.shape:", left_fitx.shape, ", right_fitx.shape:", right_fitx.shape)
    filledLaneImage=visualizationRgbImage.copy()
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(filledLaneImage, np.int_([pts]), fillColor)
    #cv2.fillPoly(overlayImage, np.int_([pts]), (255,0,0))

    return filledLaneImage 


# transformedRgbImage: 720x 1280 x3
# visualizationImage:  720x 1180 x3
def fillLaneInTransformImage(transformedRgbImage, visualizationRgbImage, resultImageTrim, ploty, left_fit, right_fit):
    #rgbImage=incrementalImages['undistortedAndTransformedImage'] # 720,1280,3
    #print("fillLaneInTransformImage-transformedRgbImage.shape:", transformedRgbImage.shape, ", type:", transformedRgbImage.dtype)
    #print("fillLaneInTransformImage-visualizationRgbImage.shape:", visualizationRgbImage.shape, ", type:", visualizationRgbImage.dtype)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] # 2nd order prediction of x from y
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #filledLaneImage=fillLaneInVisualizationImage(visualizationRgbImage, ploty, left_fitx, right_fitx, fillColor=(241, 66, 244))
    filledLaneImage=fillLaneInVisualizationImage(visualizationRgbImage, ploty, left_fitx, right_fitx)
    #print("fillLaneInTransformImage-filledLaneImage.shape:", filledLaneImage.shape, ", type:", filledLaneImage.dtype)
    overlayImage=np.zeros_like(transformedRgbImage)
    #print("fillLaneInTransformImage-overlayImage.shape:", overlayImage.shape, ", type:", overlayImage.dtype)
   
    # position smaller filledLaneImage in an overlay, so it can be combined with larger visualizationRgbImage
    trimY=resultImageTrim[1]
    trimX=resultImageTrim[0]
    #print("fillLaneInTransformImage-resultImageTrim:", resultImageTrim, ", trimX:", trimX, ", trimY:", trimY)
    overlayImage[trimY:trimY+filledLaneImage.shape[0], trimX:trimX+filledLaneImage.shape[1]]=filledLaneImage
    #print("fillLaneInTransformImage-overlayImage.shape:", overlayImage.shape, ", type:", overlayImage.dtype)
    
    combinedImage=combineImages(overlayImage, transformedRgbImage, α=.85)
    return combinedImage

def calculateRadiusCurveInPixelsSigned(y_eval, left_fit, right_fit):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / (2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / (2*right_fit[0])
    #print("calculateRadiusCurveInPixels-left_curverad:", left_curverad, ", right_curverad:", right_curverad)
    # Example values: 1926.74 1908.48
    return left_curverad,right_curverad

def calculateRadiusCurveInMeters(y_eval, ploty, left_fit, right_fit):
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] # 2nd order prediction of x from y
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    leftx = left_fitx[::-1]  # Reverse to match top-to-bottom in y
    rightx = right_fitx[::-1]  # Reverse to match top-to-bottom in y

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print("calculateRadiusCurveInMeters-left_curverad:", left_curverad, 'm, right_curverad:', right_curverad, 'm')
    return left_curverad,right_curverad

def createLanePolygonImage(visualizationShape, margin, ploty, left_fit, right_fit):
    # Generate new x and y values for plotting (the new) 2nd order poly
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] # 2nd order prediction of x from y
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # no idea how this works!
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    #print("left_line_window1.shape:", left_line_window1.shape, ", type:", left_line_window1.dtype)
    #lm=left_fitx-margin
    #print("lm.shape:", lm.shape, ", type:", lm.dtype)
    #print("ploty.shape:", ploty.shape, ", ploty:", left_fitx.dtype)
    #left_line_window1 = np.hstack((lm, ploty))
    #print("left_line_window1.shape:", left_line_window1.shape, ", type:", left_line_window1.dtype)
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    #print("left_line_window2.shape:", left_line_window2.shape, ", type:", left_line_window2.dtype)
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    lanePolygons = np.zeros(visualizationShape, dtype=np.uint8)
    cv2.fillPoly(lanePolygons, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(lanePolygons, np.int_([right_line_pts]), (0,255, 0))

    return lanePolygons

# for the transformed cropped image
def plotPolygon(plotReference, limits, ploty, left_fit, right_fit, lineColor='yellow', linewidth=1):
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] # 2nd order prediction of x from y
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    plotReference.plot(left_fitx, ploty, lineColor, linewidth)
    plotReference.plot(right_fitx, ploty, lineColor, linewidth)
    plotReference.set_xlim(0, limits[0])
    plotReference.set_ylim(limits[1], 0)
    
# for the transformed cropped image
def plotPolygonOnTransformedImage(plotReference, limits, trim, ploty, left_fit, right_fit, lineColor='yellow', linewidth=1):
    trimX=trim[0]
    trimY=trim[1]
    plotPolygon(plotReference, limits, trimY+ploty, left_fit, right_fit, lineColor, linewidth)

import matplotlib.gridspec as gridspec

def plotFigure(filledLaneVideoFrame, annotatedVisualizationImage, filledLaneWarpedImage, lambdaAx3):
    # Plot figure with subplots of different sizes
    fig = plt.figure(figsize=(20,10))
    # set up subplot grid: 3x2
    gridspec.GridSpec(3,2)

    # large subplot — filledLaneVideoFrame 2x2@0,0 on a 3x2
    ax1=plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=2)
    #print("ax1:", ax1)
    #ax1.text(0.5, 0.5, "-1\n1\n1\n1\n1\n1-",
    #        fontsize=36,
    #        color='w',
    #        transform = ax1.transAxes)
    plt.imshow(filledLaneVideoFrame)

    # small subplot 1: annotatedVisualizationImage 1x1@2,0 on a 3x2
    ax2=plt.subplot2grid((3,2), (2,0))
    #print("ax2:", ax2)
    #ax2.text(0., 1., "-2\n2\n2\n2-",
    #        fontsize=36,
    #        color='w',
    #        transform = ax2.transAxes,
    #        horizontalalignment='left',
    #    verticalalignment='top')
    plt.imshow(annotatedVisualizationImage)

    # small subplot 2: annotatedVisualizationImage 1x1@2,1 on a 3x2
    ax3=plt.subplot2grid((3,2), (2,1))
    #print("ax3:", ax3)
    #ax3.text(0., 0.25, "-3\n3\n3\n3-",
    lambdaAx3(ax3)
    plt.imshow(filledLaneWarpedImage)

    fig.tight_layout()

    return fig

