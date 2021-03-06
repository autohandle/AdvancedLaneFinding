import matplotlib.image as mpimage
import EnhanceLaneMarkers
import ProcessImage
import cv2
import numpy as np

TRIMSTARTINGROWS=(1./10.)
TRIMENDINGROWS=(1./5.)
TRIMSTARTINGCOLUMNS=(1./10.)

MINPIX=50

def calculateRadiusCurveInPixelsSigned(y_eval, left_fit, right_fit):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / (2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / (2*right_fit[0])
    #print("ProcessImage-calculateRadiusCurveInPixels-left_curverad:", left_curverad, ", right_curverad:", right_curverad)
    # Example values: 1926.74 1908.48
    return left_curverad,right_curverad

def locateLaneMarkerIndex(binaryImage):
    #print("locateLaneMarkerIndex - binaryImage.shape:", binaryImage.shape, ", type:", binaryImage.dtype)
    startingInRow=int(binaryImage.shape[0]*TRIMSTARTINGROWS)
    endingInRow=int(binaryImage.shape[0]-binaryImage.shape[0]*TRIMENDINGROWS)
    startingInColumn=int(binaryImage.shape[1]*TRIMSTARTINGCOLUMNS)
    #print("locateLaneMarkerIndex - startingInRow: ",startingInRow, ", startingInColumn:", startingInColumn, ", endingInRow:", endingInRow)
    
    croppedBinaryImage=binaryImage[startingInRow:endingInRow,startingInColumn:]
    #print("locateLaneMarkerIndex - croppedBinaryImage.shape:", croppedBinaryImage.shape, ", type:", croppedBinaryImage.dtype)
    #croppedBinaryImages[binaryImageName]=croppedBinaryImage
       
    histogram = np.sum(croppedBinaryImage, axis=0)
    #histograms[binaryImageName]=histogram
    alignedHistogram = np.zeros(binaryImage.shape[1], dtype=histogram.dtype)
    #print("locateLaneMarkerIndex - histogram.shape: ",histogram.shape, ", alignedHistogram.shape:", alignedHistogram.shape, ", type:", alignedHistogram.dtype)
    alignedHistogram[startingInColumn:startingInColumn+histogram.shape[0]]=histogram
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    #histogram=histograms[binaryImageName]
    midpoint = np.int(alignedHistogram.shape[0]/2)
    leftx_base = np.argmax(alignedHistogram[:midpoint])
    rightx_base = np.argmax(alignedHistogram[midpoint:]) + midpoint
    #print("locateLaneMarkerIndex - leftx_base:", leftx_base, ", rightx_base:", rightx_base)
    
    return [leftx_base, rightx_base], alignedHistogram

def initializeSlidingWindows(binaryImage):
    print("initializeSlidingWindows - binaryImage.shape:", binaryImage.shape, ", type:", binaryImage.dtype)
    # Identify the x and y positions of all nonzero pixels in the image
    white = binaryImage.nonzero()
    whiteY = np.array(white[0])
    whiteX = np.array(white[1])
    print("total pixels:", binaryImage.shape[0]*binaryImage.shape[1], ", whiteX:", len(whiteX), ", whiteY:", len(whiteY))

    # Set the width of the windows +/- margin
    # margin = 100
    # Set minimum number of pixels found to recenter window
    # minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binaryImage.shape[0]/nwindows)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    [leftx_base, rightx_base], histogram=locateLaneMarkerIndex(binaryImage)
    #histogram=histograms[binaryImageName]
    #midpoint = np.int(histogram.shape[0]/2)
    #leftx_base = np.argmax(histogram[:midpoint])
    #rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #print("initializeSlidingWindows-leftx_base:", leftx_base, ", rightx_base:", rightx_base)
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Identify the x and y positions of all nonzero pixels in the image
    white = binaryImage.nonzero()
    whiteY = np.array(white[0])
    whiteX = np.array(white[1])
    #print("total pixels:", binaryImage.shape[0]*binaryImage.shape[1], ", whiteX:", len(whiteX), ", whiteY:", len(whiteY))

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    # minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binaryImage.shape[0]/nwindows)
    #print("initializeSlidingWindows-nwindows: ",nwindows, ", window_height:", window_height)

    #print("initializeSlidingWindows-binaryImage.shape:", binaryImage.shape, ", type:", binaryImage.dtype)
    visualizationImage = np.dstack((binaryImage, binaryImage, binaryImage))*255
    #visualizationImage = croppedTestImages[testImageName]
    #print("initializeSlidingWindows-visualizationImage.shape:", visualizationImage.shape, ", type:", visualizationImage.dtype)

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binaryImage.shape[0] - (window+1)*window_height
        win_y_high = binaryImage.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(visualizationImage,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(visualizationImage,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        #print("window: ",window, ", window.shape: (",
        #      win_xleft_low,",",win_xleft_high,"),(",
        #      win_xright_low,",", win_xright_high, ") x (",
        #      win_y_low, ",", win_y_high, ")")

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((whiteY >= win_y_low) & (whiteY < win_y_high)
                          & (whiteX >= win_xleft_low) & (whiteX < win_xleft_high)).nonzero()[0]
        good_right_inds = ((whiteY >= win_y_low) & (whiteY < win_y_high) & (whiteX >= win_xright_low) & (whiteX < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > MINPIX:
            leftx_current = np.int(np.mean(whiteX[good_left_inds]))
        if len(good_right_inds) > MINPIX:        
            rightx_current = np.int(np.mean(whiteX[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = whiteX[left_lane_inds]
    lefty = whiteY[left_lane_inds] 
    rightx = whiteX[right_lane_inds]
    righty = whiteY[right_lane_inds] 

    # Fit a second order polynomial to each: returns oolynomial coefficients, highest power first for: leftx = f(lefty)
    left_fit = np.polyfit(lefty, leftx, 2) # i.e. x = left_fit[0] y**2 + left_fit[1] y + left_fit[2]
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting, evenly spaced numbers over a specified interval: start, stop, count
    #ploty = np.linspace(0, binaryImage.shape[0]-1, binaryImage.shape[0] ) #  y/row values
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    visualizationImage[whiteY[left_lane_inds], whiteX[left_lane_inds]] = [255, 0, 0]
    visualizationImage[whiteY[right_lane_inds], whiteX[right_lane_inds]] = [0, 0, 255]
    return [left_fit, right_fit], visualizationImage # x = left_fit[0] y**2 + left_fit[1] y + left_fit[2]
    
def processInitialImage(binaryImage):
    [left_fit, right_fit], visualizationImage = initializeSlidingWindows(binaryImage)
    #ploty = np.linspace(0, binaryImage.shape[0]-1, binaryImage.shape[0] ) 
    #p=showVideoFrames.add_subplot(totalImageRows, VIDEOIMAGECOLUMNCOUNT, videoFrameIndex+1)
    #p.set_title("frame: "+videoImageName+"\n("+str(visualizationImage.shape)+" - "+str(visualizationImage.dtype)+")")
    #p.imshow(visualizationImage)
    return [left_fit, right_fit], visualizationImage

import math

MARGIN = 100
MINPIXELCURVE = 1200 # otherwise, keep the old fit
MINCURVEDRATIO = .5 # otherwise, (re)initialize

def processImage(binaryImage, left_fit, right_fit):
    print("processImage-left_fit:", left_fit, ", right_fit:", right_fit)
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binaryImage.nonzero() # [y-row,x-col] that are not 0 (i.e. 1)
    nonzeroy = np.array(nonzero[0]) # only the nonzero y
    #print("processImage-nonzeroy.shape:", nonzeroy.shape, ", type:", nonzeroy.dtype)
    nonzerox = np.array(nonzero[1]) # only the nonzero x
    #print("processImage-nonzerox.shape:", nonzerox.shape, ", type:", nonzerox.dtype)
    
    # left lane x columns for non zero y rows from last polynomial fit
    left_laneX = left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2]
    #print("processImage-left_laneX.shape:", left_laneX.shape, ", type:", left_laneX.dtype)
    # mark x values between margins in the new frame using the old polynomial as either in(true) or out(false)
    left_lane_inds = ((nonzerox > (left_laneX - MARGIN)) & (nonzerox < (left_laneX + MARGIN))) 
    #print("processImage-left_lane_inds.shape:", left_lane_inds.shape, ", type:", left_lane_inds.dtype, ", counts:", np.unique(left_lane_inds, return_counts=True))
    
    # right lane x columns for non zero y rows from last polynomial fit
    right_laneX = right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2]
    # mark x values between margins in the new frame using the old polynomial as either in(true) or out(false)
    right_lane_inds = ((nonzerox > (right_laneX - MARGIN)) & (nonzerox < (right_laneX + MARGIN)))
    #print("processImage-right_lane_inds.shape:", right_lane_inds.shape, ", type:", right_lane_inds.dtype, ", counts:", np.unique(right_lane_inds, return_counts=True))
   
    # extract corresponding left and right line pixel positions that fall within (are true in left_lane_inds) old 2nd order poly margin
    leftx = nonzerox[left_lane_inds]
    #print("processImage-leftx.shape:", leftx.shape, ", type:", leftx.dtype, ", mean:", np.mean(leftx), ", max:", np.amax(leftx), ", minimum:", np.amin(leftx), ", counts:", np.unique(leftx, return_counts=True))
    lefty = nonzeroy[left_lane_inds] 
    #print("processImage-lefty.shape:", lefty.shape, ", type:", lefty.dtype, ", mean:", np.mean(lefty), ", max:", np.amax(lefty), ", minimum:", np.amin(lefty), ", counts:", np.unique(lefty, return_counts=True))
    rightx = nonzerox[right_lane_inds]
    #print("processImage-rightx.shape:", rightx.shape, ", type:", rightx.dtype, ", mean:", np.mean(rightx), ", max:", np.amax(rightx), ", minimum:", np.amin(rightx), ", counts:", np.unique(rightx, return_counts=True))
    righty = nonzeroy[right_lane_inds]
    #print("processImage-righty.shape:", righty.shape, ", type:", righty.dtype, ", mean:", np.mean(righty), ", max:", np.amax(righty), ", minimum:", np.amin(righty), ", counts:", np.unique(righty, return_counts=True))

    # (re)fit a second order polynomial (in pixels) to each lane : left & right
    # leftFitForXintercept = np.polyfit(lefty, leftx, 2) # leftx = f(lefty)
    # rightFitForXintercept = np.polyfit(righty, rightx, 2) # rightx = f(righty)
    
    
    #leftFitForX = np.polyfit(leftx, lefty, 2) # lefty = f(leftx)
    #rightFitForX = np.polyfit(righty, rightx, 2) # righty = f(rightx)
    
    # adjust all values to left
    #allX = leftx
    #allY = lefty
    #bothLanesWithinMargin=left_lane_inds
    # adjust right x values to left and append
    
    # x = 1/2a*(-b+sqrt(b**2-4ac))
    #leftA=leftFitForX[0]
    #leftB=leftFitForX[1]
    #leftC=leftFitForX[2]
    #forCount=0
    #for i in range(1,len(leftx)):
    #    if (lefty[i]!=0):
    #        print("processImage-leftx["+str(i)+"]:", leftx[i], ", leftA*leftx["+str(i)+"]**2+leftB*leftx["+str(i)+"]+leftC:", (leftA*leftx[i]**2+leftB*leftx[i]+leftC), ", lefty["+str(i)+"]:", lefty[i])
    #    forCount+=1
    #    if (forCount>100):
    #        break
    #print("processImage-leftB*leftB-4.*leftA*leftC:", leftB*leftB-4.*leftA*leftC)
    #leftInterceptX = (-leftB+math.sqrt(leftB*leftB-4.*leftA*leftC))/(2.*leftA)
    #print("processImage-leftInterceptX:",leftInterceptX, ", ax*x+b*x+c:", str(leftA*leftInterceptX*leftInterceptX+leftB*leftInterceptX+leftC))
    # d = -(ax*x+bx+c)
    #rightA=rightFitForX[0]
    #rightB=rightFitForX[1]
    #rightC=rightFitForX[2]

    #adjustRightX = rightA*leftInterceptX*leftInterceptX+rightB*leftInterceptX+rightC
    #print("processImage-adjustRightX:",adjustRightX, ", ax*x+b*x+c-d:", str(rightA*leftInterceptX*leftInterceptX+rightB*leftInterceptX+rightC-adjustRightX))
    
    #allX=np.append(allX, rightx-adjustRightX)
    #print("processImage-allX.shape:", allX.shape, ", type:", allX.dtype, ", mean:", np.mean(allX), ", max:", np.amax(allX), ", minimum:", np.amin(allX))
    #allY=np.append(allY, righty)
    #bothLanesWithinMargin=np.logical_or(bothLanesWithinMargin, right_lane_inds)
    #print("processImage-bothLanesWithinMargin.shape:", bothLanesWithinMargin.shape, ", type:", bothLanesWithinMargin.dtype, ", counts:", np.unique(bothLanesWithinMargin, return_counts=True))
    # (re)fit a second order polynomial to points from both lanes
    #allPointFit = np.polyfit(allY, allX, 2)
    #print("processImage-allPointFit:", allPointFit)
    
    # move polynomial fit for both lanes to left and right intercept
    #left_fit=[allPointFit[0], allPointFit[1], leftFitForX[2]]
    #right_fit=[allPointFit[0], allPointFit[1], rightFitForX[2]+adjustRightX]
    #print("processImage-allPointFit-left_fit:", left_fit,", right_fit:", right_fit)

    new_left_fit = np.polyfit(lefty, leftx, 2)
    new_right_fit = np.polyfit(righty, rightx, 2)
    #print("processImage-new_left_fit:", new_left_fit,", new_right_fit:", new_right_fit)
    
    y_eval = binaryImage.shape[0]
    left_curverad, right_curverad=calculateRadiusCurveInPixelsSigned(y_eval, new_left_fit, new_right_fit)
    #print("processImage-left_curverad:", left_curverad, ", right_curverad:", right_curverad)
    if (math.fabs(left_curverad) > MINPIXELCURVE):
        left_fit=new_left_fit
    if (math.fabs(right_curverad) > MINPIXELCURVE):
        right_fit=new_right_fit
    left_curverad, right_curverad=calculateRadiusCurveInPixelsSigned(y_eval, new_left_fit, new_right_fit)
    #print("processImage-left_curverad:", left_curverad, ", right_curverad:", right_curverad)
    maxCurve=max(left_curverad,right_curverad)
    minCurve=min(left_curverad,right_curverad)
    curvedDifferently=(math.fabs(minCurve/maxCurve) < MINCURVEDRATIO)
    leftTooSmall=(math.fabs(left_curverad) < MINPIXELCURVE) # even with new fit, curve is still too tight
    rightTooSmall=(math.fabs(right_curverad) < MINPIXELCURVE) # even with new fit, curve is still too tight
    if (curvedDifferently or leftTooSmall or rightTooSmall):
        print("processImage-processInitialImage-left_curverad:", left_curverad, ", right_curverad:", right_curverad)
        print("processImage-processInitialImage-curvedDifferently:", curvedDifferently, ", leftTooSmall:", leftTooSmall, ", rightTooSmall:", rightTooSmall)
        [left_fit, right_fit], visualizationImage=ProcessImage.processInitialImage(binaryImage)
        left_curverad, right_curverad=calculateRadiusCurveInPixelsSigned(y_eval, left_fit, right_fit)
        print("processImage-processInitialImage-left_curverad:", left_curverad, ", right_curverad:", right_curverad)
    #coefficient0=(left_fit[0]+right_fit[0])/2.
    #coefficient1=(left_fit[1]+right_fit[1])/2.
    #left_fit = [coefficient0,coefficient1,left_fit[2]]
    #right_fit = [coefficient0,coefficient1,right_fit[2]]
    print("processImage-left_fit:", left_fit,", right_fit:", right_fit)


    # Create an image to draw on and an image to show the selection window
    visualizationImage = np.dstack((binaryImage, binaryImage, binaryImage))*255
    # Color in left and right line pixels
    visualizationImage[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0] # left lane pixels are red
    visualizationImage[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255] # right lane pixels are blue
    
    #ploty = np.linspace(0, visualizationImage.shape[0]-1, visualizationImage.shape[0] ).astype(int)
    #left_fitx = (left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]).astype(int) # 2nd order prediction of x from y
    #right_fitx = (right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]).astype(int)

    #visualizationImage[ploty, left_fitx] = [0, 255, 255] # both lane pixels are green
    #visualizationImage[ploty, right_fitx] = [0, 255, 255] # both lane pixels are green

    return [left_fit, right_fit], visualizationImage


def testProcessImage():
    TRIMLEFT=100
    TRIMTOP=0
    TRIMBOTTOM=0

    ploty=None
    
    videoImages= ['./test_images/video_frames/frame0001.jpg',    './test_images/video_frames/frame0002.jpg']
    for videoFrameName, videoFrameIndex in zip(videoImages, range(0, len(videoImages))):
        videoFrame=mpimage.imread(videoFrameName)
        print("testProcessImage-videoFrameName: ",videoFrameName, ", videoFrame.shape:", videoFrame.shape, ", type:", videoFrame.dtype)
        binaryImage,incrementalImages=EnhanceLaneMarkers.enhanceLaneMarkers(videoFrame, trimLeft=TRIMLEFT, trimTop=TRIMTOP, trimBottom=TRIMBOTTOM)
        print("testProcessImage-videoFrameName: ",videoFrameName, ", binaryImage.shape:", binaryImage.shape, ", type:", binaryImage.dtype)
    
        if ploty==None :
            [left_fit, right_fit], visualizationImage=processInitialImage(binaryImage)
        else: # process a new frame by assuming the old 2nd order polynomial still fits the new frame
            [left_fit, right_fit], visualizationImage=processImage(binaryImage, left_fit, right_fit)
        print("testProcessImage-left_fit: ",left_fit, ", right_fit:", right_fit)
        print("testProcessImage-videoFrameName: ",videoFrameName, ", visualizationImage.shape:", visualizationImage.shape, ", type:", visualizationImage.dtype)
        
testProcessImage()