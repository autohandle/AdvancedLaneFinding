import cv2
import numpy as np

def doConvertToGray(rgbImage):
    #print("rgbImage.shape:", rgbImage.shape, ", type:", type(rgbImage))
    return cv2.cvtColor(rgbImage, cv2.COLOR_RGB2GRAY)

def abs_sobel_thresh(grayImage, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient=='x':
        sobel = cv2.Sobel(grayImage, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(grayImage, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    absSobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaledSobel = np.uint8(255.*absSobel/float(np.max(absSobel)))
    #print("abs_sobel_thresh-orient:", orient,", scaledSobel counts:", np.unique(scaledSobel, return_counts=True), ", shape:",scaledSobel.shape)
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    scaledSobelBinary = np.zeros_like(scaledSobel)
    #print("abs_sobel_thresh-orient:", orient,", counts:", np.unique(scaledSobelBinary, return_counts=True), ", shape:",scaledSobelBinary.shape)
    scaledSobelBinary[(scaledSobel >= thresh[0]) & (scaledSobel <= thresh[1])] = 1
    #print("abs_sobel_thresh-orient:", orient,", scaledSobelBinary counts:", np.unique(scaledSobelBinary, return_counts=True), ", shape:",scaledSobelBinary.shape)
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    #return binary_output
    #
    return scaledSobelBinary

def channelThreshold(rgbImage, channel=0, thresh=(0, 255)):
    # 1) Convert to grayscale
    oneChannel = rgbImage[:,:,channel]
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaledChannel = np.uint8(255.*oneChannel/float(np.max(oneChannel)))
    # 5) Create a binary mask where direction thresholds are met
    channelBinary = np.zeros_like(scaledChannel)
    channelBinary[(scaledChannel >= thresh[0]) & (scaledChannel <= thresh[1])] = 1
    #print("channelThreshold-counts:", np.unique(grayBinary, return_counts=True), ", shape:",grayBinary.shape)
    return channelBinary

def unscaledChannelThreshold(rgbImage, channel=0, thresh=(0, 255)):
    # 1) Convert to grayscale
    oneChannel = rgbImage[:,:,channel]
    # 5) Create a binary mask where direction thresholds are met
    channelBinary = np.zeros_like(oneChannel)
    channelBinary[(oneChannel >= thresh[0]) & (oneChannel <= thresh[1])] = 1
    #print("channelThreshold-counts:", np.unique(grayBinary, return_counts=True), ", shape:",grayBinary.shape)
    return channelBinary

def channelBlueThreshold(rgbImage, thresh=(0, 255)):
    return unscaledChannelThreshold(rgbImage, 2, thresh)

def channelGreenThreshold(rgbImage, thresh=(0, 255)):
    return unscaledChannelThreshold(rgbImage, 1, thresh)

def channelRedThreshold(rgbImage, thresh=(0, 255)):
    return unscaledChannelThreshold(rgbImage, 0, thresh)

def channelYellow(rgbImage):
    redChannel=channelRedThreshold(rgbImage, (225,255))
    #print("channelYellow-redChannel counts:", np.unique(redChannel, return_counts=True), ", shape:",redChannel.shape)
    blueChannel=channelBlueThreshold(rgbImage, (0,175))
    #print("channelYellow-blueChannel counts:", np.unique(blueChannel, return_counts=True), ", shape:",blueChannel.shape)
    greenChannel=channelGreenThreshold(rgbImage, (170,225))
    #print("channelYellow-greenChannel counts:", np.unique(greenChannel, return_counts=True), ", shape:",greenChannel.shape)
    channelBinary = np.zeros_like(redChannel)
    channelBinary[(redChannel==1) & (greenChannel==1)] = 1
    #print("channelYellow-channelBinary counts:", np.unique(channelBinary, return_counts=True), ", shape:",channelBinary.shape)
    return channelBinary

def channelWhite(rgbImage):
    LOWER=190
    UPPER=255
    redChannel=channelRedThreshold(rgbImage, (LOWER,UPPER))
    #print("channelYellow-redChannel counts:", np.unique(redChannel, return_counts=True), ", shape:",redChannel.shape)
    greenChannel=channelGreenThreshold(rgbImage, (LOWER,UPPER))
    #print("channelYellow-greenChannel counts:", np.unique(greenChannel, return_counts=True), ", shape:",greenChannel.shape)
    blueChannel=channelBlueThreshold(rgbImage, (LOWER,UPPER))
    #print("channelYellow-blueChannel counts:", np.unique(blueChannel, return_counts=True), ", shape:",blueChannel.shape)
    channelBinary = np.zeros_like(redChannel)
    channelBinary[(redChannel==1) & (greenChannel==1) & (blueChannel==1)] = 1
    #print("channelYellow-channelBinary counts:", np.unique(channelBinary, return_counts=True), ", shape:",channelBinary.shape)
    return channelBinary

DIALATEKERNELSIZE=5
def closeRegions(image):
    kernel = np.ones((DIALATEKERNELSIZE,DIALATEKERNELSIZE),np.uint8)
    dialatedImage=cv2.dilate(image.copy(), kernel, iterations=7)
    dialatedImage=cv2.erode(dialatedImage, kernel, iterations=7)
    imageWidth=dialatedImage.shape[1] # X
    imageHeight=dialatedImage.shape[0] # Y
    #lostPixels=int(DIALATEKERNELSIZE/2)
    #print("lostPixels:",lostPixels, ", DIALATEKERNELSIZE:", DIALATEKERNELSIZE, ". dialatedImage.shape", dialatedImage.shape)
    #return dialatedImage[lostPixels:imageHeight-lostPixels, lostPixels:imageWidth-lostPixels]
    return dialatedImage

BLURKERNELSIZE = 5
def doBlur(image):
    blurred = cv2.GaussianBlur(image,(BLURKERNELSIZE, BLURKERNELSIZE), 0)
    return blurred

MASKFILLVALUE = 0;
CONNECTIVITY = 4 # 4 way
def floodLaneMarker(rgbImage, theSeedPoints):
    #print("floodRoad-rgbImage.shape:", rgbImage.shape, ", theSeedPoints:", theSeedPoints)
    grayscaleImage=doConvertToGray(rgbImage.copy())
    blurredGrayImage = doBlur(grayscaleImage)
    imageWidth=blurredGrayImage.shape[1] # X
    imageHeight=blurredGrayImage.shape[0] # Y
    roadFlags = CONNECTIVITY | (MASKFILLVALUE << 1)
    # for gray scale image
    roadMask = np.zeros((imageHeight+2, imageWidth+2), np.uint8)
    #print('grayScale:', type(grayScale), 'with dimensions:', grayScale.shape, 'road flags: <', hex(roadFlags), '>, theSeedPoints:', theSeedPoints)
    for seedPoint in theSeedPoints:
        #print('seedPoint:', seedPoint," type: ", type(seedPoint)) 
        #print('road seed color:', grayScale[seedPoint[1], seedPoint[0]]," at ", seedPoint) 
        cv2.floodFill(blurredGrayImage, roadMask, seedPoint, 255, 1, 1, roadFlags)
    return grayscaleImage, roadMask

def floodColorLaneMarker(rgbImage, theSeedPoints):
    #print("floodRoad-rgbImage.shape:", rgbImage.shape, ", theSeedPoints:", theSeedPoints)
    imageWidth=rgbImage.shape[1] # X
    imageHeight=rgbImage.shape[0] # Y
    flags = CONNECTIVITY | (MASKFILLVALUE << 1)
    # for gray scale image
    mask = np.zeros((imageHeight+2, imageWidth+2), np.uint8)
    #print('grayScale:', type(grayScale), 'with dimensions:', grayScale.shape, 'road flags: <', hex(roadFlags), '>, theSeedPoints:', theSeedPoints)
    for seedPoint in theSeedPoints:
        print('seedPoint:', seedPoint," type: ", type(seedPoint)) 
        print('rgbImage color:', rgbImage[seedPoint[0], seedPoint[1]]," at ", seedPoint) 
        cv2.floodFill(rgbImage, mask, seedPoint, 255, 1, 1, flags)
    return mask

# cropping 100 pixels off the left and off the top
import PerspectiveTransform
print ("dir(PerspectiveTransform):", dir(PerspectiveTransform))

def enhanceLaneMarkers(rgbImage, trimLeft=100, trimTop=0, trimBottom=0):
    undistortedAndTransformedImage=PerspectiveTransform.undistortAndTransform(rgbImage)
    grayScaleImage=doConvertToGray(undistortedAndTransformedImage[trimTop:undistortedAndTransformedImage.shape[0]-trimBottom,trimLeft:]) 
    sobelImage=abs_sobel_thresh(grayScaleImage, orient='x', sobel_kernel=3, thresh=(20, 255))
    #yellowImage=channelYellow(transformedImage)
    #p=showMaskImages.add_subplot(maskTotalImageRows, maskImageColumnCount, testImageIndex+4)
    #p.set_title("yellowImage ("+str(sobelImage.shape[0])+"x"+str(sobelImage.shape[1])+")")
    #p.imshow(yellowImage, cmap='gray')
    
    #whiteImage=channelWhite(transformedImage)
    #p=showMaskImages.add_subplot(maskTotalImageRows, maskImageColumnCount, testImageIndex+5)
    #p.set_title("whiteImage ("+str(sobelImage.shape[0])+"x"+str(sobelImage.shape[1])+")")
    #p.imshow(whiteImage, cmap='gray')
    closeRegion=closeRegions(sobelImage)
    return closeRegion,{"undistortedAndTransformedImage":undistortedAndTransformedImage,"grayScaleImage":grayScaleImage, "sobelImage":sobelImage, "closeRegion":closeRegion}

import matplotlib.image as mpimage
def testEnhanceLaneMarkers():
    testImageName='./test_images/test1.jpg'
    print("testImageName:", testImageName)
    testRgbImage=mpimage.imread(testImageName)
    binaryImage,imageDictionary=enhanceLaneMarkers(testRgbImage)
    print("imageDictionary: ",imageDictionary.keys())
    print("testImageName: ",testImageName, ", binaryImage.shape:", binaryImage.shape, ", type:", binaryImage.dtype)

#testEnhanceLaneMarkers()
