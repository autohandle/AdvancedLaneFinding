import numpy as np
import matplotlib.image as mpimage
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import ProcessVideoFrame

# convert figure to image
# https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
def XconvertFigureToImage(figure):
    canvas = FigureCanvas(figure)
    ax = figure.gca()

    #ax.text(0.0,0.0,"Test", fontsize=45)
    #ax.axis('off')

    canvas.draw()       # draw the canvas, cache the renderer

    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    return image

# convert figure to image
# https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
def convertFigureToImage(figure):
    figure.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))

    return data

left_fit = None
right_fit = None
frameNumber=0

def process_image(videoFrame):
    global left_fit
    global right_fit
    global frameNumber
    # processVideoFrame(videoFrame, frameNumber, left_fit, right_fit) ->[left_fit,right_fit], figure
    [left_fit,right_fit], figure=ProcessVideoFrame.processVideoFrame(videoFrame, frameNumber, left_fit, right_fit)
    frameNumber+=1
    return convertFigureToImage(figure)


def processChallengeVideo():
    CHALLENGEVIDEOOUTPUT = 'test_videos_output/ChallengeVideoOutput.mp4'
    PROJECTVIDEOOUTPUT = 'test_videos_output/ProjectVideoOutput.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
    videoClip = VideoFileClip('project_video.mp4').subclip(20,30)
    annotatedClip = videoClip.fl_image(process_image)
    annotatedClip.write_videofile(PROJECTVIDEOOUTPUT, audio=False)

def testProcessImage():
    left_fit = None # x = left_fit[0] y**2 + left_fit[1] y + left_fit[2]
    right_fit = None
    videoFrameName='./test_images/video_frames/frame0001.jpg'
    videoFrame=mpimage.imread(videoFrameName)
    print("testProcessImage-videoFrameName: ",videoFrameName, ", videoFrame.shape:", videoFrame.shape, ", type:", videoFrame.dtype)
    image=process_image(videoFrame)
    print("testProcessImage-image.shape:", image.shape, ", type:", image.dtype)
    plt.imshow(image)
    plt.show()

#testProcessImage()
processChallengeVideo()