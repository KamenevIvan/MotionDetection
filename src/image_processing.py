import numpy as np
import math
import cv2 as cv
import settings

# image preprocessing functions


def fix_image(foreground):
    # applying threshold to foreground for shadows elimination
    ret, fg_t = cv.threshold(foreground, 128, 255, cv.THRESH_BINARY)

    # erosion of foreground for noise removal
    fg_e = cv.erode(fg_t, cv.getStructuringElement(2, (3, 3)))

    # dilation for filling the gaps
    dstr = cv.getStructuringElement(2, (10, 20))  # was 9,18
    return cv.dilate(fg_e, dstr, iterations=2)


def nightVision(frame):
    """This function determines if picture was taken by a camera in
    night vision mode. To this end, pixels of each channel are summed and the sums are compared.
    If the difference between them is less than ntol, True is returned
    Input: 3 channel array (picture)
    Output: True or False
    Global parameters: ntol - threshold for average difference between pixel channels
    """
    width = frame.shape[1]
    height = frame.shape[0]
    npixels = width * height
    Blue = frame[:, :, 0]
    average_b = np.sum(Blue) / npixels
    Green = frame[:, :, 1]
    average_g = np.sum(Green) / npixels
    Red = frame[:, :, 2]
    average_r = np.sum(Red) / npixels
    nightMode = (math.fabs(average_b - average_g) < settings.ntol) and (
        math.fabs(average_g - average_r) < settings.ntol
    )
    # print(average_b, average_g, average_r)
    #    ifNightMode = np.allclose(Blue, Green, atol = ntol) and np.allclose(Green, Red, atol = ntol)
    return nightMode


def gammac(frame, tavg=155.0):
    """Function to perform gamma correction of image. Gamma computation to aim at the target average grayscale pixel values
    Gamma correction of the image in BGR 3 channel format

    Input parameters:
      frame - 3 channel BGR image
      tavg - target mean value of the corresponding grayscale image
    Returns: gamma corrected 3 channel BGR image
    """
    # conversion of BGR image into grayscale image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # mean pixel value in the grayscale frame
    mean = np.mean(gray)
    # print('mean pixel value:', mean, 'target mean: ', gavg)
    # computation of gamma
    gamma = math.log(tavg / 255.0) / math.log(mean / 255.0)
    # print('gamma=', gamma)
    # apply gamma correction
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    framegc = cv.LUT(frame, lookUpTable)
    # text = f"Gamma: {gamma:.4f}"
    # cv.putText(framegc, text, (20, 50), 0, 2, (255, 0, 0), 3)
    return framegc


def adaptiveHe(frame, contrast=2.0, tile=(8, 8)):
    """CLAHE histogram equalization of BGR image
    correction of Y channel of YCrCb image

    Input: frame - BGR 3-channel picture (ndarray)
           contrast - threshold for limiting contrast
           tile - size of image parts for applying histogram equalization
    Returns: BGR frame with corrected histogram
    """
    # CLAHE class object creation
    clahe = cv.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
    # convert BGR into YCrCb color space
    ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    # split the image into 3 arrays
    y, cr, cb = cv.split(ycrcb)
    # apply adaptive histogram equalization
    ya = clahe.apply(y)
    yacrcb = cv.merge([ya, cr, cb])
    framea = cv.cvtColor(yacrcb, cv.COLOR_YCrCb2BGR)
    return framea
