import cv2
import numpy as np


# Display pictures
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def target_detect(imgDir):
    # Read image
    cardImg = cv2.imread(imgDir)
    # cardImg = cv2.imread(imageDir)
    cardImg = cv2.resize(cardImg, (800, int(float(800 / cardImg.shape[1]) * cardImg.shape[0])),
                             interpolation=cv2.INTER_AREA)

    # Eliminate useless numbers in the picture
    # Color space conversion, converted to grayscale
    cardImg_gray = cv2.cvtColor(cardImg, cv2.COLOR_BGR2GRAY)
    # cv_show('cardImg_gray', cardImg_gray)
    cardImg_blur = cv2.GaussianBlur(cardImg_gray, (3, 3), 0)
    # A Method of Detecting Edges Using a Multi-Level Edge Detection Algorithm
    cardImg_canny = cv2.Canny(cardImg_blur, 100, 300)

    # Specify the convolution kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Top-hat operation to highlight brighter areas
    cardImg_tophat = cv2.morphologyEx(cardImg_canny, cv2.MORPH_TOPHAT, kernel)
    # cv_show('cardImg_tophat', cardImg_tophat)

    # Use the sobel operator for edge detection, which is only applicable to the gradient in the x direction
    # cv2.CV_64F makes the output data type set higher
    sobelx = cv2.Sobel(cardImg_tophat, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    (minX, maxX) = (np.min(sobelx), np.max(sobelx))
    sobelx = (255 * ((sobelx - minX) / (maxX - minX)))
    sobelx = sobelx.astype('uint8')

    # y direction
    sobely = cv2.Sobel(cardImg_tophat, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    (minY, maxY) = (np.min(sobely), np.max(sobely))
    sobely = (255 * ((sobely - minY) / (maxY - minY)))
    sobely = sobely.astype('uint8')

    # Merge gradients (approximately
    sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0, dst=None, dtype=None)

    # Closing operation to connect adjacent numbers, which is convenient for screening
    cardImg_close = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, kernel)
    cardImg_binary = cv2.threshold(cardImg_close, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
    # Close operation
    cardImg_close = cv2.morphologyEx(cardImg_binary, cv2.MORPH_CLOSE, kernel)
    # cv_show('cardImg_close', cardImg_close)

    # CCOMP detects all contours, but all contours only establish two hierarchical relationships.
    # The outer periphery is the top layer. If the inner contours in the outer periphery contain other contour
    # information, all contours in the inner periphery belong to the top layer.
    a, contours, hierarchy = cv2.findContours(cardImg_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    i = 0
    for c in contours:  # c stands for each digital small outline
        # Calculate the x, y, w, h of each digital small contour
        (x, y, w, h) = cv2.boundingRect(c)
        # Subtracts each number field in a number combination
        cardImg_rec = cardImg_blur[y:y + h, x:x + w]
        cardImg_rec = cv2.threshold(cardImg_rec, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
        # Fill the edge with the specified pixel value
        cardImg_rec = cv2.copyMakeBorder(cardImg_rec, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
        # cv_show('cardImg_rec', cardImg_rec) 
        cv2.imwrite('test/detection/' + str(i) + '.png', cardImg_rec)
        i = i+1
