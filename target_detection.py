import turtle

import cv2
import numpy as np


def cv_show(name, img):  # 定义一个函数，显示图片
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def target_detect(imgDir):
    # 读取信用卡图片
    cardImg = cv2.imread(imgDir)
    # cardImg = cv2.imread(imageDir)
    cardImg = cv2.resize(cardImg, (800, int(float(800 / cardImg.shape[1]) * cardImg.shape[0])),
                             interpolation=cv2.INTER_AREA)

    # 对图片中无用的数字进行剔除
    # 色彩空间转换，转换成灰度图
    cardImg_gray = cv2.cvtColor(cardImg, cv2.COLOR_BGR2GRAY)
    # cv_show('cardImg_gray', cardImg_gray)
    cardImg_blur = cv2.GaussianBlur(cardImg_gray, (3, 3), 0)
    cardImg_canny = cv2.Canny(cardImg_blur, 100, 300)  # 使用多级边缘检测算法检测边缘的方

    # 指定卷积核大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 进行礼帽操作,突出更明亮的区域
    cardImg_tophat = cv2.morphologyEx(cardImg_canny, cv2.MORPH_TOPHAT, kernel)
    # cv_show('cardImg_tophat', cardImg_tophat)

    # 使用sobel算子进行边缘检测，这里仅适用于x方向的梯度
    # 如果这两种边界你都想检测到，最好的的办法就是将输出的数据类型设置的更高，比如cv2.CV_16S，cv2.CV_64F 等
    sobelx = cv2.Sobel(cardImg_tophat, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    (minX, maxX) = (np.min(sobelx), np.max(sobelx))
    sobelx = (255 * ((sobelx - minX) / (maxX - minX)))
    sobelx = sobelx.astype('uint8')

    # y方向
    sobely = cv2.Sobel(cardImg_tophat, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    (minY, maxY) = (np.min(sobely), np.max(sobely))
    sobely = (255 * ((sobely - minY) / (maxY - minY)))
    sobely = sobely.astype('uint8')

    # 合并梯度（近似
    sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0, dst=None, dtype=None)

    # 进行闭运算，使相邻的数字连接起来，这样便于筛选
    cardImg_close = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, kernel)
    # 自动二值化 OTSU的处理
    # cardImg_binary = cv2.adaptiveThreshold(cardImg_close, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 10)
    # cv_show('cardImg_binary', cardImg_binary)
    cardImg_binary = cv2.threshold(cardImg_close, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
    # 闭运算
    cardImg_close = cv2.morphologyEx(cardImg_binary, cv2.MORPH_CLOSE, kernel)
    # cv_show('cardImg_close', cardImg_close)

    # 轮廓检测，检测出每个数字区块;寻找数字轮廓
    # CCOMP检测所有的轮廓，但所有轮廓只建立两个等级关系，外围为顶层，若外围内的内围轮廓还包含了其他的轮廓信息，则内围内的所有轮廓均归属于顶层
    # c, contours, hierarchy = cv2.findContours(cardImg_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # back = cv2.drawContours(cardImg_close.copy(), contours, -1, (255, 255, 255), -1)
    # background = cv2.bitwise_not(back)
    # b = cv2.add(background, c)
    # cv_show('a', c)
    # cv_show('a', back)
    # cv_show('a', b)

    # cardImg_close_rev = cv2.bitwise_not(cardImg_close)
    a, contours, hierarchy = cv2.findContours(cardImg_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cardImg_cnts = cv2.drawContours(cardImg.copy(), contours, -1, (0, 0, 255), 1)
    # cv_show('a', cardImg_cnts)


    # h, w = cardImg_binary.shape[:2]
    # mask = np.zeros((h+2, w+2), np.uint8)
    #
    # im_floodfill = cardImg_binary.copy()
    # # Floodfill from point (0, 0)
    # cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    #
    # # Invert floodfilled image
    # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    #
    # # Combine the two images to get the foreground.
    # im_out = cardImg_binary | im_floodfill_inv
    # cv_show('a', im_out)

    i = 0
    for c in contours:  # c代表每一个数字小轮廓
        (x, y, w, h) = cv2.boundingRect(c)  # 计算每一个数字小轮廓的x,y,w,h
        cardImg_rec = cardImg_blur[y:y + h, x:x + w]  # 在数字组合中扣除每一个数字区域

        cardImg_rec = cv2.threshold(cardImg_rec, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
        # 填充边缘，用指定的像素值
        cardImg_rec = cv2.copyMakeBorder(cardImg_rec, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
        cv_show('cardImg_rec', cardImg_rec)  # 扣出了所有的数字
        cv2.imwrite('test/detection/' + str(i) + '.png', cardImg_rec)
        i = i+1

    # for i in range(len(contours)):
    #     cardImg_cnt1 = cv2.drawContours(cardImg_close.copy(), contours, -1, (255, 255, 255), -1, offset=)
    #     # cardImg_cnt1 = cv2.drawContours(cardImg_close.copy(), contours, i, (255, 255, 255), -1)
    #     cv_show('cardImg_cnt1', cardImg_cnt1)
    # print(hierarchy)
