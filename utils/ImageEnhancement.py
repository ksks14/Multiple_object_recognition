import cv2 as cv
import numpy as np
from utils.times import func_time


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def SSR(src_img, size):
    L_blur = cv.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv.log(img / 255.0)
    dst_Lblur = cv.log(L_blur / 255.0)
    dst_IxL = cv.multiply(dst_Img, dst_Lblur)
    log_R = cv.subtract(dst_Img, dst_IxL)

    dst_R = cv.normalize(log_R, None, 0, 255, cv.NORM_MINMAX)
    log_uint8 = cv.convertScaleAbs(dst_R)
    return log_uint8

@func_time
def SSR_image(image):
    size = 5
    b_gray, g_gray, r_gray = cv.split(image)
    b_gray = SSR(b_gray, size)
    g_gray = SSR(g_gray, size)
    r_gray = SSR(r_gray, size)
    result = cv.merge([b_gray, g_gray, r_gray])
    return result

def MSR(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv.log(img / 255.0)
        dst_Lblur = cv.log(L_blur / 255.0)
        dst_Ixl = cv.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv.subtract(dst_Img, dst_Ixl)
    dst_R = cv.normalize(log_R, None, 0, 255, cv.NORM_MINMAX)
    log_uint8 = cv.convertScaleAbs(dst_R)
    return log_uint8

@func_time
def MSR_image(image):
    scales = [15, 101, 301]  # [3,5,9]
    b_gray, g_gray, r_gray = cv.split(image)
    b_gray = MSR(b_gray, scales)
    g_gray = MSR(g_gray, scales)
    r_gray = MSR(r_gray, scales)
    result = cv.merge([b_gray, g_gray, r_gray])
    return result


@func_time
def hist(image):
    r, g, b = cv.split(image)
    r1 = cv.equalizeHist(r)
    g1 = cv.equalizeHist(g)
    b1 = cv.equalizeHist(b)
    image_equal_clo = cv.merge([r1, g1, b1])
    return image_equal_clo
