import cv2 as cv
from utils.ImageEnhancement import SSR_image, MSR_image, hist

if __name__ == '__main__':
    src_path = './src/data_test/test_4.jpg'
    img = cv.imread(src_path)
    cv.imshow('before', img)
    cv.imshow('after_1_hist', hist(img))
    cv.imshow('after_2_msr', MSR_image(img))
    cv.imshow('after_3_ssr', SSR_image(img))
    cv.waitKey(0)
    cv.destroyAllWindows()
