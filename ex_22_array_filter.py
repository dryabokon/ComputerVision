import time
import numpy
import cv2
import tools_filter
import tools_image
#----------------------------------------------------------------------------------------------------------------------
def example_01():
    image = cv2.imread('./data/ex_filter/image.png')
    image = tools_image.desaturate_2d(image)/255
    h_neg, h_pos, w_neg, w_pos = -2,2,-2,2

    result = tools_filter.sliding_2d_v2(image, h_neg, h_pos, w_neg, w_pos, stat='cnt', mode='constant')
    cv2.imwrite('./data/output/result.png',result.astype(numpy.uint8))
    return
#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    example_01()



