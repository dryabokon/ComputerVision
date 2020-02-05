import time
import numpy
import cv2
import tools_filter
#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    A = cv2.imread('./data/ex_detector/bike/Image.png')
    start_time = time.time()
    for i in range(100):
        B = tools_filter.sliding_2d(A[:,:,0],10,10)
    print(time.time()-start_time)


    cv2.imwrite('./data/output/res.png',B)


