import cv2
import numpy
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage import data
import sknw
from numba.errors import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
#----------------------------------------------------------------------------------------------------------------------
import tools_image
import tools_filter
#----------------------------------------------------------------------------------------------------------------------
def line_length(x1, y1, x2, y2):return numpy.sqrt((x1-x2)**2 + (y1-y2)**2)
# ----------------------------------------------------------------------------------------------------------------------
blur_kernel = 2
min_line_len = 100
# ----------------------------------------------------------------------------------------------------------------------
filename_in = './data/ex_lines/Image1.jpg'
#filename_in = './data/ex_lines/H.png'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def detect_01(filename_in,folder_out,blur_kernel=5):
    img = cv2.imread(filename_in)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = tools_filter.sliding_2d(gray,blur_kernel,blur_kernel,stat='avg',mode='reflect').astype(numpy.uint8)
    threshholded = 255-cv2.adaptiveThreshold(255-blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

    #edges = threshholded
    #edges = cv2.Canny(threshholded, 10, 50, apertureSize=3)
    edges = (255*skeletonize(threshholded/255)).astype(numpy.uint8)


    lines = cv2.HoughLinesP(edges, 1, numpy.pi / 180, 40, min_line_len)

    result = img.copy()
    result = tools_image.desaturate(result)
    res_bin = numpy.zeros(img.shape,dtype=numpy.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            if (line_length(x1, y1, x2, y2) > 0 * min_line_len):
                result = cv2.line(result, (x1, y1), (x2, y2), (0, 32, 255), 2)
                res_bin = cv2.line(res_bin, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv2.imwrite(folder_out + '1-blur.png', blur)
    cv2.imwrite(folder_out + '2-threshholded.png', threshholded)
    cv2.imwrite(folder_out + '3-edges.png', edges)
    cv2.imwrite(folder_out + '4-res.png', result)
    cv2.imwrite(folder_out + '5-res_bin.png', res_bin)
    return
# ----------------------------------------------------------------------------------------------------------------------
def detect_02(filename_in,folder_out):
    img = cv2.imread(filename_in)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    result = img.copy()
    result = tools_image.desaturate(result)

    lines = cv2.HoughLines(edges, 1, numpy.pi / 180, min_line_len)

    for line in lines:
        for rho, theta in line:
            a = numpy.cos(theta)
            b = numpy.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 - 100 * b)
            y1 = int(y0 + 100 * a)
            x2 = int(x0 + 100 * b)
            y2 = int(y0 - 100 * a)


            cv2.line(result, (x1, y1), (x2, y2), (0, 12, 255), 2)

    cv2.imwrite(folder_out + 'res02.png', result)
    return
# ----------------------------------------------------------------------------------------------------------------------
def detect_03(filename_in,folder_out):

    img = cv2.imread(filename_in)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = tools_filter.sliding_2d(gray, blur_kernel, blur_kernel, stat='avg', mode='reflect').astype(numpy.uint8)
    threshholded = 255 - cv2.adaptiveThreshold(255 - blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)
    ske = skeletonize(threshholded>0).astype(numpy.uint16)
    result  = img.copy()
    result = tools_image.desaturate(result)

    graph = sknw.build_sknw(ske)

    for (s, e) in graph.edges():
        ps = graph[s][e]['pts']
        xx = ps[:, 1]
        yy = ps[:, 0]
        if len(xx)>20 and line_length(xx[0],yy[0],xx[-1],yy[-1]) > 50:
            for i in range(len(xx)-1):
                result = cv2.line(result, (xx[i],yy[i]), (xx[i+1],yy[i+1]), (0, 32, 255),thickness=4)
                #result = cv2.circle(result, (int(ps[:, 1][i]), int(ps[:, 0][i])), 3, (255, 128, 0), -1)


    cv2.imwrite(folder_out + '4-res.png', result)
    plt.show()
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #detect_01(filename_in, folder_out,blur_kernel=4)
    #detect_01(folder_out + 'input.png', folder_out,blur_kernel=4)
    #detect_02(folder_out + '5-res_bin.png', folder_out)
    detect_03(filename_in,folder_out)

