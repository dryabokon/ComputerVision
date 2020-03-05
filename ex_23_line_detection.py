import math
import cv2
import numpy
import cv2
import tools_soccer_field
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
filename_in = './data/ex_lines/frame000269.jpg'
SFP = tools_soccer_field.Soccer_Field_Processor()
# ----------------------------------------------------------------------------------------------------------------------
def example_01(filename_in):
    image = cv2.imread(filename_in)
    skeleton = SFP.skelenonize_slow(image,do_debug=True)
    SFP.get_hough_lines(skeleton,do_debug=True)
    return
# ----------------------------------------------------------------------------------------------------------------------
def lines_to_chart(filename_in,folder_out):

    img = cv2.imread(filename_in)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, numpy.pi / 180, 10)

    image_map = numpy.zeros((img.shape[0]*2,180))

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

            angle = 90+math.atan((x2-x1)/(y2-y1))*180/math.pi
            image_map [int(y0),int(angle)]=255
            print(y0, angle)

    cv2.imwrite(folder_out + 'map.png',image_map)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    example_01(filename_in)

