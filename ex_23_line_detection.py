import numpy
import cv2
import tools_soccer_field
import tools_IO
from PIL import Image, ImageDraw, ImageOps
#----------------------------------------------------------------------------------------------------------------------
filename_in = './data/ex_lines/frame000269.jpg'
SFP = tools_soccer_field.Soccer_Field_Processor()
# ----------------------------------------------------------------------------------------------------------------------
def example_01(filename_in):
    image = cv2.imread(filename_in)
    skeleton = SFP.skelenonize_slow(image,do_debug=True)
    SFP.get_hough_lines(skeleton,do_debug=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #example_01(filename_in)
    lines = tools_IO.load_mat( './data/ex_lines/ske.txt')
    SFP.dr


