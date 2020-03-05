import math
import cv2
import numpy
import cv2
import tools_soccer_field
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
filename_in = './data/ex_lines/frame000626.jpg'
folder_in = './data/ex_lines/'
filename_in = './data/ex_lines/Image1.jpg'
SFP = tools_soccer_field.Soccer_Field_Processor()
# ----------------------------------------------------------------------------------------------------------------------
def example_01(filename_in,do_debug=True):

    image = cv2.imread(filename_in)
    result = SFP.process_left_view(image,do_debug)
    cv2.imwrite(folder_out + filename_in.split('/')[-1], result)
    return result
# ----------------------------------------------------------------------------------------------------------------------
def example_02(folder_in,folder_out):
    SFP.process_folder(folder_in, folder_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    example_01(filename_in)
    #example_02(folder_in,folder_out)


