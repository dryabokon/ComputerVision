import cv2
import tools_soccer_field
#----------------------------------------------------------------------------------------------------------------------
filename_in = './data/ex_lines/Image1.jpg'
#filename_in = './data/ex_lines/frame001710.jpg'
SFP = tools_soccer_field.Soccer_Field_Processor()
# ----------------------------------------------------------------------------------------------------------------------
def example_01(filename_in):
    image = cv2.imread(filename_in)
    skeleton = SFP.skelenonize_slow(image,do_debug=True)
    SFP.get_hough_lines(skeleton,do_debug=True)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    example_01(filename_in)


