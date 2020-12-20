import numpy
import cv2
# ----------------------------------------------------------------------------------------------------------------------

import tools_image
import tools_draw_numpy
import tools_Mask
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_convolution/'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_01():
    binarized = cv2.imread(folder_in + 'Image1.png', 0)
    mask255 = cv2.imread(folder_in + 'mask.png', 0)
    mask_pn = numpy.zeros_like(mask255, dtype=numpy.int32)
    mask_pn[mask255 > 128] = +1
    mask_pn[mask255 <= 128] = -1

    C = tools_image.convolve_with_mask(binarized, mask_pn, do_normalize=True)

    cv2.imwrite(folder_out + 'B.png', binarized)
    cv2.imwrite(folder_out + 'conv.png', C)
    cv2.imwrite(folder_out + 'mask.png', mask255)
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    mask255 = cv2.imread(folder_in + 'mask.png', 0)
    M = tools_Mask.Mask(mask255, folder_out)
    M.save_debug()
    binarized = cv2.imread(folder_in + 'Image1.png', 0)

    result = M.convolve(binarized)
    cv2.imwrite(folder_out+'res.png',result)


