# ----------------------------------------------------------------------------------------------------------------------
import time
import detector_Mask_RCNN
# ----------------------------------------------------------------------------------------------------------------------
filename_weights = './data/ex_rmask_cnn/mask_rcnn_coco.h5'
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def example_Mask_RCNN_on_file():

    filename_image = './data/ex_detector/bike/Image.png'
    D = detector_Mask_RCNN.detector_Mask_RCNN(filename_weights, folder_out)
    D.process_file(filename_image, './data/output/res_MASK_RCNN.jpg',draw_spline=True)

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_Mask_RCNN_on_folder():

    #folder_in = './data/ex16g/LON/'
    folder_in = 'D:/LM/ex03/'
    D = detector_Mask_RCNN.detector_Mask_RCNN(filename_weights, folder_out)
    start_time = time.time()
    D.process_folder(folder_in, folder_out)
    print('%s sec\n\n' % (time.time() - start_time))
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #example_Mask_RCNN_on_file()
    example_Mask_RCNN_on_folder()









