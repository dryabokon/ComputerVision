# ----------------------------------------------------------------------------------------------------------------------
import time
import detector_SSD300
# ----------------------------------------------------------------------------------------------------------------------
model_weights_h5    = './data/ex70/VGG_coco_SSD_300x300.h5'
# ----------------------------------------------------------------------------------------------------------------------
def example_SSD_on_file():
    filename_image = './data/ex16/LON/frame000777.jpg'
    filename_out = './data/output/res_SSD.jpg'
    D = detector_SSD300.detector_SSD300(model_weights_h5)
    D.process_file(filename_image, filename_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_SSD_on_folder():

    D = detector_SSD300.detector_SSD300(model_weights_h5)
    start_time = time.time()
    D.process_folder('./data/ex16/LON/', './data/output/')
    print('%s sec\n\n' % (time.time() - start_time))
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    example_SSD_on_file()








