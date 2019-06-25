# ----------------------------------------------------------------------------------------------------------------------
import time
import detector_SSD300
# ----------------------------------------------------------------------------------------------------------------------
filename_image = './data/ex16/LON/frame000777.jpg'
filename_out = 'data/output/res_SSD_TF.jpg'
default_model_in    = './data/ex70/VGG_coco_SSD_300x300.h5'
default_metadata_in = './data/ex70/metadata_default.txt'
# ----------------------------------------------------------------------------------------------------------------------
def example_SSD_on_file():

    D = detector_SSD300.detector_SSD300(default_model_in, default_metadata_in)
    D.process_file(filename_image, filename_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_SSD_on_folder():

    D = detector_SSD300.detector_SSD300(default_model_in, default_metadata_in, obj_threshold=0.50)
    start_time = time.time()
    D.process_folder('./data/ex16/LON/', './data/output/')
    print('%s sec\n\n' % (time.time() - start_time))
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #do_train()
    example_SSD_on_file()








