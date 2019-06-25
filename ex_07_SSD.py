# ----------------------------------------------------------------------------------------------------------------------
import time
import detector_SSD300
# ----------------------------------------------------------------------------------------------------------------------
#filename_image = './data/ex16g/LON/frame000777.jpg'
filename_image      = './data/ex70/bike/Image.png'
filename_out        = './data/output/res_SSD_TF.jpg'
default_model_in    = './data/ex71/MobileNetSSD300weights_voc_2007_class20.hdf5'
default_metadata_in = './data/ex71/metadata_default.txt'
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








