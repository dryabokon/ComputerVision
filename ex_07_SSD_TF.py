# ----------------------------------------------------------------------------------------------------------------------
import time
import detector_SSD_TF
import detector_YOLO3
# ----------------------------------------------------------------------------------------------------------------------
filename_image = './data/ex16g/LON/frame000777.jpg'
filename_out = 'data/output/res_SSD_TF.jpg'
filename_topology  = './data/ex71/ssdlite/frozen_inference_graph.pb'
# ----------------------------------------------------------------------------------------------------------------------
def example_SSD_TF_on_file():
    filename_topology  = './data/ex71/ssd/frozen_inference_graph.pb'
    D = detector_SSD_TF.detector_SSD_TF(filename_topology)
    D.process_file(filename_image, 'data/output/res_yolo.jpg')


    return
# ----------------------------------------------------------------------------------------------------------------------
def example_SSD_TF_on_folder():

    D = detector_SSD_TF.detector_SSD_TF(filename_topology)
    start_time = time.time()
    D.process_folder('./data/ex16g/LON/', './data/output/')
    print('%s sec\n\n' % (time.time() - start_time))
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #example_SSD_TF_on_file()
    example_SSD_TF_on_folder()