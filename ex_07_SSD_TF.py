# ----------------------------------------------------------------------------------------------------------------------
import time
import keras
from keras.applications.mobilenet_v2 import preprocess_input
import numpy
import tensorflow as tf
import cv2
import tools_YOLO
import tools_image
import detector_SSD_TF
# ----------------------------------------------------------------------------------------------------------------------
filename_topology  = './data/ex71/ssd/frozen_inference_graph.pb'
# ----------------------------------------------------------------------------------------------------------------------
def example_SSD_TF_on_file():
    filename_image = './data/ex16/LON/frame000777.jpg'
    filename_out = 'data/output/res_SSD_TF.jpg'
    D = detector_SSD_TF.detector_SSD_TF(filename_topology)
    D.process_file(filename_image, filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_SSD_TF_on_folder():
    folder_in= './data/ex16/LON/'
    folder_out = 'data/output/'
    D = detector_SSD_TF.detector_SSD_TF(filename_topology)
    start_time = time.time()
    D.process_folder(folder_in,folder_out)
    print('%s sec\n\n' % (time.time() - start_time))
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    example_SSD_TF_on_file()
    #example_SSD_TF_on_folder()

