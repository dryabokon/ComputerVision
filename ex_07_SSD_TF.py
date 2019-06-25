# ----------------------------------------------------------------------------------------------------------------------
import keras
from keras.applications.mobilenet_v2 import preprocess_input
import numpy
import tensorflow as tf
import cv2
import tools_YOLO
import tools_image
import detector_SSD_TF
# ----------------------------------------------------------------------------------------------------------------------
filename_image = './data/ex16/LON/frame000777.jpg'
filename_out = 'data/output/res_SSD_TF.jpg'
#filename_image = './data/ex70/bike/Image.png'
#filename_topology  = './data/ex71/ssdlite/frozen_inference_graph.pb'
# ----------------------------------------------------------------------------------------------------------------------
filename_image = './data/ex16/LON/frame000777.jpg'
filename_out = 'data/output/res_SSD_TF.jpg'
#filename_image = './data/ex70/bike/Image.png'
#filename_topology  = './data/ex71/ssdlite/frozen_inference_graph.pb'
filename_topology  = './data/ex71/ssd/frozen_inference_graph.pb'
# ----------------------------------------------------------------------------------------------------------------------
def example_SSD_TF_on_file():
    filename_topology  = './data/ex71/ssd/frozen_inference_graph.pb'
    D = detector_SSD_TF.detector_SSD_TF(model_in, metadata_in)
    D.process_file(filename_image, 'data/output/res_yolo.jpg')


    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    example_SSD_TF_on_file()