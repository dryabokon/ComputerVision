# ----------------------------------------------------------------------------------------------------------------------
import os
import cv2
# ----------------------------------------------------------------------------------------------------------------------
from keras.applications.xception import Xception
from keras.applications import MobileNet
from keras import backend as K
#K.set_image_dim_ordering('tf')
# ----------------------------------------------------------------------------------------------------------------------
import classifier_FC_Keras
import CNN_VGG16_Keras
import CNN_AlexNet_TF
import tools_IO
import tools_CNN_view
import detector_YOLO3
import detector_Zebra
import CNN_App_Keras
# ----------------------------------------------------------------------------------------------------------------------
def visualize_layers_detector_Zebra():

    filename_input = './data/ex_LPR/CZE_20150501071848723_VD.jpg'
    path_out = './data/output/'

    #if not os.path.exists(path_out):
    #    os.makedirs(path_out)
    #else:tools_IO.remove_files(path_out)

    filename_weights = './data/output/A_model.h5'
    CNN = detector_Zebra.detector_Zebra(filename_weights)

    tools_CNN_view.visualize_layers(CNN.model,filename_input, path_out)
    tools_CNN_view.visualize_filters(CNN.model,path_out)


    return
# ----------------------------------------------------------------------------------------------------------------------
def visualize_layers_TF_Alexnet():

    filename_input = './data/ex_natural_images/dog/dog_0100.jpg'
    path_output = './data/output/'

    if not os.path.exists(path_output):
        os.makedirs(path_output)
    else:tools_IO.remove_files(path_output)

    CNN = CNN_AlexNet_TF.CNN_AlexNet_TF('../_weights/bvlc_alexnet.npy')
    image = cv2.imread(filename_input)
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    print(CNN.predict(image))

    CNN.visualize_filters(path_output)
    CNN.visualize_layers(filename_input, path_output)

    return
# ----------------------------------------------------------------------------------------------------------------------
def visualize_layers_detector_YOLO3():

    filename_image = 'data/ex_detector/bike/Image.png'
    path_out = 'data/output/'

    if not os.path.exists(path_out):
        os.makedirs(path_out)
    else:tools_IO.remove_files(path_out)

    CNN = detector_YOLO3.detector_YOLO3('./data/ex_yolo/MODELS/model_default.h5','./data/ex_YOLO/models/metadata_default.txt')

    tools_CNN_view.visualize_layers(CNN.model,filename_image, path_out)
    #tools_CNN_view.visualize_filters(CNN.model,path_out)


    return
# ----------------------------------------------------------------------------------------------------------------------
def visualize_layers_keras_VGG16():

    filename_image = 'data/ex_natural_images/dog/dog_0000.jpg'
    path_out = 'data/output/'

    if not os.path.exists(path_out):
        os.makedirs(path_out)
    else:tools_IO.remove_files(path_out)

    CNN = CNN_VGG16_Keras.CNN_VGG16_Keras()
    tools_CNN_view.visualize_filters(CNN.model, path_out)
    tools_CNN_view.visualize_layers(CNN.model,filename_image, path_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
def visualize_layers_keras_MobileNet():

    filename_image = 'data/ex_natural_images/dog/dog_0101.jpg'
    path_out = 'data/output/'

    if not os.path.exists(path_out):
        os.makedirs(path_out)
    else:tools_IO.remove_files(path_out)

    CNN = CNN_App_Keras.CNN_App_Keras()
    tools_CNN_view.visualize_filters(CNN.model, path_out)
    tools_CNN_view.visualize_layers(CNN.model,filename_image, path_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #visualize_layers_detector_Zebra()
    visualize_layers_TF_Alexnet()
    #visualize_layers_keras_MobileNet()
    #visualize_layers_detector_YOLO3()
