# ----------------------------------------------------------------------------------------------------------------------
import detector_YOLO3
import detector_YOLO3_core
# ----------------------------------------------------------------------------------------------------------------------
model_in    = './data/ex_YOLO/models/model_default.h5'
metadata_in = './data/ex_YOLO/models/metadata_default.txt'
# ----------------------------------------------------------------------------------------------------------------------
def convert_to_Keras():
    D = detector_YOLO3.detector_YOLO3(None,None)
    filename_config  = 'data/ex70/darknet_to_keras/yolov3.cfg'
    filename_weights = 'data/ex70/darknet_to_keras/yolov3.weights'
    filename_output  = 'data/ex70/darknet_to_keras/_yolov3.h5'
    detector_YOLO3_core.darknet_to_keras(filename_config, filename_weights, filename_output)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_YOLO3_on_file():
    filename_image = './data/ex_detector/bike/Image.png'
    D = detector_YOLO3.detector_YOLO3(model_in,metadata_in)
    D.process_file(filename_image, './data/output/res_yolo.jpg')
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_YOLO3_on_folder():

    folder_in   = './data/ex_detector/LON/'
    D = detector_YOLO3.detector_YOLO3(model_in, metadata_in, obj_threshold=0.50)
    D.process_folder(folder_in, './data/output/')
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    example_YOLO3_on_file()
    #example_YOLO3_on_folder()







