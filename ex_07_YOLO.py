# ----------------------------------------------------------------------------------------------------------------------
import time
import detector_YOLO3
import tools_video
# ----------------------------------------------------------------------------------------------------------------------
def convert_to_Keras():
    D = detector_YOLO3.detector_YOLO3(None)
    filename_config  = 'data/ex70/darknet_to_keras/yolov3.cfg'
    filename_weights = 'data/ex70/darknet_to_keras/yolov3.weights'
    filename_output  = 'data/ex70/darknet_to_keras/_yolov3.h5'
    D.darknet_to_keras(filename_config, filename_weights, filename_output)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_YOLO3_on_file():
    model_in = './data/ex70/model_default_full.h5'
    metadata_in = './data/ex70/metadata_default_full.txt'
    filename_image = './data/ex16/LON/frame000777.jpg'
    D = detector_YOLO3.detector_YOLO3(model_in,metadata_in)
    D.process_file(filename_image, 'data/output/res_yolo.jpg')

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_YOLO3_on_folder():

    #model_in = './data/ex70/model_default.h5'
    #metadata_in = './data/ex70/metadata_default.txt'
    model_in = './data/ex70/model_default_full.h5'
    metadata_in = './data/ex70/metadata_default_full.txt'

    D = detector_YOLO3.detector_YOLO3(model_in, metadata_in, obj_threshold=0.50)
    start_time = time.time()
    D.process_folder('./data/ex16/LON/', './data/output/')
    print('%s sec\n\n' % (time.time() - start_time))
    return
# ----------------------------------------------------------------------------------------------------------------------
def do_train():
    file_annotations = 'data/ex70/annotation_racoons.txt'   #D.prepare_annotation_file('data/ex09-natural/','data/annotation.txt')
    path_out = 'data/output/'
    D = detector_YOLO3.detector_YOLO3('data/ex70/yolov3a-tiny.h5')
    D.learn(file_annotations, path_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #do_train()
    example_YOLO3_on_file()
    #example_YOLO3_on_folder()







