# ----------------------------------------------------------------------------------------------------------------------
#import time
#import detector_YOLO3
#import detector_YOLO3_core
import h5py
import numpy
import tools_HDF5
import detector_YOLO3_core
# ----------------------------------------------------------------------------------------------------------------------
def convert_to_Keras():
    D = detector_YOLO3.detector_YOLO3(None)
    filename_config  = 'data/ex70/darknet_to_keras/yolov3.cfg'
    filename_weights = 'data/ex70/darknet_to_keras/yolov3.weights'
    filename_output  = 'data/ex70/darknet_to_keras/_yolov3.h5'
    detector_YOLO3_core.darknet_to_keras(filename_config, filename_weights, filename_output)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_YOLO3_on_file():

    D = detector_YOLO3.detector_YOLO3('data/ex70/yolov3-tiny.h5')
    D.process_image('data/ex70/bike/Image.png', 'data/output/res.jpg')

    return
# ----------------------------------------------------------------------------------------------------------------------
def example_YOLO3_on_folder():

    #D = detector_YOLO3.detector_YOLO3('data/ex70/yolov3-tiny.h5')
    D = detector_YOLO3.detector_YOLO3('data/ex70/racoon_model.h5')
    start_time = time.time()
    D.process_folder('data/ex70/octavia/', 'data/output/')
    print('%s sec\n\n' % (time.time() - start_time))
    return
# ----------------------------------------------------------------------------------------------------------------------
def do_train():
    file_annotations = 'data/ex70/annotation_racoons.txt'   #D.prepare_annotation_file('data/ex09-natural/','data/annotation.txt')
    path_out = 'data/output/'
    D = detector_YOLO3.detector_YOLO3('data/ex70/yolov3a-tiny.h5')
    D.learn_tiny(file_annotations, path_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #do_train()
    #example_YOLO3_on_file()

    anchors = numpy.array([[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]).astype(numpy.float)
    anchor_mask = [[3, 4, 5], [1, 2, 3]]
    list_of_boxes = [[[1, 2, 2, 4, 0],[1, 2, 2, 4, 0]],[[1, 2, 2, 4, 0],[1, 2, 2, 4, 0],[1, 2, 2, 4, 0],],[[1, 2, 2, 4, 0]]]

    store0 = tools_HDF5.HDF5_store(filename='target_0.hdf5', object_shape=(13, 13, 3, 85),dtype=numpy.float32)
    store1 = tools_HDF5.HDF5_store(filename='target_1.hdf5', object_shape=(26, 26, 3, 85),dtype=numpy.float32)




    for k,boxes in enumerate(list_of_boxes):
        true_boxes = numpy.array([boxes])
        y = detector_YOLO3_core.preprocess_true_boxes(true_boxes, (416,416), anchors,anchor_mask, 80)
        store0.append(y[0])
        store1.append(y[1])

    new_store0 = tools_HDF5.HDF5_store(filename='target_0.hdf5')

    for i in range(store0.size):
        val = store0.get(i)
        print(val.shape)


