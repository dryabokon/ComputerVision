# ----------------------------------------------------------------------------------------------------------------------
import detector_Zebra
import ex_03_05_visualize_layers
# ----------------------------------------------------------------------------------------------------------------------
def example_on_file():
    filename_image = './data/ex_LPR/CZE_20150501071848723_VD.jpg'
    filename_weights = './data/A_model.h5'
    D = detector_Zebra.detector_Zebra()
    D.process_file(filename_image, 'data/output/res_zebra2.jpg')

    return
# ----------------------------------------------------------------------------------------------------------------------
def learn():
    D = detector_Zebra.detector_Zebra()
    folder_annotation = './data/ex_LPR_CZ/'
    file_annotations = folder_annotation + 'markup.txt'
    folder_out = './data/output/'

    D.learn(file_annotations, folder_out,folder_annotation)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #example_on_file()
    learn()
    ex_03_05_visualize_layers.visualize_layers_detector_Zebra()