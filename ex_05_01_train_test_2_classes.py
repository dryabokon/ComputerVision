import os
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
import tools_ML
import CNN_AlexNet_TF
import CNN_App_Keras
import classifier_FC_Keras
# ---------------------------------------------------------------------------------------------------------------------
def classify_images(folder_in,folder_out,mask):
    C = classifier_FC_Keras.classifier_FC_Keras(folder_debug=folder_out)
    ML = tools_ML.tools_ML(C)
    ML.E2E_images(folder_in, folder_out, mask = mask,limit_classes=3,resize_W=64, resize_H =64)

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    folder_in = 'data/ex_LPR/'
    mask = '*.bmp'
    folder_out = 'data/output/'

    classify_images(folder_in,folder_out,mask)


