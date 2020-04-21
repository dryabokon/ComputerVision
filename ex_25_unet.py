import cv2
import detector_Unet
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def do_train(train_folder_in):

    CNN = detector_Unet.detector_Unet()
    X_train, Y_train = CNN.get_data_train(train_folder_in)
    CNN.fit(X_train, Y_train)

    return
# ----------------------------------------------------------------------------------------------------------------------
def do_predict(filename_in):

    CNN = detector_Unet.detector_Unet(folder_out+'trained.h5')
    image = cv2.imread(filename_in)
    result = CNN.predict(image)
    cv2.imwrite(folder_out+ 'result.png', result)
    return
# ----------------------------------------------------------------------------------------------------------------------
train_folder_in ='./data/ex_Unet/train_02/'
# ----------------------------------------------------------------------------------------------------------------------
filename_in = './data/ex_Unet/train_02/200118_TSG_HOFFENHEIM_EINTRACHT_FRANKFURT_00050100.jpg'
if __name__ == '__main__':

    #do_train(train_folder_in)
    do_predict(filename_in)
