# ----------------------------------------------------------------------------------------------------------------------
import CNN_AlexNet_TF
import CNN_Inception_TF
import CNN_App_Keras
import classifier_FC_Keras
import face_descriptor
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
import classifier_FC_Keras
# ----------------------------------------------------------------------------------------------------------------------
def example_feature_extraction(path_input, path_output, mask):

	#CNN = CNN_AlexNet_TF.CNN_AlexNet_TF('../_weights/bvlc_alexnet.npy')
	#CNN = CNN_Inception_TF.CNN_Inception_TF()
	#CNN = CNN_App_Keras.CNN_App_Keras()
	#CNN = classifier_FC_Keras.classifier_FC_Keras()
	#CNN = classifier_FC_Keras.classifier_FC_Keras(filename_weights=filename_model)
	CNN = face_descriptor.face_descriptor()

	CNN.generate_features(path_input, path_output+CNN.name+'/',mask=mask)


	return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	path_input  = 'data/ex_faces/'
	path_output = 'data/features_faces/'

	mask = '*.png,*.jpg'

	example_feature_extraction(path_input,path_output,mask)
