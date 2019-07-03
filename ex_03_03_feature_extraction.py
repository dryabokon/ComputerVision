# ----------------------------------------------------------------------------------------------------------------------
import CNN_AlexNet_TF
import CNN_Inception_TF
import CNN_App_Keras
import classifier_FC_Keras
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
import classifier_FC_Keras
# ----------------------------------------------------------------------------------------------------------------------
def example_feature_extraction(path_input, path_output, mask):

	CNN = CNN_AlexNet_TF.CNN_AlexNet_TF()
	#CNN = CNN_Inception_TF.CNN_Inception_TF()
	#CNN = CNN_App_Keras.CNN_App_Keras()
	#CNN = classifier_FC_Keras.classifier_FC_Keras()
	#CNN = classifier_FC_Keras.classifier_FC_Keras(filename_weights=filename_model)

	CNN.generate_features(path_input, path_output+CNN.name+'/',mask=mask,limit=100)


	return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	#path_input  = 'data/ex_digits_mnist/'
	#path_output = 'data/features_ex_digits_mnist/'
	path_input  = 'data/ex_natural_images/'
	path_output = 'data/features_ex_natural_images/'
	mask = '*.jpg'

	example_feature_extraction(path_input,path_output,mask)
