# ----------------------------------------------------------------------------------------------------------------------
def healthcheck_CV():
    import cv2
    t = cv2.min(0, 0)
    return
# ----------------------------------------------------------------------------------------------------------------------
def healthcheck_Keras_GPU():
    from keras import backend

    available_gpus = backend.tensorflow_backend._get_available_gpus()
    if len(available_gpus):
        print('available_gpus', available_gpus)
    return
# ----------------------------------------------------------------------------------------------------------------------
def healthcheck_GPU():

    from tensorflow.python.client import device_lib

    list_local_devices = str(device_lib.list_local_devices())
    print(list_local_devices)
    if 'GPU' in list_local_devices:
        print('GPU OK')

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    healthcheck_CV()
    healthcheck_Keras_GPU()
    healthcheck_CV()

