import os
from argparse import Namespace
#----------------------------------------------------------------------------------------------------------------------
from face_filter import Face_filter
#----------------------------------------------------------------------------------------------------------------------
log_dir         = './data/output/'
sample_dir      = './data/output/'
result_dir      = './data/output/'
#----------------------------------------------------------------------------------------------------------------------
def get_config():
    config = Namespace(c_dim=5, c2_dim=8)
    config.model_save_dir = './data/ex_face_filter/stargan_celeba_128/models'
    config.celeba_image_dir = './data/ex_face_filter/celeba/images'
    config.attr_path = './data/ex_face_filter/celeba/list_attr_celeba.txt'
    config.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    config.image_size = 128
    config.g_conv_dim = 64
    config.d_conv_dim = 64
    config.g_repeat_num = 6
    config.d_repeat_num = 6
    config.lambda_cls = float(1.0)
    config.lambda_rec = float(10.0)
    config.lambda_gp = float(10.0)
    config.dataset = 'CelebA'
    config.batch_size = 1
    config.num_iters = 200000
    config.num_iters_decay = 100000
    config.g_lr = float(0.0001)
    config.d_lr = float(0.0001)
    config.n_critic = 5
    config.beta1 = float(0.5)
    config.beta2 = float(0.999)
    config.resume_iters = None

    config.test_iters = 200000
    config.use_tensorboard = False
    config.device = 'cpu'
    config.log_dir = log_dir
    config.sample_dir = sample_dir

    config.result_dir = result_dir
    config.log_step = 10
    config.sample_step = 1000
    config.model_save_step = 10000
    config.lr_update_step = 1000
    return config
#----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    config = get_config()
    P = Face_filter(config)
    #P.process_file('./data/ex_face_filter/celeba/images/000006.jpg','./data/output/res.jpg')
    P.process_folder('./data/ex_face_filter/celeba/images/','./data/output/')


