import argparse
from argparse import Namespace
#----------------------------------------------------------------------------------------------------------------------
from face_filter import Face_filter
#----------------------------------------------------------------------------------------------------------------------
def get_config():
    config = Namespace(c_dim=5, c2_dim=8)
    config.result_dir = './data/output/'

    config.model_save_dir   = './data/ex_face_filter/stargan_celeba_128/models'
    config.celeba_image_dir = './data/ex_face_filter/celeba'
    config.attr_path        = './data/ex_face_filter/celeba/list_attr_celeba.txt'
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

    config.log_dir = config.result_dir
    config.sample_dir = config.result_dir

    config.log_step = 10
    config.sample_step = 1000
    config.model_save_step = 10000
    config.lr_update_step = 1000
    return config
#----------------------------------------------------------------------------------------------------------------------
def main(command,filename_in,folder_in,folder_out):
    config = get_config()
    P = Face_filter(config)

    if command=='process_file':
        P.process_file(filename_in,folder_out+'result.jpg')
    if command == 'process_folder':
        P.process_folder(folder_in,folder_out)
    #if command == 'train':
    #    P.train()
    return
#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--command', default='process_folder')
    parser.add_argument('--filename_in', default='./data/ex_face_filter/celeba/000006.jpg')
    parser.add_argument('--folder_in'  , default='./data/ex_face_filter/celeba/')
    parser.add_argument('--folder_out' , default='./data/output/')
    args = parser.parse_args()

    main(args.command, args.filename_in, args.folder_in,args.folder_out)






