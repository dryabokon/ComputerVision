import argparse
#----------------------------------------------------------------------------------------------------------------------
from face_filter import Face_filter
#----------------------------------------------------------------------------------------------------------------------
def main(command,filename_in,folder_in,filename_G_weights,filename_D_weights,folder_out,folder_train_images=None,file_train_attrib=None):

    if command=='process_file':
        P = Face_filter(folder_out, filename_G_weights, filename_D_weights)
        P.process_file(filename_in,folder_out+'result.jpg')
    if command == 'process_folder':
        P = Face_filter(folder_out, filename_G_weights, filename_D_weights)
        P.process_folder(folder_in,folder_out)

    if command == 'train':
        P = Face_filter(folder_out)
        P.train(folder_train_images,file_train_attrib)
    return
#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--command', default='train')
    parser.add_argument('--filename_D_weights', default='./data/ex_face_filter/stargan_celeba_128/models/200000-D.ckpt')
    parser.add_argument('--filename_G_weights', default='./data/ex_face_filter/stargan_celeba_128/models/200000-G.ckpt')
    parser.add_argument('--filename_in', default='./data/ex_face_filter/celeba/000006.jpg')
    parser.add_argument('--folder_in'  , default='./data/ex_face_filter/celeba/')
    parser.add_argument('--folder_out' , default='./data/output/')
    parser.add_argument('--folder_train_images', default='./data/ex_face_filter/celeba')
    parser.add_argument('--file_train_attrib', default='./data/ex_face_filter/celeba/list_attr_celeba.txt')

    args = parser.parse_args()

    main(args.command, args.filename_in, args.folder_in,args.filename_G_weights,args.filename_D_weights,args.folder_out,args.folder_train_images,args.file_train_attrib)






