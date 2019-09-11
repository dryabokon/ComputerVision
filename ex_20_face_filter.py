import argparse
#----------------------------------------------------------------------------------------------------------------------
from face_filter import Face_filter
#----------------------------------------------------------------------------------------------------------------------
def main(command,filename_in,folder_in,filename_G_weights,filename_D_weights,folder_out):

    P = Face_filter(folder_out,filename_G_weights,filename_D_weights)

    if command=='process_file':
        P.process_file(filename_in,folder_out+'result.jpg')
    if command == 'process_folder':
        P.process_folder(folder_in,folder_out)

    #if command == 'train':
    #   image_dir ='./data/ex_face_filter/celeba'
    #   attr_path = './data/ex_face_filter/celeba/list_attr_celeba.txt'
    #   P.train(image_dir, attr_path)
    return
#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--command', default='process_folder')
    parser.add_argument('--filename_D_weights', default='./data/ex_face_filter/stargan_celeba_128/models/200000-D.ckpt')
    parser.add_argument('--filename_G_weights', default='./data/ex_face_filter/stargan_celeba_128/models/200000-G.ckpt')
    parser.add_argument('--filename_in', default='./data/ex_face_filter/celeba/000006.jpg')
    parser.add_argument('--folder_in'  , default='./data/ex_face_filter/celeba/')
    parser.add_argument('--folder_out' , default='./data/output/')

    args = parser.parse_args()

    main(args.command, args.filename_in, args.folder_in,args.filename_G_weights,args.filename_D_weights,args.folder_out)






