# ----------------------------------------------------------------------------------------------------------------------
import tools_video
import tools_animation
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    path_input = './data/output/'
    #path_input = 'D:/Projects/Telefonica/flow2/'
    filename_out = path_input + 'anim.gif'

    tools_animation.folder_to_animated_gif_imageio(path_input, filename_out, mask='*.jpg', framerate=60,resize_H=360, resize_W=640)