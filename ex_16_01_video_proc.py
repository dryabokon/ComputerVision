# ----------------------------------------------------------------------------------------------------------------------
import tools_video
import tools_animation
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    path_input = './data/output_ped/'
    filename_out = path_input + 'anim_ped.gif'
    #tools_video.extract_frames(filename_in,folder_out)
    tools_animation.folder_to_animated_gif_imageio(path_input, filename_out, mask='*.jpg', framerate=10,resize_H=270, resize_W=480)