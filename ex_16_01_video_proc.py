# ----------------------------------------------------------------------------------------------------------------------
import tools_video
import tools_animation
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
def cap_06():
    URL = 'https://www.youtube.com/watch?v=ulpIL6KhD50'
    out_path = 'D:/'
    out_filename = 'res'
    tools_video.grab_youtube_video(URL, out_path, out_filename)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #filename_in = 'D:/Projects/VFS/Cashiers 1_IP169_port/02.07.2019 14_59_59 (UTC+03_00).mkv'
    #folder_out = 'D:/Projects/VFS/output/'

    #tools_video.extract_frames('D:/res.mp4','D:/3/',prefix='',start_time_sec=70)
    #tools_animation.crop_images_in_folder(path_input,path_output,115, 123, 832, 1400)
    #tools_animation.folder_to_animated_gif_imageio('D:/1/', 'D:/ani.gif', mask='*.jpg', framerate=25,resize_W=210,resize_H=120)

    #tools_animation.folder_to_animated_gif_imageio('D:/JB/', 'D:/JB.gif', mask='*.jpg', framerate=25,resize_W=1120//4,resize_H=540//4)
    #tools_animation.folder_to_video('D:/Projects/Telefonica/flow3_croped/all/','D:/TF_ani4.mp4',mask='*.jpg',resize_W=2154//4,resize_H=1200//4)



    tools_video.extract_frames('D:/soccer_dataset.mp4','D:/soccer/',prefix='')

    #tools_animation.folder_to_video('D:/Projects/Telefonica/flow3_croped/all/', 'D:/TF_ani4.mp4', mask='*.jpg',resize_W=2154 // 4, resize_H=1200 // 4)
    #tools_video.grab_youtube_video('https://www.youtube.com/watch?v=CAUUi8zEZBk','D:/','bbc.avi')


