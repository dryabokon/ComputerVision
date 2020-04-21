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

    #tools_video.grab_youtube_video('https://www.youtube.com/watch?v=CAUUi8zEZBk','D:/','bbc.avi')
    #tools_video.extract_frames('D:/IMG_0563.MOV','D:/xxx/',prefix='')

    #tools_animation.merge_images_in_folders('D:/xxx/','D:/www/','D:/ttt/',rotate_first=True)
    #tools_animation.folder_to_animated_gif_imageio('D:/ttt/', 'D:/flow.gif', mask='*.jpg', framerate=6,resize_W=3120//4,resize_H=1920//4,do_reverce=False)


#    tools_animation.merge_images_in_folders_temp('D:/iii/','D:/ooo/','D:/ttt/')
    #tools_animation.folder_to_video('D:/ttt/','D:/unet2_small.mp4',mask='*.jpg',resize_W=640//2,resize_H=320//2)

    #tools_video.extract_frames('D:/ball_detavi.webm','D:/xxx/',prefix='',start_time_sec=70)

    tools_animation.folder_to_animated_gif_imageio('D:/xxx/', 'D:/ball_det_pub.gif', mask='*.jpg', framerate=20,resize_W=320,resize_H=200,do_reverce=False)

