import cv2
import numpy
import math
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

    #tools_video.grab_youtube_video('https://www.youtube.com/watch?v=g-skPkW75mQ','D:/','dashcam.avi')
    tools_video.extract_frames('D:/vlc-record-2020-12-16-14h26m55s-rtsp___192.168.1.160_554_h264-.mp4','D:/mm/',prefix='')

    #tools_animation.merge_images_in_folders('D:/s_origs/','D:/s_hist/','D:/ex05/',rotate_first=False)
    #tools_animation.folder_to_animated_gif_imageio('D:/ttt/', 'D:/flow.gif', mask='*.jpg', framerate=6,resize_W=3120//4,resize_H=1920//4,do_reverce=False)


    #tools_animation.merge_images_in_folders_temp('D:/iii/','D:/ooo/','D:/ttt/')


    #tools_animation.crop_images_in_folder('D:/ys/','D:/PP2/',220,135, 765,1050,mask='*.jpg')
    #tools_animation.folder_to_video('D:/SS/', 'D:/ex05.mp4', mask='*.jpg',framerate=5,resize_W=1920//2,resize_H=1080//2)
    #tools_animation.folder_to_animated_gif_imageio('D:/pp2/', 'D:/YSS.gif', mask='*.png,*.jpg', framerate=36,resize_W=915//4,resize_H=545//4,do_reverce=True)





