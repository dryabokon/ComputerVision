# ----------------------------------------------------------------------------------------------------------------------
import tools_video
import tools_animation
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
def cap_01():
    prefix = '01_'
    filename_in = 'D:/Projects/Nice/streets/London01.mp4'
    folder_out = 'D:/Projects/Nice/streets/'+prefix+'/'
    tools_video.extract_frames(filename_in, folder_out, prefix=prefix,start_time_sec=60 * 35 + 9, end_time_sec=60 * 35 + 20)
    return
# ----------------------------------------------------------------------------------------------------------------------
def cap_02():
    prefix = '02_'
    filename_in = 'D:/Projects/Nice/streets/London01.mp4'
    folder_out = 'D:/Projects/Nice/streets/'+prefix+'/'
    tools_video.extract_frames(filename_in, folder_out,prefix=prefix, start_time_sec=60 * 43 + 25, end_time_sec=60 * 43 + 40)
    return
# ----------------------------------------------------------------------------------------------------------------------
def cap_03():
    prefix = '03_'
    filename_in = 'D:/Projects/Nice/streets/London02.mp4'
    folder_out = 'D:/Projects/Nice/streets/'+prefix+'/'
    tools_video.extract_frames(filename_in, folder_out,prefix=prefix, start_time_sec=60 * 16 + 10, end_time_sec=60 * 16 + 26)
    return
# ----------------------------------------------------------------------------------------------------------------------
def cap_04():
    prefix = '04_'
    filename_in = 'D:/Projects/Nice/streets/London02.mp4'
    folder_out = 'D:/Projects/Nice/streets/'+prefix+'/'
    tools_video.extract_frames(filename_in, folder_out,prefix=prefix, start_time_sec=60*60*1 + 60 * 12 + 20, end_time_sec=60*60*1 + 60 * 12 + 30)
    return
# ----------------------------------------------------------------------------------------------------------------------
def cap_05():
    prefix = '05_'
    filename_in = 'D:/Projects/Nice/streets/London02.mp4'
    folder_out = 'D:/Projects/Nice/streets/'+prefix+'/'
    tools_video.extract_frames(filename_in, folder_out,prefix=prefix, start_time_sec= 60 * 24 + 10, end_time_sec= 60 * 24 + 25)
    return
# ----------------------------------------------------------------------------------------------------------------------
def cap_06():
    URL = 'https://www.youtube.com/watch?v=ulpIL6KhD50'
    out_path = 'D:/'
    out_filename = 'res'
    tools_video.grab_youtube_video(URL, out_path, out_filename)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #cap_01()
    #cap_02()
    #cap_03()
    #cap_04()
    #cap_05()
    #cap_06()

    #filename_in = 'D:/Projects/VFS/Cashiers 1_IP169_port/02.07.2019 14_59_59 (UTC+03_00).mkv'
    #folder_out = 'D:/Projects/VFS/output/'

    tools_video.extract_frames('D:/res.mp4','D:/3/',prefix='',start_time_sec=70)
    #tools_animation.crop_images_in_folder(path_input,path_output,115, 123, 832, 1400)

    #tools_animation.folder_to_animated_gif_imageio('D:/1/', 'D:/ani.gif', mask='*.jpg', framerate=25,resize_W=210,resize_H=120)

