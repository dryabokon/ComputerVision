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
if __name__ == '__main__':
    #cap_01()
    #cap_02()
    #cap_03()
    #cap_04()
    #cap_05()

    #filename_in = 'D:/Projects/VFS/Cashiers 1_IP169_port/02.07.2019 14_59_59 (UTC+03_00).mkv'
    #folder_out = 'D:/Projects/VFS/output/'



    #tools_video.extract_frames('D:/dove.mp4','D:/2/',prefix='')
    #tools_animation.crop_images_in_folder('D:/1/','D:/2/',0, 600, 700, 1250)

    tools_animation.folder_to_animated_gif_imageio('D:/2/', 'D:/dove.gif', mask='*.jpg', framerate=30,resize_W=1920//8,resize_H=1080//8)
    #tools_animation.folder_to_video('D:/Projects/Telefonica/flow3_croped/all/','D:/TF_ani4.mp4',mask='*.jpg',resize_W=2154//4,resize_H=1200//4)