# ----------------------------------------------------------------------------------------------------------------------
import tools_video
import tools_animation
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    #tools_video.grab_youtube_video('https://www.youtube.com/watch?v=ibIKS84ETpM','./data/output/res.mp4')
    #tools_video.grab_youtube_video('https://www.youtube.com/watch?v=-XPuWK-TBMo','./data/output/','res.mp4')

    #tools_video.extract_frames('D://foam.mp4','D://1/')
    tools_animation.folder_to_animated_gif_imageio('D://2/', 'D://fpam.gif', mask='*.jpg', framerate=10,resize_H=720//4, resize_W=1280//4,do_reverce=False)
    #tools_animation.folder_to_animated_gif_ffmpeg('D://2/', 'D://', 'foam.gif', mask='.jpg', framerate=10,resize_H=720//2, resize_W=1280//2)

