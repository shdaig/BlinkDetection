import moviepy.editor as mp

clip = mp.VideoFileClip("../../data/cam1.20230911T145010337317.avi")
clip = clip.subclip(0, 60)

clip.write_videofile("../data/cam1_converted_1m.mp4")
