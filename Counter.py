import os

source = "Train_AFEW\AlignedFaces_LBPTOP_Points\AlignedFaces_LBPTOP_Points"

def countFrames(source):
    count = 0
    count_max = 0
    name = ''

    for label in os.listdir(source):
        for video in os.listdir(source+'/'+label):
            for frame in os.listdir(source+'/'+label+'/'+video):
                count += 1
    
            if count > count_max:
                count_max = count
                name = label +'/'+ video
            count = 0
    
    print(count_max)
    print(name)


def countVideos(source):
    count = 0
    count_max = 0
    name = ''

    for label in os.listdir(source):
        for video in os.listdir(source + '/' + label):
            count += 1

            if count > count_max:
                count_max = count
                name = label + '/' + video

    print(count_max)
    print(name)

countVideos(source)
    