import os

source_val = "Train_AFEW\AlignedFaces_LBPTOP_Points_Val\AlignedFaces_LBPTOP_Points_Val"
count = 0
count_max = 0
name = ''

for label in os.listdir(source_val):
    for video in os.listdir(source_val+'/'+label):
        for frame in os.listdir(source_val+'/'+label+'/'+video):
            count += 1

        if count > count_max:
            count_max = count
            name = label +'/'+ video
        count = 0

print(count_max)
print(name)
