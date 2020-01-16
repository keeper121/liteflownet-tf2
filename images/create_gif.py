import cv2
import imageio
images = []

filenames = [
    'flow_official_caffe.png',
    'flow_pytorch.png',
    'flow_tf2.png'
]

names = [
    'Official Caffe',
    'Pytorch',
    'This TF2'
]

for i, filename in enumerate(filenames):
    im = imageio.imread(filename)
    cv2.putText(im, names[i], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
    for i in xrange(5):
        images.append(im)

imageio.mimsave('./compare.gif', images)