import math

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

from model import LiteFlowNet

from draw_flow import *


def pad_image(image):
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1

    nh = int(math.ceil(h / 32.) * 32)
    nw = int(math.ceil(w / 32.) * 32)

    pad_i = np.zeros([nh, nw, c])
    pad_i[:h, :w] = image
    return pad_i

tf.disable_eager_execution()

sess = tf.Session()
model = LiteFlowNet()
tens1 = tf.placeholder(tf.float32, shape=[None, None, None, 3])
tens2 = tf.placeholder(tf.float32, shape=[None, None, None, 3])
out = model(tens1, tens2)

saver = tf.train.Saver()
saver.restore(sess, './model')

inp1 = cv2.imread('images/first.png')
inp2 = cv2.imread('images/second.png')

h, w = inp1.shape[:2]
inp1 = np.float32(np.expand_dims(pad_image(inp1), 0)) / 255.0
inp2 = np.float32(np.expand_dims(pad_image(inp2), 0)) / 255.0

flow = sess.run(out, feed_dict={tens1: inp1, tens2: inp2})[0, :h, :w, :]

# visualise as image
flow_color = flow_to_color(flow, convert_to_bgr=True)
cv2.imwrite('./flow.png', flow_color)

# save optical flow to file
objectOutput = open('./out.flo', 'wb')
np.array([80, 73, 69, 72], np.uint8).tofile(objectOutput)
np.array([flow.shape[1], flow.shape[0]], np.int32).tofile(objectOutput)
np.array(np.transpose(flow).transpose([1, 2, 0]), np.float32).tofile(objectOutput)
objectOutput.close()

