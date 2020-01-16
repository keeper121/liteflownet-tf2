import math
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
from model import LiteFlowNet
import argparse

from draw_flow import *
tf.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--img1', default='images/first.png')
parser.add_argument('--img2', default='images/second.png')
parser.add_argument('--model', default='./model')
parser.add_argument('--flow', default='out.flow')
parser.add_argument('--display_flow', default=True)

args = parser.parse_args()

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



sess = tf.Session()
model = LiteFlowNet()
tens1 = tf.placeholder(tf.float32, shape=[None, None, None, 3])
tens2 = tf.placeholder(tf.float32, shape=[None, None, None, 3])
out = model(tens1, tens2)

saver = tf.train.Saver()
saver.restore(sess, args.model)

inp1 = Image.open(args.img1)
inp2 = Image.open(args.img2)

w, h = inp1.size[:2]
inp1 = np.float32(np.expand_dims(pad_image(np.asarray(inp1)[..., ::-1]), 0)) / 255.0
inp2 = np.float32(np.expand_dims(pad_image(np.asarray(inp2)[..., ::-1]), 0)) / 255.0

# input in bgr format
flow = sess.run(out, feed_dict={tens1: inp1, tens2: inp2})[0, :h, :w, :]


if args.display_flow:
    # visualise flow with color model as image
    flow_color = flow_to_color(flow, convert_to_bgr=False)
    flow_image = Image.fromarray(flow_color)
    flow_image.show()

# save optical flow to file
objectOutput = open(args.flow, 'wb')
np.array([80, 73, 69, 72], np.uint8).tofile(objectOutput)
np.array([flow.shape[1], flow.shape[0]], np.int32).tofile(objectOutput)
np.array(np.transpose(flow).transpose([1, 2, 0]), np.float32).tofile(objectOutput)
objectOutput.close()
