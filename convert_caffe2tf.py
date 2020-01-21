import numpy as np
import tensorflow.compat.v1 as tf
import argparse

from model import LiteFlowNet
from google.protobuf import text_format

from caffe.proto.caffe_pb2 import NetParameter
#from caffe_pb2 import NetParameter
tf.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--input_model', default='liteflownet.caffemodel')
parser.add_argument('--output_model', default='model')

args = parser.parse_args()

caffe_weights = {}

with open(args.input_model, 'r') as fp:
    net = NetParameter()
    net.ParseFromString(fp.read())
    #text_format.Merge(fp.read(), net)


for idx, layer in enumerate(net.layer):
    #print layer.type
    if layer.type == 'Convolution' or layer.type == 'Deconvolution':
        conv_param = layer.convolution_param
        weights = net.layer[idx].blobs[0].data
        #print net.layer[idx].blobs[0].shape.dim
        weights = np.reshape(weights, net.layer[idx].blobs[0].shape.dim)
        caffe_weights[layer.name + '/weight'] = np.array(weights)

        if len(net.layer[idx].blobs) > 1:
            bias = net.layer[idx].blobs[1].data
            caffe_weights[layer.name + '/bias'] = np.array(bias)

        print layer.name, net.layer[idx].blobs[0].shape.dim


for i in [2]:
    caffe_weights['F0_F1_64_L%i/weight_1' % i] = caffe_weights['F0_F1_64_L%i/weight' % i]
    caffe_weights['F0_F1_64_L%i/bias_1' % i] = caffe_weights['F0_F1_64_L%i/bias' % i]

weights_mapping = {
    'conv1/weight': 'flownet/feature_extractor/sequential/conv2d/kernel',
    'conv1/bias': 'flownet/feature_extractor/sequential/conv2d/bias',
    'conv2_1/weight': 'flownet/feature_extractor/sequential_1/conv2d_1/kernel',
    'conv2_1/bias': 'flownet/feature_extractor/sequential_1/conv2d_1/bias',
    'conv2_2/weight': 'flownet/feature_extractor/sequential_1/conv2d_2/kernel',
    'conv2_2/bias': 'flownet/feature_extractor/sequential_1/conv2d_2/bias',
    'conv2_3/weight': 'flownet/feature_extractor/sequential_1/conv2d_3/kernel',
    'conv2_3/bias': 'flownet/feature_extractor/sequential_1/conv2d_3/bias',
    'conv3_1/weight': 'flownet/feature_extractor/sequential_2/conv2d_4/kernel',
    'conv3_1/bias': 'flownet/feature_extractor/sequential_2/conv2d_4/bias',
    'conv3_2/weight': 'flownet/feature_extractor/sequential_2/conv2d_5/kernel',
    'conv3_2/bias': 'flownet/feature_extractor/sequential_2/conv2d_5/bias',
    'conv4_1/weight': 'flownet/feature_extractor/sequential_3/conv2d_6/kernel',
    'conv4_1/bias': 'flownet/feature_extractor/sequential_3/conv2d_6/bias',
    'conv4_2/weight': 'flownet/feature_extractor/sequential_3/conv2d_7/kernel',
    'conv4_2/bias': 'flownet/feature_extractor/sequential_3/conv2d_7/bias',
    'conv5/weight': 'flownet/feature_extractor/sequential_4/conv2d_8/kernel',
    'conv5/bias': 'flownet/feature_extractor/sequential_4/conv2d_8/bias',
    'conv6/weight': 'flownet/feature_extractor/sequential_5/conv2d_9/kernel',
    'conv6/bias': 'flownet/feature_extractor/sequential_5/conv2d_9/bias',
}


model = LiteFlowNet()
mono = tf.placeholder(tf.float32, shape=[None, None, None, 3])
color = tf.placeholder(tf.float32, shape=[None, None, None, 3])
out = model(mono, color)

c = 10
m_weights = {}
for j in [-1, -2, -3, -4, -5]:
    i = abs(j)
    lvl = [2, 3, 4, 5, 6][j]

    if lvl < 6:
        m_weights['scaled_flow_R_L%ito%i/weight' % (lvl+1, lvl)] = 'flownet/matching_%i/moduleUpflow/filter_w' % i
        m_weights['corr_L%i/weight' % lvl] = 'flownet/matching_%i/moduleUpcorr/filter_w' % i

    if lvl == 2:
        m_weights['F0_F1_64_L%i/weight' % lvl] = 'flownet/matching_%i/module_feat/conv2d_%i/kernel' % (i, c)
        m_weights['F0_F1_64_L%i/bias' % lvl] = 'flownet/matching_%i/module_feat/conv2d_%i/bias' % (i, c)
        c += 1

    m_weights['conv1_D1_L%i/weight' % lvl] = 'flownet/matching_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['conv1_D1_L%i/bias' % lvl] = 'flownet/matching_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['conv2_D1_L%i/weight' % lvl] = 'flownet/matching_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['conv2_D1_L%i/bias' % lvl] = 'flownet/matching_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['conv3_D1_L%i/weight' % lvl] = 'flownet/matching_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['conv3_D1_L%i/bias' % lvl] = 'flownet/matching_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    if lvl == 6:
        m_weights['scaled_flow_D1_L%i/weight' % lvl] = 'flownet/matching_%i/module_main/conv2d_%i/kernel' % (i, c)
        m_weights['scaled_flow_D1_L%i/bias' % lvl] = 'flownet/matching_%i/module_main/conv2d_%i/bias' % (i, c)
        c += 1
    else:
        m_weights['scaled_flow_D1_res_L%i/weight' % lvl] = 'flownet/matching_%i/module_main/conv2d_%i/kernel' % (i, c)
        m_weights['scaled_flow_D1_res_L%i/bias' % lvl] = 'flownet/matching_%i/module_main/conv2d_%i/bias' % (i, c)
        c += 1

    if lvl == 2:
        m_weights['F0_F1_64_L%i/weight_1' % lvl] = 'flownet/subpixel_%i/module_feat/conv2d_%i/kernel' % (i, c)
        m_weights['F0_F1_64_L%i/bias_1' % lvl] = 'flownet/subpixel_%i/module_feat/conv2d_%i/bias' % (i, c)
        c += 1

    m_weights['conv1_D2_L%i/weight' % lvl] = 'flownet/subpixel_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['conv1_D2_L%i/bias' % lvl] = 'flownet/subpixel_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['conv2_D2_L%i/weight' % lvl] = 'flownet/subpixel_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['conv2_D2_L%i/bias' % lvl] = 'flownet/subpixel_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['conv3_D2_L%i/weight' % lvl] = 'flownet/subpixel_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['conv3_D2_L%i/bias' % lvl] = 'flownet/subpixel_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['scaled_flow_D2_res_L%i/weight' % lvl] = 'flownet/subpixel_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['scaled_flow_D2_res_L%i/bias' % lvl] = 'flownet/subpixel_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1

    if lvl < 5:
        m_weights['F0_128_L%i/weight' % lvl] = 'flownet/regularization_%i/module_feat/conv2d_%i/kernel' % (i, c)
        m_weights['F0_128_L%i/bias' % lvl] = 'flownet/regularization_%i/module_feat/conv2d_%i/bias' % (i, c)
        c += 1

    m_weights['conv1_R_L%i/weight' % lvl] = 'flownet/regularization_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['conv1_R_L%i/bias' % lvl] = 'flownet/regularization_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1

    m_weights['conv2_R_L%i/weight' % lvl] = 'flownet/regularization_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['conv2_R_L%i/bias' % lvl] = 'flownet/regularization_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['conv3_R_L%i/weight' % lvl] = 'flownet/regularization_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['conv3_R_L%i/bias' % lvl] = 'flownet/regularization_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['conv4_R_L%i/weight' % lvl] = 'flownet/regularization_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['conv4_R_L%i/bias' % lvl] = 'flownet/regularization_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['conv5_R_L%i/weight' % lvl] = 'flownet/regularization_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['conv5_R_L%i/bias' % lvl] = 'flownet/regularization_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['conv6_R_L%i/weight' % lvl] = 'flownet/regularization_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['conv6_R_L%i/bias' % lvl] = 'flownet/regularization_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    if lvl < 5:
        m_weights['distH_R_L%i/weight' % lvl] = 'flownet/regularization_%i/module_dist/conv2d_%i/kernel' % (i, c)
        m_weights['distH_R_L%i/bias' % lvl] = 'flownet/regularization_%i/module_dist/conv2d_%i/bias' % (i, c)
        c += 1
        m_weights['distW_R_L%i/weight' % lvl] = 'flownet/regularization_%i/module_dist/conv2d_%i/kernel' % (i, c)
        m_weights['distW_R_L%i/bias' % lvl] = 'flownet/regularization_%i/module_dist/conv2d_%i/bias' % (i, c)
        c += 1
    else:
        m_weights['dist_R_L%i/weight' % lvl] = 'flownet/regularization_%i/module_dist/conv2d_%i/kernel' % (i, c)
        m_weights['dist_R_L%i/bias' % lvl] = 'flownet/regularization_%i/module_dist/conv2d_%i/bias' % (i, c)
        c += 1

    m_weights['scaled_flow_R_L%i_x/weight' % lvl] = 'flownet/regularization_%i/moduleScaleX/conv2d_%i/kernel' % (i, c)
    m_weights['scaled_flow_R_L%i_x/bias' % lvl] = 'flownet/regularization_%i/moduleScaleX/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['scaled_flow_R_L%i_y/weight' % lvl] = 'flownet/regularization_%i/moduleScaleY/conv2d_%i/kernel' % (i, c)
    m_weights['scaled_flow_R_L%i_y/bias' % lvl] = 'flownet/regularization_%i/moduleScaleY/conv2d_%i/bias' % (i, c)
    c += 1

weights_mapping.update(m_weights)

for v in sorted(weights_mapping.values()):
    print v

sess = tf.Session()

tfvarsg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='flownet')
tfvars = {v.name[:-2]: v for v in tfvarsg}

for v in sorted(tfvarsg, key=lambda x: x.name):
    print v.name

for state in sorted(caffe_weights):
    data = caffe_weights[state]
    if len(data.shape) > 3:
        shapes = data.shape
        data = np.transpose(data, [2, 3, 1, 0])

    if state in weights_mapping:
        print ("Assign: " + state + "  ====>  " + weights_mapping[state])
        sess.run(tf.assign(tfvars[weights_mapping[state]], data))

# save model
saver = tf.train.Saver(tfvars)
saver.save(sess, './model')
