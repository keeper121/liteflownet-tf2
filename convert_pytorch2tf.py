import numpy as np
import tensorflow.compat.v1 as tf
import torch
import argparse

from model import LiteFlowNet

tf.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--input_model', default='network-default.pytorch')
parser.add_argument('--output_model', default='model')

args = parser.parse_args()


def ToTensor(sample, transfrm=lambda x: np.transpose(x, [0, 3, 1, 2])):
    return torch.from_numpy(transfrm(sample))


weights_mapping = {
    'moduleFeatures.moduleOne.0.weight': 'flownet/feature_extractor/sequential/conv2d/kernel',
    'moduleFeatures.moduleOne.0.bias': 'flownet/feature_extractor/sequential/conv2d/bias',
    'moduleFeatures.moduleTwo.0.weight': 'flownet/feature_extractor/sequential_1/conv2d_1/kernel',
    'moduleFeatures.moduleTwo.0.bias': 'flownet/feature_extractor/sequential_1/conv2d_1/bias',
    'moduleFeatures.moduleTwo.2.weight': 'flownet/feature_extractor/sequential_1/conv2d_2/kernel',
    'moduleFeatures.moduleTwo.2.bias': 'flownet/feature_extractor/sequential_1/conv2d_2/bias',
    'moduleFeatures.moduleTwo.4.weight': 'flownet/feature_extractor/sequential_1/conv2d_3/kernel',
    'moduleFeatures.moduleTwo.4.bias': 'flownet/feature_extractor/sequential_1/conv2d_3/bias',
    'moduleFeatures.moduleThr.0.weight': 'flownet/feature_extractor/sequential_2/conv2d_4/kernel',
    'moduleFeatures.moduleThr.0.bias': 'flownet/feature_extractor/sequential_2/conv2d_4/bias',
    'moduleFeatures.moduleThr.2.weight': 'flownet/feature_extractor/sequential_2/conv2d_5/kernel',
    'moduleFeatures.moduleThr.2.bias': 'flownet/feature_extractor/sequential_2/conv2d_5/bias',
    'moduleFeatures.moduleFou.0.weight': 'flownet/feature_extractor/sequential_3/conv2d_6/kernel',
    'moduleFeatures.moduleFou.0.bias': 'flownet/feature_extractor/sequential_3/conv2d_6/bias',
    'moduleFeatures.moduleFou.2.weight': 'flownet/feature_extractor/sequential_3/conv2d_7/kernel',
    'moduleFeatures.moduleFou.2.bias': 'flownet/feature_extractor/sequential_3/conv2d_7/bias',
    'moduleFeatures.moduleFiv.0.weight': 'flownet/feature_extractor/sequential_4/conv2d_8/kernel',
    'moduleFeatures.moduleFiv.0.bias': 'flownet/feature_extractor/sequential_4/conv2d_8/bias',
    'moduleFeatures.moduleSix.0.weight': 'flownet/feature_extractor/sequential_5/conv2d_9/kernel',
    'moduleFeatures.moduleSix.0.bias': 'flownet/feature_extractor/sequential_5/conv2d_9/bias',
}

pytorch_model_path = args.input_model
pytorch_state_dict = torch.load(pytorch_model_path)

model = LiteFlowNet()
frame1 = tf.placeholder(tf.float32, shape=[None, None, None, 3])
frame2 = tf.placeholder(tf.float32, shape=[None, None, None, 3])
out = model(frame1, frame2)

c = 10
m_weights = {}
for j in [-1, -2, -3, -4, -5]:
    i = abs(j)
    lvls = [2, 3, 4, 5, 6][j]

    if lvls < 6:
        m_weights['moduleMatching.%i.moduleUpflow.weight' % (5 - i)] = 'flownet/matching_%i/moduleUpflow/filter_w' % i
        m_weights['moduleMatching.%i.moduleUpcorr.weight' % (5 - i)] = 'flownet/matching_%i/moduleUpcorr/filter_w' % i

    if lvls == 2:
        m_weights['moduleMatching.%i.moduleFeat.0.weight' % (5 - i)] = 'flownet/matching_%i/module_feat/conv2d_%i/kernel' % (i, c)
        m_weights['moduleMatching.%i.moduleFeat.0.bias' % (5 - i)] = 'flownet/matching_%i/module_feat/conv2d_%i/bias' % (i, c)
        c += 1

    m_weights['moduleMatching.%i.moduleMain.0.weight' % (5 - i)] = 'flownet/matching_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleMatching.%i.moduleMain.0.bias' % (5 - i)] = 'flownet/matching_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['moduleMatching.%i.moduleMain.2.weight' % (5 - i)] = 'flownet/matching_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleMatching.%i.moduleMain.2.bias' % (5 - i)] = 'flownet/matching_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['moduleMatching.%i.moduleMain.4.weight' % (5 - i)] = 'flownet/matching_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleMatching.%i.moduleMain.4.bias' % (5 - i)] = 'flownet/matching_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['moduleMatching.%i.moduleMain.6.weight' % (5 - i)] = 'flownet/matching_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleMatching.%i.moduleMain.6.bias' % (5 - i)] = 'flownet/matching_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1

    if lvls == 2:
        m_weights['moduleSubpixel.%i.moduleFeat.0.weight' % (5 - i)] = 'flownet/subpixel_%i/module_feat/conv2d_%i/kernel' % (i, c)
        m_weights['moduleSubpixel.%i.moduleFeat.0.bias' % (5 - i)] = 'flownet/subpixel_%i/module_feat/conv2d_%i/bias' % (i, c)
        c += 1

    m_weights['moduleSubpixel.%i.moduleMain.0.weight' % (5 - i)] = 'flownet/subpixel_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleSubpixel.%i.moduleMain.0.bias' % (5 - i)] = 'flownet/subpixel_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['moduleSubpixel.%i.moduleMain.2.weight' % (5 - i)] = 'flownet/subpixel_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleSubpixel.%i.moduleMain.2.bias' % (5 - i)] = 'flownet/subpixel_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['moduleSubpixel.%i.moduleMain.4.weight' % (5 - i)] = 'flownet/subpixel_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleSubpixel.%i.moduleMain.4.bias' % (5 - i)] = 'flownet/subpixel_%i/module_main/conv2d_%i/bias' % (
        i, c)
    c += 1
    m_weights['moduleSubpixel.%i.moduleMain.6.weight' % (5 - i)] = 'flownet/subpixel_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleSubpixel.%i.moduleMain.6.bias' % (5 - i)] = 'flownet/subpixel_%i/module_main/conv2d_%i/bias' % (
        i, c)
    c += 1

    if lvls < 5:
        m_weights['moduleRegularization.%i.moduleFeat.0.weight' % (5 - i)] = 'flownet/regularization_%i/module_feat/conv2d_%i/kernel' % (i, c)
        m_weights['moduleRegularization.%i.moduleFeat.0.bias' % (5 - i)] = 'flownet/regularization_%i/module_feat/conv2d_%i/bias' % (i, c)
        c += 1

    m_weights['moduleRegularization.%i.moduleMain.0.weight' % (5 - i)] = 'flownet/regularization_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleRegularization.%i.moduleMain.0.bias' % (5 - i)] = 'flownet/regularization_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1

    m_weights['moduleRegularization.%i.moduleMain.2.weight' % (5 - i)] = 'flownet/regularization_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleRegularization.%i.moduleMain.2.bias' % (5 - i)] = 'flownet/regularization_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['moduleRegularization.%i.moduleMain.4.weight' % (5 - i)] = 'flownet/regularization_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleRegularization.%i.moduleMain.4.bias' % (5 - i)] = 'flownet/regularization_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['moduleRegularization.%i.moduleMain.6.weight' % (5 - i)] = 'flownet/regularization_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleRegularization.%i.moduleMain.6.bias' % (5 - i)] = 'flownet/regularization_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['moduleRegularization.%i.moduleMain.8.weight' % (5 - i)] = 'flownet/regularization_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleRegularization.%i.moduleMain.8.bias' % (5 - i)] = 'flownet/regularization_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['moduleRegularization.%i.moduleMain.10.weight' % (5 - i)] = 'flownet/regularization_%i/module_main/conv2d_%i/kernel' % (i, c)
    m_weights['moduleRegularization.%i.moduleMain.10.bias' % (5 - i)] = 'flownet/regularization_%i/module_main/conv2d_%i/bias' % (i, c)
    c += 1
    if lvls < 5:
        m_weights['moduleRegularization.%i.moduleDist.0.weight' % (5 - i)] = 'flownet/regularization_%i/module_dist/conv2d_%i/kernel' % (i, c)
        m_weights['moduleRegularization.%i.moduleDist.0.bias' % (5 - i)] = 'flownet/regularization_%i/module_dist/conv2d_%i/bias' % (i, c)
        c += 1
        m_weights['moduleRegularization.%i.moduleDist.1.weight' % (5 - i)] = 'flownet/regularization_%i/module_dist/conv2d_%i/kernel' % (i, c)
        m_weights['moduleRegularization.%i.moduleDist.1.bias' % (5 - i)] = 'flownet/regularization_%i/module_dist/conv2d_%i/bias' % (i, c)
        c += 1
    else:
        m_weights['moduleRegularization.%i.moduleDist.0.weight' % (5 - i)] = 'flownet/regularization_%i/module_dist/conv2d_%i/kernel' % (i, c)
        m_weights['moduleRegularization.%i.moduleDist.0.bias' % (5 - i)] = 'flownet/regularization_%i/module_dist/conv2d_%i/bias' % (i, c)
        c += 1

    m_weights['moduleRegularization.%i.moduleScaleX.weight' % (5 - i)] = 'flownet/regularization_%i/moduleScaleX/conv2d_%i/kernel' % (i, c)
    m_weights['moduleRegularization.%i.moduleScaleX.bias' % (5 - i)] = 'flownet/regularization_%i/moduleScaleX/conv2d_%i/bias' % (i, c)
    c += 1
    m_weights['moduleRegularization.%i.moduleScaleY.weight' % (5 - i)] = 'flownet/regularization_%i/moduleScaleY/conv2d_%i/kernel' % (i, c)
    m_weights['moduleRegularization.%i.moduleScaleY.bias' % (5 - i)] = 'flownet/regularization_%i/moduleScaleY/conv2d_%i/bias' % (i, c)
    c += 1

weights_mapping.update(m_weights)

for v in sorted(weights_mapping.values()):
    print v

sess = tf.Session()

tfvarsg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='flownet')
tfvars = {v.name[:-2]: v for v in tfvarsg}

for v in sorted(tfvarsg, key=lambda x: x.name):
    print v.name

for state in pytorch_state_dict:
    pytorch_data = pytorch_state_dict[state].cpu().detach().numpy()
    if len(pytorch_data.shape) > 3:
        shapes = pytorch_data.shape
        pytorch_data = np.transpose(pytorch_data, [2, 3, 1, 0])

    if state in weights_mapping:
        print ("Assing: " + state + "  ====>  " + weights_mapping[state])
        sess.run(tf.assign(tfvars[weights_mapping[state]], pytorch_data))

# save model
saver = tf.train.Saver(tfvars)
saver.save(sess, args.output_model)
