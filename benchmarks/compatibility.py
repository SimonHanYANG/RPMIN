from pathlib import Path


ROOT = Path(__file__).resolve().parent


def replace(path, edits):
    path = ROOT / path
    source = path.read_text()
    for old, new in edits:
        if old in source:
            source = source.replace(old, new)
        elif new and new not in source:
            raise ValueError(f'Unexpected upstream source in {path}: {old}')
    path.write_text(source)


replace('AESOP-SR/AESOP/basicsr/utils/registry.py', [('import pyiqa\n', '')])
replace('AESOP-SR/AESOP/basicsr/data/degradations.py', [
    ('from torchvision.transforms.functional_tensor import rgb_to_grayscale',
     'from torchvision.transforms.functional import rgb_to_grayscale'),
])
replace('AESOP-SR/AESOP/basicsr/archs/rrdbnet_arch.py', [
    ("        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))\n"
     "        # if self.scale == 8:\n",
     "        if getattr(self, 'upsample_second', True):\n"
     "            feat = F.interpolate(feat, scale_factor=2, mode='nearest')\n"
     "        feat = self.lrelu(self.conv_up2(feat))\n"
     "        # if self.scale == 8:\n"),
])
replace('SeD/models/model_rrdb.py', [
    ("        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))",
     "        if getattr(self, 'upsample_second', True):\n"
     "            fea = F.interpolate(fea, scale_factor=2, mode='nearest')\n"
     "        fea = self.lrelu(self.upconv2(fea))"),
])
replace('NeXtSRGAN/modules/nextsrgan.py', [
    ('import tensorflow_addons as tfa\n', ''),
    ('import tensorflow as tf\nfrom tensorflow.keras import Model\nfrom tensorflow.keras.layers import Dense, Input, Conv2D',
     'import keras\nfrom keras import Model\nfrom keras.layers import Dense, Input, Conv2D'),
    ('        if training is None:\n'
     '            training = tf.constant(False)\n'
     '        training = tf.logical_and(training, self.trainable)\n'
     '        return super().call(x, training)',
     '        return super().call(x, training=training)'),
    ('        gelu_f = functools.partial(tfa.layers.GELU)\n', ''),
    ('    gelu_f = functools.partial(tfa.layers.GELU)\n', ''),
    ('gelu_f()', 'keras.activations.gelu'),
    ('tf.keras.', 'keras.'),
    ('tf.concat(', 'keras.ops.concatenate('),
    ("    size_fea_h = tf.shape(fea)[1] if size is None else size\n"
     "    size_fea_w = tf.shape(fea)[2] if size is None else size\n"
     "    fea_resize = tf.image.resize(fea, [size_fea_h * 2, size_fea_w * 2],\n"
     "                                 method='nearest', name='upsample_nn_1')",
     "    fea_resize = keras.layers.UpSampling2D(interpolation='nearest', name='upsample_nn_1')(fea)"),
    ("    fea_resize = tf.image.resize(fea, [size_fea_h * 4, size_fea_w * 4],\n"
     "                                 method='nearest', name='upsample_nn_2')",
     "    fea_resize = keras.layers.UpSampling2D(interpolation='nearest', name='upsample_nn_2')(fea)"),
    ("def RRDB_Model(size, channels, cfg_net, gc=32, wd=0., name='RRDB_model'):",
     "def RRDB_Model(size, channels, cfg_net, gc=32, wd=0., name='RRDB_model', scale=4):"),
    ("    fea_resize = keras.layers.UpSampling2D(interpolation='nearest', name='upsample_nn_2')(fea)\n"
     "    fea = conv_f(filters=nf, activation=keras.activations.gelu, name='upconv_2')(fea_resize)",
     "    if scale == 4:\n"
     "        fea_resize = keras.layers.UpSampling2D(interpolation='nearest', name='upsample_nn_2')(fea)\n"
     "    else:\n"
     "        fea_resize = fea\n"
     "    fea = conv_f(filters=nf, activation=keras.activations.gelu, name='upconv_2')(fea_resize)"),
])
replace('NeXtSRGAN/modules/losses.py', [
    ('import tensorflow as tf\nfrom tensorflow.keras.applications.vgg19 import preprocess_input, VGG19',
     'import keras\nfrom keras import ops\nfrom keras.applications.vgg19 import preprocess_input, VGG19'),
    ('tf.keras.', 'keras.'),
    ('tf.sigmoid', 'ops.sigmoid'),
    ('tf.reduce_mean', 'ops.mean'),
    ('tf.ones_like', 'ops.ones_like'),
    ('tf.zeros_like', 'ops.zeros_like'),
    ('    @tf.function\n', ''),
    ('    fea_extrator = keras.Model(vgg.input, vgg.layers[pick_layer].output)\n\n',
     '    fea_extrator = keras.Model(vgg.input, vgg.layers[pick_layer].output)\n'
     '    fea_extrator.trainable = False\n\n'),
])
