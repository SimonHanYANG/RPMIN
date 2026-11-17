import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('KERAS_HOME', str(ROOT / 'keras-cache'))
os.environ['KERAS_BACKEND'] = 'torch'

import keras
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(ROOT / 'NeXtSRGAN'))

from modules.losses import ContentLoss, DiscriminatorLoss, GeneratorLoss, PixelLoss
from modules.nextsrgan import DiscriminatorVGG128, RRDB_Model


def read_training_pairs(manifest, scale):
    with manifest.open(newline='', encoding='utf-8-sig') as stream:
        rows = [row for row in csv.DictReader(stream) if row['split'] == 'train']
    pairs = []
    for row in rows:
        with Image.open(manifest.parent / row['lr']) as lr, Image.open(manifest.parent / row['hr']) as hr:
            if (scale * lr.width, scale * lr.height) != hr.size:
                raise ValueError(f"Invalid paired image sizes for {row['id']}")
            pairs.append((np.asarray(lr.convert('RGB')), np.asarray(hr.convert('RGB'))))
    return pairs


def sample_batch(pairs, rng, batch, scale):
    low, high = [], []
    for index in rng.integers(len(pairs), size=batch):
        lr, hr = pairs[index]
        y = rng.integers(lr.shape[0] - 31)
        x = rng.integers(lr.shape[1] - 31)
        lr = lr[y:y + 32, x:x + 32]
        hr = hr[scale * y:scale * (y + 32), scale * x:scale * (x + 32)]
        if rng.integers(2):
            lr, hr = lr[::-1], hr[::-1]
        if rng.integers(2):
            lr, hr = lr[:, ::-1], hr[:, ::-1]
        if rng.integers(2):
            lr, hr = lr.transpose(1, 0, 2), hr.transpose(1, 0, 2)
        low.append(lr)
        high.append(hr)
    return [torch.as_tensor(np.stack(images), device='cuda').float().div_(255) for images in (low, high)]


def load_rrdb_initialization(generator, checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    names = {
        'conv_first': 'conv_first', 'conv_trunk': 'trunk_conv', 'upconv_1': 'upconv1',
        'upconv_2': 'upconv2', 'conv_hr': 'HRconv', 'conv_last': 'conv_last',
    }
    layers = [(generator.get_layer(target), source) for target, source in names.items()]
    for index, block in enumerate(generator.get_layer('RRDB_trunk').layers):
        for number in range(1, 4):
            dense = getattr(block, f'rdb_{number}')
            for conv in range(1, 6):
                layers.append((getattr(dense, f'conv{conv}'), f'RRDB_trunk.{index}.RDB{number}.conv{conv}'))
    expected = {f'{source}.{part}' for _, source in layers for part in ('weight', 'bias')}
    if set(state) != expected:
        raise ValueError('Expected all original ESRGAN RRDB ×4 convolution kernels and biases')
    for layer, source in layers:
        kernel = state[f'{source}.weight'].permute(2, 3, 1, 0).numpy()
        layer.set_weights([kernel, state[f'{source}.bias'].numpy()])


def train(args):
    args.out.mkdir(parents=True, exist_ok=False)
    models = ROOT / 'keras-cache' / 'models'
    models.mkdir(parents=True, exist_ok=True)
    vgg_name = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
    if not (models / vgg_name).exists():
        (models / vgg_name).symlink_to(ROOT / 'weights' / vgg_name)

    generator = RRDB_Model(None, 3, {'nf': 64, 'nb': 23}, scale=args.scale).to('cuda')
    provenance = {}
    if args.init_rrdb:
        load_rrdb_initialization(generator, args.init_rrdb)
        with args.init_rrdb.open('rb') as stream:
            provenance['initial_checkpoint_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    discriminator = DiscriminatorVGG128(32 * args.scale, 3).to('cuda')
    pixel = PixelLoss('l1')
    feature = ContentLoss('l1')
    gen_loss = GeneratorLoss('ragan')
    dis_loss = DiscriminatorLoss('ragan')
    pre_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.9, beta_2=0.99)
    optimizer_g = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.99)
    optimizer_d = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.99)
    variables_g = generator.trainable_variables
    variables_d = discriminator.trainable_variables
    parameters_g = [variable.value for variable in variables_g]
    parameters_d = [variable.value for variable in variables_d]

    def pretrain(lr, hr):
        loss = pixel(hr, generator(lr, training=True))
        pre_optimizer.apply(torch.autograd.grad(loss, parameters_g), variables_g)
        return (loss.detach().item(),)

    def adversarial(lr, hr):
        sr = generator(lr, training=True)
        real = discriminator(hr, training=True)
        fake = discriminator(sr, training=True)
        g = 0.01 * pixel(hr, sr) + feature(hr, sr) + 0.005 * gen_loss(real, fake)
        d = dis_loss(real, fake)
        dr_g = torch.autograd.grad(g, parameters_g, retain_graph=True)
        dr_d = torch.autograd.grad(d, parameters_d)
        optimizer_g.apply(dr_g, variables_g)
        optimizer_d.apply(dr_d, variables_d)
        return g.detach().item(), d.detach().item()

    pairs = read_training_pairs(args.manifest, args.scale)
    rng = np.random.default_rng(args.seed)
    (args.out / 'config.json').write_text(json.dumps({**vars(args), **provenance, 'training_fields': len(pairs),
                                                     'keras_backend': keras.backend.backend()},
                                                    default=str, indent=2) + '\n')
    for label, count, update in [('pretrain', args.pretrain, pretrain), ('gan', args.gan, adversarial)]:
        for step in range(1, count + 1):
            scores = update(*sample_batch(pairs, rng, args.batch, args.scale))
            if not np.isfinite(scores).all():
                raise FloatingPointError(f'{label} step {step} had non-finite loss')
            if step == 1 or step % 50 == 0 or step == count:
                print(label, step, scores, flush=True)
            if (label == 'gan' and step == count // 2) or step == count:
                generator.save_weights(args.out / f'{label}-{step}.weights.h5')


@torch.no_grad()
def infer(args):
    generator = RRDB_Model(None, 3, {'nf': 64, 'nb': 23}, scale=args.scale).to('cuda')
    generator.load_weights(args.weights)
    inputs = np.load(args.inputs, mmap_mode='r')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix('.partial.npy')
    output = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32,
                                      shape=(len(inputs), inputs.shape[1] * args.scale,
                                             inputs.shape[2] * args.scale, 3))
    for start in range(0, len(inputs), args.batch):
        array = np.array(inputs[start:start + args.batch])
        batch = torch.as_tensor(array, device='cuda').float().div_(255)
        output[start:start + len(batch)] = generator(batch, training=False).cpu().numpy()
    output.flush()
    partial.replace(args.out)


@torch.no_grad()
def interpolate(args):
    if not 0 <= args.alpha <= 1:
        raise ValueError('Interpolation alpha must be in [0, 1]')
    generator = RRDB_Model(None, 3, {'nf': 64, 'nb': 23}, scale=args.scale).to('cuda')
    generator.load_weights(args.psnr)
    initial = [variable.value.detach().clone() for variable in generator.trainable_variables]
    generator.load_weights(args.gan)
    for psnr, variable in zip(initial, generator.trainable_variables, strict=True):
        variable.assign(torch.lerp(psnr, variable.value, args.alpha))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    generator.save_weights(args.out)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--manifest', type=Path, required=True)
    training.add_argument('--out', type=Path, required=True)
    training.add_argument('--pretrain', type=int, default=800)
    training.add_argument('--gan', type=int, default=200)
    training.add_argument('--batch', type=int, default=8)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--init-rrdb', type=Path)
    inference = commands.add_parser('infer')
    inference.add_argument('--weights', type=Path, required=True)
    inference.add_argument('--inputs', type=Path, required=True)
    inference.add_argument('--out', type=Path, required=True)
    inference.add_argument('--batch', type=int, default=4)
    blending = commands.add_parser('interpolate')
    blending.add_argument('--psnr', type=Path, required=True)
    blending.add_argument('--gan', type=Path, required=True)
    blending.add_argument('--alpha', type=float, required=True)
    blending.add_argument('--out', type=Path, required=True)
    for command in (training, inference, blending):
        command.add_argument('--scale', default=4, type=int, choices=[2, 4])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch GPU unavailable')
    with torch.device('cuda'):
        if args.command == 'train':
            keras.utils.set_random_seed(args.seed)
            train(args)
        elif args.command == 'infer':
            infer(args)
        else:
            interpolate(args)


if __name__ == '__main__':
    main()
