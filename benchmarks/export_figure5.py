import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image
import torch

from compare_x4 import load_network
from rpmin.data import read_manifest
from rpmin.paired_reference import lift_transform


MFA_REF = Path('/root/autodl-tmp/rpmin-project/RP-MIN-fresh/runs/full/references/x4')
MITOS_RUN = Path('runs/mitos14-h10-train/x4')
MFA_RUN = Path('runs/x4-pilot')
DATA = Path('/root/autodl-tmp/rpmin-project/SuperResolutionDataset')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ensure_checkpoint(path, digest):
    if sha(path) != digest:
        raise ValueError(f'Checkpoint digest mismatch for {path}')


def save(image, filename):
    Image.fromarray(image).save(filename)


def pixels(image):
    return np.clip(image * 255, 0, 255).round().astype(np.uint8)


def prepare(record, root):
    mfa = record['dataset'] == 'MFA-Cell'
    refs = MFA_REF if mfa else MITOS_RUN / 'references'
    key = record['raw_file'].split('_')[0] if mfa else Path(record['raw_file']).stem.lower().replace('_','-')
    for partition in ('test', 'train-val'):
        directory = refs / partition
        matches = [row for row in read_manifest(directory/'aligned.csv') if row.id.rsplit('-',1)[0] == key]
        if matches:
            pair, = matches
            break
    else:
        raise ValueError(f'Missing field {key}')

    registration = json.loads((directory/'registration.json').read_text())
    transform, = [item['lr_to_reduced_hr'] for item in registration['pairs'] if item['id'] == pair.id]
    lr_to_raw = lift_transform(np.array(transform), 4)
    raw_to_untrimmed = np.linalg.inv(lr_to_raw)
    with Image.open(pair.lr) as im:
        x,y,_,_ = record['overview']
        expected = np.array([x-3,y-2,1.])
        point = raw_to_untrimmed @ expected
        margin = registration['margin_lr']
        left, top = np.round(point[:2]/(4*point[2])).astype(int) - margin
        left, top = int(np.clip(left,0,im.width-128)), int(np.clip(top,0,im.height-128))
        lr = np.array(im.convert('RGB').crop((left,top,left+128,top+128)))
    with Image.open(pair.hr) as im:
        aligned = np.array(im.convert('RGB').crop((4*left,4*top,4*(left+128),4*(top+128))))
    x,y,width,height = record['raw_roi']
    # Image registration maps the paper's raw-HR pixels into model-output coordinates.
    translate = np.array([[1,0,x],[0,1,y],[0,0,1.]])
    offset = np.array([[1,0,-4*(margin+left)],[0,1,-4*(margin+top)],[0,0,1.]])
    mapping = offset @ raw_to_untrimmed @ translate
    edges = cv2.perspectiveTransform(np.array([[[0,0],[width-1,0],[0,height-1],[width-1,height-1]]], dtype=np.float64),mapping)
    if edges.min() < 1 or edges.max() > 510:
        raise ValueError(f'The matched region exceeds inference context for {pair.id}: {edges}')
    folder = root / f"row_{record['row']:02d}_{record['dataset']}_{key}"
    folder.mkdir()
    save(lr,folder/'LR_input_128.png')
    save(aligned,folder/'HR_aligned_context_512.png')
    dataset = 'ciliated epithelial cell for super resolution' if mfa else 'MITOS-14 for super resolution'
    raw_path = DATA / dataset / 'x40' / record['raw_file']
    with Image.open(raw_path) as raw:
        raw = raw.convert('RGB')
        save(np.array(raw.crop((x,y,x+width,y+height))),folder/'HR_raw_roi.png')
        rx,ry,_,_ = record['overview']
        save(np.array(raw.crop((rx-3,ry-2,rx+509,ry+510))),folder/'HR_raw_overview_512.png')
    # Independently validate that inverse mapping returns the same cell structure from aligned HR.
    matched = cv2.warpPerspective(aligned,mapping,(width,height),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    raw_crop = np.array(Image.open(folder/'HR_raw_roi.png'))
    corr = np.corrcoef(cv2.cvtColor(matched,cv2.COLOR_RGB2GRAY).ravel(),cv2.cvtColor(raw_crop,cv2.COLOR_RGB2GRAY).ravel())[0,1]
    if corr < 0.80:
        raise ValueError(f'HR registration verification failed for {pair.id}: {corr:.3f}')
    record.update(id=pair.id, split=pair.split, folder=folder.name,
                  lr_patch_aligned=[left,top,128,128], raw_roi_to_output=mapping.tolist(), alignment_ncc=float(corr),
                  source_lr=str(pair.lr), source_hr=str(raw_path))
    return lr, folder, mapping


def export_predictions(method, values, prepared, records):
    if values.shape != (len(records),512,512,3) or not np.isfinite(values).all():
        raise ValueError(f'Invalid {method} prediction shape or values.')
    for sr,(_,folder,mapping),record in zip(values,prepared,records,strict=True):
        out = pixels(sr)
        _,_,w,h = record['raw_roi']
        roi = cv2.warpPerspective(out,mapping,(w,h),flags=cv2.INTER_CUBIC|cv2.WARP_INVERSE_MAP)
        save(out,folder/f'{method}_SR_512.png')
        save(roi,folder/f'{method}_roi.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('coordinates',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    root = args.out
    root.mkdir(parents=True,exist_ok=False)
    records = json.loads(args.coordinates.read_text())
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    provenance = {}
    for dataset in ('MFA-Cell','Mitos-Atypia-14'):
        subset = [item for item in records if item['dataset'] == dataset]
        prepared = [prepare(item,root) for item in subset]
        inputs = np.stack([t[0] for t in prepared])
        mfa = dataset == 'MFA-Cell'
        provenance[dataset] = {}
        for method in ('AESOP','SeD'):
            weights = (MFA_RUN/f'{method}-transfer' if mfa else MITOS_RUN/method)
            checkpoint = weights/'adapted.pt'
            ensure_checkpoint(checkpoint,json.loads((weights/'summary.json').read_text())['checkpoint_sha256'])
            model, _ = load_network(method,checkpoint)
            with torch.inference_mode():
                batch = torch.from_numpy(inputs).permute(0,3,1,2).cuda().float().div(255)
                scores = model(batch).permute(0,2,3,1).cpu().numpy()
            export_predictions(method,scores,prepared,subset)
            provenance[dataset][method] = {'checkpoint':str(checkpoint),'sha256':sha(checkpoint)}
            del model
        next_run = Path('runs/x4-next-transfer') if mfa else MITOS_RUN/'next'
        selected = json.loads((next_run/'interpolated/selection.json').read_text())
        checkpoint = Path(selected['checkpoint'])
        ensure_checkpoint(checkpoint,selected['checkpoint_sha256'])
        inp,out = root/'next-input.npy',root/'next-output.npy'
        np.save(inp,inputs)
        subprocess.run(['next-env/bin/python','next_x4.py','infer','--scale','4','--weights',str(checkpoint),
                        '--inputs',str(inp),'--out',str(out)],check=True)
        export_predictions('NeXtSRGAN',np.load(out),prepared,subset)
        inp.unlink()
        out.unlink()
        provenance[dataset]['NeXtSRGAN'] = {'checkpoint':str(checkpoint),'sha256':sha(checkpoint)}
    (root/'manifest.json').write_text(json.dumps(dict(scale=4,rows=records,models=provenance),indent=2)+'\n')


if __name__ == '__main__':
    main()
