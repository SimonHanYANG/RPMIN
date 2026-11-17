import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


CONFIG = Path(__file__).resolve().parent
PROJECT = CONFIG.parents[1]
PIPELINE = CONFIG / 'pipeline'
SETTINGS = json.loads((CONFIG / 'experiment.json').read_text(encoding='utf-8'))
SCALES = (2, 4)


def record(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def check_result(scale, result, *, reference):
    test = SETTINGS['test']
    for key in ('metric', 'border_hr', 'patch_input'):
        if result[key] != test[key]:
            raise ValueError(f'x{scale} test setting differs: {key}')
    if result['pairs'] != test['patches']:
        raise ValueError(f'x{scale}: unexpected patch count.')
    if reference:
        expected = test['scales'][str(scale)]
        if result['selected']['weights'] != expected['bands']:
            raise ValueError(f'x{scale}: unexpected locked frequency weights.')
        for metric, tolerance in test['tolerance'].items():
            if not abs(result[metric] - expected[metric]) <= tolerance:
                raise ValueError(f'x{scale}: {metric} differs from the recorded result: {result[metric]}')


def archive(destination):
    """Copy original model bytes, support parameters, evidence and checksums."""
    readapt = Path(SETTINGS['readapt_run'])
    if destination.exists():
        raise FileExistsError(f'Archive already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        staging = Path(temp) / destination.name
        files = {}

        def preserve(source, name=None):
            name = Path('support') / source if name is None else Path(name)
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            files[name.as_posix()] = {'sha256': sha256(target), 'source': str(source)}

        runs = Path('runs')
        context = runs / 'mitos14-rpmin-context'
        mixed = Path(SETTINGS['mixed_run'])
        prior = Path(SETTINGS['prior_run'])
        choice = mixed / 'final-choice.json'
        decision = record(choice)
        preserve(choice)
        preserve(mixed / 'comparison.csv')
        preserve(context / 'luma-residual.pt')
        for scale in SCALES:
            tag = f'x{scale}'
            root = readapt / tag
            best = root / 'transferred/best.pt'
            selected = root / 'selection.json'
            report = root / 'test/summary.json'
            fitting = root / 'transferred/selection.json'
            summary = record(report)
            check_result(scale, summary, reference=True)
            checkpoint_sha = sha256(best)
            test_manifest = Path(SETTINGS['h10_run']) / tag / 'references/test/aligned.csv'
            if (decision[tag]['source'] != 'readapt' or checkpoint_sha != record(fitting)['checkpoint_sha256']
                    or checkpoint_sha != record(selected)['sources'][str(scale)]['weights']
                    or sha256(test_manifest) != summary['manifest_sha256']):
                raise ValueError(f'x{scale}: model or manifest differs from the original test.')
            preserve(best, SETTINGS['test']['scales'][str(scale)]['weight'])
            for path in (selected, report, fitting, root / 'test/per_image.csv',
                         root / 'transferred/validation.csv', test_manifest):
                preserve(path)

            locked = runs / 'mitos14-rpmin-opt' / tag / 'locked'
            sources = record(locked / 'selection.json')
            for name in sources['selected']['checkpoints']:
                source = sources['weights'][name]
                path = Path(source['path'])
                if sha256(path) != source['sha256']:
                    raise ValueError(f'Changed support checkpoint: {path}')
                preserve(path, Path('support/base') / tag / name / path.name)
            if sources['selected']['strength']:
                preserve(locked / 'camera-residual.pt')
            luma_root = runs / 'mitos14-rpmin-luma' / tag
            luma_choice = luma_root / 'locked/selection.json'
            if record(luma_choice)['selected']['blend']:
                preserve(luma_root / 'camera/best.pt')
                preserve(luma_root / 'camera/selection.json')
            for path in (locked / 'selection.json', luma_choice, context / tag / 'selection.json',
                         runs / 'mitos14-rpmin-cross-scale' / tag / 'selection.json',
                         runs / 'mitos14-rpmin-camera' / tag / 'camera/camera.json',
                         Path(SETTINGS['h10_run']) / tag / 'references/train-val/aligned.csv',
                         mixed / tag / 'augmented.csv', mixed / tag / 'transferred/selection.json',
                         mixed / tag / 'transferred/best.pt', prior / tag / 'selection.json',
                         prior / tag / 'context-trained/selection.json', prior / tag / 'context-trained/best.pt'):
                preserve(path)
        for name in ('experiment.json', 'results.csv'):
            preserve(CONFIG / name, name)
        (staging / 'inventory.json').write_text(json.dumps(files, indent=2) + '\n', encoding='utf-8')
        staging.rename(destination)
    print(destination)


def train(workdir, output):
    from rpmin.data import read_manifest, write_manifest

    config = SETTINGS['train']
    mixed = output.with_name(output.name + '-mixed')
    if mixed.exists():
        raise FileExistsError(mixed)
    env = dict(os.environ, PYTHONPATH=str(PROJECT), OMP_NUM_THREADS='2')
    python = workdir / SETTINGS['python']

    def run(*args):
        subprocess.run([python, *map(str, args)], check=True, env=env)

    def fit(scale, manifest, init, target, phase):
        run(PIPELINE / 'train_mitos_camera.py', '--luminance', '--native-grid',
            '--context', config['context_lr'], '--scale', scale, '--train', manifest, '--init', init,
            '--real-steps', config[f'{phase}_steps'], '--real-lr', config[f'{phase}_lr'], '--out', target)

    def fields(rows, split):
        # H10 identifiers include a short source hash after the field name.
        return {row.id.upper().replace('-', '_')[:6] for row in rows if row.split == split}

    for scale in SCALES:
        tag = f'x{scale}'
        references = Path(SETTINGS['h10_run']) / tag / 'references'
        h10 = references / 'train-val/aligned.csv'
        registered_h07 = Path(SETTINGS['mixed_run']) / tag / 'h07-references/aligned.csv'
        original = read_manifest(h10)
        rows = original + read_manifest(registered_h07)
        test_rows = read_manifest(references / 'test/aligned.csv')
        split = SETTINGS['split']
        if (fields(rows, 'train') != set(split['mixed_train']) or fields(rows, 'val') != {split['val']}
                or fields(original, 'train') != set(split['readapt_train']) or fields(test_rows, 'test') != {split['test']}
                or len(rows) != len(split['mixed_train']) + 1 or len(test_rows) != 1
                or {row.group for row in rows} & {row.group for row in test_rows}):
            raise ValueError(f'x{scale}: unexpected field split.')
        augmented = mixed / tag / 'augmented.csv'
        write_manifest(augmented, rows)
        initial = Path(SETTINGS['prior_run']) / tag / 'context-trained/best.pt'
        stage1 = mixed / tag / 'transferred'
        fit(scale, augmented, initial, stage1, 'mixed')
        fit(scale, h10, stage1 / 'best.pt', output / tag / 'transferred', 'readapt')
    run(PIPELINE / 'select_mitos_transfer.py', 'select', '--root', output)


def test(root, output, archive_path, *, reference):
    sys.path.insert(0, str(PIPELINE))
    import torch
    from select_mitos_transfer import BASE, GRID, Comparison
    from rpmin.engine import dump_csv, dump_json, summarize

    torch.set_num_threads(2)
    checkpoints = None
    if archive_path is not None:
        inventory = record(archive_path / 'inventory.json')
        checkpoints = {}
        for scale in SCALES:
            name = SETTINGS['test']['scales'][str(scale)]['weight']
            checkpoints[scale] = archive_path / name
            if sha256(checkpoints[scale]) != inventory[name]['sha256']:
                raise ValueError(f'Changed archived checkpoint: {name}')
    comparison = Comparison(root, checkpoints=checkpoints)
    choices = {scale: record(root / f'x{scale}/selection.json') for scale in SCALES}
    if any(choice['sources'] != comparison.digests for choice in choices.values()):
        raise ValueError('A locked source changed.')
    reports = comparison.score('test', {scale: [choice['selected']['weights']] for scale, choice in choices.items()})
    for scale, tracks in reports.items():
        manifest = BASE / f'x{scale}/references/test/aligned.csv'
        original = record(Path(SETTINGS['readapt_run']) / f'x{scale}/test/summary.json')
        if sha256(manifest) != original['manifest_sha256']:
            raise ValueError(f'x{scale}: test manifest changed.')
        result = dict(scale=scale, split='test', metric='BT.601 Y', border_hr=0, manifest_sha256=sha256(manifest),
                      selected=choices[scale]['selected'], patch_input=dict(GRID, batch=4, fields=1), **summarize(tracks[0]))
        target = output / f'x{scale}'
        dump_csv(target / 'per_image.csv', tracks[0])
        dump_json(target / 'summary.json', result)
        check_result(scale, result, reference=reference)
        print(f"x{scale}: {result['psnr']:.8f} dB / {result['ssim']:.8f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Replay the saved MITOS-14 H07 RP-MIN experiment.')
    parser.add_argument('stage', choices=('train', 'test', 'archive'))
    parser.add_argument('--workdir', type=Path, default=Path(SETTINGS['server_workdir']))
    parser.add_argument('--run', type=Path, help='Re-training output or the run to test. Omit for original weights.')
    parser.add_argument('--out', type=Path, help='New test output directory or archive destination.')
    parser.add_argument('--archived-weights', type=Path, help='Use the named checkpoints from this archive for original testing.')
    args = parser.parse_args()
    if args.stage == 'train' and (args.run is None or args.out is not None):
        parser.error('train requires --run and writes its output there.')
    if args.stage == 'test' and args.out is None:
        parser.error('test requires a new --out directory.')
    if args.archived_weights and (args.stage != 'test' or args.run):
        parser.error('--archived-weights is for testing the original results.')
    workdir = args.workdir.resolve(strict=True)
    os.chdir(workdir)
    sys.path.insert(0, str(PROJECT))
    if args.stage == 'archive':
        archive((args.out or (PROJECT / SETTINGS['archive'])).resolve())
    elif args.stage == 'train':
        if args.run.exists():
            raise FileExistsError(args.run)
        train(workdir, args.run)
    else:
        if args.out.exists():
            raise FileExistsError(args.out)
        test(args.run or Path(SETTINGS['readapt_run']), args.out, args.archived_weights, reference=args.run is None)


if __name__ == '__main__':
    main()
