import argparse
import re
import shutil
import tarfile
from pathlib import Path

import requests


API = 'https://eapi.pcloud.com'
CODE = 'kZxjh1ZX32mlJHd9l4XH2hmoGY1thHoeaNk'
FIELD = re.compile(r'H07/frames/(x10|x20|x40)/(H07_0[01][A-Za-z0-9_-]*\.tiff)')
EXPECTED = {'x10': 2, 'x20': 8, 'x40': 32}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    counts = dict.fromkeys(EXPECTED, 0)
    args.out.mkdir(parents=True, exist_ok=True)
    with requests.Session() as http:
        link = {'code': CODE}
        meta = http.get(f'{API}/showpublink', params=link, timeout=20)
        meta.raise_for_status()
        archive = next(row for row in meta.json()['metadata']['contents'] if row['name'] == 'H07.tar.gz')
        details = http.get(f'{API}/getpublinkdownload', params={**link, 'fileid': archive['fileid']}, timeout=20)
        details.raise_for_status()
        offer = details.json()
        with http.get('https://' + offer['hosts'][0] + offer['path'], stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            with tarfile.open(fileobj=response.raw, mode='r|gz') as entries:
                for entry in entries:
                    match = FIELD.fullmatch(entry.name)
                    if not match or not entry.isfile():
                        continue
                    scale, name = match.groups()
                    path = args.out / scale / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists() and path.stat().st_size == entry.size:
                        counts[scale] += 1
                    else:
                        partial = path.with_suffix('.partial')
                        with entries.extractfile(entry) as source, partial.open('wb') as dest:
                            shutil.copyfileobj(source, dest, 1024 * 1024)
                        if partial.stat().st_size != entry.size:
                            raise ValueError(f'Incomplete archive entry {entry.name}')
                        partial.replace(path)
                        counts[scale] += 1
                    print(scale, name, counts[scale], flush=True)
                    if counts == EXPECTED:
                        break
    if counts != EXPECTED:
        raise ValueError(f'Incomplete H07 field families: {counts}')


if __name__ == '__main__':
    main()
