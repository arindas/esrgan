import os
import shutil
from concurrent.futures import (
    ThreadPoolExecutor)
import yaml
import wget

from esrgan.util.helpers import load_config

def is_url(url):
    try:
        from urllib.parse import urlparse
        result = urlparse(url)
        return all([result.scheme,
                    result.netloc,
                    result.path])
    except ValueError:
        return False


def fetch_archive(folder, url):
    if not is_url(url):
        raise ValueError("Malformed URL")

    os.makedirs(folder, exist_ok=True)
    print("[-]", folder, "created.")
    filename = wget.download(url=url, out=folder)
    print("[-]", filename, "downloaded.")
    shutil.unpack_archive(filename, folder)
    print("[-]", filename, "unpacked.")

def fetch_datasets(config=load_config()):
    prefix = config['datadir']
    with ThreadPoolExecutor(max_workers=8) as executor:
        for dataset in config['datasets']:
            subdir = list(dataset.keys())[0]
            urls = dataset[subdir]
            for item, url in urls.items():
                if not is_url(url):
                    continue
                folder = f'{prefix}/{subdir}/{item}'
                executor.submit(fetch_archive, folder, url)
