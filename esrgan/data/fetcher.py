import os
import shutil
from concurrent.futures import (
    ThreadPoolExecutor)
import yaml
import wget


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


def fetch_datasets(config_path="config/data.yaml"):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    prefix = config['datadir']
    with ThreadPoolExecutor(max_workers=8) as executor:
        for dataset in config['datasets'].keys():
            subdir = list(dataset.keys())[0]
            urls = dataset[subdir]
            print(dataset, urls)

            for item, url in urls.items():
                if not is_url(url):
                    continue
                folder = f'{prefix}/{subdir}/{item}'
                executor.submit(fetch_archive, folder, url)
