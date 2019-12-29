"""
Module containing functions for fetching and extracting
esrgan datasets into configuration specified locations.
"""

from concurrent.futures import (
    ThreadPoolExecutor)
from esrgan.util.net import (
    fetch_archive,
    is_url
)


def fetch_datasets(config):
    """Fetches all datasets specified in the config.

    Keyword arguments:
    config -- loaded configuration dict
    """
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
