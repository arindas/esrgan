"""
Module for network operations required to download
and maintain archives.
"""

import os
from urllib.parse import urlparse
import shutil
import wget


def is_url(url):
    """Checks if given url is valid.

    Keyword arguments:
    url -- url to be checked
    """
    try:
        result = urlparse(url)
        return all([result.scheme,
                    result.netloc,
                    result.path])
    except ValueError:
        return False


def fetch_archive(folder, url):
    """Fetches an archive from the given url into
    the given folder an extracts it.

    Keyword arguments:
    folder -- folder to place and extract the archive
    url -- url to fetch the archive from
    """
    if not is_url(url):
        raise ValueError("Malformed URL")

    os.makedirs(folder, exist_ok=True)
    print("[-]", folder, "created.")
    filename = wget.download(url=url, out=folder)
    print("[-]", filename, "downloaded.")
    shutil.unpack_archive(filename, folder)
    print("[-]", filename, "unpacked.")
