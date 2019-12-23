import tensorflow as tf
import shutil
import wget
import yaml
import os

def is_valid_url (url):
    try:
        from urllib.parse import urlparse 
        result = urlparse (url)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

class DatasetFetcher:
    def __init__ (self, config_path="config/data.yaml"):
        self.load_config (config_path)

    def load_config (self, config_path): 
        with open (config_path, "r") as stream:
            try:
                self.config = yaml.safe_load (stream)
            except yaml.YAMLError as exception: 
                print (exception)

    def fetch_datasets (self):
        prefix = self.config['datadir']

        for dataset in self.config['datasets']:
            urls = dataset[list(dataset)[0]]
            print (urls)

            for item, url in urls.items():
                if not is_valid_url(url):
                    continue

                folder = f'{prefix}/{dataset}/{item}'
                os.makedirs (folder, exist_ok=True)
                filename = wget.download (url=url, out=folder)
                shutil.unpack_archive (filename, folder)
