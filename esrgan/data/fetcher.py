import tensorflow as tf
import pathlib
import yaml

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
        for dataset in self.config['datasets']:
            links = dataset[list(dataset)[0]]
            print (links)

            for folder, link in links.items():
                if len(link) == 0:
                    continue

                tf.keras.utils.get_file(
                    origin=link, 
                    fname=f'{dataset}/{folder}',
                    untar=True)
                pass
