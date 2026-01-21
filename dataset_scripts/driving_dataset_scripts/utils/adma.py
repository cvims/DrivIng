import json
from driving_dataset_scripts.utils.common import utc_string_to_rostime, rostime_to_float


def load_adma_file_data(adma_file_path):
    with open(adma_file_path, 'r') as file:
        data = json.load(file)

    return data
