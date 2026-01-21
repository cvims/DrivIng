import os
import argparse
import json
import copy
import shutil
import random
import tqdm
import numpy as np

from driving_dataset_scripts.utils.annotations import load_annotation_data, convert_labels_from_tracks_wise_to_timestamp_wise, convert_from_timestamp_wise_to_tracks_wise
from driving_dataset_scripts.utils.common import read_timesync_file, create_time_sync_file


# name=(train,val,test)
SPLIT_SIZES = (0.8,0.1,0.1)


def symlink_timesync_files(time_sync_content, seq_path, root_path):
    # reference all from root path to seq path
    for sensor_name in list(time_sync_content.keys()):
        if sensor_name == 'timestamp_nanoseconds':
            continue

        os.makedirs(os.path.join(seq_path, sensor_name), exist_ok=True)

        for sensor_ref in time_sync_content[sensor_name]:
            org_sensor_filename = os.path.join(str(sensor_ref[0]) + sensor_ref[1])
            if org_sensor_filename:
                new_sensor_path = os.path.join(seq_path, sensor_name, org_sensor_filename)
                os.symlink(os.path.join(root_path, sensor_name, org_sensor_filename), new_sensor_path)


def create_timesync_file_for_sequence(timesync_content, timestamps):
    new_timesync_content = {k: [] for k in list(timesync_content.keys())}

    ts_index_map = {k: i for i, k in enumerate(timesync_content['timestamp_nanoseconds'])}

    for ts in timestamps:
        ts = int(ts)
        _index = ts_index_map[ts]
        for key in list(new_timesync_content.keys()):
            new_timesync_content[key].append(copy.deepcopy(timesync_content[key][_index]))

    return new_timesync_content


def split_by_chunk_size(chunks, timesync_content, ts_wise_annotations, split_sizes):
    """
    Split timestamps into equally sized chunks and then divide each chunk
    into train/val/test according to split_sizes.

    Args:
        chunks (int): Number of timestamps per chunk.
        timesync_content: Full time sync content reference
        ts_wise_annotations (dict or iterable): Dictionary or list of timestamps.
        split_sizes (tuple or list): Ratios for (train, test, val), e.g. (0.7, 0.15, 0.15)

    Returns:
        tuple: (train_timestamps, test_timestamps, val_timestamps)
    """
    all_timestamps = timesync_content['timestamp_nanoseconds']

    # Sort timestamps
    train_timestamps = []
    train_timesync = []
    test_timestamps = []
    test_timesync = []
    val_timestamps = []
    val_timesync = []

    chunks_list = np.array_split(all_timestamps, chunks)

    # Split all timestamps into equally sized chunks
    for chunk in chunks_list:
        # Compute split indices
        n = len(chunk)
        n_train = int(n * split_sizes[0])
        n_test = int(n * split_sizes[2])
        # Ensure all timestamps are used
        n_val = n - n_train - n_test

        # Slice chunk into splits
        train_timestamps.append({int(ts): ts_wise_annotations[int(ts)] if int(ts) in ts_wise_annotations else {} for ts in chunk[:n_train]})
        train_timesync.append(create_timesync_file_for_sequence(timesync_content, chunk[:n_train]))

        val_timestamps.append({int(ts): ts_wise_annotations[int(ts)] if int(ts) in ts_wise_annotations else {} for ts in chunk[n_train:n_train + n_val]})
        val_timesync.append(create_timesync_file_for_sequence(timesync_content, chunk[n_train:n_train + n_val]))

        test_timestamps.append({int(ts): ts_wise_annotations[int(ts)] if int(ts) in ts_wise_annotations else {} for ts in chunk[n_train + n_val:]})
        test_timesync.append(create_timesync_file_for_sequence(timesync_content, chunk[n_train + n_val:]))

    return (train_timestamps, train_timesync), (val_timestamps, val_timesync), (test_timestamps, test_timesync)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split annotations into train/val/test chunks and prepare dataset structure."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Root folder containing the converted rosbags."
    )
    parser.add_argument(
        "--sequence-name",
        type=str,
        help="Name of the sequence to process."
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default='',
        required=False,
        help="Directory to place the split information. By default it uses the data-path/splits"
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=50,
        required=False,
        help="Number of chunks to split each sequence into."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()


def process_split(splits, root_data_path, out_path, file_counter, padding):
    """Process a single split (train/val/test) and write files to disk."""
    for split, time_sync_content in tqdm.tqdm(zip(*splits), total=len(splits), desc=f'Creating split.'):
        seq_tracks = convert_from_timestamp_wise_to_tracks_wise(split)
        seq_name = f"{file_counter:0{padding}d}"
        seq_path = os.path.join(out_path, seq_name)
        os.makedirs(seq_path, exist_ok=False)

        # Save annotations
        with open(os.path.join(seq_path, 'annotations.json'), 'w') as f:
            json.dump(seq_tracks, f)

        # Copy calibration
        shutil.copy(os.path.join(root_data_path, 'calibration.json'), os.path.join(seq_path, 'calibration.json'))

        # Save timesync file
        create_time_sync_file(time_sync_content, seq_path, 'timesync_info.csv', False)

        # Symlink original files
        symlink_timesync_files(time_sync_content, seq_path, root_data_path)

        file_counter += 1

    return file_counter


def main():
    args = parse_args()
    random.seed(args.seed)

    data_path = args.data_path
    sequence_name = args.sequence_name
    sequence_root_path = os.path.join(data_path, sequence_name)

    out_path = args.out_path
    if not out_path:
        out_path = os.path.join(data_path, '..', 'splits')

    annot_path = os.path.join(sequence_root_path, 'annotations.json')

    os.makedirs(out_path, exist_ok=True)

    padding = int(np.log10(args.n_chunks * 3)) + 1

    seq_out_path = os.path.join(out_path, sequence_name)

    # Load timesync and annotation data
    timesync_content = read_timesync_file(os.path.join(sequence_root_path, 'timesync_info.csv'))
    annotation_data = load_annotation_data(annot_path)
    annotation_data = convert_labels_from_tracks_wise_to_timestamp_wise(annotation_data)

    # Split sequences
    train_splits, val_splits, test_splits = split_by_chunk_size(
        args.n_chunks, timesync_content, ts_wise_annotations=annotation_data, split_sizes=SPLIT_SIZES
    )

    # Process splits
    file_counter = 0
    if train_splits:
        train_out_path = os.path.join(seq_out_path, 'train')
        os.makedirs(train_out_path, exist_ok=False)
        file_counter = process_split(train_splits, sequence_root_path, train_out_path, file_counter, padding)

    if val_splits:
        val_out_path = os.path.join(seq_out_path, 'val')
        os.makedirs(val_out_path, exist_ok=False)
        file_counter = process_split(val_splits, sequence_root_path, val_out_path, file_counter, padding)

    if test_splits:
        test_out_path = os.path.join(seq_out_path, 'test')
        os.makedirs(test_out_path, exist_ok=False)
        file_counter = process_split(test_splits, sequence_root_path, test_out_path, file_counter, padding)


if __name__ == "__main__":
    main()
