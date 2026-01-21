import os
import json


def load_annotation_data(labels_path):
    if labels_path is None:
        return None

    with open(os.path.join(labels_path), 'r') as f:
        labels_trackwise = json.load(f)
    
    return labels_trackwise


def convert_labels_from_tracks_wise_to_timestamp_wise(labels, skip_tracks=[]):
    if labels is None:
        return None

    labels_new = {}

    for track in labels['tracks']:
        if track['track_id'] in skip_tracks:
            continue
        for timestamp_index, timestamp in enumerate(track['timestamps']):
            if timestamp not in labels_new.keys():
                labels_new[timestamp] = []

            this_dict = {
                'track_id': track['track_id'],
                'object_type': track['object_type'],
                'position': track['positions'][timestamp_index],
                'orientation': track['orientations'][timestamp_index],
                'dimension': track['dimensions'][0] if len(track['dimensions']) == 1 else track['dimensions'][timestamp_index],
                'attributes': dict(),
            }

            for k, v in track['attributes'].items():
                if isinstance(v, list):
                    this_dict['attributes'][k] = v[timestamp_index]
                else:
                    this_dict['attributes'][k] = v

            labels_new[timestamp].append(this_dict)

    return labels_new


def convert_from_timestamp_wise_to_tracks_wise(ts_wise_labels):
    """
    Converts timestamp-wise labels back into track-wise format.

    Args:
        ts_wise_labels (dict): {timestamp: [object_dicts]} — as produced by
                               convert_labels_from_tracks_wise_to_timestamp_wise().

    Returns:
        dict: {
            "tracks": [
                {
                    "track_id": ...,
                    "object_type": ...,
                    "timestamps": [...],
                    "positions": [...],
                    "orientations": [...],
                    "dimensions": [...],
                    "attributes": {...}
                },
                ...
            ]
        }
    """
    if ts_wise_labels is None:
        return None

    tracks_dict = {}

    # Iterate through each timestamp and its annotations
    for timestamp, objects in ts_wise_labels.items():
        for obj in objects:
            tid = obj["track_id"]

            # Initialize a new track if not seen before
            if tid not in tracks_dict:
                tracks_dict[tid] = {
                    "track_id": tid,
                    "object_type": obj["object_type"],
                    "timestamps": [],
                    "positions": [],
                    "orientations": [],
                    "dimensions": [],
                    "attributes": {},
                }

            # Append timestamp data
            tracks_dict[tid]["timestamps"].append(timestamp)
            tracks_dict[tid]["positions"].append(obj["position"])
            tracks_dict[tid]["orientations"].append(obj["orientation"])
            tracks_dict[tid]["dimensions"].append(obj["dimension"])

            # Handle attributes (per-frame or static)
            for k, v in obj["attributes"].items():
                # static
                if k not in ['4', '5']:
                    tracks_dict[tid]["attributes"][k] = v
                    continue

                if k not in tracks_dict[tid]["attributes"]:
                    # Initialize as list if attribute varies per timestamp
                    tracks_dict[tid]["attributes"][k] = [v]
                else:
                    tracks_dict[tid]["attributes"][k].append(v)

    # Convert dictionary to track list format
    tracks_list = {"tracks": list(tracks_dict.values())}

    return tracks_list
