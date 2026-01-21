# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

from typing import List, Optional


def category_to_detection_name(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :return: nuScenes detection class.
    """
    detection_mapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None


def detection_name_to_rel_attributes(detection_name: str) -> List[str]:
    """
    Returns a list of relevant attributes for a given detection class.
    :param detection_name: The detection class.
    :return: List of relevant attributes.
    """
    if detection_name in ['pedestrian']:
        rel_attributes = ['pedestrian.standing', 'pedestrian.walking', 'pedestrian.pushing']
    elif detection_name in ['motorcycle']:
        rel_attributes = ['cycle.with_rider']
    elif detection_name in ['bicycle']:
        rel_attributes = ['cycle.with_rider', 'cycle.without_rider']
    elif detection_name in ['trailer']:
        rel_attributes = ['vehicle.car_trailer', 'vehicle.truck_trailer', 'vehicle.cyclist_trailer']
    elif detection_name in ['car', 'bus', 'truck']:
        rel_attributes = ['vehicle.emergency', 'vehicle.regular', 'vehicle.public_transport']
    elif detection_name in ['barrier']:
        # Classes without attributes: barrier, traffic_cone.
        rel_attributes = []
    else:
        raise ValueError('Error: %s is not a valid detection class.' % detection_name)

    return rel_attributes
