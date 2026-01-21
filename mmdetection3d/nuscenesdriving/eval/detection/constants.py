# nuScenes dev-kit.
# Code written by Oscar Beijbom and Varun Bankiti, 2019.

DETECTION_NAMES = ['car', 'truck', 'bus', 'trailer', 'pedestrian', 'motorcycle', 'bicycle','barrier']

PRETTY_DETECTION_NAMES = {'car': 'Car',
                          'truck': 'Truck',
                          'bus': 'Bus',
                          'trailer': 'Trailer',
                          'pedestrian': 'Pedestrian',
                          'motorcycle': 'Motorcycle',
                          'bicycle': 'Bicycle',
                          'barrier': 'Barrier'}

DETECTION_COLORS = {'car': 'C0',
                    'truck': 'C1',
                    'bus': 'C2',
                    'trailer': 'C3',
                    'pedestrian': 'C4',
                    'motorcycle': 'C5',
                    'bicycle': 'C6',
                    'barrier': 'C7'}

ATTRIBUTE_NAMES = [
    'vehicle.emergency', 'vehicle.regular', 'vehicle.public_transport',
    'vehicle.car_trailer', 'vehicle.truck_trailer', 'vehicle.cyclist_trailer',
    'pedestrian.standing', 'pedestrian.walking', 'pedestrian.sitting',
    'cycle.with_rider', 'cycle.without_rider'
    ]

PRETTY_ATTRIBUTE_NAMES = {
    'vehicle.emergency': 'Emergency Vehicle',
    'vehicle.regular': 'Regular Vehicle',
    'vehicle.public_transport': 'Public Transport Vehicle',
    'vehicle.car_trailer': 'Car Trailer',
    'vehicle.truck_trailer': 'Truck Trailer',
    'vehicle.cyclist_trailer': 'Cyclist Trailer',
    'pedestrian.standing': 'Pedestrian Standing',
    'pedestrian.walking': 'Pedestrian Walking',
    'pedestrian.sitting': 'Pedestrian Sitting',
    'cycle.with_rider': 'Cycle w/ Rider',
    'cycle.without_rider': 'Cycle w/o Rider'
}


TP_METRICS = [
    'trans_err',
    'scale_err',
    'orient_err',
    'vel_err',
    # 'attr_err'
]

PRETTY_TP_METRICS = {'trans_err': 'Trans.', 'scale_err': 'Scale', 'orient_err': 'Orient.', 'vel_err': 'Vel.',
                     'attr_err': 'Attr.'}

TP_METRICS_UNITS = {'trans_err': 'm',
                    'scale_err': '1-IOU',
                    'orient_err': 'rad.',
                    'vel_err': 'm/s',
                    #'attr_err': '1-acc.'
                    }
