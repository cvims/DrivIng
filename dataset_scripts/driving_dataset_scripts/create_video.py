import os
from driving_dataset_scripts.utils.video import create_lidar_camera_video, create_lidar_camera_video, create_camera_video, create_lidar_bev_video
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create LiDAR–Camera videos from converted ROS bag data."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Root directory containing the rosbags_converted sequences."
    )
    parser.add_argument(
        "--sequence-name",
        type=str,
        help="Name of the sequence."
    )
    parser.add_argument(
        "--video-output-dir",
        type=str,
        help="Directory where the output videos will be saved."
    )
    parser.add_argument(
        "--video-filename",
        type=str,
        help="Video filename without mp4."
    )

    return parser.parse_args()


if __name__ == '__main__':
    ##############################################################
    ############### CREATE LIDAR-CAMERA VIDEO FILE ###############
    ##############################################################

    args = parse_args()

    data_path = args.data_path
    sequence_name = args.sequence_name
    sequence_root_path = os.path.join(data_path, sequence_name)

    annot_path = os.path.join(sequence_root_path, 'annotations.json')
    video_output_dir = args.video_output_dir
    video_filename = args.video_filename
    video_output_file = os.path.join(video_output_dir, video_filename + '.mp4')

    # Day, Dusk, Dawn are synced at 10 Hz
    fps = 10

    create_lidar_camera_video(
        sequence_root_path, video_output_file, annot_path,
        fps=fps, res=0.1, left_right_range=(-100, 100), forward_backward_range=(-150, 150),
        undistort_images=True,
        num_workers=16,
        min_chunk_size=300,
        annotations_to_images=True
    )

    # create_camera_video(
    #     sequence_root_path, video_output_file, annot_path,
    #     fps=fps,
    #     undistort_images=True,
    #     annotations_to_images=True,
    #     points_to_image=True,
    #     num_workers=16,
    #     min_chunk_size=300,
    #     img_size=(1080, 1920)
    # )

    ##############################################################