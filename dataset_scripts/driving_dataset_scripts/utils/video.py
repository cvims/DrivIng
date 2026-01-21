import os
import numpy as np
import cv2
import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Pool, Manager, Process
from driving_dataset_scripts.utils.lidar import read_extended_point_cloud_format_npz, project_point_cloud_to_image
from driving_dataset_scripts.utils.camera import load_image, annotation_to_image, plot_images_combined
from driving_dataset_scripts.utils.common import read_timesync_file, load_calibration_file
from driving_dataset_scripts.utils.annotations import load_annotation_data, convert_labels_from_tracks_wise_to_timestamp_wise
from driving_dataset_scripts.utils.codes import CLASS_COLOR_CODES
import gc


_UNIT_CUBE_CORNERS = np.array([
    [ 0.5,  0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5,  0.5, -0.5],
    [ 0.5,  0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [-0.5, -0.5,  0.5],
    [-0.5,  0.5,  0.5]
], dtype=np.float32)


def annotation_to_corners_3d(annotation):
    """
    Compute oriented 3D bounding box corners from annotation dict.
    Faster version with preallocated templates and vectorized math.
    """
    pos = np.asarray(annotation["position"], dtype=np.float32)
    yaw = float(annotation["orientation"])
    l, w, h = map(float, annotation["dimension"])

    # Scale precomputed unit cube to actual box size
    corners = _UNIT_CUBE_CORNERS * np.array([l, w, h], dtype=np.float32)

    # Rotation around Z (yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)

    # Apply rotation and translation
    return corners @ R.T + pos


def draw_annotation_data_onto_bev(bev, annotation_data, left_right_range=(-50, 50), forward_backward_range=(-100, 100), res=0.1, Ry=None, custom_center_point=None):
    """
    Creates a bounding box in BEV view for all annotated objects.

    Args:
        bev (np.ndarray): 2D Image
        annotation_data: list of dict entries of usual annotation style (timestamp_wise) {'track_id', 'object_type', 'position', 'orientation', 'dimension', 'attribute'}
    """
    if not annotation_data:
        return bev

    for ann in annotation_data:
        obj_type = ann['object_type']
        # if obj_type == 'Other':
        #     continue

        world_corners = annotation_to_corners_3d(ann)
    
        if custom_center_point is not None:
            cx, cy = custom_center_point
            world_corners[:, 0] -= cx
            world_corners[:, 1] -= cy

        if Ry is not None:
            world_corners = (Ry @ world_corners.T).T

        # --- BEV projection (ignore z, but we’ll use it for top vs bottom) ---
        x = ((world_corners[:,0] - left_right_range[0]) / res).astype(int)
        y = ((world_corners[:,1] - forward_backward_range[0]) / res).astype(int)
        bev_pts = np.stack([x, y], axis=-1)

        # Draw bottom face (indices 0–3)
        bottom_face = bev_pts[0:4]
        cv2.polylines(bev, [bottom_face], True, CLASS_COLOR_CODES[obj_type], 2)

        # Draw top face (indices 4–7)
        top_face = bev_pts[4:8]
        cv2.polylines(bev, [top_face], True, CLASS_COLOR_CODES[obj_type], 2)

        # Draw vertical edges
        for b, t in zip(range(4), range(4,8)):
            cv2.line(bev, tuple(bev_pts[b]), tuple(bev_pts[t]), CLASS_COLOR_CODES[obj_type], 2)

    return bev


def create_point_cloud_bev_image(pcd, left_right_range=(-50, 50), forward_backward_range=(-100, 100), res=0.1, z_clip=(-3, 5), annotation_data=None, custom_center_point=None, background_color=None):
    """
    Erstellt ein BEV-Bild aus einem LiDAR-Punkt-Array.

    Args:
        pcd (np.ndarray): (N, 5) LiDAR Punkte (x, y, z, intensity, time).
        left_right_range (tuple): Bereich (min, max) in Metern nach links/rechts (X).
        forward_backward_range (tuple): Bereich (min, max) in Metern nach vorne (Y).
        res (float): Aufloesung in m/pixel.
        z_clip (tuple): (min, max) Hoehenbegrenzung fuer BEV-Clipping.
        annotation_data: list of dict entries of usual annotation style {'track_id', 'object_type', 'position', 'orientation', 'dimension', 'attribute'}
        custom_center_point: [x, y] shift point cloud by those values.
    Returns:
        bev (np.ndarray): Bild als 2D BEV.
    """
    assert pcd.shape[1] >= 3, "pcd must have at least 3 columns (x,y,z)"

    if custom_center_point is not None:
        cx, cy = custom_center_point
        pcd = pcd.copy()
        pcd[:, 0] -= cx
        pcd[:, 1] -= cy

    # Filter Punkte im Sichtbereich
    x_mask = (pcd[:, 0] >= left_right_range[0]) & (pcd[:, 0] <= left_right_range[1])
    y_mask = (pcd[:, 1] >= forward_backward_range[0]) & (pcd[:, 1] <= forward_backward_range[1])
    z_mask = (pcd[:, 2] >= z_clip[0]) & (pcd[:, 2] <= z_clip[1])
    mask = x_mask & y_mask & z_mask
    pcd = pcd[mask]

    # Rotation for 3D effect
    theta = np.deg2rad(30)
    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,            1, 0           ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    pcd_rotated = (Ry @ pcd[:, :3].T).T

    # In Pixel-Koordinaten umrechnen
    x_img = ((pcd_rotated[:, 0] - left_right_range[0]) / res).astype(np.int32)
    y_img = ((pcd_rotated[:, 1] - forward_backward_range[0]) / res).astype(np.int32)

    # Bildgroesse
    width = int((left_right_range[1] - left_right_range[0]) / res)
    height = int((forward_backward_range[1] - forward_backward_range[0]) / res)

    x_img = np.clip(x_img, 0, width - 1)
    y_img = np.clip(y_img, 0, height - 1)

    # Grauwert ueber Hoehe oder Intensitaet
    if pcd.shape[1] >= 4:
        intensity = pcd[:, 3]
    else:
        # Hoehe als Grauwert
        z = pcd[:, 2]
        intensity = ((z - z_clip[0]) / (z_clip[1] - z_clip[0]))

    # Bild initialisieren
    background_color = 30 if background_color is None else background_color
    bev = np.full((height, width), fill_value=background_color, dtype=np.uint8)

    # Punkte einzeichnen
    bev[y_img, x_img] = intensity

    mask_bg = (bev == background_color)

    bev = cv2.applyColorMap(bev, cv2.COLORMAP_TURBO)

    bev[mask_bg] = background_color

    if annotation_data:
        bev = draw_annotation_data_onto_bev(bev, annotation_data, left_right_range, forward_backward_range, res, Ry, custom_center_point)

    bev = np.flipud(bev)
    bev = cv2.rotate(bev, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return bev


def process_single_pcd(pcd_path, x_range, y_range, res, z_clip):
    if isinstance(pcd_path, np.ndarray):
        pcd = pcd_path
    else:
        pcd = read_extended_point_cloud_format_npz(pcd_path)
    bev = create_point_cloud_bev_image(pcd, left_right_range=x_range, forward_backward_range=y_range, res=res, z_clip=z_clip)

    if isinstance(pcd_path, np.ndarray):
        del pcd_path
        del pcd
        gc.collect()
    return bev


def create_lidar_bev_video(pcd_file_paths, output_file_path, fps=10, x_range=(-100, 100), y_range=(-100, 100), res=0.1, z_clip=(-3, 3), num_workers=16):
    """
    Creates a BEV video from multiple point cloud (.npz) files in parallel.
    """
    if len(pcd_file_paths) == 0:
        raise ValueError("pcd_file_paths is empty.")

    # Process one frame to determine frame size
    if isinstance(pcd_file_paths[0], np.ndarray):
        first_pcd = pcd_file_paths[0]
    else:
        first_pcd = read_extended_point_cloud_format_npz(pcd_file_paths[0])
    height, width, _ = create_point_cloud_bev_image(first_pcd, left_right_range=x_range, forward_backward_range=y_range, res=res, z_clip=z_clip).shape

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=True)

    # Prepare parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        process_fn = partial(process_single_pcd, x_range=x_range, y_range=y_range, res=res, z_clip=z_clip)
        bev_frames = list(tqdm.tqdm(executor.map(process_fn, pcd_file_paths), total=len(pcd_file_paths), desc="Processing BEV frames"))

    # Write frames in order
    for frame in bev_frames:
        writer.write(frame)

    writer.release()
    print(f"Saved BEV video of lidar to: {output_file_path}.")


def process_lidar_camera_frame(i, sync_data, calibration_data, annotation_data, base_path, camera_names, lidar_name,
                  camera_steps, lidar_w, lidar_h, left_right_range, forward_backward_range,
                  res, z_clip, undistort_images, annotations_to_images):
    cam_h, cam_w = camera_steps[1], camera_steps[0]

    if annotation_data:
        _ts = sync_data['timestamp_nanoseconds'][i]
        annotation_data = annotation_data.get(_ts, None)

    # Load camera images
    camera_data_file_names = [str(sync_data[cam][i][0]) + sync_data[cam][i][1] for cam in camera_names]
    camera_data_file_paths = [os.path.join(base_path, cam, fname) for cam, fname in zip(camera_names, camera_data_file_names)]
    images = [load_image(p) for p in camera_data_file_paths]

    # Load lidar
    lidar_file_name = str(sync_data[lidar_name][i][0]) + sync_data[lidar_name][i][1]
    lidar_file_path = os.path.join(base_path, lidar_name, lidar_file_name)
    pcd = read_extended_point_cloud_format_npz(lidar_file_path)
    lidar_bev = create_point_cloud_bev_image(pcd, left_right_range, forward_backward_range, res, z_clip, annotation_data=annotation_data)
    vTl = calibration_data['lidars'][lidar_name]['vTl']

    for j, cam_name in enumerate(camera_names):
        if images[j] is not None:
            K, D, K_undistorted, image_size, undistort_function, cTv = calibration_data['cameras'][cam_name].values()
            # Undistort if needed
            if undistort_images:
                images[j] = undistort_function(images[j])
                if annotations_to_images:
                    images[j] = annotation_to_image(images[j], annotation_data, vTl, cTv, K_undistorted, None)
            else:
                if annotations_to_images:
                    images[j] = annotation_to_image(images[j], annotation_data, vTl, cTv, K, D)

    # Resize and combine images
    images = [cv2.resize(img, camera_steps) if img is not None else np.zeros((cam_h, cam_w, 3), dtype=np.uint8) for img in images]
    lidar_bev = cv2.resize(lidar_bev, (lidar_w, lidar_h))

    frame = np.zeros((cam_h * 3, cam_w * len(camera_names), 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        x_start = idx * cam_w
        frame[0:cam_h, x_start:x_start + cam_w] = img
    frame[cam_h:, :] = lidar_bev

    return frame


def process_camera_frame(i, sync_data, calibration_data, annotation_data, base_path, camera_names,
                         undistort_images, annotations_to_images: bool, img_size,
                         rows, lidar_name=None, points_to_image: bool = False):

    if annotation_data:
        _ts = sync_data['timestamp_nanoseconds'][i]
        annotation_data = annotation_data.get(_ts, None)

    # Load camera images
    camera_data_file_names = [str(sync_data[cam][i][0]) + sync_data[cam][i][1] for cam in camera_names]
    camera_data_file_paths = [os.path.join(base_path, cam, fname) for cam, fname in zip(camera_names, camera_data_file_names)]
    images = [load_image(p) for p in camera_data_file_paths]

    # Load lidar
    pcd = None
    if points_to_image and lidar_name:
        lidar_file_name = str(sync_data[lidar_name][i][0]) + sync_data[lidar_name][i][1]
        lidar_file_path = os.path.join(base_path, lidar_name, lidar_file_name)
        pcd = read_extended_point_cloud_format_npz(lidar_file_path)

    vTl = calibration_data['lidars']['middle_lidar']['vTl']

    for j, cam_name in enumerate(camera_names):
        if images[j] is not None:
            K, D, K_undistorted, image_size, undistort_function, cTv = calibration_data['cameras'][cam_name].values()
            # Undistort if needed
            if undistort_images:
                images[j] = undistort_function(images[j])
                if pcd is not None:
                    images[j] = project_point_cloud_to_image(images[j], pcd, K_undistorted, cTv, vTl, None)
                if annotations_to_images:
                    images[j] = annotation_to_image(images[j], annotation_data, vTl, cTv, K_undistorted, None)
            else:
                if pcd is not None:
                    images[j] = project_point_cloud_to_image(images[j], pcd, K, cTv, vTl, D)
                if annotations_to_images:
                    images[j] = annotation_to_image(images[j], annotation_data, vTl, cTv, K, D)
        else:
            images[j] = np.zeros(shape=(1080, 1920, 3), dtype=np.uint8)

    # Resize and combine images
    frame = plot_images_combined(images, rows=rows, out_path=None, img_size=img_size)

    return frame


def process_lidar_camera_chunk(indices, sync_data, calibration_data, annotation_data, base_path, camera_names, lidar_name,
                  camera_steps, lidar_w, lidar_h, left_right_range, forward_backward_range,
                  res, z_clip, undistort_images, annotations_to_images, frame_queue, progress_queue):
    try:
        for i in indices:
            frame = process_lidar_camera_frame(i, sync_data, calibration_data, annotation_data, base_path, camera_names, lidar_name,
                                  camera_steps, lidar_w, lidar_h, left_right_range, forward_backward_range,
                                  res, z_clip, undistort_images, annotations_to_images)
            frame_queue.put((i, frame))   # send frame to writer
            progress_queue.put(1)         # update progress
    except Exception as e:
        print(f"Error processing chunk: {e}")
    finally:
        frame_queue.put(None)  # signal this worker is done


def process_camera_chunk(indices, sync_data, calibration_data, annotation_data, base_path, camera_names,
                         undistort_images, annotations_to_images, frame_queue, progress_queue, img_size, rows,
                         lidar_name, points_to_image):
    try:
        for i in indices:
            frame = process_camera_frame(i, sync_data, calibration_data, annotation_data, base_path, camera_names,
                                         undistort_images, annotations_to_images, img_size, rows, lidar_name, points_to_image)
            frame_queue.put((i, frame))   # send frame to writer
            progress_queue.put(1)         # update progress
    except Exception as e:
        print(f"Error processing chunk: {e}")
    finally:
        frame_queue.put(None)  # signal this worker is done


def frame_writer(frame_queue, output_file_path, total_frames, video_size, fps, num_workers):
    writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size)
    next_frame_idx = 0
    buffer = {}
    finished_workers = 0

    while True:
        item = frame_queue.get()
        if item is None:
            finished_workers += 1
            if finished_workers == num_workers:
                break
            continue

        idx, frame = item
        buffer[idx] = frame

        # Write frames in order
        while next_frame_idx in buffer:
            writer.write(buffer.pop(next_frame_idx))
            next_frame_idx += 1

    writer.release()


def monitor_progress(progress_queue, total_frames):
    with tqdm.tqdm(total=total_frames, desc="Processing video frames") as pbar:
        for _ in range(total_frames):
            progress_queue.get()
            pbar.update(1)


def chunk_indices_with_min_size(lst, max_workers, min_chunk_size):
    """
    Teilt `lst` in Chunks auf, sodass:
    - Die Anzahl der Prozesse höchstens `max_workers` ist.
    - Kein Chunk kleiner als `min_chunk_size`.
    """
    total_items = len(lst)

    if total_items <= min_chunk_size:
        return [lst]

    max_chunks = total_items // min_chunk_size
    num_chunks = min(max_workers, max_chunks)
    if num_chunks == 0:
        num_chunks = 1  # fallback für sehr kleine Listen
    k, m = divmod(total_items, num_chunks)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks)]


def process_chunk_lidar(args):
    return process_lidar_camera_chunk(*args)

def process_chunk_camera(args):
    return process_camera_chunk(*args)


def create_lidar_camera_video(
        base_path, output_file_path, annotation_file_path=None, fps=10,
        left_right_range=(-50, 50), forward_backward_range=(-100, 100),
        res=0.1, z_clip=(-3, 5), undistort_images=True, annotations_to_images=False,
        num_workers=0, min_chunk_size=50):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Load metadata
    timesync_file_path = os.path.join(base_path, 'timesync_info.csv')
    calibration_file_path = os.path.join(base_path, 'calibration.json')

    sync_data = read_timesync_file(timesync_file_path)
    calibration_data = load_calibration_file(calibration_file_path)
    annotation_data = load_annotation_data(annotation_file_path)
    annotation_data = convert_labels_from_tracks_wise_to_timestamp_wise(annotation_data)

    camera_names = [
        'back_left_camera', 'left_camera', 'front_left_camera',
        'front_right_camera', 'right_camera', 'back_right_camera'
    ]
    lidar_name = 'middle_lidar'

    assert all(cam in calibration_data['cameras'] for cam in camera_names)
    assert lidar_name in calibration_data['lidars']
    assert all(cam in sync_data for cam in camera_names)
    assert lidar_name in sync_data

    total_frames = len(sync_data['timestamp_nanoseconds'])
    frame_indices = list(range(total_frames))

    video_image_size = (1080, 1920, 3)
    cam_h = video_image_size[0] // 3
    cam_w = video_image_size[1] // len(camera_names)
    lidar_h = cam_h * 2
    lidar_w = video_image_size[1]
    camera_steps = (cam_w, cam_h)

    n_workers = num_workers or os.cpu_count()
    chunks = chunk_indices_with_min_size(frame_indices, n_workers, min_chunk_size=min_chunk_size)
    n_workers = min(n_workers, len(chunks))  # <- Final verwendete Anzahl an Prozessen

    with Manager() as manager:
        frame_queue = manager.Queue(maxsize=1000)
        progress_queue = manager.Queue()

        # Start monitor and writer processes as before...
        monitor_proc = Process(target=monitor_progress, args=(progress_queue, total_frames))
        writer_proc = Process(target=frame_writer, args=(frame_queue, output_file_path, total_frames,
                                                        (video_image_size[1], video_image_size[0]), fps, n_workers))
        monitor_proc.start()
        writer_proc.start()

        with Pool(processes=n_workers) as pool:
            # Prepare arguments per chunk for pool.map or pool.starmap
            args_list = [(chunk, sync_data, calibration_data, annotation_data, base_path, camera_names, lidar_name,
                          camera_steps, lidar_w, lidar_h, left_right_range, forward_backward_range,
                          res, z_clip, undistort_images, annotations_to_images, frame_queue, progress_queue) for chunk in chunks]

            pool.map(process_chunk_lidar, args_list)

        # Wait for workers and monitor
        writer_proc.join()
        monitor_proc.join()
    
    print(f'Lidar-camera video created: {output_file_path}.')


def create_camera_video(
        base_path, output_file_path, annotation_file_path=None, fps=10,
        undistort_images=True, annotations_to_images=False, points_to_image=False,
        num_workers=0, min_chunk_size=50,
        img_size=(1080, 1920)):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Load metadata
    timesync_file_path = os.path.join(base_path, 'timesync_info.csv')
    calibration_file_path = os.path.join(base_path, 'calibration.json')

    sync_data = read_timesync_file(timesync_file_path)
    calibration_data = load_calibration_file(calibration_file_path)
    annotation_data = load_annotation_data(annotation_file_path)
    annotation_data = convert_labels_from_tracks_wise_to_timestamp_wise(annotation_data)

    camera_names = [
        'back_left_camera', 'left_camera', 'front_left_camera',
        'front_right_camera', 'right_camera', 'back_right_camera'
    ]
    lidar_name = 'middle_lidar'
    rows = 2

    assert all(cam in calibration_data['cameras'] for cam in camera_names)
    assert all(cam in sync_data for cam in camera_names)

    total_frames = len(sync_data['timestamp_nanoseconds'])
    frame_indices = list(range(total_frames))

    n_workers = num_workers or os.cpu_count()
    chunks = chunk_indices_with_min_size(frame_indices, n_workers, min_chunk_size=min_chunk_size)
    n_workers = min(n_workers, len(chunks))  # <- Final verwendete Anzahl an Prozessen

    with Manager() as manager:
        frame_queue = manager.Queue(maxsize=1000)
        progress_queue = manager.Queue()

        # Start monitor and writer processes as before...
        monitor_proc = Process(target=monitor_progress, args=(progress_queue, total_frames))
        writer_proc = Process(target=frame_writer, args=(frame_queue, output_file_path, total_frames,
                                                        (img_size[1], img_size[0]), fps, n_workers))
        monitor_proc.start()
        writer_proc.start()

        with Pool(processes=n_workers) as pool:
            # Prepare arguments per chunk for pool.map or pool.starmap
            args_list = [(chunk, sync_data, calibration_data, annotation_data, base_path, camera_names,
                          undistort_images, annotations_to_images, frame_queue, progress_queue, img_size, rows, lidar_name, points_to_image) for chunk in chunks]

            pool.map(process_chunk_camera, args_list)

        # Wait for workers and monitor
        writer_proc.join()
        monitor_proc.join()
    
    print(f'Camera-only video created: {output_file_path}.')


def create_video_static(output_file_path, fps=10, left_right_range=(-50, 50), forward_backward_range=(-100, 100), res=0.1, z_clip=(-3, 5)):
    "Made for 1 lidar and 6 images. An example usage is creating a video during reading a rosbag."

    # Initialize video writer BEFORE loop
    video_image_size = (1080, 1920, 3)  # h, w, c
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (video_image_size[1], video_image_size[0]))

    cam_h = video_image_size[0] // 3
    cam_w = video_image_size[1] // 6
    camera_steps = (cam_w, cam_h)  # w, h

    lidar_h = cam_h * 2
    lidar_w = video_image_size[1]

    def write_frame(lidar_pcd, images):
        lidar_bev = create_point_cloud_bev_image(
            lidar_pcd,
            left_right_range=left_right_range, forward_backward_range=forward_backward_range,
            res=res, z_clip=z_clip
        )
        lidar_bev = cv2.resize(lidar_bev, (lidar_w, lidar_h))

        images = [cv2.resize(img, camera_steps) if img is not None else np.zeros((cam_h, cam_w, 3), dtype=np.uint8) for img in images]

        combined_frame = np.zeros(shape=video_image_size, dtype=np.uint8)

        for idx, img in enumerate(images):
            x_start = idx * cam_w
            combined_frame[0:cam_h, x_start:x_start+cam_w] = img
        
        combined_frame[cam_h:, :] = lidar_bev

        writer.write(combined_frame)
    
    def close_writer():
        writer.release()
        print(f"Saved combined lidar-camera video to: {output_file_path}.")

    return write_frame, close_writer
