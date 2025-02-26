import cv2 as cv
import numpy as np
import sys
from scipy import linalg
import yaml
import os

# Global variable for calibration settings loaded from a YAML file.
calibration_settings = {}

# Given projection matrices P1 and P2, and pixel coordinates point1 and point2,
# return the triangulated 3D point.
def DLT(P1, P2, point1, point2):
    A = [point1[1]*P1[2, :] - P1[1, :],
         P1[0, :] - point1[0]*P1[2, :],
         point2[1]*P2[2, :] - P2[1, :],
         P2[0, :] - point2[0]*P2[2, :]]
    A = np.array(A).reshape((4, 4))
    B = A.T @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]

# Open and load the calibration_settings.yaml file.
def parse_calibration_settings_file(filename):
    global calibration_settings
    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    
    print('Using calibration settings from:', filename)
    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)
    
    # Rudimentary check to ensure the YAML file looks valid.
    if 'checkerboard_rows' not in calibration_settings.keys():
        print('The settings file appears to be missing required keys (e.g., "checkerboard_rows").')
        quit()

# This function processes both videos in sync and saves a side-by-side image for every
# sampled frame in which at least one checkerboard is found.
def save_checkerboard_detection_frames(video_path0, video_path1, output_folder, frame_sample_interval=30):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    cap0 = cv.VideoCapture(video_path0)
    cap1 = cv.VideoCapture(video_path1)
    if not cap0.isOpened() or not cap1.isOpened():
        print("Error opening one of the video files.")
        return

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
    frame_idx = 0
    saved_count = 0
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 and not ret1:
            break  # End if both videos have ended.
        # If one video ends before the other, use a black image.
        if not ret0:
            frame0 = np.zeros_like(frame1)
        if not ret1:
            frame1 = np.zeros_like(frame0)
        
        if frame_idx % frame_sample_interval == 0:
            # Process first video.
            gray0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
            ret_cb0, corners0 = cv.findChessboardCorners(gray0, (rows, columns), None)
            if ret_cb0:
                frame0_disp = frame0.copy()
                corners0 = cv.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
                cv.drawChessboardCorners(frame0_disp, (rows, columns), corners0, ret_cb0)
            else:
                frame0_disp = frame0  # Use original frame if no detection.
            
            # Process second video.
            gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            ret_cb1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
            if ret_cb1:
                frame1_disp = frame1.copy()
                corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                cv.drawChessboardCorners(frame1_disp, (rows, columns), corners1, ret_cb1)
            else:
                frame1_disp = frame1  # Use original frame if no detection.
            
            # Only save the frame if at least one checkerboard was found.
            if ret_cb0 or ret_cb1:
                combined = cv.hconcat([frame0_disp, frame1_disp])
                filename = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
                cv.imwrite(filename, combined)
                saved_count += 1

        frame_idx += 1

    cap0.release()
    cap1.release()
    print("Saved", saved_count, "checkerboard detection frames to folder:", output_folder)

# Given two synchronized video files, extract calibration frames from both videos.
def calibrate_from_videos(video_path0, video_path1, frame_sample_interval=30):
    cap0 = cv.VideoCapture(video_path0)
    cap1 = cv.VideoCapture(video_path1)
    if not cap0.isOpened() or not cap1.isOpened():
        print("Error opening one of the video files.")
        quit()
    
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
    # Prepare object points for the checkerboard corners.
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    objpoints_cam0, imgpoints_cam0 = [], []
    objpoints_cam1, imgpoints_cam1 = [], []
    objpoints_stereo, imgpoints_left, imgpoints_right = [], [], []
    
    frame_idx = 0
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            break
        
        if frame_idx % frame_sample_interval == 0:
            gray0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
            gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            
            ret_cb0, corners0 = cv.findChessboardCorners(gray0, (rows, columns), None)
            ret_cb1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
            
            if ret_cb0:
                corners0 = cv.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
                objpoints_cam0.append(objp)
                imgpoints_cam0.append(corners0)
            else:
                print(f"Checkerboard not detected in video0 frame {frame_idx}")
            
            if ret_cb1:
                corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                objpoints_cam1.append(objp)
                imgpoints_cam1.append(corners1)
            else:
                print(f"Checkerboard not detected in video1 frame {frame_idx}")
            
            if ret_cb0 and ret_cb1:
                objpoints_stereo.append(objp)
                imgpoints_left.append(corners0)
                imgpoints_right.append(corners1)
        frame_idx += 1

    cap0.release()
    cap1.release()
    
    if len(objpoints_cam0) < 1 or len(objpoints_cam1) < 1:
        print("Insufficient calibration frames detected in one or both videos.")
        quit()
    
    img_shape = gray0.shape[::-1]  # (width, height)

    ret0, cmtx0, dist0, rvecs0, tvecs0 = cv.calibrateCamera(objpoints_cam0, imgpoints_cam0, img_shape, None, None)
    print("Camera0 intrinsic calibration RMSE:", ret0)
    ret1, cmtx1, dist1, rvecs1, tvecs1 = cv.calibrateCamera(objpoints_cam1, imgpoints_cam1, img_shape, None, None)
    print("Camera1 intrinsic calibration RMSE:", ret1)
    
    if len(objpoints_stereo) < 1:
        print("Insufficient stereo calibration pairs detected.")
        quit()
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret_stereo, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(
        objpoints_stereo, imgpoints_left, imgpoints_right,
        cmtx0, dist0, cmtx1, dist1, img_shape,
        criteria=criteria, flags=stereocalibration_flags)
    print("Stereo calibration RMSE:", ret_stereo)
    
    return cmtx0, dist0, cmtx1, dist1, R, T

# Converts a rotation matrix R and translation vector T into a homogeneous representation matrix.
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1
    return P

# Turn camera calibration data into a projection matrix.
def get_projection_matrix(cmtx, R, T):
    return cmtx @ _make_homogeneous_rep_matrix(R, T)[:3, :]

if __name__ == '__main__':
    # Expected usage:
    # python3 calibrate.py calibration_settings.yaml <video_path0> <video_path1>
    if len(sys.argv) != 4:
        print("Usage: python3 calibrate.py calibration_settings.yaml <video_path0> <video_path1>")
        quit()
    
    settings_file = sys.argv[1]
    video_path0 = sys.argv[2]
    video_path1 = sys.argv[3]
    
    parse_calibration_settings_file(settings_file)
    
    # Use frame_sample_interval from YAML if available, default to 30.
    frame_sample_interval = calibration_settings.get('video_frame_interval', 30)
    
    # Save side-by-side frames where at least one checkerboard is found.
    output_folder = "checkerboard_frames"
    save_checkerboard_detection_frames(video_path0, video_path1, output_folder, frame_sample_interval)
    
    # Next, perform calibration.
    cmtx0, dist0, cmtx1, dist1, R, T = calibrate_from_videos(video_path0, video_path1, frame_sample_interval)
    
    # Save calibration parameters.
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')
    def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):
        out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
        with open(out_filename, 'w') as outf:
            outf.write('Intrinsic:\n')
            for row in camera_matrix:
                outf.write(' '.join(map(str, row)) + '\n')
            outf.write('Distortion:\n')
            outf.write(' '.join(map(str, distortion_coefs[0])) + '\n')
    save_camera_intrinsics(cmtx0, dist0, 'camera0')
    save_camera_intrinsics(cmtx1, dist1, 'camera1')
    
    def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix=''):
        cam0_file = os.path.join('camera_parameters', prefix + 'camera0_rot_trans.dat')
        with open(cam0_file, 'w') as outf:
            outf.write('R:\n')
            for row in R0:
                outf.write(' '.join(map(str, row)) + '\n')
            outf.write('T:\n')
            outf.write(' '.join(map(str, T0.flatten())) + '\n')
        cam1_file = os.path.join('camera_parameters', prefix + 'camera1_rot_trans.dat')
        with open(cam1_file, 'w') as outf:
            outf.write('R:\n')
            for row in R1:
                outf.write(' '.join(map(str, row)) + '\n')
            outf.write('T:\n')
            outf.write(' '.join(map(str, T1.flatten())) + '\n')
        return R0, T0, R1, T1
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.zeros((3, 1), dtype=np.float32)
    save_extrinsic_calibration_parameters(R0, T0, R, T)
    R1, T1 = R, T
    
    # Display calibration parameters.
    print("\n--- Calibration Parameters ---")
    print("Camera0 Intrinsic Matrix:\n", cmtx0)
    print("Camera0 Distortion Coefficients:\n", dist0)
    print("Camera1 Intrinsic Matrix:\n", cmtx1)
    print("Camera1 Distortion Coefficients:\n", dist1)
    print("Stereo Rotation Matrix (Camera0 -> Camera1):\n", R)
    print("Stereo Translation Vector (Camera0 -> Camera1):\n", T)
    print("------------------------------\n")
