#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
import argparse
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Button, Slider

# -----------------------------------------
# Global Navigation State
# -----------------------------------------
current_frame = 0
paused = True
show_names = True

# -----------------------------------------
# Hard-coded Camera Calibration
# -----------------------------------------
# Camera0 (front)
K0 = np.array([
    [229.75032642,   0.,         200.44588524],
    [  0.,         169.79368238, 246.51861282],
    [  0.,           0.,           1.        ]
])
dist0 = np.array([[-0.14493007,  0.03117784,  0.09789049, -0.290851,  0.17057335]])

# Camera1 (side)
K1 = np.array([
    [185.66584519,   0.,         206.46116634],
    [  0.,         159.65919947, 216.49924977],
    [  0.,           0.,           1.        ]
])
dist1 = np.array([[-0.04283384, 0.04385798, -0.05661369, 0.03800788, -0.02296988]])

# Stereo calibration (Camera0 -> Camera1)
R = np.array([
    [ 0.07658793, -0.24328972,  0.96692523],
    [-0.42846919,  0.86763286,  0.25224466],
    [-0.90030464, -0.43361656, -0.03779194]
])
T = np.array([[-124.13846774],
              [ -20.27274023],
              [  14.12344011]])

# Compute Projection Matrices:
P0 = K0 @ np.hstack((np.eye(3), np.zeros((3,1))))
P1 = K1 @ np.hstack((R, T))

# -----------------------------------------
# Landmark Triangulation (DLT)
# -----------------------------------------
def triangulate_point(PA, PB, xA, yA, xB, yB):
    """
    DLT-based triangulation of one landmark from two cameras.
    (xA, yA) are coordinates in camera0, (xB, yB) in camera1.
    """
    A = [
        (yA * PA[2, :] - PA[1, :]),
        (PA[0, :] - xA * PA[2, :]),
        (yB * PB[2, :] - PB[1, :]),
        (PB[0, :] - xB * PB[2, :])
    ]
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1, :]
    X /= X[-1]
    return X[:3]

# -----------------------------------------
# Dog-Skeleton Connection Functions
# -----------------------------------------
def draw_dog_3d(data_3d):
    rules = [
        {"parts": ['nose', 'left_eye', 'right_eye', 'nose'], "color": 'grey'},
        {"parts": ['left_eye', 'left_ear_bottom', 'right_ear_bottom', 'right_eye'], "color": 'grey'},
        {"parts": ['left_ear_bottom', 'left_ear_middle', 'left_ear_top'], "color": 'grey'},
        {"parts": ['right_ear_bottom', 'right_ear_middle', 'right_ear_top'], "color": 'grey'},
        {"parts": ['nose', 'neck'], "color": 'cyan'},
        {"parts": ['neck', 'tail_bottom'], "color": 'black'},
        {"parts": ['tail_bottom', 'tail_middle', 'tail_top'], "color": 'pink'},
    ]
    limb_colors = {
        "right_front": "blue",
        "left_front": "orange",
        "right_back": "green",
        "left_back": "purple",
    }
    for d1 in ["right", "left"]:
        for d2p1 in [["front", "neck"], ["back", "tail_bottom"]]:
            current_limb = [d2p1[1]]
            for d3 in ["elbow", "knee", "paw"]:
                current_limb.append(f"{d1}_{d2p1[0]}_{d3}")
            color_key = f"{d1}_{d2p1[0]}"
            color = limb_colors.get(color_key, "black")
            rules.append({"parts": current_limb, "color": color})
    segments = []
    for rule in rules:
        xs, ys, zs = [], [], []
        missing = False
        for part in rule["parts"]:
            if part not in data_3d:
                missing = True
                break
            pt = data_3d[part]
            xs.append(pt[0])
            ys.append(pt[1])
            zs.append(pt[2])
        if not missing:
            segments.append((xs, ys, zs, rule["color"]))
    return segments

def draw_dog_2d(row, view, ax):
    prefix = "front_" if view == "front" else "side_"
    rules = [
        {"parts": ['nose', 'left_eye', 'right_eye', 'nose'], "color": 'grey'},
        {"parts": ['left_eye', 'left_ear_bottom', 'right_ear_bottom', 'right_eye'], "color": 'grey'},
        {"parts": ['left_ear_bottom', 'left_ear_middle', 'left_ear_top'], "color": 'grey'},
        {"parts": ['right_ear_bottom', 'right_ear_middle', 'right_ear_top'], "color": 'grey'},
        {"parts": ['nose', 'neck'], "color": 'cyan'},
        {"parts": ['neck', 'tail_bottom'], "color": 'black'},
        {"parts": ['tail_bottom', 'tail_middle', 'tail_top'], "color": 'pink'},
    ]
    limb_colors = {
        "right_front": "blue",
        "left_front": "orange",
        "right_back": "green",
        "left_back": "purple",
    }
    for d1 in ["right", "left"]:
        for d2p1 in [["front", "neck"], ["back", "tail_bottom"]]:
            current_limb = [d2p1[1]]
            for d3 in ["elbow", "knee", "paw"]:
                current_limb.append(f"{d1}_{d2p1[0]}_{d3}")
            color_key = f"{d1}_{d2p1[0]}"
            color = limb_colors.get(color_key, "black")
            rules.append({"parts": current_limb, "color": color})
    for rule in rules:
        xs, ys = [], []
        missing = False
        for part in rule["parts"]:
            x_col = f"{prefix}{part}_x"
            y_col = f"{prefix}{part}_y"
            if x_col not in row.index or y_col not in row.index:
                missing = True
                break
            valx = row[x_col]
            valy = row[y_col]
            if pd.isna(valx) or pd.isna(valy):
                missing = True
                break
            xs.append(valx)
            ys.append(valy)
        if not missing:
            ax.plot(xs, ys, color=rule["color"], linewidth=2)

# -----------------------------------------
# Triangulate 3D Landmarks for a Frame
# -----------------------------------------
def process_frame_3d(row, landmarks):
    data_3d = {}
    for lm in landmarks:
        fx_col = f"front_{lm}_x"
        fy_col = f"front_{lm}_y"
        sx_col = f"side_{lm}_x"
        sy_col = f"side_{lm}_y"
        if fx_col in row.index and fy_col in row.index and sx_col in row.index and sy_col in row.index:
            fx = row[fx_col]
            fy = row[fy_col]
            sx = row[sx_col]
            sy = row[sy_col]
            if not (pd.isna(fx) or pd.isna(fy) or pd.isna(sx) or pd.isna(sy)):
                X = triangulate_point(P0, P1, fx, fy, sx, sy)
                data_3d[lm] = X
    return data_3d

# -----------------------------------------
# Update Plots for a Given Frame
# -----------------------------------------
def update_all_plots(ax3d, ax_front, ax_side, row, landmarks, num_frame, num_frames):
    # --- 3D Plot ---
    ax3d.cla()
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    data_3d = process_frame_3d(row, landmarks)
    if data_3d:
        for lm, pt in data_3d.items():
            ax3d.scatter(pt[0], pt[1], pt[2], c='b', marker='o')
            if show_names:
                ax3d.text(pt[0], pt[1], pt[2], lm, fontsize=8, color='black')
        segments = draw_dog_3d(data_3d)
        for xs, ys, zs, color in segments:
            ax3d.plot(xs, ys, zs, color=color, linewidth=2)
    else:
        ax3d.text(0.5, 0.5, 0.5, "No 3D Data", fontsize=12)
    ax3d.set_title(f"3D Reconstruction\nFrame {num_frame}/{num_frames}")

    # --- Front 2D Plot ---
    ax_front.cla()
    ax_front.set_xlabel("X (Front)")
    ax_front.set_ylabel("Y (Front)")
    ax_front.set_xlim(0, 1)
    ax_front.set_ylim(0, 1)
    for lm in landmarks:
        x_key = f"front_{lm}_x"
        y_key = f"front_{lm}_y"
        if x_key in row and y_key in row:
            x_val = row[x_key]
            y_val = row[y_key]
            if not (pd.isna(x_val) or pd.isna(y_val)):
                ax_front.scatter(x_val, y_val, c='r', marker='o')
                if show_names:
                    ax_front.text(x_val, y_val, lm, fontsize=8)
    draw_dog_2d(row, "front", ax_front)
    ax_front.set_title("Front View")

    # --- Side 2D Plot ---
    ax_side.cla()
    ax_side.set_xlabel("X (Side)")
    ax_side.set_ylabel("Y (Side)")
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, 1)
    for lm in landmarks:
        x_key = f"side_{lm}_x"
        y_key = f"side_{lm}_y"
        if x_key in row and y_key in row:
            x_val = row[x_key]
            y_val = row[y_key]
            if not (pd.isna(x_val) or pd.isna(y_val)):
                ax_side.scatter(x_val, y_val, c='r', marker='o')
                if show_names:
                    ax_side.text(x_val, y_val, lm, fontsize=8)
    draw_dog_2d(row, "side", ax_side)
    ax_side.set_title("Side View")

# -----------------------------------------
# Key & Timer Callbacks
# -----------------------------------------
def render_frame(df, landmarks, ax3d, ax_front, ax_side, num_frames, slider=None):
    global current_frame
    row = df.iloc[current_frame]
    update_all_plots(ax3d, ax_front, ax_side, row, landmarks, current_frame+1, num_frames)
    if slider:
        slider.eventson = False
        slider.set_val(current_frame)
        slider.eventson = True
    plt.draw()

def on_key(event, df, landmarks, ax3d, ax_front, ax_side, num_frames, slider):
    global current_frame, paused
    if event.key == 'down':
        if current_frame < num_frames - 1:
            current_frame += 1
            render_frame(df, landmarks, ax3d, ax_front, ax_side, num_frames, slider)
    elif event.key == 'up':
        if current_frame > 0:
            current_frame -= 1
            render_frame(df, landmarks, ax3d, ax_front, ax_side, num_frames, slider)
    elif event.key == ' ':
        paused = not paused
        print("Paused" if paused else "Resumed")

def timer_event(df, landmarks, ax3d, ax_front, ax_side, num_frames, slider):
    global current_frame, paused
    if not paused:
        if current_frame < num_frames - 1:
            current_frame += 1
        else:
            current_frame = 0
        render_frame(df, landmarks, ax3d, ax_front, ax_side, num_frames, slider)

# -----------------------------------------
# Main Interactive Code
# -----------------------------------------
def main():
    global current_frame, paused
    parser = argparse.ArgumentParser(
        description="3D Dog Reconstruction with pre-calibrated cameras and dog skeleton."
    )
    parser.add_argument("csv_file", help="Path to the CSV file with landmark data.")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        sys.exit(f"Error reading CSV file: {e}")

    num_frames = len(df)
    print(f"Loaded {num_frames} frames from '{args.csv_file}'.")
    print("Use Up/Down arrow keys, space to pause/resume, or the slider to navigate frames.")

    # Define landmarks (CSV columns: front_{lm}_x, front_{lm}_y, side_{lm}_x, side_{lm}_y)
    landmarks = [
        "nose", "left_eye", "right_eye",
        "left_ear_top", "right_ear_top",
        "left_ear_middle", "right_ear_middle",
        "left_ear_bottom", "right_ear_bottom",
        "neck", "tail_bottom", "tail_middle", "tail_top",
        "right_front_elbow", "right_front_knee", "right_front_paw",
        "left_front_elbow", "left_front_knee", "left_front_paw",
        "right_back_elbow", "right_back_knee", "right_back_paw",
        "left_back_elbow", "left_back_knee", "left_back_paw"
    ]

    # We want 2D plots of 360x480 pixels and a larger 3D plot.
    # At 100 dpi, 360px = 3.6 in and 480px = 4.8 in.
    # We'll create a figure that is 1080x960 pixels (10.8x9.6 inches) in total.
    fig = plt.figure(figsize=(10.8, 9.6), dpi=100)

    # Define positions in normalized coordinates:
    # Left column: width = 360/1080 = 0.3333, split vertically into two 2D plots (each 480/960 = 0.5 height).
    ax_front = fig.add_axes([0.0, 0.5, 0.3333, 0.5])  # Top-left (Front view 2D)
    ax_side = fig.add_axes([0.0, 0.0, 0.3333, 0.5])   # Bottom-left (Side view 2D)

    # Right column: occupies the remaining area (720/1080 = 0.6667 width, full height) for the 3D plot.
    ax3d = fig.add_axes([0.3333, 0.0, 0.6667, 1.0], projection='3d')

    # Create slider axes (positioned above the bottom of the 3D plot)
    slider_ax = fig.add_axes([0.35, 0.02, 0.3, 0.03])
    frame_slider = Slider(
        slider_ax,
        'Frame',
        0,
        num_frames - 1,
        valinit=current_frame,
        valstep=1,
        valfmt='%0.0f'
    )

    def slider_update(val):
        global current_frame
        current_frame = int(frame_slider.val)
        render_frame(df, landmarks, ax3d, ax_front, ax_side, num_frames, frame_slider)

    frame_slider.on_changed(slider_update)

    render_frame(df, landmarks, ax3d, ax_front, ax_side, num_frames, frame_slider)

    # Connect key press events.
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, df, landmarks, ax3d, ax_front, ax_side, num_frames, frame_slider))
    timer = fig.canvas.new_timer(interval=200)
    timer.add_callback(timer_event, df, landmarks, ax3d, ax_front, ax_side, num_frames, frame_slider)
    timer.start()

    # Create toggle button for names.
    button_ax = fig.add_axes([0.70, 0.02, 0.1, 0.05])
    toggle_button = Button(button_ax, "Toggle Names")

    def toggle_names(event):
        global show_names
        show_names = not show_names
        render_frame(df, landmarks, ax3d, ax_front, ax_side, num_frames, frame_slider)

    toggle_button.on_clicked(toggle_names)

    plt.show()

if __name__ == "__main__":
    main()
