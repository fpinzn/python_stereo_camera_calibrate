import json
import csv

# Define the mapping for each file.
# For each landmark you want to extract, specify the NDJSON key.
# For instance, for front video:
front_map = {
    "left_eye": "animal_joint_left_eye",
    "left_ear_middle": "animal_joint_left_ear_middle",
    "right_eye": "animal_joint_right_eye",
    "right_ear_top": "animal_joint_right_ear_top",
    "right_ear_middle": "animal_joint_right_ear_middle",
    "nose": "animal_joint_nose",
    "left_ear_top": "animal_joint_left_ear_top",
    "right_ear_bottom": "animal_joint_right_ear_bottom",
    "right_back_paw": "animal_joint_right_back_paw",
    "tail_top": "animal_joint_tail_top",
    "left_back_elbow": "animal_joint_left_back_elbow",
    "left_front_knee": "animal_joint_left_front_knee",
    "left_back_knee": "animal_joint_left_back_knee",
    "right_front_paw": "animal_joint_right_front_paw",
    "left_front_paw": "animal_joint_left_front_paw",
    "left_ear_bottom": "animal_joint_left_ear_bottom",
    # The following are additional landmarks – adjust as needed.
    "tail_middle": "animal_joint_tail_middle",
    "right_front_elbow": "animal_joint_right_front_elbow",
    "neck": "animal_joint_heck",  # if "heck" corresponds to the neck
    "tail_bottom": "animal_joint_tail_bottom",
    "right_back_elbow": "animal_joint_right_back_elbow"
}

# Similarly, for the side video.
side_map = {
    "left_eye": "animal_joint_left_eye",
    "left_ear_middle": "animal_joint_left_ear_middle",
    "right_eye": "animal_joint_right_eye",
    "right_ear_top": "animal_joint_right_ear_top",
    "right_ear_middle": "animal_joint_right_ear_middle",
    "nose": "animal_joint_nose",
    "left_ear_top": "animal_joint_left_ear_top",
    "right_ear_bottom": "animal_joint_right_ear_bottom",
    "right_back_paw": "animal_joint_right_back_paw",
    "tail_top": "animal_joint_tail_top",
    "left_back_elbow": "animal_joint_left_back_elbow",
    "front_left_knee": "animal_joint_left_front_knee",  # note: adjust name if needed
    "left_back_knee": "animal_joint_left_back_knee",
    "front_right_paw": "animal_joint_right_front_paw",
    "front_left_paw": "animal_joint_left_front_paw",
    "left_ear_bottom": "animal_joint_left_ear_bottom",
    # Additional landmarks for side – adjust as needed.
    "tail_middle": "animal_joint_tail_middle",
    "front_right_elbow": "animal_joint_right_front_elbow",
    "neck": "animal_joint_heck",
    "tail_bottom": "animal_joint_tail_bottom",
    "right_back_elbow": "animal_joint_right_back_elbow"
}

# Define the complete CSV header as provided.
header = [
    "absolute_time",
    "front_left_eye_x", "front_left_eye_y", "front_left_eye_c",
    "front_left_ear_middle_x", "front_left_ear_middle_y", "front_left_ear_middle_c",
    "front_right_eye_x", "front_right_eye_y", "front_right_eye_c",
    "front_right_ear_top_x", "front_right_ear_top_y", "front_right_ear_top_c",
    "front_right_ear_middle_x", "front_right_ear_middle_y", "front_right_ear_middle_c",
    "front_nose_x", "front_nose_y", "front_nose_c",
    "front_left_ear_top_x", "front_left_ear_top_y", "front_left_ear_top_c",
    "front_right_ear_bottom_x", "front_right_ear_bottom_y", "front_right_ear_bottom_c",
    "front_right_back_paw_x", "front_right_back_paw_y", "front_right_back_paw_c",
    "front_tail_top_x", "front_tail_top_y", "front_tail_top_c",
    "front_left_back_elbow_x", "front_left_back_elbow_y", "front_left_back_elbow_c",
    "front_left_knee_x", "front_left_knee_y", "front_left_knee_c",
    "front_left_back_knee_x", "front_left_back_knee_y", "front_left_back_knee_c",
    "front_right_paw_x", "front_right_paw_y", "front_right_paw_c",
    "front_left_paw_x", "front_left_paw_y", "front_left_paw_c",
    "front_left_ear_bottom_x", "front_left_ear_bottom_y", "front_left_ear_bottom_c",
    "front_left_back_paw_x", "front_left_back_paw_y", "front_left_back_paw_c",
    "front_right_knee_x", "front_right_knee_y", "front_right_knee_c",
    "front_right_back_knee_x", "front_right_back_knee_y", "front_right_back_knee_c",
    "front_left_elbow_x", "front_left_elbow_y", "front_left_elbow_c",
    "front_tail_middle_x", "front_tail_middle_y", "front_tail_middle_c",
    "front_right_elbow_x", "front_right_elbow_y", "front_right_elbow_c",
    "front_neck_x", "front_neck_y", "front_neck_c",
    "front_tail_bottom_x", "front_tail_bottom_y", "front_tail_bottom_c",
    "front_right_back_elbow_x", "front_right_back_elbow_y", "front_right_back_elbow_c",
    "side_distance",
    "side_left_eye_x", "side_left_eye_y", "side_left_eye_c",
    "side_left_ear_middle_x", "side_left_ear_middle_y", "side_left_ear_middle_c",
    "side_right_eye_x", "side_right_eye_y", "side_right_eye_c",
    "side_right_ear_top_x", "side_right_ear_top_y", "side_right_ear_top_c",
    "side_right_ear_middle_x", "side_right_ear_middle_y", "side_right_ear_middle_c",
    "side_nose_x", "side_nose_y", "side_nose_c",
    "side_left_ear_top_x", "side_left_ear_top_y", "side_left_ear_top_c",
    "side_right_ear_bottom_x", "side_right_ear_bottom_y", "side_right_ear_bottom_c",
    "side_right_back_paw_x", "side_right_back_paw_y", "side_right_back_paw_c",
    "side_tail_top_x", "side_tail_top_y", "side_tail_top_c",
    "side_left_back_elbow_x", "side_left_back_elbow_y", "side_left_back_elbow_c",
    "side_front_left_knee_x", "side_front_left_knee_y", "side_front_left_knee_c",
    "side_left_back_knee_x", "side_left_back_knee_y", "side_left_back_knee_c",
    "side_front_right_paw_x", "side_front_right_paw_y", "side_front_right_paw_c",
    "side_front_left_paw_x", "side_front_left_paw_y", "side_front_left_paw_c",
    "side_left_ear_bottom_x", "side_left_ear_bottom_y", "side_left_ear_bottom_c",
    "side_left_back_paw_x", "side_left_back_paw_y", "side_left_back_paw_c",
    "side_front_right_knee_x", "side_front_right_knee_y", "side_front_right_knee_c",
    "side_right_back_knee_x", "side_right_back_knee_y", "side_right_back_knee_c",
    "side_front_left_elbow_x", "side_front_left_elbow_y", "side_front_left_elbow_c",
    "side_tail_middle_x", "side_tail_middle_y", "side_tail_middle_c",
    "side_front_right_elbow_x", "side_front_right_elbow_y", "side_front_right_elbow_c",
    "side_neck_x", "side_neck_y", "side_neck_c",
    "side_tail_bottom_x", "side_tail_bottom_y", "side_tail_bottom_c",
    "side_right_back_elbow_x", "side_right_back_elbow_y", "side_right_back_elbow_c"
]

def get_landmark_values(landmarks, key):
    """Given the landmarks dict and a landmark key, return x, y, and c.
       Returns empty strings if the key is not found."""
    if key in landmarks:
        data = landmarks[key]
        return data.get("x", ""), data.get("y", ""), data.get("c", "")
    else:
        return "", "", ""

# Example function to compute side_distance from the side NDJSON.
# You might adjust this depending on what “side_distance” means.
def compute_side_distance(data):
    # For example, use the bounding box values if available:
    bbox = data.get("bounding_box", {})
    try:
        x0 = float(bbox.get("x0", 0))
        x1 = float(bbox.get("x1", 0))
        # Simple horizontal distance (modify as needed)
        return abs(x1 - x0)
    except Exception:
        return ""

# Open both NDJSON files (one per camera)
with open("front_video.json", "r") as front_file, open("side_video.json", "r") as side_file, open("video_landmarks.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()

    # Process each synchronized line
    for front_line, side_line in zip(front_file, side_file):
        front_json = json.loads(front_line)
        side_json = json.loads(side_line)
        row = {}

        # Use absolute_time from one file (they are synchronized)
        row["absolute_time"] = front_json.get("absolute_time", "")

        # For the front file, extract landmarks values based on the mapping.
        # The following helper fills in three CSV columns (x, y, c) for a given landmark.
        def fill_front(col_prefix, map_key):
            x, y, c = get_landmark_values(front_json.get("landmarks", {}), front_map.get(map_key, ""))
            row[f"front_{col_prefix}_x"] = x
            row[f"front_{col_prefix}_y"] = y
            row[f"front_{col_prefix}_c"] = c

        # Fill in front values.
        fill_front("left_eye", "left_eye")
        fill_front("left_ear_middle", "left_ear_middle")
        fill_front("right_eye", "right_eye")
        fill_front("right_ear_top", "right_ear_top")
        fill_front("right_ear_middle", "right_ear_middle")
        fill_front("nose", "nose")
        fill_front("left_ear_top", "left_ear_top")
        fill_front("right_ear_bottom", "right_ear_bottom")
        fill_front("right_back_paw", "right_back_paw")
        fill_front("tail_top", "tail_top")
        fill_front("left_back_elbow", "left_back_elbow")
        fill_front("left_knee", "left_front_knee")
        fill_front("left_back_knee", "left_back_knee")
        fill_front("right_paw", "right_front_paw")
        fill_front("left_paw", "left_front_paw")
        fill_front("left_ear_bottom", "left_ear_bottom")
        # If the same landmark appears twice in the header (for example, left_back_paw),
        # you may choose to copy the same value or adjust the mapping if needed.
        fill_front("left_back_paw", "left_back_paw")
        fill_front("right_knee", "right_front_knee")  # Adjust if your mapping differs
        fill_front("right_back_knee", "right_back_knee")  # This may come from another key if available
        fill_front("left_elbow", "left_front_elbow")
        fill_front("tail_middle", "tail_middle")
        fill_front("right_elbow", "right_front_elbow")
        fill_front("neck", "neck")
        fill_front("tail_bottom", "tail_bottom")
        fill_front("right_back_elbow", "right_back_elbow")

        # For the side file:
        # First, compute side_distance.
        row["side_distance"] = compute_side_distance(side_json)

        def fill_side(col_prefix, map_key):
            x, y, c = get_landmark_values(side_json.get("landmarks", {}), side_map.get(map_key, ""))
            row[f"side_{col_prefix}_x"] = x
            row[f"side_{col_prefix}_y"] = y
            row[f"side_{col_prefix}_c"] = c

        fill_side("left_eye", "left_eye")
        fill_side("left_ear_middle", "left_ear_middle")
        fill_side("right_eye", "right_eye")
        fill_side("right_ear_top", "right_ear_top")
        fill_side("right_ear_middle", "right_ear_middle")
        fill_side("nose", "nose")
        fill_side("left_ear_top", "left_ear_top")
        fill_side("right_ear_bottom", "right_ear_bottom")
        fill_side("right_back_paw", "right_back_paw")
        fill_side("tail_top", "tail_top")
        fill_side("left_back_elbow", "left_back_elbow")
        fill_side("front_left_knee", "front_left_knee")
        fill_side("left_back_knee", "left_back_knee")
        fill_side("front_right_paw", "front_right_paw")
        fill_side("front_left_paw", "front_left_paw")
        fill_side("left_ear_bottom", "left_ear_bottom")
        fill_side("left_back_paw", "left_back_paw")
        fill_side("front_right_knee", "right_front_knee")  # Adjust mapping if needed
        fill_side("right_back_knee", "right_back_knee")
        fill_side("front_left_elbow", "left_front_elbow")
        fill_side("tail_middle", "tail_middle")
        fill_side("front_right_elbow", "right_front_elbow")
        fill_side("neck", "neck")
        fill_side("tail_bottom", "tail_bottom")
        fill_side("right_back_elbow", "right_back_elbow")

        # Write the row to CSV.
        writer.writerow(row)
