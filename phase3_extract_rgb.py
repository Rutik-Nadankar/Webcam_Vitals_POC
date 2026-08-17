import cv2
import csv
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = (
    BASE_DIR
    / "videos"
    / "test_face_withoutspecs.mp4"
)

CASCADE_PATH = (
    BASE_DIR
    / "haarcascade_frontalface_default.xml"
)

DATA_DIR = BASE_DIR / "data"

OUTPUT_CSV = (
    DATA_DIR
    / "test_face_withoutspecs_rgb.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

DETECTION_WIDTH = 640

SMOOTHING = 0.85

SCALE_FACTOR = 1.1

MIN_NEIGHBORS = 6


# ============================================================
# SMOOTH FACE BOX
# ============================================================

def smooth_box(previous, current):

    if previous is None:
        return current

    px, py, pw, ph = previous
    cx, cy, cw, ch = current

    x = int(
        SMOOTHING * px
        + (1 - SMOOTHING) * cx
    )

    y = int(
        SMOOTHING * py
        + (1 - SMOOTHING) * cy
    )

    w = int(
        SMOOTHING * pw
        + (1 - SMOOTHING) * cw
    )

    h = int(
        SMOOTHING * ph
        + (1 - SMOOTHING) * ch
    )

    return x, y, w, h


# ============================================================
# CALCULATE ROIS
# ============================================================

def calculate_rois(face):

    x, y, w, h = face

    # --------------------------------------------------------
    # FOREHEAD
    # --------------------------------------------------------

    forehead = (
        int(x + 0.30 * w),
        int(y + 0.12 * h),

        int(x + 0.70 * w),
        int(y + 0.30 * h),
    )

    # --------------------------------------------------------
    # LEFT CHEEK
    # --------------------------------------------------------

    left_cheek = (
        int(x + 0.12 * w),
        int(y + 0.48 * h),

        int(x + 0.38 * w),
        int(y + 0.72 * h),
    )

    # --------------------------------------------------------
    # RIGHT CHEEK
    # --------------------------------------------------------

    right_cheek = (
        int(x + 0.62 * w),
        int(y + 0.48 * h),

        int(x + 0.88 * w),
        int(y + 0.72 * h),
    )

    return {

        "forehead": forehead,

        "left_cheek": left_cheek,

        "right_cheek": right_cheek,
    }


# ============================================================
# CLAMP ROI
# ============================================================

def clamp_roi(
    roi,
    frame_width,
    frame_height
):

    x1, y1, x2, y2 = roi

    x1 = max(
        0,
        min(frame_width - 1, x1)
    )

    y1 = max(
        0,
        min(frame_height - 1, y1)
    )

    x2 = max(
        x1 + 1,
        min(frame_width, x2)
    )

    y2 = max(
        y1 + 1,
        min(frame_height, y2)
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# EXTRACT RGB FROM ROI
# ============================================================

def extract_rgb(frame, roi):

    x1, y1, x2, y2 = roi

    region = frame[
        y1:y2,
        x1:x2
    ]

    if region.size == 0:

        return (
            0.0,
            0.0,
            0.0,
            0
        )

    # OpenCV uses:
    #
    # BGR
    #
    # not RGB.

    b, g, r, _ = cv2.mean(
        region
    )

    pixel_count = (
        region.shape[0]
        * region.shape[1]
    )

    return (
        float(r),
        float(g),
        float(b),
        pixel_count
    )


# ============================================================
# WEIGHTED COMBINED RGB
# ============================================================

def combine_rgb(signals):

    total_pixels = sum(
        signal[3]
        for signal in signals
    )

    if total_pixels == 0:

        return (
            0.0,
            0.0,
            0.0
        )

    combined_r = sum(
        signal[0] * signal[3]
        for signal in signals
    ) / total_pixels

    combined_g = sum(
        signal[1] * signal[3]
        for signal in signals
    ) / total_pixels

    combined_b = sum(
        signal[2] * signal[3]
        for signal in signals
    ) / total_pixels

    return (
        combined_r,
        combined_g,
        combined_b
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("CONTACTLESS VITALS POC")
    print("PHASE 3 - RGB SIGNAL EXTRACTION")
    print("=" * 72)


    # ========================================================
    # CHECK FILES
    # ========================================================

    if not VIDEO_PATH.exists():

        print()
        print(
            "ERROR: Video not found."
        )

        print(
            VIDEO_PATH
        )

        return


    if not CASCADE_PATH.exists():

        print()
        print(
            "ERROR: Face detector XML not found."
        )

        print(
            CASCADE_PATH
        )

        return


    # ========================================================
    # CREATE DATA DIRECTORY
    # ========================================================

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print()
    print(
        f"Video: {VIDEO_PATH.name}"
    )

    print(
        f"Output: {OUTPUT_CSV}"
    )


    # ========================================================
    # LOAD FACE DETECTOR
    # ========================================================

    face_detector = cv2.CascadeClassifier(
        str(CASCADE_PATH)
    )


    if face_detector.empty():

        print()
        print(
            "ERROR: Could not load face detector."
        )

        return


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    video = cv2.VideoCapture(
        str(VIDEO_PATH)
    )


    if not video.isOpened():

        print()
        print(
            "ERROR: Could not open video."
        )

        return


    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        video.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        video.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    print()
    print("-" * 72)

    print(
        f"Resolution : {width} x {height}"
    )

    print(
        f"FPS        : {fps:.2f}"
    )

    print(
        f"Frames     : {total_frames}"
    )

    print("-" * 72)


    # ========================================================
    # CSV HEADERS
    # ========================================================

    headers = [

        "frame",

        "time_sec",

        "face_detected",

        "face_x",
        "face_y",
        "face_width",
        "face_height",

        # Forehead

        "forehead_r",
        "forehead_g",
        "forehead_b",

        # Left cheek

        "left_cheek_r",
        "left_cheek_g",
        "left_cheek_b",

        # Right cheek

        "right_cheek_r",
        "right_cheek_g",
        "right_cheek_b",

        # Combined signal

        "combined_r",
        "combined_g",
        "combined_b",
    ]


    # ========================================================
    # VARIABLES
    # ========================================================

    previous_face = None

    frame_number = 0

    detected_frames = 0

    missed_frames = 0


    # ========================================================
    # OPEN CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            headers
        )


        # ====================================================
        # PROCESS VIDEO
        # ====================================================

        while True:

            success, frame = video.read()


            if not success:

                break


            frame_number += 1


            # =================================================
            # TIMESTAMP
            # =================================================

            if fps > 0:

                time_sec = (
                    (frame_number - 1)
                    / fps
                )

            else:

                time_sec = 0.0


            # =================================================
            # RESIZE FOR FACE DETECTION
            # =================================================

            scale = (
                DETECTION_WIDTH
                / frame.shape[1]
            )


            detection_height = int(
                frame.shape[0]
                * scale
            )


            small_frame = cv2.resize(
                frame,
                (
                    DETECTION_WIDTH,
                    detection_height
                ),
                interpolation=cv2.INTER_AREA
            )


            # =================================================
            # GRAYSCALE
            # =================================================

            gray = cv2.cvtColor(
                small_frame,
                cv2.COLOR_BGR2GRAY
            )


            gray = cv2.equalizeHist(
                gray
            )


            # =================================================
            # FACE DETECTION
            # =================================================

            faces = face_detector.detectMultiScale(

                gray,

                scaleFactor=SCALE_FACTOR,

                minNeighbors=MIN_NEIGHBORS,

                minSize=(70, 70)
            )


            # =================================================
            # FACE FOUND
            # =================================================

            if len(faces) > 0:

                face_small = max(

                    faces,

                    key=lambda f:
                    f[2] * f[3]
                )


                sx, sy, sw, sh = (
                    face_small
                )


                # =============================================
                # MAP TO ORIGINAL FRAME
                # =============================================

                x = int(
                    sx / scale
                )

                y = int(
                    sy / scale
                )

                w = int(
                    sw / scale
                )

                h = int(
                    sh / scale
                )


                current_face = (
                    x,
                    y,
                    w,
                    h
                )


                previous_face = smooth_box(

                    previous_face,

                    current_face
                )


                detected_frames += 1

                face_detected = 1


            else:

                missed_frames += 1

                face_detected = 0


            # =================================================
            # NO FACE AVAILABLE
            # =================================================

            if previous_face is None:

                writer.writerow([

                    frame_number,

                    f"{time_sec:.6f}",

                    0,

                    "", "", "", "",

                    "", "", "",

                    "", "", "",

                    "", "", "",

                    "", "", "",
                ])

                continue


            # =================================================
            # FACE INFORMATION
            # =================================================

            x, y, w, h = previous_face


            # =================================================
            # CALCULATE ROI COORDINATES
            # =================================================

            rois = calculate_rois(
                previous_face
            )


            forehead_roi = clamp_roi(

                rois["forehead"],

                width,

                height
            )


            left_roi = clamp_roi(

                rois["left_cheek"],

                width,

                height
            )


            right_roi = clamp_roi(

                rois["right_cheek"],

                width,

                height
            )


            # =================================================
            # EXTRACT RGB VALUES
            # =================================================

            forehead_signal = extract_rgb(

                frame,

                forehead_roi
            )


            left_signal = extract_rgb(

                frame,

                left_roi
            )


            right_signal = extract_rgb(

                frame,

                right_roi
            )


            # =================================================
            # COMBINED FACE SIGNAL
            # =================================================

            combined_r, combined_g, combined_b = combine_rgb([

                forehead_signal,

                left_signal,

                right_signal,
            ])


            # =================================================
            # SAVE CSV ROW
            # =================================================

            writer.writerow([

                frame_number,

                f"{time_sec:.6f}",

                face_detected,

                x,
                y,
                w,
                h,

                # Forehead

                f"{forehead_signal[0]:.6f}",
                f"{forehead_signal[1]:.6f}",
                f"{forehead_signal[2]:.6f}",

                # Left cheek

                f"{left_signal[0]:.6f}",
                f"{left_signal[1]:.6f}",
                f"{left_signal[2]:.6f}",

                # Right cheek

                f"{right_signal[0]:.6f}",
                f"{right_signal[1]:.6f}",
                f"{right_signal[2]:.6f}",

                # Combined

                f"{combined_r:.6f}",
                f"{combined_g:.6f}",
                f"{combined_b:.6f}",
            ])


            # =================================================
            # TERMINAL PROGRESS
            # =================================================

            if (
                frame_number % 100 == 0
                or frame_number == total_frames
            ):

                progress = (
                    frame_number
                    / total_frames
                    * 100
                )

                print(

                    f"\rProcessing: "
                    f"{frame_number}/{total_frames} "
                    f"({progress:.1f}%)",

                    end="",

                    flush=True
                )


    # ========================================================
    # CLEANUP
    # ========================================================

    video.release()


    # ========================================================
    # RESULTS
    # ========================================================

    processed = (
        detected_frames
        + missed_frames
    )


    if processed > 0:

        detection_rate = (

            detected_frames
            / processed
            * 100
        )

    else:

        detection_rate = 0


    print()
    print()

    print("=" * 72)
    print("PHASE 3 COMPLETE")
    print("=" * 72)

    print(
        f"Frames processed : {frame_number}"
    )

    print(
        f"Face detected    : {detected_frames}"
    )

    print(
        f"Face missed      : {missed_frames}"
    )

    print(
        f"Detection rate   : {detection_rate:.2f}%"
    )

    print()

    print(
        "RGB data saved to:"
    )

    print(
        OUTPUT_CSV
    )

    print("=" * 72)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()