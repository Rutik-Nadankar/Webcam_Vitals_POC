import cv2
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


# ============================================================
# CONFIGURATION
# ============================================================

# Face detection is performed on a smaller copy for speed.
# RGB extraction later will use the original high-resolution frame.
DETECTION_WIDTH = 640

# Higher value = smoother face box
SMOOTHING = 0.85

# Haar face detector parameters
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 6


# ============================================================
# SMOOTH FACE BOX
# ============================================================

def smooth_box(previous, current):
    """
    Smooth face-box movement between frames.
    """

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
# CALCULATE SKIN REGIONS
# ============================================================

def calculate_rois(face):
    """
    Create forehead, left-cheek and right-cheek regions
    relative to the detected face.
    """

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
# KEEP ROI INSIDE FRAME
# ============================================================

def clamp_roi(roi, frame_width, frame_height):
    """
    Prevent ROI coordinates from going outside the frame.
    """

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
        1,
        min(frame_width, x2)
    )

    y2 = max(
        1,
        min(frame_height, y2)
    )

    return x1, y1, x2, y2


# ============================================================
# DRAW ROI
# ============================================================

def draw_roi(frame, roi, label):
    """
    Draw one skin region.
    """

    x1, y1, x2, y2 = roi

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        4
    )

    cv2.putText(
        frame,
        label,
        (x1, max(30, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CONTACTLESS VITALS POC")
    print("PHASE 2 - FACE + ROI DETECTION")
    print("=" * 70)


    # ========================================================
    # CHECK VIDEO
    # ========================================================

    if not VIDEO_PATH.exists():

        print()
        print("ERROR: Video file not found.")

        print(
            f"Expected here:\n{VIDEO_PATH}"
        )

        return


    print()
    print(
        f"Video found: {VIDEO_PATH}"
    )


    # ========================================================
    # CHECK CASCADE XML
    # ========================================================

    if not CASCADE_PATH.exists():

        print()
        print(
            "ERROR: Face detector XML file not found."
        )

        print(
            f"Expected here:\n{CASCADE_PATH}"
        )

        return


    print(
        f"Detector found: {CASCADE_PATH}"
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
            "ERROR: Could not load face detector XML."
        )

        return


    print(
        "Face detector loaded successfully."
    )


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    video = cv2.VideoCapture(
        str(VIDEO_PATH)
    )


    if not video.isOpened():

        print()
        print(
            "ERROR: OpenCV could not open the video."
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

    original_width = int(
        video.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    original_height = int(
        video.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    if fps > 0:

        duration = (
            total_frames / fps
        )

    else:

        duration = 0


    print()
    print("-" * 70)

    print(
        f"Resolution : "
        f"{original_width} x {original_height}"
    )

    print(
        f"FPS        : {fps:.2f}"
    )

    print(
        f"Frames     : {total_frames}"
    )

    print(
        f"Duration   : {duration:.2f} seconds"
    )

    print("-" * 70)

    print()
    print("Controls:")
    print("Q     = Quit")
    print("SPACE = Pause / Resume")
    print("R     = Restart")
    print()


    # ========================================================
    # VARIABLES
    # ========================================================

    previous_face = None

    frame_number = 0

    detected_frames = 0

    missed_frames = 0

    paused = False

    frame = None


    # ========================================================
    # VIDEO PROCESSING LOOP
    # ========================================================

    while True:

        if not paused:

            success, frame = video.read()


            if not success:

                print()
                print(
                    "End of video reached."
                )

                break


            frame_number += 1


            # =================================================
            # CREATE SMALL FRAME FOR FACE DETECTION
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
            # CONVERT TO GRAYSCALE
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

                # Pick largest face
                face_small = max(
                    faces,
                    key=lambda f: (
                        f[2] * f[3]
                    )
                )


                sx, sy, sw, sh = face_small


                # =============================================
                # MAP SMALL FRAME COORDINATES BACK TO ORIGINAL
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


                # =============================================
                # SMOOTH BOX MOVEMENT
                # =============================================

                previous_face = smooth_box(
                    previous_face,
                    current_face
                )


                detected_frames += 1


            # =================================================
            # FACE NOT FOUND
            # =================================================

            else:

                missed_frames += 1


        # ====================================================
        # NOTHING TO DISPLAY YET
        # ====================================================

        if frame is None:
            continue


        # ====================================================
        # DISPLAY COPY
        # ====================================================

        display = frame.copy()

        frame_height, frame_width = (
            display.shape[:2]
        )


        # ====================================================
        # DRAW FACE + SKIN ROIS
        # ====================================================

        if previous_face is not None:

            x, y, w, h = previous_face


            # ------------------------------------------------
            # FACE BOX
            # ------------------------------------------------

            cv2.rectangle(
                display,
                (x, y),
                (
                    x + w,
                    y + h
                ),
                (0, 255, 0),
                5
            )


            # ------------------------------------------------
            # CALCULATE ROIS
            # ------------------------------------------------

            rois = calculate_rois(
                previous_face
            )


            forehead = clamp_roi(
                rois["forehead"],
                frame_width,
                frame_height
            )


            left_cheek = clamp_roi(
                rois["left_cheek"],
                frame_width,
                frame_height
            )


            right_cheek = clamp_roi(
                rois["right_cheek"],
                frame_width,
                frame_height
            )


            # ------------------------------------------------
            # DRAW ROIS
            # ------------------------------------------------

            draw_roi(
                display,
                forehead,
                "FOREHEAD"
            )


            draw_roi(
                display,
                left_cheek,
                "LEFT CHEEK"
            )


            draw_roi(
                display,
                right_cheek,
                "RIGHT CHEEK"
            )


        # ====================================================
        # DETECTION RATE
        # ====================================================

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


        # ====================================================
        # CURRENT TIME
        # ====================================================

        if fps > 0:

            current_time = (
                frame_number
                / fps
            )

        else:

            current_time = 0


        # ====================================================
        # DISPLAY TEXT
        # ====================================================

        cv2.putText(
            display,
            "CONTACTLESS VITALS POC",
            (60, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            4
        )


        cv2.putText(
            display,
            "PHASE 2: FACE + ROI",
            (60, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            3
        )


        cv2.putText(
            display,
            f"Time: {current_time:.1f}s",
            (60, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            3
        )


        cv2.putText(
            display,
            f"Detection: {detection_rate:.1f}%",
            (60, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            3
        )


        # ====================================================
        # DISPLAY RESIZE
        #
        # Original 2160 x 3840 remains untouched.
        # This resize is only so the portrait video fits screen.
        # ====================================================

        DISPLAY_HEIGHT = 800


        display_scale = (
            DISPLAY_HEIGHT
            / display.shape[0]
        )


        display_width = int(
            display.shape[1]
            * display_scale
        )


        display = cv2.resize(
            display,
            (
                display_width,
                DISPLAY_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )


        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(
            "Phase 2 - Face and Skin ROIs",
            display
        )


        # ====================================================
        # PLAYBACK SPEED
        # ====================================================

        if fps > 0:

            delay = max(
                1,
                int(1000 / fps)
            )

        else:

            delay = 16


        if paused:

            key = (
                cv2.waitKey(30)
                & 0xFF
            )

        else:

            key = (
                cv2.waitKey(delay)
                & 0xFF
            )


        # ====================================================
        # CONTROLS
        # ====================================================

        # Q = Quit
        if key == ord("q"):

            break


        # SPACE = Pause
        elif key == 32:

            paused = not paused


        # R = Restart
        elif key == ord("r"):

            video.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            previous_face = None

            frame_number = 0

            detected_frames = 0

            missed_frames = 0

            paused = False

            print(
                "Video restarted."
            )


    # ========================================================
    # CLEANUP
    # ========================================================

    video.release()

    cv2.destroyAllWindows()


    # ========================================================
    # FINAL RESULTS
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
    print("=" * 70)
    print("PHASE 2 RESULTS")
    print("=" * 70)

    print(
        f"Frames processed : {processed}"
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

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()