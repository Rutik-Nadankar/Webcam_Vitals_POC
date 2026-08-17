import cv2
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_FOLDER = Path("videos")

VIDEO_NAME = "test_face_withoutspecs"

SUPPORTED_EXTENSIONS = [
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".MOV",
    ".MP4",
]


# ============================================================
# FIND VIDEO
# ============================================================

def find_video():

    for extension in SUPPORTED_EXTENSIONS:

        path = VIDEO_FOLDER / f"{VIDEO_NAME}{extension}"

        if path.exists():
            return path

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print("CONTACTLESS VITALS POC")
    print("PHASE 1 - VIDEO INPUT TEST")
    print("=" * 65)

    # --------------------------------------------------------
    # Find video
    # --------------------------------------------------------

    video_path = find_video()

    if video_path is None:

        print()
        print("ERROR: Could not find your video.")

        print()
        print("Expected something like:")

        print(
            VIDEO_FOLDER
            / f"{VIDEO_NAME}.mp4"
        )

        print()
        print("Files currently inside videos folder:")

        if VIDEO_FOLDER.exists():

            for file in VIDEO_FOLDER.iterdir():
                print(f"  - {file.name}")

        else:

            print("  videos folder does not exist.")

        return


    print()
    print(f"Video found: {video_path}")


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    video = cv2.VideoCapture(
        str(video_path)
    )


    if not video.isOpened():

        print()
        print("ERROR: OpenCV could not open the video.")

        return


    # ========================================================
    # GET VIDEO PROPERTIES
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


    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    if fps > 0:

        duration = (
            total_frames / fps
        )

    else:

        duration = 0


    # ========================================================
    # PRINT VIDEO INFORMATION
    # ========================================================

    print()
    print("-" * 65)

    print(f"Filename       : {video_path.name}")

    print(
        f"Resolution     : "
        f"{width} x {height}"
    )

    print(
        f"FPS            : "
        f"{fps:.2f}"
    )

    print(
        f"Total frames   : "
        f"{total_frames}"
    )

    print(
        f"Duration       : "
        f"{duration:.2f} seconds"
    )

    print("-" * 65)


    # ========================================================
    # BASIC QUALITY CHECK
    # ========================================================

    print()
    print("Initial video check:")


    if duration >= 30:

        print(
            "[OK] Duration is sufficient for POC."
        )

    else:

        print(
            "[WARNING] A 30-60 second video "
            "is preferable."
        )


    if fps >= 25:

        print(
            "[OK] Frame rate is suitable."
        )

    else:

        print(
            "[WARNING] FPS is below 25."
        )


    if width >= 720:

        print(
            "[OK] Resolution is suitable."
        )

    else:

        print(
            "[WARNING] Resolution is relatively low."
        )


    print()
    print("Controls:")
    print("Q     = Quit")
    print("SPACE = Pause / Resume")
    print("R     = Restart video")


    # ========================================================
    # PLAYBACK VARIABLES
    # ========================================================

    paused = False

    frame_number = 0

    frame = None


    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        if not paused:

            success, frame = video.read()


            # ------------------------------------------------
            # End of video
            # ------------------------------------------------

            if not success:

                print()
                print("End of video reached.")

                break


            frame_number += 1


        if frame is None:
            continue


        # ====================================================
        # CREATE DISPLAY FRAME
        # ====================================================

        display = frame.copy()


        # ----------------------------------------------------
        # Resize large phone videos for display only
        #
        # IMPORTANT:
        # We are NOT modifying the original video.
        # ----------------------------------------------------

        display_height, display_width = (
            display.shape[:2]
        )


        max_display_width = 1000


        if display_width > max_display_width:

            scale = (
                max_display_width
                / display_width
            )

            new_width = int(
                display_width * scale
            )

            new_height = int(
                display_height * scale
            )

            display = cv2.resize(
                display,
                (
                    new_width,
                    new_height
                )
            )


        # ====================================================
        # CURRENT TIME
        # ====================================================

        if fps > 0:

            current_time = (
                frame_number / fps
            )

        else:

            current_time = 0


        screen_height, screen_width = (
            display.shape[:2]
        )


        # ====================================================
        # TEXT OVERLAY
        # ====================================================

        cv2.putText(
            display,
            "Contactless Vitals POC",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            "PHASE 1: VIDEO INPUT",
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        cv2.putText(
            display,
            f"Time: {current_time:.1f}s / {duration:.1f}s",
            (25, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            f"FPS: {fps:.1f}",
            (25, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            f"Frame: {frame_number} / {total_frames}",
            (25, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Paused indicator
        # ----------------------------------------------------

        if paused:

            cv2.putText(
                display,
                "PAUSED",
                (25, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


        cv2.putText(
            display,
            "Q: Quit | SPACE: Pause | R: Restart",
            (
                25,
                screen_height - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )


        # ====================================================
        # SHOW VIDEO
        # ====================================================

        cv2.imshow(
            "Contactless Vitals POC",
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

            delay = 33


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
        # KEYBOARD CONTROLS
        # ====================================================

        # Q
        if key == ord("q"):

            break


        # SPACE
        elif key == 32:

            paused = not paused


        # R
        elif key == ord("r"):

            video.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            frame_number = 0

            paused = False

            print(
                "Video restarted."
            )


    # ========================================================
    # CLEANUP
    # ========================================================

    video.release()

    cv2.destroyAllWindows()


    print()
    print("=" * 65)
    print("PHASE 1 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()