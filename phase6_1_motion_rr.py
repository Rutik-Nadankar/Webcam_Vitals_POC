from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import (
    butter,
    filtfilt,
    detrend,
    welch,
    find_peaks,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = (
    BASE_DIR
    / "videos"
    / "test_face_withoutspecs.mp4"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SIGNAL_PLOT = (
    RESULTS_DIR
    / "phase6_1_shoulder_respiration.png"
)

SPECTRUM_PLOT = (
    RESULTS_DIR
    / "phase6_1_rr_spectrum.png"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "phase6_1_rr_summary.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Respiratory range:
#
# 0.10 Hz = 6 breaths/min
# 0.50 Hz = 30 breaths/min

RESP_LOW_HZ = 0.10
RESP_HIGH_HZ = 0.50

FILTER_ORDER = 4


# ============================================================
# VIDEO PROCESSING SIZE
# ============================================================

# Your original video is 2160 x 3840.
#
# Optical flow does NOT need full 4K resolution.
# Processing a smaller frame makes tracking much faster.

TRACK_HEIGHT = 800


# ============================================================
# OPTICAL FLOW SETTINGS
# ============================================================

MAX_FEATURES = 250

QUALITY_LEVEL = 0.01

MIN_DISTANCE = 8

MIN_TRACKED_POINTS = 10

REDETECT_INTERVAL_SECONDS = 2


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass(signal, fs):

    nyquist = (
        fs / 2
    )

    low = (
        RESP_LOW_HZ
        / nyquist
    )

    high = (
        RESP_HIGH_HZ
        / nyquist
    )

    b, a = butter(
        FILTER_ORDER,
        [low, high],
        btype="band"
    )

    return filtfilt(
        b,
        a,
        signal
    )


# ============================================================
# NORMALIZE SIGNAL
# ============================================================

def normalize(signal):

    signal = np.asarray(
        signal,
        dtype=float
    )

    signal = detrend(
        signal
    )

    std = np.std(
        signal
    )

    if std == 0:

        return np.zeros_like(
            signal
        )

    return (
        signal
        - np.mean(signal)
    ) / std


# ============================================================
# RESIZE FRAME FOR TRACKING
# ============================================================

def resize_for_tracking(frame):

    original_height, original_width = (
        frame.shape[:2]
    )

    scale = (
        TRACK_HEIGHT
        / original_height
    )

    new_width = int(
        original_width
        * scale
    )

    resized = cv2.resize(
        frame,
        (
            new_width,
            TRACK_HEIGHT
        ),
        interpolation=cv2.INTER_AREA
    )

    return resized


# ============================================================
# CREATE ROI MASK
# ============================================================

def create_roi_mask(
    gray_frame,
    roi
):

    x, y, w, h = roi

    mask = np.zeros_like(
        gray_frame
    )

    mask[
        y:y + h,
        x:x + w
    ] = 255

    return mask


# ============================================================
# FIND TRACKING FEATURES
# ============================================================

def detect_features(
    gray_frame,
    roi
):

    mask = create_roi_mask(
        gray_frame,
        roi
    )

    points = cv2.goodFeaturesToTrack(

        gray_frame,

        mask=mask,

        maxCorners=MAX_FEATURES,

        qualityLevel=QUALITY_LEVEL,

        minDistance=MIN_DISTANCE,

        blockSize=7
    )

    return points


# ============================================================
# PSD RESPIRATORY RATE
# ============================================================

def estimate_rr_psd(
    signal,
    fs
):

    frequencies, power = welch(

        signal,

        fs=fs,

        nperseg=len(signal),

        nfft=16384
    )

    valid = (
        (frequencies >= RESP_LOW_HZ)
        &
        (frequencies <= RESP_HIGH_HZ)
    )

    respiratory_frequencies = (
        frequencies[valid]
    )

    respiratory_power = (
        power[valid]
    )

    if len(
        respiratory_power
    ) == 0:

        return None


    # ========================================================
    # FIND DOMINANT PEAK
    # ========================================================

    peaks, _ = find_peaks(

        respiratory_power,

        prominence=(
            np.max(
                respiratory_power
            )
            * 0.05
        )
    )


    if len(peaks) > 0:

        strongest_peak = peaks[
            np.argmax(
                respiratory_power[
                    peaks
                ]
            )
        ]

    else:

        strongest_peak = np.argmax(
            respiratory_power
        )


    peak_frequency = (
        respiratory_frequencies[
            strongest_peak
        ]
    )

    rr = (
        peak_frequency
        * 60
    )


    # ========================================================
    # SPECTRAL QUALITY
    # ========================================================

    mean_power = np.mean(
        respiratory_power
    )

    if mean_power > 0:

        quality = (
            respiratory_power[
                strongest_peak
            ]
            /
            mean_power
        )

    else:

        quality = 0


    return {

        "rr": float(
            rr
        ),

        "quality": float(
            quality
        ),

        "frequencies": (
            respiratory_frequencies
        ),

        "power": (
            respiratory_power
        ),
    }


# ============================================================
# AUTOCORRELATION RESPIRATORY RATE
# ============================================================

def estimate_rr_autocorr(
    signal,
    fs
):

    correlation = np.correlate(

        signal,

        signal,

        mode="full"
    )

    correlation = correlation[
        len(signal) - 1:
    ]


    if correlation[0] != 0:

        correlation = (
            correlation
            /
            correlation[0]
        )


    # ========================================================
    # RESPIRATORY PERIOD RANGE
    #
    # 30 breaths/min = 2 seconds
    # 6 breaths/min  = 10 seconds
    # ========================================================

    minimum_lag = int(
        fs
        / RESP_HIGH_HZ
    )

    maximum_lag = int(
        fs
        / RESP_LOW_HZ
    )


    search_region = correlation[
        minimum_lag:
        maximum_lag + 1
    ]


    if len(
        search_region
    ) == 0:

        return None


    peaks, _ = find_peaks(
        search_region
    )


    if len(peaks) == 0:

        best_peak = np.argmax(
            search_region
        )

    else:

        best_peak = peaks[
            np.argmax(
                search_region[
                    peaks
                ]
            )
        ]


    lag = (
        minimum_lag
        + best_peak
    )


    period_seconds = (
        lag
        / fs
    )


    if period_seconds <= 0:

        return None


    rr = (
        60
        / period_seconds
    )


    quality = (
        search_region[
            best_peak
        ]
    )


    return {

        "rr": float(
            rr
        ),

        "quality": float(
            quality
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 76)

    print(
        "CONTACTLESS VITALS POC"
    )

    print(
        "PHASE 6.1 - SHOULDER MOTION RESPIRATORY RATE"
    )

    print("=" * 76)


    # ========================================================
    # CHECK VIDEO
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


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    video = cv2.VideoCapture(
        str(VIDEO_PATH)
    )


    if not video.isOpened():

        print(
            "ERROR: Could not open video."
        )

        return


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


    print()

    print(
        f"Original resolution : "
        f"{original_width} x "
        f"{original_height}"
    )

    print(
        f"FPS                 : "
        f"{fps:.2f}"
    )

    print(
        f"Frames              : "
        f"{total_frames}"
    )


    # ========================================================
    # READ FIRST FRAME
    # ========================================================

    success, first_frame = (
        video.read()
    )


    if not success:

        print(
            "ERROR: Could not read first frame."
        )

        video.release()

        return


    # ========================================================
    # RESIZE FOR TRACKING
    # ========================================================

    tracking_frame = (
        resize_for_tracking(
            first_frame
        )
    )


    tracking_height, tracking_width = (
        tracking_frame.shape[:2]
    )


    print()

    print(
        f"Tracking resolution : "
        f"{tracking_width} x "
        f"{tracking_height}"
    )


    # ========================================================
    # SELECT SHOULDER ROI
    # ========================================================

    print()

    print(
        "A window will open."
    )

    print(
        "Draw ONE rectangle around both shoulders "
        "and upper chest."
    )

    print(
        "Try to exclude most of the face."
    )

    print(
        "Then press ENTER or SPACE."
    )

    print()


    selected_roi = cv2.selectROI(

        "Select Shoulder / Upper Chest ROI",

        tracking_frame,

        fromCenter=False,

        showCrosshair=True
    )


    cv2.destroyWindow(
        "Select Shoulder / Upper Chest ROI"
    )


    roi_x, roi_y, roi_w, roi_h = (
        map(
            int,
            selected_roi
        )
    )


    if (
        roi_w <= 0
        or roi_h <= 0
    ):

        print(
            "No ROI selected."
        )

        video.release()

        return


    ROI = (
        roi_x,
        roi_y,
        roi_w,
        roi_h
    )


    print()

    print(
        f"Tracking ROI: "
        f"x={roi_x}, "
        f"y={roi_y}, "
        f"w={roi_w}, "
        f"h={roi_h}"
    )


    # ========================================================
    # RESET VIDEO
    # ========================================================

    video.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )


    # ========================================================
    # FIRST FRAME
    # ========================================================

    success, frame = (
        video.read()
    )


    if not success:

        print(
            "ERROR: Could not restart video."
        )

        video.release()

        return


    frame = resize_for_tracking(
        frame
    )


    previous_gray = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # INITIAL FEATURES
    # ========================================================

    points = detect_features(

        previous_gray,

        ROI
    )


    if points is None:

        print(
            "ERROR: No tracking points found "
            "inside selected ROI."
        )

        video.release()

        return


    points = points.astype(
        np.float32
    )


    print()

    print(
        f"Tracking points: "
        f"{len(points)}"
    )


    # ========================================================
    # STORAGE
    # ========================================================

    motion_signal = []

    time_signal = []

    point_counts = []


    frame_number = 1


    # ========================================================
    # FEATURE RE-DETECTION INTERVAL
    # ========================================================

    redetect_interval = max(

        1,

        int(
            fps
            * REDETECT_INTERVAL_SECONDS
        )
    )


    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        success, frame = (
            video.read()
        )


        if not success:

            break


        frame_number += 1


        # ====================================================
        # RESIZE FRAME
        # ====================================================

        frame = resize_for_tracking(
            frame
        )


        current_gray = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2GRAY
        )


        # ====================================================
        # OPTICAL FLOW
        # ====================================================

        next_points, status, error = (
            cv2.calcOpticalFlowPyrLK(

                previous_gray,

                current_gray,

                points,

                None,

                winSize=(21, 21),

                maxLevel=3,

                criteria=(
                    cv2.TERM_CRITERIA_EPS
                    |
                    cv2.TERM_CRITERIA_COUNT,

                    30,

                    0.01
                )
            )
        )


        # ====================================================
        # FLOW FAILURE
        # ====================================================

        if (
            next_points is None
            or status is None
        ):

            new_points = detect_features(

                current_gray,

                ROI
            )


            if new_points is None:

                motion_signal.append(
                    0.0
                )

                time_signal.append(
                    frame_number
                    / fps
                )

                point_counts.append(
                    0
                )

                previous_gray = (
                    current_gray
                )

                continue


            points = (
                new_points.astype(
                    np.float32
                )
            )

            previous_gray = (
                current_gray
            )

            continue


        # ====================================================
        # FIX OPENCV POINT SHAPE
        #
        # OpenCV normally returns:
        #
        # (N, 1, 2)
        #
        # We convert to:
        #
        # (N, 2)
        # ====================================================

        old_points_2d = (
            points.reshape(
                -1,
                2
            )
        )

        new_points_2d = (
            next_points.reshape(
                -1,
                2
            )
        )


        status_mask = (
            status.reshape(-1)
            == 1
        )


        good_old = (
            old_points_2d[
                status_mask
            ]
        )

        good_new = (
            new_points_2d[
                status_mask
            ]
        )


        # ====================================================
        # TOO FEW GOOD POINTS
        # ====================================================

        if (
            len(good_new)
            < MIN_TRACKED_POINTS
        ):

            new_points = detect_features(

                current_gray,

                ROI
            )


            if new_points is not None:

                points = (
                    new_points.astype(
                        np.float32
                    )
                )


            previous_gray = (
                current_gray
            )

            continue


        # ====================================================
        # VERTICAL MOVEMENT
        # ====================================================

        vertical_motion = (

            good_new[:, 1]

            -

            good_old[:, 1]
        )


        # Median is robust against badly tracked points.

        median_vertical_motion = (
            np.median(
                vertical_motion
            )
        )


        motion_signal.append(
            median_vertical_motion
        )


        time_signal.append(
            frame_number
            / fps
        )


        point_counts.append(
            len(good_new)
        )


        # ====================================================
        # UPDATE TRACKING STATE
        # ====================================================

        previous_gray = (
            current_gray
        )


        points = (
            good_new
            .reshape(
                -1,
                1,
                2
            )
            .astype(
                np.float32
            )
        )


        # ====================================================
        # PERIODIC FEATURE RE-DETECTION
        # ====================================================

        if (
            frame_number
            % redetect_interval
            == 0
        ):

            new_points = detect_features(

                current_gray,

                ROI
            )


            if new_points is not None:

                points = (
                    new_points.astype(
                        np.float32
                    )
                )


        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            frame_number % 250
            == 0
        ):

            progress = (

                frame_number
                / total_frames
                * 100
            )


            print(

                f"\rProcessing: "
                f"{frame_number}/"
                f"{total_frames} "
                f"({progress:.1f}%)",

                end="",

                flush=True
            )


    # ========================================================
    # CLEANUP VIDEO
    # ========================================================

    video.release()


    print()

    print()


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    motion_signal = np.asarray(

        motion_signal,

        dtype=float
    )


    time_signal = np.asarray(

        time_signal,

        dtype=float
    )


    point_counts = np.asarray(

        point_counts,

        dtype=float
    )


    # ========================================================
    # CHECK LENGTH
    # ========================================================

    if (
        len(motion_signal)
        < fps * 15
    ):

        print(
            "ERROR: Not enough motion data "
            "was successfully tracked."
        )

        return


    print(
        f"Motion samples collected : "
        f"{len(motion_signal)}"
    )


    if len(
        point_counts
    ) > 0:

        print(
            f"Average tracked points   : "
            f"{np.mean(point_counts):.1f}"
        )


    # ========================================================
    # FRAME-TO-FRAME MOVEMENT -> POSITION-LIKE SIGNAL
    # ========================================================

    cumulative_motion = (
        np.cumsum(
            motion_signal
        )
    )


    cumulative_motion = normalize(
        cumulative_motion
    )


    # ========================================================
    # RESPIRATORY FILTER
    # ========================================================

    filtered_motion = bandpass(

        cumulative_motion,

        fps
    )


    filtered_motion = normalize(
        filtered_motion
    )


    # ========================================================
    # PSD ESTIMATE
    # ========================================================

    psd_result = estimate_rr_psd(

        filtered_motion,

        fps
    )


    # ========================================================
    # AUTOCORRELATION ESTIMATE
    # ========================================================

    autocorr_result = (
        estimate_rr_autocorr(

            filtered_motion,

            fps
        )
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print()

    print("=" * 76)

    print(
        "RESPIRATORY MOTION RESULTS"
    )

    print("=" * 76)


    if psd_result:

        print(
            f"PSD RR             : "
            f"{psd_result['rr']:.1f} "
            f"breaths/min"
        )

        print(
            f"Spectral quality   : "
            f"{psd_result['quality']:.2f}"
        )


    if autocorr_result:

        print(
            f"Autocorrelation RR : "
            f"{autocorr_result['rr']:.1f} "
            f"breaths/min"
        )

        print(
            f"Autocorr quality   : "
            f"{autocorr_result['quality']:.2f}"
        )


    # ========================================================
    # CONSENSUS
    # ========================================================

    final_rr = np.nan

    quality_label = "LOW"


    if (
        psd_result is not None
        and autocorr_result is not None
    ):

        difference = abs(

            psd_result["rr"]

            -

            autocorr_result["rr"]
        )


        print(
            f"Method difference  : "
            f"{difference:.1f} "
            f"breaths/min"
        )


        # ----------------------------------------------------
        # GOOD
        # ----------------------------------------------------

        if (
            difference <= 2.5
            and
            psd_result[
                "quality"
            ] >= 3.0
        ):

            final_rr = np.mean([

                psd_result["rr"],

                autocorr_result[
                    "rr"
                ]
            ])

            quality_label = "GOOD"


        # ----------------------------------------------------
        # MODERATE
        # ----------------------------------------------------

        elif (
            difference <= 4.0
            and
            psd_result[
                "quality"
            ] >= 2.5
        ):

            final_rr = np.mean([

                psd_result["rr"],

                autocorr_result[
                    "rr"
                ]
            ])

            quality_label = (
                "MODERATE"
            )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()

    print("=" * 76)

    print(
        "FINAL MOTION-BASED RESPIRATORY RATE"
    )

    print("=" * 76)


    if np.isnan(
        final_rr
    ):

        print(
            "No reliable respiratory consensus."
        )

    else:

        print(
            f"Estimated RR   : "
            f"{final_rr:.1f} "
            f"breaths/min"
        )


    print(
        f"Signal quality : "
        f"{quality_label}"
    )


    # ========================================================
    # MOTION SIGNAL GRAPH
    # ========================================================

    # Adjust time array to match final signal length.

    graph_time = (
        time_signal[
            :len(
                filtered_motion
            )
        ]
    )


    plt.figure(
        figsize=(12, 5)
    )


    plt.plot(

        graph_time,

        filtered_motion
    )


    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Normalized vertical motion"
    )

    plt.title(
        "Shoulder / Upper Chest "
        "Respiratory Motion"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()


    plt.savefig(
        SIGNAL_PLOT,
        dpi=160
    )

    plt.close()


    # ========================================================
    # RESPIRATORY SPECTRUM
    # ========================================================

    if psd_result:

        plt.figure(
            figsize=(12, 5)
        )


        plt.plot(

            psd_result[
                "frequencies"
            ]
            * 60,

            psd_result[
                "power"
            ]
        )


        plt.xlabel(
            "Respiratory Rate "
            "(breaths/min)"
        )

        plt.ylabel(
            "Power"
        )

        plt.title(
            "Shoulder Motion "
            "Respiratory Spectrum"
        )

        plt.xlim(
            6,
            30
        )

        plt.grid(
            alpha=0.25
        )

        plt.tight_layout()


        plt.savefig(
            SPECTRUM_PLOT,
            dpi=160
        )

        plt.close()


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "CONTACTLESS VITALS POC\n"
        )

        file.write(
            "PHASE 6.1 SHOULDER MOTION "
            "RESPIRATORY RATE\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )


        file.write(
            f"Sampling rate: "
            f"{fps:.2f} Hz\n"
        )

        file.write(
            f"Motion samples: "
            f"{len(motion_signal)}\n"
        )


        if len(
            point_counts
        ) > 0:

            file.write(
                f"Average tracked points: "
                f"{np.mean(point_counts):.1f}\n"
            )


        if psd_result:

            file.write(
                f"\nPSD RR: "
                f"{psd_result['rr']:.1f} "
                f"breaths/min\n"
            )

            file.write(
                f"Spectral quality: "
                f"{psd_result['quality']:.2f}\n"
            )


        if autocorr_result:

            file.write(
                f"\nAutocorrelation RR: "
                f"{autocorr_result['rr']:.1f} "
                f"breaths/min\n"
            )

            file.write(
                f"Autocorrelation quality: "
                f"{autocorr_result['quality']:.2f}\n"
            )


        if not np.isnan(
            final_rr
        ):

            file.write(
                f"\nEstimated RR: "
                f"{final_rr:.1f} "
                f"breaths/min\n"
            )


        file.write(
            f"\nSignal quality: "
            f"{quality_label}\n"
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("=" * 76)

    print(
        "PHASE 6.1 COMPLETE"
    )

    print("=" * 76)


    print()

    print(
        f"Signal plot : "
        f"{SIGNAL_PLOT}"
    )

    print(
        f"Spectrum    : "
        f"{SPECTRUM_PLOT}"
    )

    print(
        f"Summary     : "
        f"{SUMMARY_FILE}"
    )


    print()

    print(
        "Experimental POC only - "
        "not a medical measurement."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()