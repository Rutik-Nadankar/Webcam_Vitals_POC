from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import (
    butter,
    filtfilt,
    detrend,
    welch,
    find_peaks
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    BASE_DIR
    / "data"
    / "test_face_withoutspecs_rgb.csv"
)

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = (
    RESULTS_DIR
    / "phase5_1_pos_windows.csv"
)

OUTPUT_PLOT = (
    RESULTS_DIR
    / "phase5_1_pos_hr_trend.png"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "phase5_1_pos_summary.txt"
)


# ============================================================
# PARAMETERS
# ============================================================

LOW_HZ = 0.75
HIGH_HZ = 2.5

# 45 BPM - 150 BPM

WINDOW_SECONDS = 12
STEP_SECONDS = 1

START_IGNORE_SECONDS = 3
END_IGNORE_SECONDS = 2

MIN_PEAK_RATIO = 5.0

MAX_BPM_CHANGE = 12.0


# ============================================================
# FILTER
# ============================================================

def bandpass(signal, fs):

    nyquist = fs / 2

    b, a = butter(
        4,
        [
            LOW_HZ / nyquist,
            HIGH_HZ / nyquist
        ],
        btype="band"
    )

    return filtfilt(
        b,
        a,
        signal
    )


# ============================================================
# POS rPPG
# ============================================================

def pos_signal(r, g, b):
    """
    Simplified POS-style projection.

    Uses normalized RGB variations instead
    of only the green channel.
    """

    rgb = np.vstack([
        r,
        g,
        b
    ])


    # --------------------------------------------------------
    # Normalize channels by their means
    # --------------------------------------------------------

    mean_rgb = np.mean(
        rgb,
        axis=1,
        keepdims=True
    )


    mean_rgb[
        mean_rgb == 0
    ] = 1


    normalized = (
        rgb / mean_rgb
    )


    # --------------------------------------------------------
    # POS projections
    # --------------------------------------------------------

    x = (
        normalized[1]
        - normalized[2]
    )


    y = (
        normalized[1]
        + normalized[2]
        - 2 * normalized[0]
    )


    std_y = np.std(y)


    if std_y == 0:
        alpha = 0
    else:
        alpha = (
            np.std(x)
            / std_y
        )


    signal = (
        x + alpha * y
    )


    signal = detrend(
        signal
    )


    std_signal = np.std(
        signal
    )


    if std_signal > 0:

        signal = (
            signal
            - np.mean(signal)
        ) / std_signal


    return signal


# ============================================================
# ESTIMATE BPM
# ============================================================

def estimate_bpm(signal, fs):

    if len(signal) < (
        WINDOW_SECONDS
        * fs
        * 0.8
    ):
        return None


    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    filtered = bandpass(
        signal,
        fs
    )


    # --------------------------------------------------------
    # PSD
    # --------------------------------------------------------

    frequencies, power = welch(
        filtered,
        fs=fs,
        nperseg=len(filtered)
    )


    valid = (
        (frequencies >= LOW_HZ)
        &
        (frequencies <= HIGH_HZ)
    )


    frequencies = (
        frequencies[valid]
    )

    power = (
        power[valid]
    )


    if len(power) == 0:
        return None


    # --------------------------------------------------------
    # Find spectral peaks
    # --------------------------------------------------------

    peaks, properties = find_peaks(
        power,
        prominence=np.max(power) * 0.05
    )


    if len(peaks) == 0:

        peak_index = np.argmax(
            power
        )

    else:

        strongest = np.argmax(
            power[peaks]
        )

        peak_index = peaks[
            strongest
        ]


    peak_frequency = (
        frequencies[
            peak_index
        ]
    )


    bpm = (
        peak_frequency
        * 60
    )


    mean_power = np.mean(
        power
    )


    if mean_power > 0:

        peak_ratio = (
            power[
                peak_index
            ]
            / mean_power
        )

    else:

        peak_ratio = 0


    return {
        "bpm": float(bpm),
        "quality": float(
            peak_ratio
        ),
        "filtered": filtered,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("CONTACTLESS VITALS POC")
    print("PHASE 5.1 - POS rPPG HEART RATE")
    print("=" * 75)


    # ========================================================
    # LOAD CSV
    # ========================================================

    if not CSV_PATH.exists():

        print(
            "ERROR: Phase 3 CSV not found."
        )

        return


    df = pd.read_csv(
        CSV_PATH
    )


    time = df[
        "time_sec"
    ].to_numpy()


    dt = np.median(
        np.diff(time)
    )


    fs = (
        1 / dt
    )


    print()
    print(
        f"Sampling rate : {fs:.2f} Hz"
    )


    # ========================================================
    # VALID ANALYSIS RANGE
    # ========================================================

    start = (
        time[0]
        + START_IGNORE_SECONDS
    )

    end = (
        time[-1]
        - END_IGNORE_SECONDS
    )


    print(
        f"Analysis range: "
        f"{start:.1f}s - {end:.1f}s"
    )


    # ========================================================
    # SLIDING WINDOWS
    # ========================================================

    records = []

    previous_valid_bpm = None

    window_start = start


    while (
        window_start
        + WINDOW_SECONDS
        <= end
    ):

        window_end = (
            window_start
            + WINDOW_SECONDS
        )


        window = df[
            (
                df["time_sec"]
                >= window_start
            )
            &
            (
                df["time_sec"]
                < window_end
            )
        ]


        if len(window) == 0:

            window_start += (
                STEP_SECONDS
            )

            continue


        # ====================================================
        # EXTRACT COMBINED RGB
        # ====================================================

        r = window[
            "combined_r"
        ].to_numpy(
            dtype=float
        )

        g = window[
            "combined_g"
        ].to_numpy(
            dtype=float
        )

        b = window[
            "combined_b"
        ].to_numpy(
            dtype=float
        )


        # ====================================================
        # POS SIGNAL
        # ====================================================

        signal = pos_signal(
            r,
            g,
            b
        )


        result = estimate_bpm(
            signal,
            fs
        )


        centre = (
            window_start
            + WINDOW_SECONDS / 2
        )


        accepted = False

        rejection_reason = ""


        if result is None:

            rejection_reason = (
                "no_signal"
            )


        elif (
            result["quality"]
            < MIN_PEAK_RATIO
        ):

            rejection_reason = (
                "low_quality"
            )


        else:

            candidate = (
                result["bpm"]
            )


            # =================================================
            # TEMPORAL CONSISTENCY
            # =================================================

            if previous_valid_bpm is None:

                accepted = True

            else:

                difference = abs(
                    candidate
                    - previous_valid_bpm
                )


                if (
                    difference
                    <= MAX_BPM_CHANGE
                ):

                    accepted = True

                else:

                    rejection_reason = (
                        "large_jump"
                    )


        if accepted:

            previous_valid_bpm = (
                result["bpm"]
            )


            final_bpm = (
                result["bpm"]
            )

        else:

            final_bpm = np.nan


        records.append({

            "time_sec": centre,

            "window_start": (
                window_start
            ),

            "window_end": (
                window_end
            ),

            "raw_bpm": (
                result["bpm"]
                if result
                else np.nan
            ),

            "quality": (
                result["quality"]
                if result
                else np.nan
            ),

            "accepted": (
                accepted
            ),

            "final_bpm": (
                final_bpm
            ),

            "rejection_reason": (
                rejection_reason
            ),
        })


        window_start += (
            STEP_SECONDS
        )


    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results = pd.DataFrame(
        records
    )


    results.to_csv(
        OUTPUT_CSV,
        index=False
    )


    # ========================================================
    # PRINT WINDOWS
    # ========================================================

    print()
    print("=" * 75)
    print("WINDOW RESULTS")
    print("=" * 75)


    for _, row in (
        results.iterrows()
    ):

        if row["accepted"]:

            print(

                f"{row['time_sec']:5.1f}s"

                f" | {row['final_bpm']:5.1f} BPM"

                f" | Q {row['quality']:.2f}"

                f" | ACCEPT"
            )

        else:

            print(

                f"{row['time_sec']:5.1f}s"

                f" | raw "
                f"{row['raw_bpm']:5.1f}"

                f" | Q "
                f"{row['quality']:.2f}"

                f" | REJECT: "
                f"{row['rejection_reason']}"
            )


    # ========================================================
    # FINAL BPM
    # ========================================================

    valid = results[
        "final_bpm"
    ].dropna()


    if len(valid) == 0:

        print()
        print(
            "No reliable windows detected."
        )

        return


    final_bpm = np.median(
        valid
    )


    std_bpm = np.std(
        valid
    )


    valid_percentage = (
        len(valid)
        / len(results)
        * 100
    )


    # ========================================================
    # QUALITY LABEL
    # ========================================================

    if (
        std_bpm <= 5
        and valid_percentage >= 70
    ):

        quality_label = "GOOD"


    elif (
        std_bpm <= 8
        and valid_percentage >= 50
    ):

        quality_label = "MODERATE"


    else:

        quality_label = "LOW"


    print()
    print("=" * 75)
    print("FINAL POS HEART RATE")
    print("=" * 75)


    print(
        f"Estimated HR      : "
        f"{final_bpm:.1f} BPM"
    )


    print(
        f"Window SD         : "
        f"{std_bpm:.1f} BPM"
    )


    print(
        f"Accepted windows  : "
        f"{len(valid)}/{len(results)}"
    )


    print(
        f"Accepted percent  : "
        f"{valid_percentage:.1f}%"
    )


    print(
        f"Signal quality    : "
        f"{quality_label}"
    )


    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(
        figsize=(12, 5)
    )


    plt.plot(

        results[
            "time_sec"
        ],

        results[
            "raw_bpm"
        ],

        marker="o",

        alpha=0.4,

        label="Raw candidate"
    )


    accepted = results[
        results[
            "accepted"
        ]
    ]


    plt.plot(

        accepted[
            "time_sec"
        ],

        accepted[
            "final_bpm"
        ],

        marker="o",

        linewidth=2,

        label="Accepted HR"
    )


    plt.axhline(

        final_bpm,

        linestyle="--",

        label=(
            f"Median "
            f"{final_bpm:.1f} BPM"
        )
    )


    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Heart Rate (BPM)"
    )

    plt.title(
        "POS rPPG Heart Rate Trend"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()


    plt.savefig(
        OUTPUT_PLOT,
        dpi=160
    )

    plt.close()


    # ========================================================
    # SUMMARY
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
            "PHASE 5.1 POS rPPG\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            f"Estimated HR: "
            f"{final_bpm:.1f} BPM\n"
        )

        file.write(
            f"Window SD: "
            f"{std_bpm:.1f} BPM\n"
        )

        file.write(
            f"Accepted windows: "
            f"{len(valid)}/{len(results)}\n"
        )

        file.write(
            f"Accepted percentage: "
            f"{valid_percentage:.1f}%\n"
        )

        file.write(
            f"Signal quality: "
            f"{quality_label}\n"
        )


    print()
    print("=" * 75)
    print("PHASE 5.1 COMPLETE")
    print("=" * 75)

    print()
    print(
        f"CSV   : {OUTPUT_CSV}"
    )

    print(
        f"Plot  : {OUTPUT_PLOT}"
    )

    print(
        f"Report: {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()