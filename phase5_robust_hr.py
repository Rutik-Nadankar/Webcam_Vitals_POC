from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, detrend, welch


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
    / "phase5_hr_windows.csv"
)

OUTPUT_PLOT = (
    RESULTS_DIR
    / "phase5_hr_trend.png"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "phase5_hr_summary.txt"
)


# ============================================================
# SETTINGS
# ============================================================

LOW_HZ = 0.7
HIGH_HZ = 3.0

WINDOW_SECONDS = 10
STEP_SECONDS = 2

START_IGNORE_SECONDS = 3
END_IGNORE_SECONDS = 2

MIN_PEAK_RATIO = 3.0

ROI_AGREEMENT_BPM = 6.0


# ============================================================
# SIGNALS
# ============================================================

SIGNALS = {
    "combined": "combined_g",
    "forehead": "forehead_g",
    "left_cheek": "left_cheek_g",
    "right_cheek": "right_cheek_g",
}


# ============================================================
# BANDPASS
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
# BPM ESTIMATION
# ============================================================

def estimate_bpm(signal, fs):

    if len(signal) < fs * 5:
        return None

    signal = detrend(signal)

    std = np.std(signal)

    if std == 0:
        return None

    signal = (
        signal - np.mean(signal)
    ) / std

    filtered = bandpass(
        signal,
        fs
    )

    frequencies, power = welch(
        filtered,
        fs=fs,
        nperseg=min(
            len(filtered),
            1024
        )
    )

    valid = (
        (frequencies >= LOW_HZ)
        &
        (frequencies <= HIGH_HZ)
    )

    f = frequencies[valid]
    p = power[valid]

    if len(f) == 0:
        return None

    peak_index = np.argmax(p)

    peak_frequency = f[
        peak_index
    ]

    bpm = (
        peak_frequency
        * 60
    )

    mean_power = np.mean(p)

    if mean_power > 0:

        peak_ratio = (
            p[peak_index]
            / mean_power
        )

    else:

        peak_ratio = 0


    return {
        "bpm": float(bpm),
        "peak_ratio": float(
            peak_ratio
        ),
    }


# ============================================================
# CONSENSUS
# ============================================================

def calculate_consensus(results):

    good_results = {
        name: result
        for name, result
        in results.items()
        if (
            result is not None
            and result[
                "peak_ratio"
            ] >= MIN_PEAK_RATIO
        )
    }

    if not good_results:
        return None


    bpms = np.array([
        result["bpm"]
        for result
        in good_results.values()
    ])


    median_bpm = np.median(
        bpms
    )


    agreeing = {
        name: result
        for name, result
        in good_results.items()
        if abs(
            result["bpm"]
            - median_bpm
        ) <= ROI_AGREEMENT_BPM
    }


    # We want at least two regions
    # agreeing whenever possible.

    if len(agreeing) >= 2:

        values = np.array([
            result["bpm"]
            for result
            in agreeing.values()
        ])

        weights = np.array([
            result["peak_ratio"]
            for result
            in agreeing.values()
        ])

        consensus = np.average(
            values,
            weights=weights
        )

        quality = np.mean(
            weights
        )

        return {
            "bpm": float(consensus),
            "quality": float(quality),
            "regions": list(
                agreeing.keys()
            ),
        }


    # Fallback to strongest region

    strongest = max(
        good_results,
        key=lambda name:
        good_results[name][
            "peak_ratio"
        ]
    )

    result = good_results[
        strongest
    ]

    return {
        "bpm": result["bpm"],
        "quality": result[
            "peak_ratio"
        ],
        "regions": [
            strongest
        ],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("CONTACTLESS VITALS POC")
    print("PHASE 5 - ROBUST HEART RATE")
    print("=" * 75)


    if not CSV_PATH.exists():

        print(
            "ERROR: RGB CSV not found."
        )

        return


    df = pd.read_csv(
        CSV_PATH
    )


    # ========================================================
    # SAMPLING RATE
    # ========================================================

    time = df[
        "time_sec"
    ].to_numpy()

    dt = np.median(
        np.diff(time)
    )

    fs = 1 / dt


    print()
    print(
        f"Sampling rate: {fs:.2f} Hz"
    )


    # ========================================================
    # VALID TIME RANGE
    # ========================================================

    start_time = (
        time[0]
        + START_IGNORE_SECONDS
    )

    end_time = (
        time[-1]
        - END_IGNORE_SECONDS
    )


    print(
        f"Analysis range: "
        f"{start_time:.1f}s - "
        f"{end_time:.1f}s"
    )


    # ========================================================
    # WINDOW LOOP
    # ========================================================

    records = []

    current_start = start_time


    while (
        current_start
        + WINDOW_SECONDS
        <= end_time
    ):

        current_end = (
            current_start
            + WINDOW_SECONDS
        )


        window = df[
            (
                df["time_sec"]
                >= current_start
            )
            &
            (
                df["time_sec"]
                < current_end
            )
        ]


        roi_results = {}


        for name, column in (
            SIGNALS.items()
        ):

            signal = (
                window[column]
                .astype(float)
                .to_numpy()
            )


            roi_results[name] = (
                estimate_bpm(
                    signal,
                    fs
                )
            )


        consensus = (
            calculate_consensus(
                roi_results
            )
        )


        centre_time = (
            current_start
            + WINDOW_SECONDS / 2
        )


        record = {
            "time_sec": centre_time,
            "window_start": current_start,
            "window_end": current_end,
        }


        # ----------------------------------------------------
        # Store individual regions
        # ----------------------------------------------------

        for name in SIGNALS:

            result = roi_results[
                name
            ]

            if result is None:

                record[
                    f"{name}_bpm"
                ] = np.nan

                record[
                    f"{name}_quality"
                ] = np.nan

            else:

                record[
                    f"{name}_bpm"
                ] = result["bpm"]

                record[
                    f"{name}_quality"
                ] = result[
                    "peak_ratio"
                ]


        # ----------------------------------------------------
        # Consensus
        # ----------------------------------------------------

        if consensus:

            record[
                "consensus_bpm"
            ] = consensus["bpm"]

            record[
                "quality"
            ] = consensus[
                "quality"
            ]

            record[
                "regions_used"
            ] = ",".join(
                consensus[
                    "regions"
                ]
            )

        else:

            record[
                "consensus_bpm"
            ] = np.nan

            record[
                "quality"
            ] = np.nan

            record[
                "regions_used"
            ] = ""


        records.append(
            record
        )


        current_start += (
            STEP_SECONDS
        )


    # ========================================================
    # DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        records
    )


    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )


    # ========================================================
    # VALID ESTIMATES
    # ========================================================

    valid = results_df[
        "consensus_bpm"
    ].dropna()


    print()
    print("=" * 75)
    print("WINDOW RESULTS")
    print("=" * 75)


    for _, row in (
        results_df.iterrows()
    ):

        if np.isnan(
            row[
                "consensus_bpm"
            ]
        ):

            print(
                f"{row['time_sec']:5.1f}s"
                " | rejected"
            )

        else:

            print(
                f"{row['time_sec']:5.1f}s"
                f" | "
                f"{row['consensus_bpm']:5.1f} BPM"
                f" | Quality "
                f"{row['quality']:.2f}"
                f" | "
                f"{row['regions_used']}"
            )


    # ========================================================
    # FINAL ESTIMATE
    # ========================================================

    if len(valid) > 0:

        final_bpm = np.median(
            valid
        )

        bpm_std = np.std(
            valid
        )

        minimum = np.min(
            valid
        )

        maximum = np.max(
            valid
        )

        valid_percent = (
            len(valid)
            / len(results_df)
            * 100
        )


        print()
        print("=" * 75)
        print("FINAL POC HEART RATE")
        print("=" * 75)

        print(
            f"Estimated HR     : "
            f"{final_bpm:.1f} BPM"
        )

        print(
            f"Window variation : "
            f"{bpm_std:.1f} BPM"
        )

        print(
            f"Observed range   : "
            f"{minimum:.1f} - "
            f"{maximum:.1f} BPM"
        )

        print(
            f"Valid windows    : "
            f"{valid_percent:.1f}%"
        )


        # ====================================================
        # SIMPLE QUALITY LABEL
        # ====================================================

        if (
            valid_percent >= 80
            and bpm_std <= 5
        ):

            quality_label = "GOOD"

        elif (
            valid_percent >= 60
            and bpm_std <= 10
        ):

            quality_label = "MODERATE"

        else:

            quality_label = "LOW"


        print(
            f"Signal quality   : "
            f"{quality_label}"
        )


    else:

        final_bpm = np.nan
        bpm_std = np.nan
        valid_percent = 0
        quality_label = "LOW"

        print(
            "No reliable HR windows found."
        )


    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(
        figsize=(12, 5)
    )


    plt.plot(
        results_df[
            "time_sec"
        ],
        results_df[
            "consensus_bpm"
        ],
        marker="o"
    )


    if not np.isnan(
        final_bpm
    ):

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
        "Camera-Based Heart Rate Trend"
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
            "PHASE 5 ROBUST HEART RATE\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            f"Sampling rate: "
            f"{fs:.2f} Hz\n"
        )

        file.write(
            f"Window size: "
            f"{WINDOW_SECONDS}s\n"
        )

        file.write(
            f"Window step: "
            f"{STEP_SECONDS}s\n\n"
        )


        if not np.isnan(
            final_bpm
        ):

            file.write(
                f"Estimated HR: "
                f"{final_bpm:.1f} BPM\n"
            )

            file.write(
                f"Window SD: "
                f"{bpm_std:.1f} BPM\n"
            )

            file.write(
                f"Valid windows: "
                f"{valid_percent:.1f}%\n"
            )

            file.write(
                f"Signal quality: "
                f"{quality_label}\n"
            )


    print()
    print("=" * 75)
    print("PHASE 5 COMPLETE")
    print("=" * 75)

    print()
    print(
        f"Results CSV: {OUTPUT_CSV}"
    )

    print(
        f"Trend plot : {OUTPUT_PLOT}"
    )

    print(
        f"Summary    : {SUMMARY_FILE}"
    )

    print()
    print(
        "Experimental POC only - "
        "not a medical measurement."
    )


if __name__ == "__main__":
    main()