from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import (
    butter,
    filtfilt,
    detrend,
    welch
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

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

RAW_PLOT = (
    RESULTS_DIR
    / "phase4_raw_green_signal.png"
)

FILTERED_PLOT = (
    RESULTS_DIR
    / "phase4_filtered_rppg.png"
)

SPECTRUM_PLOT = (
    RESULTS_DIR
    / "phase4_frequency_spectrum.png"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "phase4_summary.txt"
)


# ============================================================
# SETTINGS
# ============================================================

# Physiological pulse frequency range used for this POC
#
# 0.7 Hz  = 42 BPM
# 3.0 Hz  = 180 BPM

LOW_CUT_HZ = 0.7
HIGH_CUT_HZ = 3.0

FILTER_ORDER = 4


# Ignore the first few seconds because phone cameras can
# adjust exposure/white balance when recording starts.

START_IGNORE_SECONDS = 3.0

# Ignore a small amount at the end as well.

END_IGNORE_SECONDS = 2.0


# ============================================================
# SIGNALS TO ANALYSE
# ============================================================

SIGNALS = {

    "Combined Face": "combined_g",

    "Forehead": "forehead_g",

    "Left Cheek": "left_cheek_g",

    "Right Cheek": "right_cheek_g",
}


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(
    signal,
    sampling_rate,
    low_cut,
    high_cut,
    order=4
):

    nyquist = (
        sampling_rate / 2
    )

    low = (
        low_cut / nyquist
    )

    high = (
        high_cut / nyquist
    )

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    filtered = filtfilt(
        b,
        a,
        signal
    )

    return filtered


# ============================================================
# NORMALISE SIGNAL
# ============================================================

def normalise_signal(signal):

    mean = np.mean(
        signal
    )

    std = np.std(
        signal
    )

    if std == 0:
        return signal * 0

    return (
        signal - mean
    ) / std


# ============================================================
# CALCULATE BPM FROM PSD
# ============================================================

def calculate_bpm(
    signal,
    sampling_rate
):

    # --------------------------------------------------------
    # Welch power spectrum
    # --------------------------------------------------------

    nperseg = min(
        2048,
        len(signal)
    )

    frequencies, power = welch(

        signal,

        fs=sampling_rate,

        nperseg=nperseg
    )


    # --------------------------------------------------------
    # Keep only possible pulse frequencies
    # --------------------------------------------------------

    valid = (

        (frequencies >= LOW_CUT_HZ)

        &

        (frequencies <= HIGH_CUT_HZ)
    )


    pulse_frequencies = (
        frequencies[valid]
    )

    pulse_power = (
        power[valid]
    )


    if len(pulse_frequencies) == 0:

        return None


    # --------------------------------------------------------
    # Strongest spectral peak
    # --------------------------------------------------------

    peak_index = np.argmax(
        pulse_power
    )


    peak_frequency = (
        pulse_frequencies[
            peak_index
        ]
    )


    bpm = (
        peak_frequency * 60
    )


    # --------------------------------------------------------
    # Simple spectral signal-quality score
    #
    # This is NOT medical confidence.
    # It only measures how dominant the strongest peak is.
    # --------------------------------------------------------

    average_power = np.mean(
        pulse_power
    )


    if average_power > 0:

        peak_ratio = (

            pulse_power[peak_index]

            /

            average_power
        )

    else:

        peak_ratio = 0


    return {

        "bpm": float(bpm),

        "peak_frequency": float(
            peak_frequency
        ),

        "peak_power": float(
            pulse_power[peak_index]
        ),

        "peak_ratio": float(
            peak_ratio
        ),

        "frequencies": pulse_frequencies,

        "power": pulse_power,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 75)

    print(
        "CONTACTLESS VITALS POC"
    )

    print(
        "PHASE 4 - rPPG SIGNAL ANALYSIS"
    )

    print("=" * 75)


    # ========================================================
    # CHECK CSV
    # ========================================================

    if not CSV_PATH.exists():

        print()

        print(
            "ERROR: Phase 3 CSV not found."
        )

        print(
            CSV_PATH
        )

        return


    # ========================================================
    # LOAD DATA
    # ========================================================

    dataframe = pd.read_csv(
        CSV_PATH
    )


    print()

    print(
        f"Loaded: {CSV_PATH.name}"
    )

    print(
        f"Rows: {len(dataframe)}"
    )


    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    required_columns = [

        "time_sec",

        "face_detected",

        "combined_g",

        "forehead_g",

        "left_cheek_g",

        "right_cheek_g",
    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in dataframe.columns
    ]


    if missing_columns:

        print()

        print(
            "ERROR: Missing CSV columns:"
        )

        print(
            missing_columns
        )

        return


    # ========================================================
    # KEEP VALID FACE FRAMES
    # ========================================================

    dataframe = dataframe[

        dataframe[
            "face_detected"
        ] == 1

    ].copy()


    # ========================================================
    # CALCULATE SAMPLING RATE
    # ========================================================

    time_values = dataframe[
        "time_sec"
    ].to_numpy()


    time_differences = np.diff(
        time_values
    )


    median_dt = np.median(
        time_differences
    )


    sampling_rate = (
        1 / median_dt
    )


    duration = (

        time_values[-1]

        -

        time_values[0]
    )


    print()

    print("-" * 75)

    print(
        f"Sampling rate : "
        f"{sampling_rate:.2f} Hz"
    )

    print(
        f"Duration      : "
        f"{duration:.2f} sec"
    )

    print(
        f"Valid frames  : "
        f"{len(dataframe)}"
    )

    print("-" * 75)


    # ========================================================
    # REMOVE BEGINNING / END
    # ========================================================

    start_time = (

        time_values[0]

        +

        START_IGNORE_SECONDS
    )


    end_time = (

        time_values[-1]

        -

        END_IGNORE_SECONDS
    )


    analysis_data = dataframe[

        (
            dataframe["time_sec"]
            >= start_time
        )

        &

        (
            dataframe["time_sec"]
            <= end_time
        )

    ].copy()


    analysis_time = (

        analysis_data[
            "time_sec"
        ].to_numpy()
    )


    # Make graph start at zero

    analysis_time = (

        analysis_time

        -

        analysis_time[0]
    )


    print()

    print(
        f"Analysing from "
        f"{start_time:.1f}s "
        f"to {end_time:.1f}s"
    )

    print(
        f"Analysis frames: "
        f"{len(analysis_data)}"
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    filtered_signals = {}

    analysis_results = {}


    # ========================================================
    # PROCESS EACH ROI
    # ========================================================

    for label, column in SIGNALS.items():

        raw_signal = (

            analysis_data[
                column
            ]
            .astype(float)
            .to_numpy()
        )


        # ----------------------------------------------------
        # Remove linear drift
        # ----------------------------------------------------

        signal_detrended = detrend(
            raw_signal
        )


        # ----------------------------------------------------
        # Normalise
        # ----------------------------------------------------

        signal_normalised = normalise_signal(
            signal_detrended
        )


        # ----------------------------------------------------
        # Band-pass
        # ----------------------------------------------------

        signal_filtered = bandpass_filter(

            signal_normalised,

            sampling_rate,

            LOW_CUT_HZ,

            HIGH_CUT_HZ,

            FILTER_ORDER
        )


        filtered_signals[
            label
        ] = signal_filtered


        # ----------------------------------------------------
        # BPM
        # ----------------------------------------------------

        result = calculate_bpm(

            signal_filtered,

            sampling_rate
        )


        analysis_results[
            label
        ] = result


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print("=" * 75)

    print(
        "CANDIDATE PULSE RESULTS"
    )

    print("=" * 75)


    for label, result in analysis_results.items():

        if result is None:

            print(
                f"{label:15} : "
                f"No result"
            )

            continue


        print(

            f"{label:15} : "

            f"{result['bpm']:6.1f} BPM"

            f" | Peak quality: "
            f"{result['peak_ratio']:.2f}"
        )


    # ========================================================
    # FIND STRONGEST REGION
    # ========================================================

    valid_results = {

        label: result

        for label, result
        in analysis_results.items()

        if result is not None
    }


    if valid_results:

        best_region = max(

            valid_results,

            key=lambda label:
            valid_results[
                label
            ]["peak_ratio"]
        )


        best_result = (

            valid_results[
                best_region
            ]
        )


        print()

        print("-" * 75)

        print(
            f"Strongest spectral region : "
            f"{best_region}"
        )

        print(
            f"Candidate pulse estimate  : "
            f"{best_result['bpm']:.1f} BPM"
        )

        print(
            f"Peak quality ratio        : "
            f"{best_result['peak_ratio']:.2f}"
        )

        print("-" * 75)


    # ========================================================
    # PLOT 1 - RAW GREEN SIGNAL
    # ========================================================

    plt.figure(
        figsize=(12, 5)
    )


    plt.plot(

        analysis_time,

        analysis_data[
            "combined_g"
        ].to_numpy()
    )


    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Average green intensity"
    )

    plt.title(
        "Raw Combined Facial Green Signal"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        RAW_PLOT,
        dpi=160
    )

    plt.close()


    # ========================================================
    # PLOT 2 - FILTERED rPPG
    # ========================================================

    plt.figure(
        figsize=(12, 5)
    )


    for label, signal in filtered_signals.items():

        plt.plot(

            analysis_time,

            signal,

            label=label,

            alpha=0.8
        )


    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Normalised amplitude"
    )

    plt.title(
        "Filtered Facial rPPG Signals"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        FILTERED_PLOT,
        dpi=160
    )

    plt.close()


    # ========================================================
    # PLOT 3 - FREQUENCY SPECTRUM
    # ========================================================

    plt.figure(
        figsize=(12, 5)
    )


    for label, result in analysis_results.items():

        if result is None:
            continue


        bpm_axis = (

            result[
                "frequencies"
            ]

            * 60
        )


        plt.plot(

            bpm_axis,

            result[
                "power"
            ],

            label=label,

            alpha=0.8
        )


    plt.xlabel(
        "Frequency (BPM)"
    )

    plt.ylabel(
        "Power"
    )

    plt.title(
        "rPPG Frequency Spectrum"
    )

    plt.xlim(
        LOW_CUT_HZ * 60,
        HIGH_CUT_HZ * 60
    )

    plt.legend()

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
            "PHASE 4 rPPG ANALYSIS\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(

            f"Sampling rate: "
            f"{sampling_rate:.2f} Hz\n"
        )

        file.write(

            f"Valid frames: "
            f"{len(dataframe)}\n"
        )

        file.write(

            f"Analysis frames: "
            f"{len(analysis_data)}\n\n"
        )


        for label, result in analysis_results.items():

            if result is None:
                continue

            file.write(

                f"{label}: "
                f"{result['bpm']:.1f} BPM "
                f"(peak ratio "
                f"{result['peak_ratio']:.2f})\n"
            )


        if valid_results:

            file.write(
                "\n"
            )

            file.write(

                f"Strongest region: "
                f"{best_region}\n"
            )

            file.write(

                f"Candidate BPM: "
                f"{best_result['bpm']:.1f}\n"
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("=" * 75)

    print(
        "PHASE 4 COMPLETE"
    )

    print("=" * 75)

    print()

    print(
        "Generated:"
    )

    print(
        f"1. {RAW_PLOT}"
    )

    print(
        f"2. {FILTERED_PLOT}"
    )

    print(
        f"3. {SPECTRUM_PLOT}"
    )

    print(
        f"4. {SUMMARY_FILE}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is an experimental camera-based "
        "pulse estimate, not a medical measurement."
    )

    print("=" * 75)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()