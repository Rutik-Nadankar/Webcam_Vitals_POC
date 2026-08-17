from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import (
    butter,
    filtfilt,
    detrend,
    welch,
    hilbert,
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

SIGNAL_PLOT = (
    RESULTS_DIR
    / "phase6_respiratory_signals.png"
)

SPECTRUM_PLOT = (
    RESULTS_DIR
    / "phase6_respiratory_spectrum.png"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "phase6_respiratory_summary.txt"
)


# ============================================================
# SETTINGS
# ============================================================

# Respiratory frequency range for POC

RESP_LOW_HZ = 0.10
RESP_HIGH_HZ = 0.50

# Pulse range

PULSE_LOW_HZ = 0.75
PULSE_HIGH_HZ = 2.50

START_IGNORE_SECONDS = 3
END_IGNORE_SECONDS = 2

FILTER_ORDER = 4


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass(
    signal,
    fs,
    low_hz,
    high_hz
):

    nyquist = fs / 2

    low = (
        low_hz / nyquist
    )

    high = (
        high_hz / nyquist
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
# NORMALISE
# ============================================================

def normalise(signal):

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
# POS SIGNAL
# ============================================================

def pos_signal(
    r,
    g,
    b
):

    rgb = np.vstack([
        r,
        g,
        b
    ])


    means = np.mean(
        rgb,
        axis=1,
        keepdims=True
    )


    means[
        means == 0
    ] = 1


    normalized_rgb = (
        rgb / means
    )


    # POS projections

    x = (
        normalized_rgb[1]
        - normalized_rgb[2]
    )


    y = (
        normalized_rgb[1]
        + normalized_rgb[2]
        - 2 * normalized_rgb[0]
    )


    std_y = np.std(
        y
    )


    if std_y == 0:

        alpha = 0

    else:

        alpha = (
            np.std(x)
            / std_y
        )


    signal = (
        x
        + alpha * y
    )


    return normalise(
        signal
    )


# ============================================================
# ESTIMATE RESPIRATORY RATE
# ============================================================

def estimate_rr(
    signal,
    fs
):

    signal = normalise(
        signal
    )


    # --------------------------------------------------------
    # Power spectrum
    # --------------------------------------------------------

    nperseg = len(
        signal
    )


    # Zero padding gives us a smoother spectrum.
    # It does not create new information,
    # but makes peak location easier to inspect.

    nfft = 16384


    frequencies, power = welch(

        signal,

        fs=fs,

        nperseg=nperseg,

        nfft=nfft
    )


    valid = (
        (frequencies >= RESP_LOW_HZ)
        &
        (frequencies <= RESP_HIGH_HZ)
    )


    resp_freq = (
        frequencies[valid]
    )

    resp_power = (
        power[valid]
    )


    if len(resp_power) == 0:

        return None


    # --------------------------------------------------------
    # Spectral peak
    # --------------------------------------------------------

    peaks, _ = find_peaks(
        resp_power,
        prominence=(
            np.max(resp_power)
            * 0.05
        )
    )


    if len(peaks) > 0:

        strongest = peaks[
            np.argmax(
                resp_power[peaks]
            )
        ]

    else:

        strongest = np.argmax(
            resp_power
        )


    peak_frequency = (
        resp_freq[
            strongest
        ]
    )


    rr_psd = (
        peak_frequency
        * 60
    )


    mean_power = np.mean(
        resp_power
    )


    if mean_power > 0:

        spectral_quality = (
            resp_power[
                strongest
            ]
            / mean_power
        )

    else:

        spectral_quality = 0


    # ========================================================
    # AUTOCORRELATION CHECK
    # ========================================================

    autocorr = np.correlate(
        signal,
        signal,
        mode="full"
    )


    autocorr = autocorr[
        len(signal) - 1:
    ]


    if autocorr[0] != 0:

        autocorr = (
            autocorr
            / autocorr[0]
        )


    # Respiration period:
    #
    # 0.5 Hz = 2 seconds
    # 0.1 Hz = 10 seconds

    min_lag = int(
        fs / RESP_HIGH_HZ
    )

    max_lag = int(
        fs / RESP_LOW_HZ
    )


    search_region = autocorr[
        min_lag:max_lag + 1
    ]


    if len(search_region) == 0:

        rr_autocorr = np.nan
        autocorr_quality = 0

    else:

        ac_peaks, _ = find_peaks(
            search_region
        )


        if len(ac_peaks) > 0:

            best_peak = ac_peaks[
                np.argmax(
                    search_region[
                        ac_peaks
                    ]
                )
            ]

        else:

            best_peak = np.argmax(
                search_region
            )


        lag = (
            min_lag
            + best_peak
        )


        period_seconds = (
            lag / fs
        )


        if period_seconds > 0:

            rr_autocorr = (
                60 / period_seconds
            )

        else:

            rr_autocorr = np.nan


        autocorr_quality = (
            search_region[
                best_peak
            ]
        )


    return {

        "rr_psd": float(
            rr_psd
        ),

        "rr_autocorr": float(
            rr_autocorr
        ),

        "spectral_quality": float(
            spectral_quality
        ),

        "autocorr_quality": float(
            autocorr_quality
        ),

        "frequencies": (
            resp_freq
        ),

        "power": (
            resp_power
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
        "PHASE 6 - RESPIRATORY RATE"
    )

    print("=" * 76)


    # ========================================================
    # LOAD CSV
    # ========================================================

    if not CSV_PATH.exists():

        print(
            "ERROR: Phase 3 RGB CSV not found."
        )

        return


    df = pd.read_csv(
        CSV_PATH
    )


    df = df[
        df[
            "face_detected"
        ] == 1
    ].copy()


    # ========================================================
    # SAMPLING RATE
    # ========================================================

    time = df[
        "time_sec"
    ].to_numpy(
        dtype=float
    )


    dt = np.median(
        np.diff(time)
    )


    fs = (
        1 / dt
    )


    print()

    print(
        f"Sampling rate : "
        f"{fs:.2f} Hz"
    )


    # ========================================================
    # REMOVE START / END
    # ========================================================

    start_time = (
        time[0]
        + START_IGNORE_SECONDS
    )


    end_time = (
        time[-1]
        - END_IGNORE_SECONDS
    )


    analysis = df[
        (
            df["time_sec"]
            >= start_time
        )
        &
        (
            df["time_sec"]
            <= end_time
        )
    ].copy()


    analysis_time = (
        analysis[
            "time_sec"
        ].to_numpy(
            dtype=float
        )
    )


    analysis_time = (
        analysis_time
        - analysis_time[0]
    )


    print(
        f"Analysis range: "
        f"{start_time:.1f}s - "
        f"{end_time:.1f}s"
    )


    print(
        f"Analysis frames: "
        f"{len(analysis)}"
    )


    # ========================================================
    # RGB
    # ========================================================

    r = analysis[
        "combined_r"
    ].to_numpy(
        dtype=float
    )


    g = analysis[
        "combined_g"
    ].to_numpy(
        dtype=float
    )


    b = analysis[
        "combined_b"
    ].to_numpy(
        dtype=float
    )


    # ========================================================
    # METHOD 1
    #
    # PULSE AMPLITUDE MODULATION
    # ========================================================

    pulse = pos_signal(
        r,
        g,
        b
    )


    pulse_filtered = bandpass(

        pulse,

        fs,

        PULSE_LOW_HZ,

        PULSE_HIGH_HZ
    )


    # Hilbert envelope

    pulse_envelope = np.abs(
        hilbert(
            pulse_filtered
        )
    )


    pulse_envelope = normalise(
        pulse_envelope
    )


    respiration_am = bandpass(

        pulse_envelope,

        fs,

        RESP_LOW_HZ,

        RESP_HIGH_HZ
    )


    am_result = estimate_rr(

        respiration_am,

        fs
    )


    # ========================================================
    # METHOD 2
    #
    # SLOW FACIAL BRIGHTNESS MODULATION
    # ========================================================

    brightness = (
        r + g + b
    ) / 3


    brightness = normalise(
        brightness
    )


    respiration_brightness = bandpass(

        brightness,

        fs,

        RESP_LOW_HZ,

        RESP_HIGH_HZ
    )


    brightness_result = estimate_rr(

        respiration_brightness,

        fs
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()

    print("=" * 76)

    print(
        "RESPIRATORY CANDIDATES"
    )

    print("=" * 76)


    if am_result:

        print()

        print(
            "Pulse amplitude modulation:"
        )

        print(
            f"  PSD estimate       : "
            f"{am_result['rr_psd']:.1f} "
            f"breaths/min"
        )

        print(
            f"  Autocorrelation    : "
            f"{am_result['rr_autocorr']:.1f} "
            f"breaths/min"
        )

        print(
            f"  Spectral quality   : "
            f"{am_result['spectral_quality']:.2f}"
        )

        print(
            f"  Autocorr quality   : "
            f"{am_result['autocorr_quality']:.2f}"
        )


    if brightness_result:

        print()

        print(
            "Facial brightness modulation:"
        )

        print(
            f"  PSD estimate       : "
            f"{brightness_result['rr_psd']:.1f} "
            f"breaths/min"
        )

        print(
            f"  Autocorrelation    : "
            f"{brightness_result['rr_autocorr']:.1f} "
            f"breaths/min"
        )

        print(
            f"  Spectral quality   : "
            f"{brightness_result['spectral_quality']:.2f}"
        )

        print(
            f"  Autocorr quality   : "
            f"{brightness_result['autocorr_quality']:.2f}"
        )


    # ========================================================
    # SIMPLE CONSENSUS
    # ========================================================

    candidates = []


    if am_result:

        if (
            am_result[
                "spectral_quality"
            ] >= 3
        ):

            candidates.append(
                am_result[
                    "rr_psd"
                ]
            )


    if brightness_result:

        if (
            brightness_result[
                "spectral_quality"
            ] >= 3
        ):

            candidates.append(
                brightness_result[
                    "rr_psd"
                ]
            )


    # ========================================================
    # FINAL ESTIMATE
    # ========================================================

    final_rr = np.nan

    quality_label = "LOW"


    if len(candidates) >= 2:

        difference = abs(
            candidates[0]
            - candidates[1]
        )


        if difference <= 3:

            final_rr = np.mean(
                candidates
            )

            quality_label = "GOOD"


        elif difference <= 5:

            final_rr = np.median(
                candidates
            )

            quality_label = "MODERATE"


        else:

            quality_label = "LOW"


    elif len(candidates) == 1:

        final_rr = candidates[0]

        quality_label = "LOW"


    print()

    print("=" * 76)

    print(
        "FINAL POC RESPIRATORY RATE"
    )

    print("=" * 76)


    if np.isnan(
        final_rr
    ):

        print(
            "No reliable consensus respiratory "
            "rate detected."
        )

    else:

        print(
            f"Estimated RR     : "
            f"{final_rr:.1f} breaths/min"
        )


    print(
        f"Signal quality   : "
        f"{quality_label}"
    )


    # ========================================================
    # SIGNAL PLOT
    # ========================================================

    plt.figure(
        figsize=(12, 5)
    )


    plt.plot(

        analysis_time,

        respiration_am,

        label=(
            "Pulse amplitude modulation"
        )
    )


    plt.plot(

        analysis_time,

        respiration_brightness,

        label=(
            "Brightness modulation"
        ),

        alpha=0.7
    )


    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Normalised amplitude"
    )

    plt.title(
        "Respiratory Candidate Signals"
    )

    plt.legend()

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
    # SPECTRUM PLOT
    # ========================================================

    plt.figure(
        figsize=(12, 5)
    )


    if am_result:

        plt.plot(

            am_result[
                "frequencies"
            ] * 60,

            am_result[
                "power"
            ],

            label=(
                "Pulse amplitude"
            )
        )


    if brightness_result:

        plt.plot(

            brightness_result[
                "frequencies"
            ] * 60,

            brightness_result[
                "power"
            ],

            label=(
                "Brightness"
            )
        )


    plt.xlabel(
        "Respiratory Rate "
        "(breaths/min)"
    )

    plt.ylabel(
        "Power"
    )

    plt.title(
        "Respiratory Frequency Spectrum"
    )


    plt.xlim(
        RESP_LOW_HZ * 60,
        RESP_HIGH_HZ * 60
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
            "PHASE 6 RESPIRATORY RATE\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            f"Sampling rate: "
            f"{fs:.2f} Hz\n"
        )


        if am_result:

            file.write(
                f"\nPulse amplitude RR: "
                f"{am_result['rr_psd']:.1f} "
                f"breaths/min\n"
            )

            file.write(
                f"Pulse amplitude quality: "
                f"{am_result['spectral_quality']:.2f}\n"
            )


        if brightness_result:

            file.write(
                f"\nBrightness RR: "
                f"{brightness_result['rr_psd']:.1f} "
                f"breaths/min\n"
            )

            file.write(
                f"Brightness quality: "
                f"{brightness_result['spectral_quality']:.2f}\n"
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
            f"Signal quality: "
            f"{quality_label}\n"
        )


    print()

    print("=" * 76)

    print(
        "PHASE 6 COMPLETE"
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