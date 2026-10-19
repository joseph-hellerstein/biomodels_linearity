'''Oscillation detection using Fast Fourier Transform (FFT).'''

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.signal import find_peaks  # type: ignore
from typing import List, Union


# Number of MAD multiples above the median power spectrum value used to set the
# noise floor.  Chosen large enough that white Gaussian noise does not produce
# spurious detections while still allowing clean oscillatory signals to pass.
_MAD_MULTIPLIER = 15


def findOscillations(
    df: pd.DataFrame,
    min_frequency: float = 0.0,
    max_frequency: Union[float, None] = None,
    height_threshold_multiplier: float = 3.0,
) -> List[float]:
    """Detect oscillation frequencies in a time-series DataFrame using FFT.

    For each column of the input DataFrame (assumed to be uniformly sampled),
    removes the mean, computes the one-sided power spectrum via ``numpy.fft.rfft``,
    and identifies peaks whose prominence exceeds an adaptive noise floor derived
    from the median absolute deviation (MAD) of the spectrum.

    Parameters
    ----------
    df : pd.DataFrame
        Time-series data.  Index is time (uniformly spaced), columns are variables.
    min_frequency : float, optional
        Lower bound on detected frequencies in Hz (cycles per unit time).
        Default ``0.0``.
    max_frequency : float or None, optional
        Upper bound on detected frequencies in Hz.  If *None*, uses Nyquist frequency
        of the data (half the sampling rate).  Default ``None``.
    height_threshold_multiplier : float, optional
        Controls detection sensitivity. A peak must have prominence at least
        ``max_power / height_threshold_multiplier``, where ``max_power`` is the
        largest spectral value for that column. Larger values require stronger
        oscillations and reject more noise; smaller values are more sensitive.
        Default ``3.0``.

    Returns
    -------
    list of float
        Sorted, deduplicated list of detected oscillation frequencies in Hz.
        Empty list if no peaks pass the threshold across all columns.

    Notes
    -----
    - The function assumes *uniform* time spacing within each column group (which is
      typical for ODE solver outputs).  It uses only one frequency axis, derived from
      the overall sampling rate of the data.
    - The DC component (frequency = 0) is always excluded from results since it
      corresponds to the mean, not an oscillation.
    - Peaks must have at least one neighbour on each side with lower power; this
      prevents flat plateaus or edge artefacts from being reported.
    - Columns containing non-finite values (NaN / Inf) are skipped silently.
    - Non-uniform time spacing or a zero time step yields an empty result rather
      than raising an error.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> t = np.linspace(0, 10, 500)
    >>> df = pd.DataFrame({"A": np.sin(2 * np.pi * 0.5 * t)}, index=t)
    >>> findOscillations(df)
    [0.5]
    """
    if len(df) < 4:
        return []

    time_index = df.index.to_numpy(dtype=float)
    dt_values = np.diff(time_index)

    # Reject non-uniform sampling or zero/negative time step.
    if (dt_values.size == 0 or not np.all(dt_values > 0)
            or not np.allclose(dt_values, dt_values[0], rtol=1e-6)):
        return []

    dt = float(np.mean(dt_values))
    if dt <= 0.0:
        return []

    n_points = len(df)
    sample_rate = 1.0 / dt
    nyquist = sample_rate / 2.0

    # Frequency axis for the one-sided FFT (includes DC at index 0).
    freqs = np.fft.rfftfreq(n_points, d=dt)

    if max_frequency is None:
        max_freq = nyquist
    else:
        if float(max_frequency) <= min_frequency:
            return []
        max_freq = float(max_frequency)

    detected_frequencies: set[float] = set()
    threshold_multiplier = float(height_threshold_multiplier)

    for col in df.columns:
        values = df[col].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            continue

        # Detrend (subtract mean): removes the DC component.
        detrended = values - np.mean(values)

        fft_values = np.fft.rfft(detrended)
        power_spectrum = (np.abs(fft_values) ** 2) / n_points

        max_power = float(np.max(power_spectrum))
        if max_power == 0.0:
            continue

        # Compute an adaptive noise floor using median + MAD (robust to outliers).
        spectrum_no_dc = power_spectrum[1:]
        median_power = float(np.median(spectrum_no_dc))
        mad = float(np.median(np.abs(spectrum_no_dc - median_power)))
        if mad <= 0.0:
            mad = max(median_power, 1e-30)

        # Noise floor: combine MAD-based estimate with a fallback to median.
        noise_floor = median_power + _MAD_MULTIPLIER * mad

        # Adaptive prominence threshold: peak must exceed both the noise floor
        # and ``max_power / multiplier`` above its local surroundings.
        prominence_threshold = max(noise_floor, max_power / threshold_multiplier)

        # Find all local peaks (excluding DC at index 0) with sufficient prominence.
        peak_indices, _ = find_peaks(
            power_spectrum[1:],
            prominence=prominence_threshold,
            distance=1,
        )

        for idx in peak_indices:
            freq = float(freqs[idx + 1])  # shift by 1 to align with full array
            if min_frequency <= freq <= max_freq:
                detected_frequencies.add(round(freq, 8))

    return sorted(detected_frequencies)
