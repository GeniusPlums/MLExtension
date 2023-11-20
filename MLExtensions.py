import numpy as np
from matplotlib import colors
import pandas as pd
from scipy.signal import lfilter
from talib import ADX

def normalizeDeriv(src, quadraticMeanLength):
    """
    Returns the smoothed hyperbolic tangent of the input series.

    Parameters:
        src (np.array): The input series (i.e., the first-order derivative for price).
        quadraticMeanLength (int): The length of the quadratic mean (RMS).

    Returns:
        nDeriv (np.array): The normalized derivative of the input series.
    """
    # Calculate the quadratic mean (RMS)
    rms = np.sqrt(np.mean(np.square(src[-quadraticMeanLength:])))

    # Calculate the normalized derivative
    nDeriv = np.tanh(src / rms)

    return nDeriv

def normalize(src, min_val, max_val):
    """
    Rescales a source value with an unbounded range to a target range.

    Parameters:
        src (np.array): The input series
        min_val (float): The minimum value of the unbounded range
        max_val (float): The maximum value of the unbounded range

    Returns:
        norm_src (np.array): The normalized series
    """
    # Calculate the minimum and maximum of the input series
    src_min = np.min(src)
    src_max = np.max(src)

    # Rescale the series to the target range
    norm_src = (src - src_min) * (max_val - min_val) / (src_max - src_min) + min_val

    return norm_src

def rescale(src, oldMin, oldMax, newMin, newMax):
    """
    Rescales a source value with a bounded range to another bounded range.

    Parameters:
        src (np.array): The input series
        oldMin (float): The minimum value of the range to rescale from
        oldMax (float): The maximum value of the range to rescale from
        newMin (float): The minimum value of the range to rescale to
        newMax (float): The maximum value of the range to rescale to

    Returns:
        rescaled_src (np.array): The rescaled series
    """
    # Rescale the series to the new range
    rescaled_src = (src - oldMin) * (newMax - newMin) / (oldMax - oldMin) + newMin

    return rescaled_src

def color_green(prediction):
    """
    Assigns varying shades of the color green based on the KNN classification.

    Parameters:
        prediction (int|float): Value of the prediction

    Returns:
        color (tuple): RGB color tuple
    """
    # Normalize the prediction to the range [0, 1]
    normalized_prediction = min(max(prediction, 0), 1)

    # Create a colormap of varying shades of green
    cmap = colors.LinearSegmentedColormap.from_list("", ["white", "green"])

    # Get the RGB color corresponding to the normalized prediction
    color = cmap(normalized_prediction)

    return color

def color_red(prediction):
    """
    Assigns varying shades of the color red based on the KNN classification.

    Parameters:
        prediction (int|float): Value of the prediction

    Returns:
        color (tuple): RGB color tuple
    """
    # Normalize the prediction to the range [0, 1]
    normalized_prediction = min(max(prediction, 0), 1)

    # Create a colormap of varying shades of red
    cmap = colors.LinearSegmentedColormap.from_list("", ["white", "red"])

    # Get the RGB color corresponding to the normalized prediction
    color = cmap(normalized_prediction)

    return color

def tanh(src):
    """
    Returns the hyperbolic tangent of the input series. The sigmoid-like hyperbolic tangent function is used to compress the input to a value between -1 and 1.

    Parameters:
        src (np.array): The input series (i.e., the normalized derivative).

    Returns:
        tanh_src (np.array): The hyperbolic tangent of the input series.
    """
    # Calculate the hyperbolic tangent of the input series
    tanh_src = np.tanh(src)

    return tanh_src

def dualPoleFilter(src, lookback):
    """
    Returns the smoothed hyperbolic tangent of the input series.

    Parameters:
        src (np.array): The input series (i.e., the hyperbolic tangent).
        lookback (int): The lookback window for the smoothing.

    Returns:
        filter_src (np.array): The smoothed hyperbolic tangent of the input series.
    """
    # Calculate the smoothing factor
    smoothing_factor = 2 / (lookback + 1)

    # Initialize the filter series
    filter_src = np.zeros_like(src)

    # Calculate the first value of the filter series
    filter_src[0] = src[0]

    # Calculate the remaining values of the filter series
    for i in range(1, len(src)):
        filter_src[i] = (1 - smoothing_factor)**2 * (filter_src[i-1] + 2*smoothing_factor*src[i] - ((1 - smoothing_factor)**2)*filter_src[i-2])

    return filter_src

def tanhTransform(src, smoothingFrequency, quadraticMeanLength):
    """
    Returns the tanh transform of the input series.

    Parameters:
        src (np.array): The input series (i.e., the result of the tanh calculation).
        smoothingFrequency (float): The frequency for smoothing.
        quadraticMeanLength (int): The length of the quadratic mean (RMS).

    Returns:
        signal (np.array): The smoothed hyperbolic tangent transform of the input series.
    """
    # Calculate the quadratic mean (RMS)
    rms = np.sqrt(np.mean(np.square(src[-quadraticMeanLength:])))

    # Calculate the smoothed hyperbolic tangent transform
    signal = np.tanh(smoothingFrequency * src / rms)

    return signal

def n_rsi(src, n1, n2):
    """
    Returns the normalized RSI ideal for use in ML algorithms.

    Parameters:
        src (np.array): The input series (i.e., the result of the RSI calculation).
        n1 (int): The length of the RSI.
        n2 (int): The smoothing length of the RSI.

    Returns:
        signal (np.array): The normalized RSI.
    """
    # Calculate the difference between consecutive elements in the series
    delta = src.diff()

    # Separate the positive and negative differences
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the average gain and average loss
    avg_gain = up.rolling(window=n1, min_periods=0).mean()
    avg_loss = abs(down.rolling(window=n1, min_periods=0).mean())

    # Calculate the RS (Relative Strength)
    rs = avg_gain / avg_loss

    # Calculate the RSI (Relative Strength Index)
    rsi = 100 - (100 / (1 + rs))

    # Normalize the RSI
    rsi = (rsi - rsi.rolling(window=n2).min()) / (rsi.rolling(window=n2).max() - rsi.rolling(window=n2).min())

    return rsi

def n_cci(src, n1, n2):
    """
    Returns the normalized CCI ideal for use in ML algorithms.

    Parameters:
        src (np.array): The input series (i.e., the result of the CCI calculation).
        n1 (int): The length of the CCI.
        n2 (int): The smoothing length of the CCI.

    Returns:
        signal (np.array): The normalized CCI.
    """
    # Calculate the CCI
    cci = (src - pd.Series(src).rolling(window=n1).mean()) / (0.015 * pd.Series(src).rolling(window=n1).std())

    # Normalize the CCI
    signal = (cci - cci.rolling(window=n2).min()) / (cci.rolling(window=n2).max() - cci.rolling(window=n2).min())

    return signal

def n_wt(src, n1, n2):
    """
    Returns the normalized WaveTrend Classic series ideal for use in ML algorithms.

    Parameters:
        src (np.array): The input series (i.e., the result of the WaveTrend Classic calculation).
        n1 (int): The length of the WaveTrend Classic series.
        n2 (int): The smoothing length of the WaveTrend Classic series.

    Returns:
        signal (np.array): The normalized WaveTrend Classic series.
    """
    # Calculate the WaveTrend Classic series
    wt = (src - pd.Series(src).rolling(window=n1).mean()) / (0.015 * pd.Series(src).rolling(window=n1).std())

    # Normalize the WaveTrend Classic series
    signal = (wt - wt.rolling(window=n2).min()) / (wt.rolling(window=n2).max() - wt.rolling(window=n2).min())

    return signal

def n_adx(highSrc, lowSrc, closeSrc, n1):
    """
    Returns the normalized ADX ideal for use in ML algorithms.

    Parameters:
        highSrc (np.array): The input series for the high price.
        lowSrc (np.array): The input series for the low price.
        closeSrc (np.array): The input series for the close price.
        n1 (int): The length of the ADX.

    Returns:
        signal (np.array): The normalized ADX.
    """
    # Calculate the ADX
    adx = ADX(highSrc, lowSrc, closeSrc, timeperiod=n1)

    # Normalize the ADX
    signal = (adx - np.min(adx)) / (np.max(adx) - np.min(adx))

    return signal
