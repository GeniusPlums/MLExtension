import numpy as np
from matplotlib import colors

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
