import numpy as np
from numpy.typing import NDArray
import soundfile as sf

def convert_AtoB_tetramic(signals_Aformat: NDArray) -> NDArray:
    """ Convert Ambisonics signals from A-format to B-format using SN3D and ACN ordering.

    Paramaters
    -------------
    signals_Aformat: NDArray
        Input Ambisonic signals from a tetramic in A-format of shape (num_samples, num_channels).

    Returns
    -------------
    NDArray
        Signals in B-format of shape (num_time_samples, num_channels) with [w, x, y, z] ordering.
    
    """
 
    # Assume 4 unit vectors for tetrahedral mic (each row is [x, y, z])
    dirs = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ])

    # Divide directions by matrix norms
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    # Break down x, y, z from directional vectors into seperate arrays
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    theta = np.arccos(z)
    phi = np.arctan2(y, x)

    # Generate SN3D-Normalized real Spherical Harmonic basis functions
    #With ACN order: [Y0_0, Y1_-1, Y0_1, Y1_1] => [W, Y, Z, X]
    Y_00 = 1 / np.sqrt(4*np.pi) * np.ones_like(theta)
    Y_1m1 = np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.sin(phi)
    Y_10 = np.sqrt(3 / (4* np.pi)) * np.cos(theta)
    Y_11 = np.sqrt(3 / (4*np.pi)) * np.sin(theta) * np.cos(phi)

    # Stack Spherical Harmonic functions into array shape (num_mic_directions, num_channels)
    Y = np.column_stack((Y_00, Y_1m1, Y_10, Y_11))

    # Get the inverse matrix for an A to B transform
    Y_inv = np.linalg.inv(Y)

    # Matrix multiply the A-format signals with the inverted matrix to get B-format signals
    #Using einsum to imply summation
    signals_Bformat = (Y_inv @ signals_Aformat.T).T

    # Return B-format of shape (num_time_samples, num_channels):
    return signals_Bformat

def scale_Bformat_signals(signals_Bformat: NDArray,
                          left_scaler: float,
                          right_scaler: float,
                          top_scaler: float,
                          bottom_scaler: float,
                          front_scaler: float,
                          back_scaler: float) -> NDArray:
    """
    vector_left = ([1,
                    1 * np.cos(np.radians(0)) * np.sin(np.radians(90)),
                    0 * np.sin(np.radians(0)),
                    0 * np.cos(np.radians(0)) * np.cos(np.radians(90))
    ])

    vector_right = ([1,
                     1 * np.cos(np.radians(0)) * np.sin(np.radians(-90)),
                     0 * np.sin(np.radians(0)),
                     0 * np.cos(np.radians(0)) * np.cos(np.radians(-90))
    ])

    vector_top = ([1,
                   0 * np.cos(np.radians(90)) * np.sin(np.radians(0)),
                   1 * np.sin(np.radians(90)),
                   0 * np.cos(np.radians(90)) *np.cos(np.radians(0))
    ])
    
    vector_bottom = ([1,
                      0 * np.cos(np.radians(-90)) * np.sin(np.radians(0)),
                      1 * np.sin(np.radians(-90)),
                      0 * np.cos(np.radians(0)) * np.cos(np.radians(0))
    ])
    
    vector_front = ([1,
                     0 * np.cos(np.radians(0)) * np.sin(np.radians(180)),
                     0 * np.sin(np.radians(0)),
                     1 * np.cos(np.radians(0)) * np.cos(np.radians(0))
    ])

    vector_back = ([1,
                    0 * np.cos(np.radians(0)) * np.sin(np.radians(-180)),
                    0 * np.sin(np.radians(0)),
                    1 * np.cos(np.radians(0)) * np.cos(np.radians(-180))
    ])
    """

    left = np.array([0, 1 * left_scaler, 0, 0])
    right = np.array([0, -1 * right_scaler, 0, 0])
    top = np.array([0, 0, 1 * top_scaler, 0])
    bottom = np.array([0, 0, -1 * bottom_scaler, 0])
    front = np.array([0, 0, 0, 1 * front_scaler])
    back = np.array([0, 0, 0, -1 * back_scaler])

    left_signal = signals_Bformat[:, 1:] @ left[1:]
    right_signal = signals_Bformat[:, 1:] @ right[1:]
    top_signal = signals_Bformat[:, 1:] @ top[1:]
    bottom_signal = signals_Bformat [:, 1:] @ bottom[1:]
    front_signal = signals_Bformat [:, 1:] @ front[1:]
    back_signal = signals_Bformat [:, 1:] @ back[1:]


    w_signal = signals_Bformat[:, 0]
    y_signal = left_signal + right_signal
    z_signal = top_signal + bottom_signal
    x_signal = front_signal + back_signal

    new_Bformat = np.column_stack((w_signal, y_signal, z_signal, x_signal))

    eps = 1e-12

    E_orig = np.mean(np.sum(signals_Bformat**2, axis=1))
    E_new = np.mean(np.sum(new_Bformat**2, axis=1))

    gain = np.sqrt (E_orig / max(E_new, eps))

    new_Bformat *= gain

    return new_Bformat



    
