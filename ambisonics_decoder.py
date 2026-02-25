import numpy as np
from numpy.typing import NDArray 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sofar as sf

def convert_AtoB_tetramic(signals_Aformat: NDArray) -> NDArray:
    """Convert Ambisonic signals from A-format to B-format using SN3D and ACN ordering.
   
    Parameters
    ----------
    signals_Aformat : NDArray
        Input Ambisonic signals from tetramic in A-Format of shape (num_samples, num_channels).
    
    Returns
    ----------
    NDArray
        Signals in B-Format of shape (num_time_samples, num_channels) with [w, x, y, z] ordering.
        
    """
    
    # Raise Value Error if array contains more or less than 4 channels

    if signals_Aformat.shape()
    # Assume 4 unit vectors for tetrahedral mic (each row is [x, y, z])
    dirs = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1], 
        ],dtype=np.float64)
    
    # Divide directions by matrix norms
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    # Break down x, y, z from direction vectors
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    theta = np.arccos(z)
    phi = np.arctan2(y, x)

    #Generate SN3D-Normalized real Spherical Harmonic basis functions
    #With ACN order: [Y0_0, Y1_-1, Y0_1, Y1_1] => [W, Y, Z, X]
    Y_00 = (1 / np.sqrt(4 * np.pi)) * np.ones_like(theta)
    Y_1m1 = np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.sin(phi)
    Y_10 = np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
    Y_11 = np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.cos(phi)

    #Stack Spherical Harmonic functions into shape (num_mic_dirs, num_channels)
    Y = np.column_stack((Y_00, Y_1m1, Y_10, Y_11))

    #Invert the matrix to get A to B transform
    Y_inv = np.linalg.inv(Y)

    #Multiply the A-format signals with the inverted matrix to get B-format signals
    #Using einsum to imply summation
    signals_Bformat = (Y_inv @ signals_Aformat.T).T

    #Return B-format signals of shape: (num_time_samples, num_channels):
    return signals_Bformat

def convert_AtoB_zylia(signals_Aformat: NDArray) -> NDArray:
    
def convert_AtoB_format(signals_Aformat: NDArray, mictype: str) -> NDArray:
    if (mictype == "Tetramic"):
        b_format = convert_AtoB_tetramic(signals_Aformat)
    elif (mictype == "Zylia"):
        b_format = convert_AtoB_zylia(signals_Aformat)
    else:
        raise ValueError("Wrong Microphone format used. Please enter either a" \
        " Tetramic A-format file or a Zylia A-format file.")
    return b_format
def combine_two_signals_BFormat(signals_Bformat1: NDArray, signals_Bformat2: NDArray,
                                z2_is_down = False, z2_is_up = False,
                                x2_is_front = False, x2_is_back = False,
                                y2_is_left = False, y2_is_right = False,
                                w_scale = 1.0) -> NDArray:
    """ Takes two Ambisonic B-format signals and returns their combined product
        to emphasize beams of different spheres based on user input.
        
        Parameters
        ----------
        
        signals_Bformat1: NDArray
            Input Ambisonic signals from one tetramic in B-format of shape (num_time_samples, num_channels).
        signals_Bformat2: NDArray
            Input Ambisonic signals from another tetramic in B-format of shape (num_time_samples, num_channels).
        z2_is_down: Boolean
            Whether or not to incorporate downwards facing Z-domain of the second signal
        z2_is_up: Boolean
            Whether or not to incorporate upwards facing Z-domain of the second signal
        x2_is_left: Boolean
            Whether or not to incorporate left facing X-domain of the second signal
        x2_is_right: Boolean
            Whether or not to incorporate right facing X-domain of the second signal
        y2_is_front: Boolean
            Whether or not to incorporate front facing Y-domain of the second signal

        If the second signal is not instantiated for each boolean, the first signal is automatically used to represent the directions.

        w_scale: Float
            Scaling factor for W domains between the two combined Ws. If set to 0, the Second W signal is represented fully.
            If set to 1, the First W signal is represented fully. Set to 0.5, both Ws are represented equally.
        Returns
        ---------
        NDArray
            Signals in B-format of shape (num_time_samples, num_channels) with combined [w, y, z, x] ordering.
        """
    
    #Generate vector arrays for each direction
    
    # Azimuth = angle in horizontal plane (radians)
    # Elevation = angle up/down from horizontal plane (radians)
   
    
    #+Y
    vector_left = ([1,
                    1 * np.cos(np.radians(0)) * np.sin(np.radians(90)),
                    0 * np.sin(np.radians(0)),
                    0 * np.cos(np.radians(0)) * np.cos(np.radians(90))
    ])
    #-Y
    vector_right = ([1,
                   1 * np.cos(np.radians(0)) * np.sin(np.radians(-90)),
                   0 * np.sin(np.radians(0)),
                   1 * np.cos(np.radians(0)) * np.cos(np.radians(-90))
    ])
    #+Z
    vector_top = ([1,
                     0 * np.cos(np.radians(90)) * np.sin(np.radians(0)),
                     1 * np.sin(np.radians(90)),
                     0 * np.cos(np.radians(90)) * np.cos(np.radians(0))
    ])
    #-Z
    vector_bottom = ([1,
                     0 * np.cos(np.radians(-90)) * np.sin(np.radians(0)),
                     1 * np.sin(np.radians(-90)),
                     0 * np.cos(np.radians(-90)) * np.cos(np.radians(0))
    ])
    #+X
    vector_front = ([1,
                   0 * np.cos(np.radians(0)) * np.sin(np.radians(0)),
                   0 * np.sin(np.radians(0)),
                   1 * np.cos(np.radians(0)) * np.cos(np.radians(0))
    ])
    #-X
    vector_back = ([1,
                    0 * np.cos(np.radians(0)) * np.sin(np.radians(180)),
                    0 * np.sin(np.radians(0)),
                    1 * np.cos(np.radians(0)) * np.cos(np.radians(180))
    ])


    signals_Bformat1 = np.nan_to_num(signals_Bformat1, nan=0.0, posinf=0.0, neginf=0.0)
    signals_Bformat2 = np.nan_to_num(signals_Bformat2, nan=0.0, posinf=0.0, neginf=0.0)
    vector_left = np.nan_to_num(vector_left, nan=0.0, posinf=0.0, neginf=0.0)
    vector_right = np.nan_to_num(vector_right, nan=0.0, posinf=0.0, neginf=0.0)
    vector_top = np.nan_to_num(vector_top, nan=0.0, posinf=0.0, neginf=0.0)
    vector_bottom = np.nan_to_num(vector_bottom, nan=0.0, posinf=0.0, neginf=0.0)
    vector_front = np.nan_to_num(vector_front, nan=0.0, posinf=0.0, neginf=0.0)
    vector_back = np.nan_to_num(vector_back, nan=0.0, posinf=0.0, neginf=0.0)

    #Extract bottom, top, left, right, front, back signals
    if (z2_is_down):
        bottom_signal = signals_Bformat2[:, 1:] @ vector_bottom[1:]
        print(np.where(np.isnan(bottom_signal)))
    else :
        bottom_signal = signals_Bformat1[:, 1:] @ vector_bottom[1:]
        print(np.where(np.isnan(bottom_signal)))
    if (z2_is_up):
        top_signal = signals_Bformat2[:, 1:] @ vector_top[1:]
        print(np.where(np.isnan(top_signal)))
    else :
        top_signal = signals_Bformat1[:, 1:] @ vector_top[1:]
        print(np.where(np.isnan(top_signal)))
    if (y2_is_left):
        left_signal = signals_Bformat2[:, 1:] @ vector_left[1:]
        print(np.where(np.isnan(left_signal)))
    else :
        left_signal = signals_Bformat1[:, 1:] @ vector_left[1:]
        print(np.where(np.isnan(left_signal)))
    if (y2_is_right):
        right_signal = signals_Bformat2[:, 1:] @ vector_right[1:]
        print(np.where(np.isnan(right_signal)))
    else :
        right_signal = signals_Bformat1[:, 1:] @ vector_right[1:]
        print(np.where(np.isnan(right_signal)))
    if (x2_is_front):
        front_signal = signals_Bformat2[:, 1:] @ vector_front[1:]
        print(np.where(np.isnan(front_signal)))
    else :
        front_signal = signals_Bformat1[:, 1:] @ vector_front[1:]
        print(np.where(np.isnan(front_signal)))
    if (x2_is_back):
        back_signal = signals_Bformat2[:, 1:] @ vector_back[1:]
        print(np.where(np.isnan(back_signal)))
    else :
        back_signal = signals_Bformat1[:, 1:] @ vector_back[1:]
        print(np.where(np.isnan(back_signal)))

    #Combine the signals together to create new X, Y, Z vectors
    w_scale2 = 1-w_scale
    combined_w = w_scale * signals_Bformat1[:, 0] + w_scale2 * signals_Bformat2[:, 0]
    combined_y = front_signal + back_signal
    combined_z = top_signal + bottom_signal
    combined_x = right_signal + left_signal

    #Generate a new B-Format signal with newly generated vectors
    num_samples = signals_Bformat1.shape[0]
    new_Bformat = np.zeros((num_samples, 4))
    new_Bformat[:, 0] = combined_w
    new_Bformat[:, 1] = combined_y
    new_Bformat[:, 2] = combined_z
    new_Bformat[:, 3] = combined_x

    #Return new B-Format Signal
    return new_Bformat

def plot_Bformat_comparison_on_sphere(signals_Bformat1: NDArray, signals_Bformat2: NDArray, signals_Combined: NDArray, title="B-format Comparison"):
    """ Plots and compares two combined signals on a color-coded sphere.

    Parameters
    ----------
    signals_Bformat1 : NDArray
        Input B-format signal from a tetramic of shape (num_samples, 4) in [W, Y, Z, X] order.
    signals_bformat2: NDArray
        Input another B-format signal from a tetramic of shape (num_samples, 4) in [W, Y, Z, X] order.
    signals_Combined: NDArray
        Input the combined result of both signals outputted by the masking function of shape (num_samples, 4) in [W, Y, Z, X] order.
    title : str
        Title of the figure.
    """
    #Create spherical grid
    n_theta, n_phi = 200, 100
    theta = np.linspace(0, 2 * np.pi, n_theta) #azimuth
    phi = np.linspace(-np.pi / 2, np.pi / 2, n_phi) #elevation
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    signals_Bformat1 = signals_Bformat1 / np.max(np.abs(signals_Bformat1))
    signals_Bformat2 = signals_Bformat2 / np.max(np.abs(signals_Bformat2))
    signals_Combined = signals_Combined / np.max(np.abs(signals_Combined))

    E1_rms = np.sqrt(np.mean(signals_Bformat1**2, axis=0))
    E2_rms = np.sqrt(np.mean(signals_Bformat2**2, axis=0))
    Ec_rms = np.sqrt(np.mean(signals_Combined**2, axis=0))

    E1_rms /= np.max(E1_rms)
    E2_rms /= np.max(E2_rms)
    Ec_rms /= np.max(Ec_rms)

    E1_grid = (E1_rms[0] 
               + E1_rms[1]*np.sin(theta_grid)*np.cos(phi_grid)
               + E1_rms[2]*np.sin(phi_grid)
               + E1_rms[3]*np.cos(theta_grid)*np.cos(phi_grid))
    E2_grid = (E2_rms[0]
               + E2_rms[1]*np.sin(theta_grid)*np.cos(phi_grid)
               + E2_rms[2]*np.sin(phi_grid)
               +E2_rms[3]*np.cos(theta_grid)*np.cos(phi_grid))
    Ec_grid = (Ec_rms[0]
               + Ec_rms[1]*np.sin(theta_grid)*np.cos(phi_grid)
               + Ec_rms[2]*np.sin(phi_grid)
               + Ec_rms[3]*np.cos(theta_grid)*np.cos(phi_grid))

    ratio1 = np.abs(E1_grid) / (np.abs(E1_grid) + np.abs(E2_grid) + 1e-8)

    cmap = cm.get_cmap('RdBu_r')
    colors = cmap(ratio1)
    r = np.ones_like(ratio1)


    Ec_norm = (Ec_grid - Ec_grid.min()) / (Ec_grid.max() - Ec_grid.min() + 1e-12)
    r = 0.3 + 0.7 * Ec_norm
    Xs = r * np.cos(theta_grid) * np.cos(phi_grid)
    Ys = r * np.sin(theta_grid) * np.cos(phi_grid)
    Zs = r * np.sin(phi_grid)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xs, Ys, Zs, facecolors=colors,
                           linewidth = 0, antialiased=True, shade=False)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(ratio1)
    fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.04, label='Signal 1 Ratio')
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='Y+')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Z+')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='X+')
    ax.legend()
   
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    plt.show()




    

