import numpy as np
import torch
from fastcospy import FastCOSPYEmulator

import matplotlib.pyplot as plt
from matplotlib import colors

B_MAX = 5.
LOCAL_SOURCE_INTERCEPT = 1.5
GLOBAL_SOURCE_INTERCEPT = 1.3
D_MAX = 32.
EPSILON = 1.
GAS_INTERCEPT = -4.5
GAS_SCALE = 3.5
ENERGY_CENTER = np.array([-20.5, -21.5, -22., -23, -23.5, -24.5, -26.]) # The output of the model is normalized to be roughly between -1 and 1. To recover the log of the flux in physical units at energy i_energy, one should add to the output ENERGY_CENTER[i_energy].

def compute_diffusion_coefficient_galprop(halo_size: float, energy: float) -> float:
    """Helper function. Computes the diffusion coefficient using the GALPROP model."""
    diffusion_coefficient = 3.311e-8 * 1.33 * halo_size * np.power(energy / 3.e9, 1. / 3.)
    return diffusion_coefficient


def compute_time_input_double(ages: np.ndarray, halo_size: float, energy: float, reduction_factor: float = 100., age_of_change: float = 1.e4, change_speed: float = 100.) -> np.ndarray:
    """Helper function. Computes the time input necessary for the calculation of the source count."""
    diffusion_coefficient_galprop = compute_diffusion_coefficient_galprop(halo_size, energy)
    reduced_diffusion_coefficient = diffusion_coefficient_galprop / reduction_factor
    alpha_parameter = reduction_factor - 1.
    argument_of_cosh = change_speed * (ages - age_of_change)

    time_input = np.logaddexp(argument_of_cosh, -argument_of_cosh) - np.log(2.)
    return reduced_diffusion_coefficient * (((alpha_parameter + 2.) / 2.) * ages + (alpha_parameter / (2. * change_speed)) * (time_input + np.log(2) - change_speed * age_of_change))

def compute_approximator(time_input: np.ndarray, halo_size: float) -> np.ndarray:
    """Helper function. Computes the approximator for the source count calculation."""
    approximator = (np.power((1 + 1.5 * time_input / (halo_size * halo_size)), 1.25) * np.exp(-np.power((1.5 * time_input / (halo_size * halo_size)), 0.97)))
    return approximator

def build_grid(n_lon: int, n_lat: int) -> tuple[np.ndarray, np.ndarray]:
    """Helper function. Builds the grid of the coordinates of the pixels."""
    lon_edges = np.linspace(-180.25, 179.75, n_lon + 1)
    lat_edges = np.linspace(-5.25,5.25, n_lat + 1)

    lon_center = (lon_edges[:-1] + lon_edges[1:]) / 2.
    lat_center = (lat_edges[:-1] + lat_edges[1:]) / 2.

    return lon_center, lat_center

def fill_geometric_channels(lon_center: np.ndarray, lat_center: np.ndarray, log_halo_size: float, number_of_channels: int) -> np.ndarray:
    """Processing function. Fills the geometric channels of the input features and normalizes them."""
    features = np.zeros((number_of_channels, len(lat_center), len(lon_center)), dtype=np.float32)
    
    features[0,:,:] = np.cos(np.radians(lon_center[None, :])) # Normalized between -1 and 1.
    features[1,:,:] = np.sin(np.radians(lon_center[None, :])) # Normalized between -1 and 1.
    features[2,:,:] = lat_center[:, None] / B_MAX # Normalized between -1 and 1 after dividing by B_MAX.
    features[3,:,:] = log_halo_size # Normalized between 0 and 1 naturally.

    return features


def compute_angular_scale(time_input: np.ndarray, source_dis: np.ndarray) -> np.ndarray:
    """Helper function. Computes the angular scale for the source count calculation."""
    return np.degrees(np.arctan(np.sqrt(2. * time_input) / source_dis))

def compute_diff_lon_and_lat(angular_scale: np.ndarray, lon_center: np.ndarray, source_lon: np.ndarray, lat_center: np.ndarray, source_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Helper function. Computes the difference in longitude and latitude for the source count calculation."""
    source_lon = np.where(2. * angular_scale < 0.5, np.round(source_lon / 0.5) * 0.5, source_lon)
    source_lat = np.where(2. * angular_scale < 0.5, np.round(source_lat / 0.5) * 0.5, source_lat)
    
    diff_lon = np.zeros((len(lat_center), len(lon_center), len(source_lon)))
    diff_lon = np.abs(lon_center[None, :, None] - source_lon[None, None, :])
    diff_lon = np.where(diff_lon > 180., 360. - diff_lon, diff_lon)
    square_diff_lon = np.power(diff_lon, 2)
    
    square_diff_lat = np.zeros((len(lat_center), len(lon_center), len(source_lon)))
    square_diff_lat = np.power(lat_center[:, None, None] - source_lat[None, None, :], 2)

    return square_diff_lon, square_diff_lat


def compute_source_count(time_input: np.ndarray, sources: np.ndarray, lon_center: np.ndarray, lat_center: np.ndarray, approximator: np.ndarray) -> np.ndarray:
    """Helper function. Computes the source count for each pixel and each source, which is necessary for the calculation of the source features."""
    angular_scale = compute_angular_scale(time_input, sources[:, 3])
    square_diff_lon, square_diff_lat = compute_diff_lon_and_lat(angular_scale, lon_center, sources[:, 1], lat_center, sources[:, 2])

    source_count_detailed = np.zeros((len(lat_center), len(lon_center), len(time_input)))

    extension = (2. * np.power(angular_scale[None, None, :], 2.))
    
    source_count_detailed = np.exp(-(square_diff_lon + square_diff_lat) / extension)
    source_count_detailed *= (approximator[None, None, :] / (extension * np.pi))
    source_count_detailed = np.clip(source_count_detailed, 0., 4.)

    return source_count_detailed

def get_source_features(time_input: np.ndarray, sources: np.ndarray, lon_center: np.ndarray, lat_center: np.ndarray, approximator: np.ndarray) -> np.ndarray:
    """Processing function. Computes the source features for the input features and normalizes them."""
    source_count_detailed = compute_source_count(time_input, sources, lon_center, lat_center, approximator)
    source_gas = (np.log10(sources[:, 4] + EPSILON) + GAS_INTERCEPT) / GAS_SCALE

    mean_gas = np.sum((source_count_detailed * source_gas[None, None, :]), axis=2) / np.sum(source_count_detailed, axis=2)
    std_gas =  np.sqrt(np.sum((source_count_detailed * source_gas[None, None, :] * source_gas[None, None, :]), axis=2) / np.sum(source_count_detailed, axis=2) - mean_gas * mean_gas)
    max_val = np.max(source_count_detailed, axis=2, keepdims=True)
    masked_counts = np.where(source_count_detailed == max_val, -np.inf, source_count_detailed)
    gas_max = source_gas[np.argmax(source_count_detailed, axis = 2)]
    gas_max_2 = source_gas[np.argmax(masked_counts, axis = 2)]

    source_features = np.zeros((9, len(lat_center), len(lon_center)), dtype=np.float32)
    source_features[0,:,:] = np.log10(np.sum(source_count_detailed, axis = 2)) + LOCAL_SOURCE_INTERCEPT # Normalized between -1 and 1 (roughly because outliers are important for the prediction but we don't want them to be too extreme).
    source_features[1,:,:] = mean_gas # Normalized between -1 and 1 (roughly because outliers are important for the prediction but we don't want them to be too extreme).
    source_features[2,:,:] = std_gas # Already stable because of the normalization of the gas.
    source_features[3,:,:] = np.log10(np.max(source_count_detailed, axis = 2)) + LOCAL_SOURCE_INTERCEPT # Normalized between -1 and 1 (roughly because outliers are important for the prediction but we don't want them to be too extreme).
    source_features[4,:,:] = gas_max # Normalized between -1 and 1 (roughly because outliers are important for the prediction but we don't want them to be too extreme).
    source_features[5,:,:] = np.log10(np.max(masked_counts, axis = 2)) + LOCAL_SOURCE_INTERCEPT # Normalized between -1 and 1 (roughly because outliers are important for the prediction but we don't want them to be too extreme).
    source_features[6,:,:] = gas_max_2 # Normalized between -1 and 1 (roughly because outliers are important for the prediction but we don't want them to be too extreme).
    source_features[7,:,:] = np.min(sources[:, 3]) / D_MAX # Normalized between 0 and 1.
    source_features[8,:,:] = np.log10(np.mean(np.sum(source_count_detailed, axis = 2))) + GLOBAL_SOURCE_INTERCEPT # Normalized between -1 and 1 (roughly because outliers are important for the prediction but we don't want them to be too extreme).

    return source_features
    
def inference(features: np.ndarray, model: torch.nn.Module, device: str = "cpu"):
    """Runs the inference for the given features."""
    input_tensor = torch.from_numpy(features).unsqueeze(0).float().to(device)

    output = torch.zeros((7,21, 720))

    with torch.no_grad():
        for index in range(0,17):
            input_for_model = input_tensor[:,:,:,(index * 41) : ((index + 1) * 41)]
            intermediate_output = model.predict(input_for_model)
            output[:,:,(index * 41) : ((index + 1) * 41)] = intermediate_output[0,:,:,:]
        input_for_model = input_tensor[:,:,:,697:720]
        features_patch = torch.cat([input_for_model, input_tensor[:,:,:, :18]], dim=3)
        intermediate_output = model.predict(features_patch)
        output[:,:,697 : 720] = intermediate_output[0,:,:,:23]

    flux_map = output.cpu().numpy()

    return flux_map

def plot_flux_map(flux_map: np.ndarray, energy_index: int):
    """Plots the flux map for a given energy index."""
    
    Z = (np.power(10.,flux_map[energy_index,:,:]* 1. + ENERGY_CENTER[energy_index]))
    lon = np.linspace(-180., 179.5, 720)
    lat = np.linspace(-5, 5., 21)
    X, Y = np.meshgrid(lon, lat)
    Z = Z.reshape(len(lon), len(lat))
    Z = np.transpose(Z)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    pcm = ax.pcolor(X, Y, Z, norm=colors.LogNorm(vmin=1.e-25, vmax=1.e-21), cmap="plasma", shading="auto")
    pcm = ax.set_aspect(aspect="equal", adjustable=None, anchor=None, share=False)
    ax.set_ylabel(r"$b(\rm{deg})$", fontsize=9)
    ax.set_xlabel(r"$l(\rm{deg})$", fontsize=9)

    ax.set_xticks(np.arange(-180., 180.5, 30))
    ax.set_yticks(np.arange(-5., 5.5, 2.5))
    ax.tick_params(axis='both', labelsize=8)
    cbar = fig.colorbar(pcm, ax=ax, cmap="plasma", extend="max", shrink=1, pad = 0.13, aspect = 45,
                         norm=colors.LogNorm(vmin=1.e-25, vmax=1.e-21), orientation = "horizontal")
    cbar.set_label(label=r"Gamma-ray flux $(\rm{eV}^{-1}\rm{m}^{-2}\rm{s}^{-1}\,\rm{sr}^{-1})$",
                    fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    plt.show()


def main():
    gas_density = np.load("gas_data.npy")
    sources = np.load("lists_of_sources.npz")

    for i_list in range(10):

        list_of_sources = sources[f"list_{i_list}"]
        number_of_channels, n_lat, n_lon = 14, 21, 720
        halo_size, energy = np.array([5.]), np.array([1.e14])
        log_halo_size = np.log10(halo_size)

        lon_center, lat_center = build_grid(n_lon, n_lat)
        time_input = compute_time_input_double(list_of_sources[:,0], halo_size[0], energy)
        approximator = compute_approximator(time_input, halo_size[0])
        features = fill_geometric_channels(lon_center, lat_center, log_halo_size, number_of_channels)
        features[4:13,:,:] = get_source_features(time_input, list_of_sources, lon_center, lat_center, approximator)
        features[13,:,:] = gas_density

        model = FastCOSPYEmulator(device="cpu")
        flux_map = inference(features, model)

        flux_map = np.transpose(flux_map, axes=(0,2,1))

        plot_flux_map(flux_map, energy_index=3)


if __name__ == "__main__":
    main()