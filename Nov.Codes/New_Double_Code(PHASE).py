import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d


def degrees(radians):
    return np.degrees(radians)

dirName = "/Users/mickaelbegon/Library/CloudStorage/Dropbox/1_EN_COURS/FALL2023/"
typeTouch = "Struck" #"Pressed" #

# Load data_1
with open(dirName + typeTouch + "_with_Thorax.pckl",
          "rb") as file:
    data_1 = pickle.load(file)

with open(dirName + typeTouch + "_without_Thorax.pckl",
          "rb") as file:
    data_2 = pickle.load(file)

# Process specific points for data_1 and data_2
specific_points_s_1 = [sum(data_1["phase_time"][: i + 1]) for i in range(len(data_1["phase_time"]))]
specific_points_s_2 = [sum(data_2["phase_time"][: i + 1]) for i in range(len(data_2["phase_time"]))]

# Labels for data_1 and data_2
label_1 = "with"
label_2 = "without"
# Processing data_1 and data_2 for q, qdot, tau
# For data_1
array_q_s_1 = [data_1["states_no_intermediate"][i]["q"] for i in range(len(data_1["states_no_intermediate"]))]
array_qdot_s_1 = [data_1["states_no_intermediate"][i]["qdot"] for i in range(len(data_1["states_no_intermediate"]))]
array_tau_s_1 = [data_1["controls"][i]["tau"] for i in range(len(data_1["controls"]))]

# Replace NaN values in tau arrays for data_1
for i in range(len(array_tau_s_1) - 1):
    array_tau_s_1[i][:, -1] = array_tau_s_1[i + 1][:, 0]
array_tau_s_1[-1][:, -1] = array_tau_s_1[-1][:, -2]

# For data_2
array_q_s_2 = [data_2["states_no_intermediate"][i]["q"] for i in range(len(data_2["states_no_intermediate"]))]
array_qdot_s_2 = [data_2["states_no_intermediate"][i]["qdot"] for i in range(len(data_2["states_no_intermediate"]))]
array_tau_s_2 = [data_2["controls"][i]["tau"] for i in range(len(data_2["controls"]))]

# Replace NaN values in tau arrays for data_2
for i in range(len(array_tau_s_2) - 1):
    array_tau_s_2[i][:, -1] = array_tau_s_2[i + 1][:, 0]
array_tau_s_2[-1][:, -1] = array_tau_s_2[-1][:, -2]

# Concatenate arrays for q, qdot, tau for both data sets
concatenated_array_q_s_1 = degrees(np.concatenate(array_q_s_1, axis=1))
concatenated_array_qdot_s_1 = degrees(np.concatenate(array_qdot_s_1, axis=1))
concatenated_array_tau_s_1 = np.concatenate(array_tau_s_1, axis=1)

concatenated_array_q_s_2 = degrees(np.concatenate(array_q_s_2, axis=1))
concatenated_array_qdot_s_2 = degrees(np.concatenate(array_qdot_s_2, axis=1))
concatenated_array_tau_s_2 = np.concatenate(array_tau_s_2, axis=1)


# # Creating finger arrays
# array1 = concatenated_array_tau_s_2[6,:]  # shorter array with 75 elements
# hand_pressed = concatenated_array_tau_s_1[6,:]  # longer array with 81 elements
#
# # Creating an interpolation function for the shorter array
# interp_function = interp1d(np.linspace(0, 1, len(array1)), array1)
#
# # Using the interpolation function to create a new array with the length of the longer array
# interpolated_array_hand_struck = interp_function(np.linspace(0, 1, len(hand_pressed)))
#
# # Creating wrist arrays
# array1 = concatenated_array_tau_s_2[5,:]  # shorter array with 75 elements
# wrist_pressed = concatenated_array_tau_s_1[5,:]  # longer array with 81 elements
#
# # Creating an interpolation function for the shorter array
# interp_function = interp1d(np.linspace(0, 1, len(array1)), array1)
#
# # Using the interpolation function to create a new array with the length of the longer array
# interpolated_array_wrist_struck = interp_function(np.linspace(0, 1, len(wrist_pressed)))


# Generate time array for plotting for both data sets
time_arrays_1 = [
    np.linspace(specific_points_s_1[i], specific_points_s_1[i + 1], len(array_q_s_1[i][0]))
    for i in range(len(specific_points_s_1) - 1)
]
concatenated_array_time_s_1 = np.concatenate(time_arrays_1)

time_arrays_2 = [
    np.linspace(specific_points_s_2[i], specific_points_s_2[i + 1], len(array_q_s_2[i][0]))
    for i in range(len(specific_points_s_2) - 1)
]
concatenated_array_time_s_2 = np.concatenate(time_arrays_2)

# Plotting
Name = [
    "Pelvic Tilt, Anterior (-) and Posterior (+) Rotation",
    "Thorax, Left (+) and Right (-) Rotation",
    "Thorax, Flexion (-) and Extension (+)",
    "Right Shoulder, Abduction (-) and Adduction (+)",
    "Right Shoulder, Internal (+) and External (-) Rotation",
    "Right Shoulder, Flexion (+) and Extension (-)",
    "Elbow, Flexion (+) and Extension (-)",
    "Elbow, Pronation (+) and Supination (-)",
    "Wrist, Flexion (-) and Extension (+)",
    "MCP, Flexion (+) and Extension (-)",
]
#
# Handle NaN values in tau arrays

for i in range(-7, 0):
    fig, axs = plt.subplots(nrows=3, ncols=1)

    # Plot for q
    axs[0].plot(concatenated_array_time_s_1, concatenated_array_q_s_1[i, :], color="red", label=label_1)
    axs[0].plot(
        concatenated_array_time_s_2, concatenated_array_q_s_2[i, :], color="blue", linestyle="--", label=label_2
    )

    axs[0].set_title(Name[i])
    axs[0].set_ylabel("θ (deg)")
    axs[0].legend()

    # Plot for qdot
    axs[1].plot(concatenated_array_time_s_1, concatenated_array_qdot_s_1[i, :], color="red", label=label_1)

    axs[1].plot(
        concatenated_array_time_s_2, concatenated_array_qdot_s_2[i, :], color="blue", linestyle="--", label=label_2
    )

    axs[1].set_ylabel(r"$\dot{\theta}$ (deg/sec)")
    axs[1].legend()

    # Plot for tau
    axs[2].step(concatenated_array_time_s_1, concatenated_array_tau_s_1[i, :], color="red", label=label_1)
    axs[2].step(concatenated_array_time_s_2, concatenated_array_tau_s_2[i, :], color="blue", label=label_2)
    axs[2].step(concatenated_array_time_s_2,
                concatenated_array_tau_s_1[i, :]-concatenated_array_tau_s_2[i, :],
                color="black", linestyle="--", label="diff")


    axs[2].set_ylabel(r"$\tau$ (N/m)")
    axs[2].set_xlabel("Time (sec)")
    axs[2].legend()

    # Set common properties for all subplots
    for ax in axs:
        ax.grid(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which="minor", linestyle=":", linewidth="0.2", color="gray")

        # Add vertical lines for specific points in data_1
        for point in specific_points_s_1:
            ax.axvline(x=point, color="k", linestyle=":")

        for point in specific_points_s_2:
            ax.axvline(x=point, color="k", linestyle=":")

    plt.tight_layout()

I1 = (np.trapz(concatenated_array_tau_s_1[-1, :]**2, x=concatenated_array_time_s_1) +
      np.trapz(concatenated_array_tau_s_1[-2, :]**2, x=concatenated_array_time_s_1))
I2 = (np.trapz(concatenated_array_tau_s_2[-1, :]**2, x=concatenated_array_time_s_2) +
      np.trapz(concatenated_array_tau_s_2[-2, :]**2, x=concatenated_array_time_s_2))
print(I1, I2, (I1-I2)/I1*100)

plt.show()

# # Process and plot Force Values for data_1
# Force_Values_V1 = np.array(data_1["Force_Values"][:, 2])
# Force_Values_V2 = np.array(data_2["Force_Values"][:, 2])
# width = 0.2
#
# x = np.arange(len(Force_Values_V1))
# plt.bar(x, Force_Values_V1.flatten(), label='1st_Version', color='blue')
# # plt.bar(x, Force_Values_V2.flatten(), label='Regularized_Version_2', color='red')
#
# plt.xlabel('Nodes')
# plt.ylabel('Force Values (N)')
# plt.title('Force Values')
# plt.legend()
# plt.tight_layout()
# plt.show()
