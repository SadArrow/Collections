# Data loading:
# data_filename_left = "C:/Users/SadArrow/Desktop/SNN/2025 Autumn/code/synthetic/spike_l/0001.dat"
data_filename_left = "C:/SpikeCV-main/SpikeCV/spkData/datasets/Spike-Stero/indoor/left/0000/0000"
# data_filename_right = "C:/Users/SadArrow/Desktop/SNN/2025 Autumn/code/synthetic/spike_r/0001.dat"
data_filename_right = "C:/SpikeCV-main/SpikeCV/spkData/datasets/Spike-Stero/indoor/right/0000/0000"
label_type = "stero_depth_estimation"


# Intensity Representation calculation:
INF = 1e12
# for LFRM:
N = 5

# for ETS:
tau = 10.0


# Epipolar search calculation:
d_min = 0
d_max = 64