import spkProc
import spkData.spike_stero
from spkData.load_dat import data_parameter_dict
from spkData.load_dat import SpikeStream
from visualization.get_video import obtain_spike_video
import numpy as np
import time
# from device.spikevision.m1k40.sdk import spikelinkapi as link
import args as r
import matplotlib.pyplot as plt

time0 = time.time()

# 获取数据集参数字典
paraDict_left = data_parameter_dict(r.data_filename_left, r.label_type)
paraDict_right = data_parameter_dict(r.data_filename_right, r.label_type)

print(paraDict_left)
print(paraDict_right)

spikestream_left = SpikeStream(**paraDict_left)
spikestream_right = SpikeStream(**paraDict_right)

#获取文件中所有脉冲流
spike_left = spikestream_left.get_spike_matrix()
spike_right = spikestream_right.get_spike_matrix()
# print("Total spikes shape:", spike_left.shape, spike_right.shape)
# print("spike_left: ", spike_left, "spike_right: ", spike_right)

# obtain_spike_video(spike_left, "left.avi", **paraDict_left)
# obtain_spike_video(spike_right, "right.avi", **paraDict_right)

# We assume that the spike streams (spike_left & spike_right) are both epipolar aligned

# Asynchronous Intensity Representation

# Method 1: Local Firing Rate Map (LFRM)
# before N spikes occur, the firing rate should be 0

left_arr = np.array(spike_left)
right_arr = np.array(spike_right)
height, width, times = left_arr.shape
# print(left_arr.shape, right_arr.shape, left_arr, right_arr)
LFRM_left = np.zeros_like(left_arr, dtype=np.float32)
LFRM_right = np.zeros_like(right_arr, dtype=np.float32)

time1 = time.time()

print(f"Time for spike stream loading: {time1 - time0:.4f}s" )

for i in range(left_arr.shape[0]):
    for j in range(left_arr.shape[1]):
        left_fires = np.where(left_arr[i, j, :])[0]
        right_fires = np.where(right_arr[i, j, :])[0]

        # calculate LFRM for left stream

        left_res = np.roll(left_fires, r.N)
        # left_res[:N] = 0
        left_d = left_fires - left_res
        left_d[:r.N] = r.INF
        left_d = np.insert(left_d, 0, r.INF)

        left_F = np.append(left_fires, left_arr.shape[2])
        left_seg_len = np.insert(np.diff(left_F), 0, left_F[0])
        # left_seg_len = np.diff(left_F)
        left_R = np.repeat(left_d, left_seg_len)
        LFRM_left[i, j, :] = r.N / left_R

        # if i <= 2 and j <= 2:
        #     print(f"with {i}, {j}: {left_F[-1]},\n {left_d},\n {left_F},\n {np.sum(left_seg_len)}\n")
        #     print("left_LFRM:", LFRM_left[i, j, :])

        # calculate LFRM for right stream

        right_res = np.roll(right_fires, r.N)
        # right_res[:N] = 0
        right_d = right_fires - right_res
        right_d[:r.N] = r.INF
        right_d = np.insert(right_d, 0, r.INF)

        right_F = np.append(right_fires, right_arr.shape[2])
        right_seg_len = np.insert(np.diff(right_F), 0, right_F[0])
        right_R = np.repeat(right_d, right_seg_len)
        LFRM_right[i, j, :] = r.N / right_R

# print("LFRM_left: ", LFRM_left.shape, LFRM_left)
# print("LFRM_right: ", LFRM_right.shape, LFRM_right)

time2 = time.time()

print(f"Time for LFRM calculation: {time2 - time1:.4f}s" )
# print("Time for np.where:", time2 - time1)

# Method 2: Eligibility Trace Surface (ETS)



ETS_left = np.zeros_like(left_arr, dtype=np.float32)
ETS_right = np.zeros_like(right_arr, dtype=np.float32)

time3 = time.time()

for i in range(left_arr.shape[0]):
    for j in range(left_arr.shape[1]):
        timestamp = np.arange(left_arr.shape[2])
        left_fires = np.insert(np.where(left_arr[i, j, :])[0], 0, 0)
        right_fires = np.insert(np.where(right_arr[i, j, :])[0], 0, 0)


        left_seg_len = np.diff(np.append(left_fires, left_arr.shape[2]))
        # lspk stands for "last spike"
        left_lspk = np.repeat(left_fires, left_seg_len)
        left_lspk[:left_fires[1]] = -r.INF

        ETS_left[i, j, :] = np.exp(-(timestamp - left_lspk) / r.tau)

        
        right_seg_len = np.diff(np.append(right_fires, right_arr.shape[2]))
        # lspk stands for "last spike"
        right_lspk = np.repeat(right_fires, right_seg_len)
        right_lspk[:right_fires[1]] = -r.INF

        ETS_right[i, j, :] = np.exp(-(timestamp - right_lspk) / r.tau)


        # if i <= 2 and j <= 2:
        #    print(f"with {i}, {j}: {left_fires}, {np.sum(left_seg_len)}, \n{left_lspk}")
        #    print("with {i}, {j}, left_ETS:", ETS_left[i, j, :])

# print("ETS_left:", ETS_left.shape, ETS_left)
# print("ETS_right:", ETS_right.shape, ETS_right)

time4 = time.time()

print(f"Time for ETS calculation: {time4 - time3:.4f}s")


# Geometric Stereo Matching
# Next, we define cost function for LFRM & ETS relatively, and perform epipolar search
# the final disparity p should be a (H, W, T) matrix

# for LFRM, the cost is the absolute difference
# LFRM_left, LFRM_right

cost_volume = np.full((height, width, times), r.INF)
chosen_disparity = np.full((height, width, times), -1)
print("cost_volume constructed")

for d in range(r.d_min, r.d_max + 1):
    full_d = np.full((height, width, times), d)
    left_slice = LFRM_left
    right_slice = np.roll(LFRM_right, d, axis=0)
    
    # print(left_slice.shape, right_slice.shape)
    
    # if d == 31:
        # print("valid_mask: ", valid_mask)
        # print("LFRM_left: ", LFRM_left[:, 0, 0], "left_slice: ", left_slice[:, 0, 0])
        # print("right_slice: ", right_slice)
    
    cost = np.abs(left_slice - right_slice)
    cost[:d, :, :] = r.INF
    cost_update = cost < cost_volume

    cost_volume[cost_update] = cost[cost_update]
    chosen_disparity[cost_update] = full_d[cost_update]


print("Cost: ", cost_volume.shape, cost_volume)
print("chosen_disparity: ", chosen_disparity.shape, chosen_disparity)


time5 = time.time()

plt.imshow(chosen_disparity[-1], cmap='jet')
# plt.imshow(depth_m, cmap='jet')
# plt.colorbar()
plt.axis('off')
plt.show()

print(f"Time for epipolar search: {time5 - time4:.4f}s")
