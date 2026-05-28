import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import time
# import vpi

from read_pfm import Read_pfm
from evaluation import compute_epe, compute_bad_pixels

pic = "0400"


# def sgbm_disparity(use_wls):

# def sgbm_disparity(use_wls, minDisp, numDisp, blockSize, 
#                    disp12MaxDiff, uniquenessRatio, speckleWindowSize, speckleRange):
def sgbm_disparity(use_wls, minDisp, numDisp, blockSize, 
                   disp12MaxDiff, uniquenessRatio, speckleWindowSize, speckleRange,
                   left_path, right_path):
    
    time1 = time.time()

    # ======  synthetic 标定参数 ======
    doffs_px   = 0.        # 像素
    baseline_m = 0.055        # 米（Middlebury 的单位）
    fx = 289.740625                 # 用左相机 fx（像素）

    # 对应的图像尺寸/范围（供 sanity check 与 SGBM 设置）
    width, height = 960, 540  
    # ndisp_gt = 270               # 数据集给出的最大视差提示（像素）

    # ====== 读入与校验 ======
    # imgL = cv.imread(f"./synthetic/rgb_l/{pic}.png" , cv.IMREAD_COLOR)   # 左图
    imgL = cv.imread(left_path , cv.IMREAD_COLOR)   # 左图
    # imgR = cv.imread(f"./synthetic/rgb_r/{pic}.png" , cv.IMREAD_COLOR)   # 右图
    imgR = cv.imread(right_path , cv.IMREAD_COLOR)   # 右图
    # print(imgR)


    if imgL.shape[:2] != imgR.shape[:2]:
        raise ValueError("左右图尺寸不一致")
    if imgL.shape[1] != width or imgL.shape[0] != height:
        print(f"[警告] 图像尺寸为 {imgL.shape[1]}x{imgL.shape[0]}，与标定的 {width}x{height} 不一致，继续按实际尺寸计算。")

    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    # print(grayL)

    # ====== SGBM 参数（传统法，非深度学习） ======
    # numDisparities 必须为 16 的倍数；用覆盖 ndisp 的最小倍数
    # numDisp = int(np.ceil(ndisp_gt/16.0))*16   # 272
    # minDisp = 0                                # 以 0 起始，覆盖到 numDisp

    # minDisp = 0
    # numDisp = 64   # 必须为16的倍数；若近距离更多，可改为 96/128
    
    
    gpu_left = cv.cuda.GpuMat()
    gpu_left.upload(grayL)
    gpu_right = cv.cuda.GpuMat()
    gpu_right.upload(grayR)
    
    # gpu_left = cv.UMat(grayL)
    # gpu_right = cv.UMat(grayR)
    # blockSize = 5
    P1 = 8  * 1 * blockSize**2
    P2 = 32 * 1 * blockSize**2

    # left_matcher = cv.StereoSGBM_create(
    # left_matcher = cv.cuda.createStereoSGM(
    left_matcher = cv.cuda.createStereoSGM(
        minDisparity=minDisp,
        numDisparities=numDisp,
        # blockSize=blockSize,
        P1=P1, P2=P2,
        # disp12MaxDiff=disp12MaxDiff,
        uniquenessRatio=uniquenessRatio,
        # speckleWindowSize=speckleWindowSize,
        # speckleRange=speckleRange,
        # mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 可选：WLS 边缘保留滤波，显著提升低纹理区域质量（需要 opencv-contrib 的 ximgproc）
    # use_wls = hasattr(cv, "ximgproc")
    '''
    if use_wls:
        right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
        wls = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.2)
    '''
    # ====== 计算视差 ======
        
    # print(type(gpu_left))
    # print(type(gpu_right))
    # grayL = cv.UMat(grayL)
    # grayR = cv.UMat(grayR)

    # disp = left_matcher.compute(grayL, grayR)    # int16, 放大16倍
    disp = left_matcher.compute(gpu_left, gpu_right)
    # dispL = left_matcher.compute()
    
    '''
    if use_wls:
        # dispR = right_matcher.compute(grayR, grayL)
        dispR = right_matcher.compute(gpu_right, gpu_left)
        disp  = wls.filter(dispL, imgL, disparity_map_right=dispR)
    else:
        disp = dispL
    '''
    disp = disp.download()
    # disp = disp.get()

    disp = disp.astype(np.float32) / 16.0         # 还原到像素单位

    # ====== 视差 -> 深度（Middlebury 公式） ======
    # Z = f_px * B_m / (d + doffs_px)
    den = disp + doffs_px
    depth_m = np.zeros_like(disp, dtype=np.float32)
    mask = den > 1e-6
    depth_m[mask] = (baseline_m * fx) / den[mask]

    # mask = (disp > vmin) & (disp < vmax)
    # depth = np.zeros_like(disp)
    # depth[mask] = fx * baseline_mm / (disp[mask])   # 单位 mm

    time2 = time.time()

    print(f"Processing Time: {time2 - time1:.3f} seconds")

    return disp, depth_m


# Grid Search for best parameters
# os.makedirs(f"output_depth_images_{pic}", exist_ok=True)

# 64 3?
# disp, depth_m = sgbm_disparity(use_wls=False)

'''

min_disp_range = [0]
num_disp_range = [140, 156, 172, 188, 204, 220, 236, 252]
# [16, 32, 48, 64, 80, 96, 112, 128]
block_size_range = [1, 3, 5, 7, 9, 11, 13, 15, 17]
disp12MaxDiff_range = [1]
# [1, 5, 10]
uniquenessRatio_range = [10]
# [0, 10, 20]
spekleWindowSize_range = [10]
# [0, 10, 100]
speckleRange_range = [1]
# [0, 1, 2]
for min_disp in min_disp_range:
    for num_disp in num_disp_range:
        for block_size in block_size_range:
            for disp12MaxDiff in disp12MaxDiff_range:
                for uniquenessRatio in uniquenessRatio_range:
                    for spekleWindowSize in spekleWindowSize_range:
                        for speckleRange in speckleRange_range:
                            print(f"Testing parameters: min_disp={min_disp}, num_disp={num_disp}, \
                                    block_size={block_size}, disp12MaxDiff={disp12MaxDiff}, \
                                    uniquenessRatio={uniquenessRatio}, spekleWindowSize={spekleWindowSize}, \
                                    speckleRange={speckleRange}")
                            disp, depth_m = sgbm_disparity(use_wls=True, minDisp=min_disp, numDisp=num_disp,
                                                            blockSize=block_size, disp12MaxDiff=disp12MaxDiff,
                                                            uniquenessRatio=uniquenessRatio, speckleWindowSize=spekleWindowSize,
                                                            speckleRange=speckleRange)
                            
                            valid = np.isfinite(disp) & (disp > 0)
                            disp_vis = np.zeros_like(disp, dtype=np.uint8)
                            if np.any(valid):
                                dmin, dmax = np.percentile(disp[valid], [1, 99.5])
                                dmin, dmax = float(dmin), float(max(dmax, dmin+1e-3))
                                disp_norm = np.clip((disp - dmin) / (dmax - dmin), 0, 1)
                                disp_vis = (disp_norm * 255).astype(np.uint8)
                            
                            plt.imshow(disp_vis, cmap='jet')
                            # plt.imshow(depth_m , cmap='jet')
                            # plt.colorbar()
                            plt.axis('off')
                            plt.savefig(f"output_depth_images_{pic}/{min_disp}-{num_disp}_{block_size}_{disp12MaxDiff}_{uniquenessRatio}_{spekleWindowSize}_{speckleRange}.png")
                            plt.close()
                            # cv.imwrite(f"output_depth_images/{min_disp}-{num_disp}_{block_size}_{disp12MaxDiff}_{uniquenessRatio}_{spekleWindowSize}_{speckleRange}.png", disp_vis)
'''  


# 现在, 从ETH3D中分别导入左右图片和标准视差图, 用SGBM估计深度并计算epe, bad 2和bad 3
'''

dirs = ["delivery_area_1l", "delivery_area_2l", "delivery_area_3l", 
        "electro_1l", "electro_2l", "electro_3l",
        "facade_1s", "forest_1s", "forest_2s",
        "playground_1l", "playground_2l", "playground_3l",
        "terrace_1s", "terrace_2s",
        "terrains_1l", "terrains_2l"]

# disp, depth_m = sgbm_disparity(use_wls=True)
minDisp = 0
numDisp = 192
blockSize = 7
disp12MaxDiff = 1
uniquenessRatio = 10
speckleWindowSize = 10
speckleRange = 1

for dir in dirs:
    left_path = f'C:/Users/SadArrow/Desktop/SNN/2025 Autumn/code/ETH3D/two_view_training/{dir}/im0.png'
    right_path = f'C:/Users/SadArrow/Desktop/SNN/2025 Autumn/code/ETH3D/two_view_training/{dir}/im1.png'
    std_path = f'C:/Users/SadArrow/Desktop/SNN/2025 Autumn/code/ETH3D/two_view_training_gt/{dir}/disp0GT.pfm'

    disp_estimated, depth_m = sgbm_disparity(use_wls=True, minDisp=minDisp, numDisp=numDisp,
                                             blockSize=blockSize, disp12MaxDiff=disp12MaxDiff,
                                             uniquenessRatio=uniquenessRatio, speckleWindowSize=speckleWindowSize,
                                             speckleRange=speckleRange,
                                             left_path=left_path, right_path=right_path)
    
    std_disp, scale = Read_pfm(std_path, dir)

    os.makedirs(f"ETH3D_result/{dir}", exist_ok=True)

    epe = compute_epe(disp_estimated, std_disp)
    bad2 = compute_bad_pixels(disp_estimated, std_disp, threshold=2.0)
    bad3 = compute_bad_pixels(disp_estimated, std_disp, threshold=3.0)

    print(f"The estimation performance of {dir} is:\n")
    print(f"EPE: {epe:.4f}\n")
    print(f"Bad 2: {bad2*100:.2f}%\n")
    print(f"Bad 3: {bad3*100:.2f}%\n\n")

    with open(f'ETH3D_result/{dir}/evaluation.txt', 'w') as f:
        f.write(f"The parameters of SGBM are:\n"
                f"minDisp = {minDisp}\nnumDisp = {numDisp}\nblockSize = {blockSize}\n"
                f"disp12MaxDiff = {disp12MaxDiff}\nuniquenessRatio = {uniquenessRatio}\n"
                f"speckleWindowSize = {speckleWindowSize}\nspeckleRange = {speckleRange}\n\n")
        f.write(f"The estimation performance of {dir} is:\n")
        f.write(f"EPE: {epe:.4f}\n")
        f.write(f"Bad 2: {bad2*100:.2f}%\n")
        f.write(f"Bad 3: {bad3*100:.2f}%\n")
    
    # np.savetxt(f'ETH3D_result/{dir}/E.txt', disp_estimated, fmt='%.4f')
    # np.savetxt(f'ETH3D_result/{dir}/S.txt', std_disp, fmt='%.4f')

    valid = np.isfinite(disp_estimated) & (disp_estimated > 0)
    disp_vis = np.zeros_like(disp_estimated, dtype=np.uint8)
    if np.any(valid):
        dmin, dmax = np.percentile(disp_estimated[valid], [1, 99.5])
        dmin, dmax = float(dmin), float(max(dmax, dmin+1e-3))
        disp_norm = np.clip((disp_estimated - dmin) / (dmax - dmin), 0, 1)
        disp_vis = (disp_norm * 255).astype(np.uint8)
                            
        plt.imshow(disp_vis, cmap='jet')
        # plt.imshow(depth_m , cmap='jet')
        # plt.colorbar()
        plt.axis('off')
        plt.savefig(f"ETH3D_result/{dir}/estimated.png")
        plt.close()
    
'''


'''
# ====== 视差可视化（仅查看） ======
valid = np.isfinite(disp) & (disp > 0)
disp_vis = np.zeros_like(disp, dtype=np.uint8)
if np.any(valid):
    dmin, dmax = np.percentile(disp[valid], [1, 99.5])
    dmin, dmax = float(dmin), float(max(dmax, dmin+1e-3))
    disp_norm = np.clip((disp - dmin) / (dmax - dmin), 0, 1)
    disp_vis = (disp_norm * 255).astype(np.uint8)
# cv.imwrite("mb_disp.png", disp_vis)

# ====== 视差可视化（按 vmin/vmax 固定范围） ======
# vmin, vmax = 23.0, 245.0  # Middlebury 提供的显示范围
# valid_disp = np.isfinite(disp) & (disp > 0.0)

# # 裁剪到 [vmin, vmax] 并线性归一化到 [0, 255]
# disp_clip = np.clip(disp, vmin, vmax)
# disp_norm = (disp_clip - vmin) / (vmax - vmin + 1e-6)
# disp_vis  = (np.clip(disp_norm, 0, 1) * 255).astype(np.uint8)

# # 无效像素设为黑（可选）
# disp_vis[~valid_disp] = 0


# 深度可视化（仅查看效果，按 0.3m~5m 映射）
dmin_m, dmax_m = 0.3, 3.0
depth_vis = np.clip((depth_m - dmin_m) / (dmax_m - dmin_m), 0, 1)
depth_vis = (depth_vis * 255).astype(np.uint8)
# cv.imwrite("mb_depth_vis.png", depth_vis)


plt.imshow(disp_vis, cmap='jet')
# plt.imshow(depth_m , cmap='jet')
# plt.colorbar()
plt.axis('off')
plt.show()
'''