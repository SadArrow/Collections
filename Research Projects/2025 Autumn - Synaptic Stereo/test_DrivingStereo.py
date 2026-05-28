import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sgbm_depth1_synthetic_rgb import sgbm_disparity
from evaluation import compute_epe, compute_bad_pixels

disp_path = "E:/BaiduNetdiskDownload/DrivingStereo/test-disparity-map/disparity-map-full-size/2018-07-11-14-48-52"
left_path = "E:/BaiduNetdiskDownload/DrivingStereo/test-left-image/left-image-full-size/2018-07-11-14-48-52"
right_path = "E:/BaiduNetdiskDownload/DrivingStereo/test-right-image/right-image-full-size/2018-07-11-14-48-52"

# os.makedirs('DrivingStereo_result', exist_ok=True)
save_path = "C:/Users/SadArrow/Desktop/SNN/2025 Autumn/code/DrivingStereo_result"

# os.makedirs(f'{path}/hey', exist_ok=True)
items = os.listdir(disp_path)
print(items)

minDisp = 0
numDisp = 64
blockSize = 5
disp12MaxDiff = 2
uniquenessRatio = 10
speckleWindowSize = 10
speckleRange = 2

def process_disparity(disp):
    valid = np.isfinite(disp) & (disp > 0)
    disp_vis = np.zeros_like(disp, dtype=np.uint8)
    if np.any(valid):
        dmin, dmax = np.percentile(disp[valid], [1, 99.5])
        dmin, dmax = float(dmin), float(max(dmax, dmin+1e-3))
        disp_norm = np.clip((disp - dmin) / (dmax - dmin), 0, 1)
        disp_vis = (disp_norm * 255).astype(np.uint8)

        '''                    
        plt.imshow(disp_vis, cmap='jet')
        # plt.imshow(depth_m , cmap='jet')
        # plt.colorbar()
        plt.axis('off')
        plt.savefig(f"ETH3D_result/{dir}/estimated.png")
        plt.close()
        '''
    
    return disp_vis

# Todo list:

# 1. read disparity map
# 2. read left and right images
# 3. calculate SGBM, with use_wls=true/false
# 4. calculate epe/bad 2/bad 3, with left_crop=true/false
# 5. save two disparity maps, standard disparity map and four results

F1 = open(f"{save_path}/results_wls_crop.txt", "w", encoding='utf-8')
F1.write("Frame Name\tepe\tBad 2\tBad 3\n")

F2 = open(f"{save_path}/results_wls_no_crop.txt", "w", encoding='utf-8')
F2.write("Frame Name\tepe\tBad 2\tBad 3\n")

F3 = open(f"{save_path}/results_no_wls_crop.txt", "w", encoding='utf-8')
F3.write("Frame Name\tepe\tBad 2\tBad 3\n")

F4 = open(f"{save_path}/results_no_wls_no_crop.txt", "w", encoding='utf-8')
F4.write("Frame Name\tepe\tBad 2\tBad 3\n")

for item in items:
    disp_file_path = os.path.join(disp_path, item)
    left_img_path = os.path.join(left_path, item)
    right_img_path = os.path.join(right_path, item)

    # 1. read disparity map
    std_disp_uint16 = cv2.imread(disp_file_path, cv2.IMREAD_UNCHANGED)
    if std_disp_uint16.dtype != 'uint16':
        print(f"警告: 图像 {disp_file_path} 的数据类型不是 uint16，而是 {std_disp_uint16.dtype}。")
        print("这可能不是DrivingStereo的原始ground truth文件。")

    # 关键步骤 2: 转换为浮点数并除以 128.0 (全分辨率)
    # 确保使用 128.0 (浮点数) 进行除法，以得到浮点数结果
    std_disp = std_disp_uint16.astype(np.float32) / 128.0

    # np.savetxt(f'{path}/hey/{item}.txt', disparity, fmt='%.1f')
    '''
    valid = np.isfinite(std_disp) & (std_disp > 0)
    disp_vis = np.zeros_like(std_disp, dtype=np.uint8)
    if np.any(valid):
        dmin, dmax = np.percentile(std_disp[valid], [1, 99.5])
        dmin, dmax = float(dmin), float(max(dmax, dmin+1e-3))
        disp_norm = np.clip((std_disp - dmin) / (dmax - dmin), 0, 1)
        disp_vis = (disp_norm * 255).astype(np.uint8)

                            
        plt.imshow(disp_vis, cmap='jet')
        plt.axis('off')
        plt.show()
        
        # np.savetxt(f'{disp_path}/hey/{item}.txt', disp_vis, fmt='%.1f')
    '''

    std_disp_vis = process_disparity(std_disp)
     
    # 2. read left and right images
    # 3. calculate SGBM, with use_wls=true/false
    # use sgbm_disparity
    disp_wls, depth_wls = sgbm_disparity(use_wls=True, minDisp=minDisp, numDisp=numDisp, blockSize=blockSize,
                                         disp12MaxDiff=disp12MaxDiff, uniquenessRatio=uniquenessRatio,
                                        speckleWindowSize=speckleWindowSize, speckleRange=speckleRange,
                                        left_path=left_img_path, right_path=right_img_path)

    disp_no_wls, depth_no_wls = sgbm_disparity(use_wls=False, minDisp=minDisp, numDisp=numDisp, blockSize=blockSize,
                                         disp12MaxDiff=disp12MaxDiff, uniquenessRatio=uniquenessRatio,
                                        speckleWindowSize=speckleWindowSize, speckleRange=speckleRange,
                                        left_path=left_img_path, right_path=right_img_path)

    disp_wls_vis = process_disparity(disp_wls)
    disp_no_wls_vis = process_disparity(disp_no_wls)
    # std_disp_vis = std_disp
    # disp_wls_vis = disp_wls
    # disp_no_wls_vis = disp_no_wls

    # 4. calculate epe/bad 2/bad 3, with left_crop=true/false
    
    epe_wls_crop = compute_epe(disp_wls, std_disp, max_disparity=numDisp, left_crop=True)
    epe_wls_no_crop = compute_epe(disp_wls, std_disp, max_disparity=numDisp, left_crop=False)
    epe_no_wls_crop = compute_epe(disp_no_wls, std_disp, max_disparity=numDisp, left_crop=True)
    epe_no_wls_no_crop = compute_epe(disp_no_wls, std_disp, max_disparity=numDisp, left_crop=False)

    bad2_wls_crop = compute_bad_pixels(disp_wls, std_disp, threshold=2, max_disparity=numDisp, left_crop=True)
    bad2_wls_no_crop = compute_bad_pixels(disp_wls, std_disp, threshold=2, max_disparity=numDisp, left_crop=False)
    bad2_no_wls_crop = compute_bad_pixels(disp_no_wls, std_disp, threshold=2, max_disparity=numDisp, left_crop=True)
    bad2_no_wls_no_crop = compute_bad_pixels(disp_no_wls, std_disp, threshold=2, max_disparity=numDisp, left_crop=False)

    bad3_wls_crop = compute_bad_pixels(disp_wls, std_disp, threshold=3, max_disparity=numDisp, left_crop=True)
    bad3_wls_no_crop = compute_bad_pixels(disp_wls, std_disp, threshold=3, max_disparity=numDisp, left_crop=False)
    bad3_no_wls_crop = compute_bad_pixels(disp_no_wls, std_disp, threshold=3, max_disparity=numDisp, left_crop=True)
    bad3_no_wls_no_crop = compute_bad_pixels(disp_no_wls, std_disp, threshold=3, max_disparity=numDisp, left_crop=False)

    # 5. save two disparity maps, standard disparity map and four results

    os.makedirs(f'{save_path}/{item[:-4]}', exist_ok=True)

    plt.imshow(std_disp_vis, cmap='jet')
    plt.axis('off')
    plt.savefig(f"{save_path}/{item[:-4]}/standard.png")
    plt.close()

    plt.imshow(disp_wls_vis, cmap='jet')
    plt.axis('off')
    plt.savefig(f"{save_path}/{item[:-4]}/estimated_wls.png")
    plt.close()

    plt.imshow(disp_no_wls_vis, cmap='jet')
    plt.axis('off')
    plt.savefig(f"{save_path}/{item[:-4]}/estimated_no_wls.png")
    plt.close()

    # test_mask = (std_disp_vis > 0) & (std_disp_vis < numDisp)
    # np.savetxt(f'{save_path}/{item[:-4]}/disparity_mask.txt', test_mask, fmt='%d')
    # np.savetxt(f'{save_path}/{item[:-4]}/estimation_mask.txt', disp_wls_vis[test_mask], fmt='%d')
    # np.savetxt(f'{save_path}/{item[:-4]}/error_mask.txt', np.abs(disp_wls_vis[test_mask] - std_disp_vis[test_mask]), fmt='%d')

    with open(f'{save_path}/{item[:-4]}/evaluation.txt', 'w') as f:
        f.write(f"The parameters of SGBM are:\n"
                f"minDisp = {minDisp}\nnumDisp = {numDisp}\nblockSize = {blockSize}\n"
                f"disp12MaxDiff = {disp12MaxDiff}\nuniquenessRatio = {uniquenessRatio}\n"
                f"speckleWindowSize = {speckleWindowSize}\nspeckleRange = {speckleRange}\n\n")
        
        f.write(f"Performance with WLS and left crop:\n"
                f"EPE: {epe_wls_crop:.4f}\n"
                f"Bad 2: {bad2_wls_crop*100:.2f}%\n"
                f"Bad 3: {bad3_wls_crop*100:.2f}%\n\n")
        
        f.write(f"Performance with WLS and no left crop:\n"
                f"EPE: {epe_wls_no_crop:.4f}\n"
                f"Bad 2: {bad2_wls_no_crop*100:.2f}%\n"
                f"Bad 3: {bad3_wls_no_crop*100:.2f}%\n\n")
        
        f.write(f"Performance with no WLS and left crop:\n"
                f"EPE: {epe_no_wls_crop:.4f}\n"
                f"Bad 2: {bad2_no_wls_crop*100:.2f}%\n"
                f"Bad 3: {bad3_no_wls_crop*100:.2f}%\n\n")
        
        f.write(f"Performance with no WLS and no left crop:\n"
                f"EPE: {epe_no_wls_no_crop:.4f}\n"
                f"Bad 2: {bad2_no_wls_no_crop*100:.2f}%\n"
                f"Bad 3: {bad3_no_wls_no_crop*100:.2f}%\n\n")
    
    F1.write(f"{item}\t{epe_wls_crop:.4f}\t{bad2_wls_crop*100:.2f}%\t{bad3_wls_crop*100:.2f}%\n")
    F2.write(f"{item}\t{epe_wls_no_crop:.4f}\t{bad2_wls_no_crop*100:.2f}%\t{bad3_wls_no_crop*100:.2f}%\n")
    F3.write(f"{item}\t{epe_no_wls_crop:.4f}\t{bad2_no_wls_crop*100:.2f}%\t{bad3_no_wls_crop*100:.2f}%\n")
    F4.write(f"{item}\t{epe_no_wls_no_crop:.4f}\t{bad2_no_wls_no_crop*100:.2f}%\t{bad3_no_wls_no_crop*100:.2f}%\n")

    print(f"Processed {item} successfully.")    


F1.close()
F2.close()
F3.close()
F4.close()

        



    
    

