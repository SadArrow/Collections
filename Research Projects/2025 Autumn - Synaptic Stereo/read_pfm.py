import cv2
import numpy as np
import matplotlib.pyplot as plt

def Read_pfm(file_path, dir):
    """
    读取PFM文件，返回一个NumPy数组和缩放比例。
    """
    with open(file_path, 'rb') as file:
        # 读取文件头
        header = file.readline().decode('utf-8').rstrip()
        if header != 'PF' and header != 'Pf':
            raise Exception('Not a PFM file.')

        dim_match = file.readline().decode('utf-8').rstrip()
        width, height = map(int, dim_match.split())
        scale = float(file.readline().decode('utf-8').rstrip())

        # PFM文件数据存储顺序是左下角开始，我们需要翻转
        data = np.fromfile(file, dtype=np.float32)
        data = data.reshape((height, width))

        # 如果缩放因子为负数，则数据是从左下角到右上角
        if scale < 0:
            data = np.flipud(data) # 上下翻转
            scale = -scale
        
        valid = np.isfinite(data) & (data > 0)
        data[~valid] = -1
        disp_vis = np.zeros_like(data, dtype=np.uint8)
        if np.any(valid):
            dmin, dmax = np.percentile(data[valid], [1, 99.5])
            dmin, dmax = float(dmin), float(max(dmax, dmin+1e-3))
            disp_norm = np.clip((data - dmin) / (dmax - dmin), 0, 1)
            disp_vis = (disp_norm * 255).astype(np.uint8)
                            

        plt.imshow(disp_vis, cmap='jet')
        # plt.imshow(depth_m , cmap='jet')
        # plt.colorbar()
        plt.axis('off')
        plt.savefig(f"ETH3D_result/{dir}/standard.png")
        plt.close()

        '''
        invalid_mask = (data <= 0) | (data > 1e6)
        valid_values = data[~invalid_mask]
        if valid_values.size > 0:
            # 创建一个副本，将无效值设为NaN
            temp_map = data.copy()
            temp_map[invalid_mask] = np.nan
    
            # 使用matplotlib保存，它会自动处理NaN值
            plt.imsave(f'ETH3D_result/{dir}/standard.png', temp_map, cmap='jet')
        '''

        return data, scale

'''
# 1. 读取PFM文件
# file_path = 'C:/Users/SadArrow/Desktop/SNN/2025 Autumn/Sampler/Driving/disparity/0400.pfm' # 替换为你的.pfm文件路径
file_path = "C:/Users/SadArrow/Desktop/SNN/2025 Autumn/code/ETH3D/two_view_training_gt/delivery_area_1l/disp0GT.pfm"
disparity_map, scale = Read_pfm(file_path)

print(f"图像尺寸: {disparity_map.shape}")
print(f"缩放因子: {scale}")
print(f"视差范围: [{np.min(disparity_map):.2f}, {np.max(disparity_map):.2f}]")

# 2. 处理无效像素（通常视差为0或无穷大的地方）
# 假设无效像素的视差为0或一个非常大的值
invalid_mask = (disparity_map <= 0) | (disparity_map > 1e6)
disparity_map_clean = np.ma.masked_where(invalid_mask, disparity_map)

# 3. 可视化
plt.figure(figsize=(15, 5))

# 原始视差图（可能包含无效值）
plt.subplot(1, 3, 1)
plt.imshow(disparity_map, cmap='jet')
plt.colorbar(label='Disparity Value')
plt.title('Raw Disparity Map (with invalid values)')
plt.axis('off')

# 清理后的视差图（隐藏无效值）
plt.subplot(1, 3, 2)
plt.imshow(disparity_map_clean, cmap='jet')
plt.colorbar(label='Disparity Value')
plt.title('Cleaned Disparity Map (invalid masked)')
plt.axis('off')

# 视差值分布直方图
plt.subplot(1, 3, 3)
# 只对有效值绘制直方图
valid_values = disparity_map[~invalid_mask]
plt.hist(valid_values.flatten(), bins=50, log=True)
plt.xlabel('Disparity Value')
plt.ylabel('Frequency (Log)')
plt.title('Disparity Value Distribution')

plt.tight_layout()
plt.show()

# 4. （可选）保存为普通图像（如PNG）以便用其他软件查看
# 将有效视差归一化到0-255范围

normalized = np.zeros_like(disparity_map, dtype=np.uint8)
if valid_values.size > 0:
    normalized_valid = cv2.normalize(disparity_map_clean, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    normalized = normalized_valid.filled(0) # 将无效区域填充为0（黑色）

if valid_values.size > 0:
    # 创建一个副本，将无效值设为NaN
    temp_map = disparity_map.copy()
    temp_map[invalid_mask] = np.nan
    
    # 使用matplotlib保存，它会自动处理NaN值
    plt.imsave('test_standard.png', temp_map, cmap='jet')
    print("视差图已保存为 'test_standard.png'")

    


'''