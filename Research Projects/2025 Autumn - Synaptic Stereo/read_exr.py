import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt

pic = "0216"

def read_exr_depth(exr_path):
    # 打开EXR文件
    exr_file = OpenEXR.InputFile(exr_path)
    
    # 获取头信息
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    # 读取深度通道（通常是R通道或Z通道）
    try:
        # 尝试读取Z通道
        z_str = exr_file.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
    except:
        # 如果Z通道不存在，使用R通道
        z_str = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    
    # 转换为numpy数组
    depth = np.frombuffer(z_str, dtype=np.float32)
    depth.shape = (size[1], size[0])
    
    return depth


disp = read_exr_depth(f'./synthetic/depth/{pic}.exr')

valid = np.isfinite(disp) & (disp > 0)
disp_vis = np.zeros_like(disp, dtype=np.uint8)
if np.any(valid):
    dmin, dmax = np.percentile(disp[valid], [1, 99.5])
    dmin, dmax = float(dmin), float(max(dmax, dmin+1e-3))
    disp_norm = np.clip((disp - dmin) / (dmax - dmin), 0, 1)
    disp_vis = (disp_norm * 255).astype(np.uint8)
# cv.imwrite("mb_disp.png", disp_vis)

plt.imshow(disp_vis, cmap='jet')
# plt.imshow(depth_m , cmap='jet')
# plt.colorbar()
plt.axis('off')
plt.savefig(f"{pic}_standard.png")
plt.close()