import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def compute_epe(pred_disp, gt_disp, mask=None, max_disparity=192, left_crop=False):
        """
        计算端点误差 (End-Point Error)
        
        Args:
            pred_disp: 预测视差图
            gt_disp: 真实视差图
            mask: 有效像素掩码
            
        Returns:
            epe: 平均端点误差
        """
        if mask is None:
            mask = (gt_disp > 0) & (gt_disp < max_disparity)
            if left_crop:
                mask[:, :max_disparity] = 0

            
        diff = np.abs(pred_disp[mask] - gt_disp[mask])
        epe = np.mean(diff)
        return epe

def compute_bad_pixels(pred_disp, gt_disp, threshold, mask=None, max_disparity=192, left_crop=False):
        """
        计算坏像素比例 (bad pixels)
        
        Args:
            pred_disp: 预测视差图
            gt_disp: 真实视差图
            threshold: 阈值 (如2, 3)
            mask: 有效像素掩码
            
        Returns:
            bad_pixels: 坏像素比例
        """
        if mask is None:
            mask = (gt_disp > 0) & (gt_disp < max_disparity)
            if left_crop:
                mask[:, :max_disparity] = 0
            
        diff = np.abs(pred_disp[mask] - gt_disp[mask])
        bad_pixels = np.sum(diff > threshold) / len(diff)
        return bad_pixels