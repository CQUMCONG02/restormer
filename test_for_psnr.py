import os
import cv2
import numpy as np


def preprocess_image(image_array):
    """
    对图像数组进行预处理：裁剪到 [-1, 1] 范围，然后转换到 [0, 1] 范围。

    Args:
        image_array (numpy.ndarray): 输入的图像数组 (H, W, C) 或 (H, W)，数值范围任意。

    Returns:
        numpy.ndarray: 预处理后的图像数组，数值范围在 [0, 1]。
    """
    # 转换为浮点数以进行数学运算
    image_float = image_array.astype(np.float32)

    # Clamp (裁剪) 数值到 [-1, 1] 范围
    image_clamped = np.clip(image_float, -1.0, 1.0)

    # 将 [-1, 1] 映射到 [0, 1]
    image_processed = (image_clamped * 0.5) + 0.5

    return image_processed


def calculate_psnr_numpy(restored_np, target_np):
    """
    计算两个 NumPy 数组图像之间的峰值信噪比 (PSNR)。

    Args:
        restored_np (numpy.ndarray): 经过预处理的生成图像数组，值范围 [0, 1]。
        target_np (numpy.ndarray): 经过预处理的目标/真实图像数组，值范围 [0, 1]。

    Returns:
        float: PSNR 值，单位为 dB。如果计算失败则返回 None。
    """
    # 确保两张图片尺寸相同
    if restored_np.shape != target_np.shape:
        print(f"错误: 图像尺寸不匹配 {restored_np.shape} vs {target_np.shape}")
        return None

    # 确保数据类型是 float32 以便精确计算
    restored_np = restored_np.astype(np.float32)
    target_np = target_np.astype(np.float32)

    # 计算 MSE (均方误差)
    mse = np.mean((restored_np - target_np) ** 2)

    if mse == 0:  # 如果 MSE 为 0，意味着两张图片完全一样
        return float('inf')  # PSNR 理论上为无穷大

    # 计算 PSNR
    # MAX_I^2 是最大像素值的平方。因为我们处理的是 [0, 1] 范围，所以 MAX_I = 1.0
    max_pixel_value = 1.0
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    return psnr_value


def calculate_psnr_from_paths(gen_path, gt_path):
    """
    从文件路径读取图像，进行预处理，然后计算 PSNR。

    Args:
        gen_path (str): 生成图像的路径。
        gt_path (str): Ground Truth 图像的路径。

    Returns:
        float: PSNR 值，单位为 dB。如果计算失败则返回 None。
    """
    # 读取图像
    img_gen = cv2.imread(gen_path, cv2.IMREAD_UNCHANGED)
    img_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

    if img_gen is None or img_gt is None:
        print(f"警告: 无法读取图像 {gen_path} 或 {gt_path}")
        return None

    # --- 应用预处理 ---
    # 注意：cv2.imread 读取的 BGR 顺序，如果您的网络输出是 RGB，可能需要 cv2.cvtColor 转换。
    # 这里假设通道顺序一致或都是灰度图。
    restored = preprocess_image(img_gen)
    target = preprocess_image(img_gt)

    # --- 调用 PSNR 计算函数 ---
    psnr_value = calculate_psnr_numpy(restored, target)

    return psnr_value


def find_image_pairs(folder_path):
    """
    在给定文件夹中查找以 'N.png' 和 'N_gt.png' 形式命名的图像对。

    Args:
        folder_path (str): 要搜索的文件夹路径。

    Returns:
        list of tuples: 包含 (generated_img_path, gt_img_path) 对的列表。
    """
    pairs = []
    all_files = os.listdir(folder_path)

    # 创建一个集合以便快速查找
    file_set = set(all_files)

    for filename in all_files:
        if filename.endswith('.png') and not filename.endswith('_gt.png'):
            # 提取基础名称 (例如 '1' from '1.png')
            base_name = filename[:-4]  # 移除 '.png'

            # 构造对应的 ground truth 文件名
            gt_filename = f"{base_name}_gt.png"

            # 检查对应的 ground truth 文件是否存在
            if gt_filename in file_set:
                gen_path = os.path.join(folder_path, filename)
                gt_path = os.path.join(folder_path, gt_filename)
                pairs.append((gen_path, gt_path))

    return pairs


def main():
    """
    主函数：获取文件夹路径，查找图像对，计算并打印 PSNR。
    """
    # 请将此处的 'your_folder_path_here' 替换为你存放图片的实际文件夹路径
    # folder_path = input("请输入包含图片的文件夹路径: ").strip()

    # 如果你不想每次都输入，可以直接在这里赋值，例如：
    folder_path = "/home/uav/DSY/Restormer/results/Dehazing_Restormer_archived_20260127_111907/visualization/TestDataset"

    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个有效的目录。")
        return

    # 查找所有图像对
    image_pairs = find_image_pairs(folder_path)

    if not image_pairs:
        print(f"在 '{folder_path}' 中没有找到匹配的 '_gt.png' 图像对。")
        return

    print(f"在 '{folder_path}' 中找到了 {len(image_pairs)} 个图像对。开始计算 PSNR...\n")

    total_psnr = 0.0
    count = 0

    for gen_path, gt_path in image_pairs:
        # 使用修改后的函数进行计算
        psnr = calculate_psnr_from_paths(gen_path, gt_path)
        if psnr is not None:
            # 提取文件名进行显示
            gen_name = os.path.basename(gen_path)
            gt_name = os.path.basename(gt_path)
            print(f"PSNR ({gen_name} vs {gt_name}): {psnr:.2f} dB")
            total_psnr += psnr
            count += 1
        else:
            # 如果 calculate_psnr_from_paths 内部已经打印了错误信息，则这里可以省略
            # print(f"计算 {gen_path} 和 {gt_path} 的 PSNR 时出错。")
            pass

    if count > 0:
        average_psnr = total_psnr / count
        print("\n" + "=" * 40)
        print(f"平均 PSNR: {average_psnr:.2f} dB (基于 {count} 个有效图像对)")
        print("=" * 40)
    else:
        print("没有成功计算任何一对图像的 PSNR。")


if __name__ == "__main__":
    main()