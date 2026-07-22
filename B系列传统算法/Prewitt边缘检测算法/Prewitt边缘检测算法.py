# 本程序实现了Prewitt边缘检测算法。
# Prewitt算子是一种基于离散微分算子的边缘检测方法，
# 通过计算图像灰度函数的梯度来识别图像中的边缘。
# 与Sobel算子不同，Prewitt算子使用均匀权重的卷积核，
# 对水平和垂直方向的边缘进行检测。
# 程序包含图像灰度转换、Prewitt卷积核定义、卷积运算、梯度计算和图像归一化等核心步骤。
# 所有核心算法均使用Python原生实现，OpenCV仅用于图像文件的读写操作。



############################################################
#   微信公众号：计算机视觉技术
#   微信视频号：计算机视觉技术
#   网站         ：BUTIANYUN.COM
############################################################



import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def butianyun_read_image(filename):
    """读取图像文件"""
    return cv2.imread(filename)

def butianyun_write_image(filename, image):
    """写入图像文件"""
    cv2.imwrite(filename, image)

def butianyun_convert_to_gray(image):
    """将彩色图像转换为灰度图像"""
    if len(image.shape) == 3:
        gray = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                gray[i, j] = int(0.299 * image[i, j, 2] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 0])
        return gray
    return image

def butianyun_pad_image(image, pad_size=1):
    """对图像进行边缘填充，便于卷积运算"""
    padded = np.pad(image, pad_size, mode='reflect')
    return padded

def butianyun_convolve(image, kernel):
    """卷积运算"""
    height, width = image.shape
    k_height, k_width = kernel.shape
    pad_size = k_height // 2
    padded = butianyun_pad_image(image, pad_size)
    result = np.zeros((height, width), dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            patch = padded[i:i+k_height, j:j+k_width]
            result[i, j] = np.sum(patch * kernel)
    
    return result

def butianyun_prewitt_kernel_x():
    """定义Prewitt水平方向卷积核"""
    return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)

def butianyun_prewitt_kernel_y():
    """定义Prewitt垂直方向卷积核"""
    return np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

def butianyun_calculate_gradient_magnitude(gradient_x, gradient_y):
    """计算梯度幅值"""
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return magnitude

def butianyun_normalize_image(image):
    """将图像归一化到0-255范围"""
    if np.max(image) > 0:
        normalized = (image / np.max(image) * 255).astype(np.uint8)
    else:
        normalized = image.astype(np.uint8)
    return normalized

def butianyun_prewitt_edge_detection(image):
    """执行Prewitt边缘检测算法"""
    gray = butianyun_convert_to_gray(image)
    
    kernel_x = butianyun_prewitt_kernel_x()
    kernel_y = butianyun_prewitt_kernel_y()
    
    gradient_x = butianyun_convolve(gray, kernel_x)
    gradient_y = butianyun_convolve(gray, kernel_y)
    
    magnitude = butianyun_calculate_gradient_magnitude(gradient_x, gradient_y)
    edges = butianyun_normalize_image(magnitude)
    
    return edges

def butianyun_display_comparison(original, processed, algorithm_name, save_path):
    """对照显示原始图像和处理后的图像"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    if len(original.shape) == 3:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        ax1.imshow(original_rgb)
    else:
        ax1.imshow(original, cmap='gray')
    ax1.set_title('原始图像', fontsize=14)
    ax1.axis('off')
    
    ax2.imshow(processed, cmap='gray')
    ax2.set_title('Prewitt边缘检测结果', fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

def butianyun_main():
    """主函数"""
    input_filename = 'butianyun_computer_vision_input.png'
    output_filename = 'butianyun_computer_vision_output.png'
    comparison_filename = 'Prewitt边缘检测算法.png'
    
    image = butianyun_read_image(input_filename)
    
    edges = butianyun_prewitt_edge_detection(image)
    
    butianyun_write_image(output_filename, edges)
    
    butianyun_display_comparison(image, edges, 'Prewitt边缘检测算法', comparison_filename)

if __name__ == '__main__':
    butianyun_main()
