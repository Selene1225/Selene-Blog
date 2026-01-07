# [AI-Algorithm] 03-Convolution

## 1. 为什么用卷积？

想象你要处理一张很普通的手机照片：1000×1000 像素，RGB 3通道。

输入向量的大小是：1000×1000×3=3,000,000 (三百万维)。

如果你用全连接层（Dense Layer），假设只有 1000 个神经元：

> 权重数量 = 3,000,000×1,000=30 亿个参数！

这不仅显存瞬间爆炸，而且根本训练不出来（过拟合风险极高）。


再考虑如果你在图片的左上角学会了识别“猫”，全连接层换到图片的右下角看到猫时，它完全不认识。因为它把每个像素的位置都“写死”了，它认为“左上角的像素”和“右下角的像素”是完全两回事。

人类的直觉： 猫就是猫，不管它在画面的哪里。

为了解决这些问题，就用到了卷积！

卷积的核心思想是：“拿着手电筒（Filter）在图片上滑动”。
1. 局部连接 (Local Connectivity)： 我们不需要看整张图，每次只看一个 3×3 的小窗口。
2. 权值共享 (Weight Sharing)： 这是核心！ 我们用同一个 3×3 的过滤器（Filter），扫描整张大图。
  - 物理含义： 如果这个过滤器是用来检测“垂直边缘”的，那么不管边缘在图片哪里，我都能检测出来。
  - 参数量对比： 同样处理上面的图，一个 3×3 的卷积核，参数只有 9个！从 30亿 降到 9，这就是卷积的威力。

核心公式
> 假设输入图是 $X$，卷积核是 $K$（大小 $k×k$），输出是 $Y$。 在位置 $(i,j)$ 的输出值是：

$Y_{i,j} = \sum_{m=0}^{k−1} \sum_{n=0}^{k−1}X_{i+m,j+n} \cdot K_{m,n}+b$
直觉翻译： 对应位置相乘，然后求和。就像把两张透明纸叠在一起看重合度。

## 2. 代码演示卷积做什么

```py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# update the image path here
image_path = "{image path}"

def simple_conv2d(image, kernel, padding=0, stride=1):
    """
    Simple 2D convolution function using only NumPy
    """
    # Add padding if needed
    if padding > 0:
        padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    else:
        padded_image = image
    
    # Get dimensions
    img_h, img_w = padded_image.shape
    ker_h, ker_w = kernel.shape
    
    # Calculate output dimensions
    out_h = (img_h - ker_h) // stride + 1
    out_w = (img_w - ker_w) // stride + 1
    
    # Initialize output array
    output = np.zeros((out_h, out_w))
    
    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            # Extract region of interest
            roi = padded_image[i*stride:i*stride+ker_h, j*stride:j*stride+ker_w]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(roi * kernel)
    
    return output

# Test with a simple 5x5 image
test_image = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
])

# 3x3 mean kernel (each value is 1/9)
mean_kernel = np.ones((3, 3)) / 9

print("Input image (5x5):")
print(test_image)
print("\nConvolution kernel (3x3 mean):")
print(mean_kernel)

result = simple_conv2d(test_image, mean_kernel)
print("\nOutput result:")
print(result)
print("\nThis is convolution!")


# Check if file exists
if os.path.exists(image_path):
    # Load and convert to grayscale
    original_image = Image.open(image_path).convert('L')
    gray_image = np.array(original_image)
    print("Successfully loaded cat image")
else:
    # Create a random image if cat image doesn't exist
    gray_image = (np.random.rand(100, 100) * 255).astype(np.uint8)
    print("Cat image not found, using random image instead")

# Create 15x15 mean kernel for stronger blur effect
blur_kernel = np.ones((15, 15)) / 225  # 225 = 15x15

# Apply convolution (blur the image)
blurred_image = simple_conv2d(gray_image.astype(float), blur_kernel)

# Display original and blurred images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(gray_image, cmap='gray')
ax1.set_title('Original Cat Image')
ax1.axis('off')

ax2.imshow(blurred_image, cmap='gray')
ax2.set_title('Strongly Blurred Cat Image')
ax2.axis('off')

plt.show()

print(f"Original image shape: {gray_image.shape}")
print(f"Blurred image shape: {blurred_image.shape}")
print("Blur kernel size: 15x15")
```

这是最普通的卷积进行图像处理，卷积核是 $3 \times 3$ 且9个值相加为1，相当于对图像做模糊处理。

这个时候如果把卷积核改一下：

```py
vertical_kernel = np.zeros((3, 3))
vertical_kernel[:, 2] = 1
vertical_kernel[:, 0] = -1
```

我们要理解，卷积核是在计算相似度，上面 vertical_kernel是很经典的边缘检测卷积核

$$
\begin{bmatrix}
-1 & 0 & 1 \\
-1 & 0 & 1 \\
-1 & 0 & 1
\end{bmatrix}
$$

场景1：遇到平坦区域，比如纯白图像像素是 $m = \begin{bmatrix} 10 & 10 & 10 \\ 10 & 10 & 10 \\ 10 & 10 & 10 \end{bmatrix}$ ，运算 $vertical_kernel \cdot m = 0$，说明这里什么特征也没有。

场景2：遇到有垂直边界（左边暗右边亮）$m = \begin{bmatrix} 0 & 0 & 10 \\ 0 & 0 & 10 \\ 0 & 0 & 10 \end{bmatrix}$ ，运算 $vertical_kernel \cdot m = 30$，说明一定有一条竖线。

所以，卷积核就是在图片上滑来滑去，一旦现在区域和卷积核长得像（特征匹配），它就输出一个大数字（激活），如果长得不像，就输出0。

> 可以运行 AI_Code/03-convolution.ipynb 看结果


## 3. 相关问题
**1. $1 \times 1$ 卷积核有什么用？**
三个核心作用：
1. 跨通道信息交互：普通卷积是在空间上聚合信息，而 $1 \times 1$ 实在通道维度上做全连接，把不同通道在同一个像素位置的信息融合在了一起。（图像RGB三通道，卷积核实际上是立体的 $3 \times 3 \times 3$，三个通道卷积核会相加得到最后的值）
2. 升维或降维：通过改变输出通道数 $C_{out}$，我们可以极大地降低计算量。例如 ResNet 的 Bottleneck结构，先用 $1 \times 1$ 把256维降到64维（节省算力），中间用 $3 \times 3$ 处理，最后再用 $1 \times 1$ 升回256维。
3. 增加非线性：虽然卷积核小，但是后面接的ReLU不会少。再不损失空间分辨率的情况下，增加了一次非线性激活，增加了网络的表达能力。

**2. 计算一个卷机层的参数量，假设输入是 $28 \times 28 \times 128$ ，使用 $3 \times 3$ 的卷积核，输出通道是256，如果使用同等维度的全连接层，参数量会差多少？**

回答模板：

1. 卷积层参数计算：卷积核的参数只与核大小核输入/输出通道数有关，与图片长宽(28*28)无关，所以公式：$(K \times K \times C_{in} + 1_{bias}) \times C_{out}$ 计算：$(3 \times 3 \times 128 + 1) \times 256 \approx 294912$（大约30w参数）

2. 全连接层参数计算：全连接层需要把输入Flatten，输入维度：$28 \times 28 \times 128 \approx 100000$, 输出维度（为了保持信息量同级）：$28 \times 28 \times 256 \approx 200000$。权重矩阵：$10^5 \times 2 \cdot 10^5 = 2 \cdot 10^{10}$（200亿参数）

3. 全连接层的参数量是卷积层近10万倍，所以卷积就是利用图像的局部性核平移等变性来实现权值共享。

**3. 架构直觉，为什么现在网络（如VGG）都喜欢用小的 $3 \times 3$ 卷积核，而不是大的如 $7 \times 7$ ?**

这里基于2个维度的考量：等效感受野（Receptive field equivalent，翻译很难懂 =_=）和参数效率。

1. 等效感受野：堆叠2个 $3 \times 3$ 的卷积层，其感受野等于1个 $5 \times 5$ 的卷积层。堆叠3个等效于 $7 \times 7$ 的卷积层
2. 参数更少：
    - 2个 $3 \times 3$ : $2 \times (3 \times 3 \times C^2 ) = 18C^2$
    - 1个 $5 \times 5$ : $1 \times (5 \times 5 \times C^2 ) = 25C^2$
    - 结论：用小核堆叠，参数量更少
3. 非线性更多：用2层可以多夹一层ReLU，也就是多一层非线性变换，网络的拟合能力比单纯的一层 $5 \times 5$ 线性层更强。

**4. 卷积具有“平移不变性”吗**
卷积层具有的是“平移等变性”，而不是“平移不变性”。
1. 平移不变性，是不管输入怎么移动，输出都不变
2. 平移等变性，是输入图像如果向右平移，卷积后的特征也会向右平移，卷积层回忠实地记录位置的变化。 