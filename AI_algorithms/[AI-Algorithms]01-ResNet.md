**ResNet 与 Batch Normalization 深度解析**

## **1\. 为什么需要 ResNet? (The "Why")**

ResNet (Deep Residual Learning) 是深度学习历史上的分水岭。

* **在 ResNet 之前**：网络超过 20 层就很难训练（梯度消失/爆炸，退化问题）。  
* **在 ResNet 之后**：网络可以轻松堆叠到 100层、1000层。

更重要的是，**Transformer (LLM 的基石) 里的核心机制之一就是 Residual Connection (残差连接)**。不理解 ResNet，你就无法真正理解为什么现在的 GPT、Llama 能做得这么深而不崩溃。

### **核心思想：残差映射 (Residual Mapping)**

ResNet 引入了 "Shortcut Connection" (捷径连接)。假设我们想学习一个目标映射 $H(x)$：

* **传统网络**：直接尝试学习 $H(x)$。  
* **ResNet**：尝试学习残差函数 $F(x) = H(x) - x$。  
  * 因此，原始映射变为：$H(x) = F(x) + x$。

学习 "什么都不做" (即 $F(x)=0$，输出等于输入) 比学习一个恒等映射 (Identity Mapping) 要容易得多。如果某一层是多余的，权重只需衰减到 0，网络就自动跳过了这一层。

**2\. 梯度高速公路 (Gradient Superhighway) \- 数学推导**

### **前向传播**

假设一个残差块的公式为 $x_{l+1} = x_l + F(x_l, W_l)$。  
如果我们堆叠了 $L$ 层，那么从第 $l$ 层到第 $L$ 层的关系是：

$$x_L = x_l + \sum_{i=l}^{L-1} F(x_i, W_i)$$

### **反向传播 (关键推导)**

根据损失函数 $\mathcal{L}$ 对 $x_l$ 求梯度：

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l}$$  
将 $x_L$ 的公式代入链式法则展开：

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \left( 1 + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F(x_i, W_i) \right)$$  
利用乘法分配律展开，总梯度被分成了两部分：

$$\frac{\partial \mathcal{L}}{\partial x_l} = \underbrace{\frac{\partial \mathcal{L}}{\partial x_L}}_{\text{gradient returned directly}} + \underbrace{\frac{\partial \mathcal{L}}{\partial x_L} \cdot \sum \frac{\partial F}{\partial x_l}}_{\text{gradient through residual layer}}$$

### **为什么这一步很重要？**

1. 传统网络 (如 VGG)：  
   梯度是连乘的：$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \prod_{i=l}^{L-1} \frac{\partial f_i}{\partial x_i}$。  
   只要中间有一层的梯度小于 1 (比如 0.9)，连乘几十次后，$\approx 0.9^{50} \approx 0.005$，梯度就消失了。  
2. ResNet 的魔法：  
   因为有了公式里的 加法结构，第一项 $\frac{\partial \mathcal{L}}{\partial x_L}$ 是 "保底梯度"。  
   这一项不包含任何权重 $W$ 或激活函数的导数。这意味着，深层的梯度可以无损地、不经过任何衰减地直接“瞬移”回浅层。  
   即使第二项（修正梯度）因为梯度消失变成了 0，总梯度依然有第一项撑着。

[Image Diagram: 梯度高速公路示意图，显示梯度分为 dOut × 1 和 dOut × dF/dx 两条路径]

## ---

**3\. 梯度爆炸与 Batch Normalization**

### **潜在风险**

公式中的第二项 $\sum \frac{\partial F}{\partial x_l}$ 有可能导致 梯度爆炸 (Explosion)。  
虽然 "1" 解决了梯度消失，但后面那个连加项 $\sum$ 如果每一层的导数都很大，加起来可能会导致总梯度变得非常大。

### **Batch Normalization (BN) 的作用**

BN 是那个把即将失控的数值强行按回地面的“镇压者”。它的核心作用是 **强行勒马**。无论前一层的输出变成了什么鬼样子（比如均值漂移到了 1000），BN 都会把它强行拉回到一个 **固定的分布** (通常是均值 0，方差 1 附近)。

BN 通过两个步骤来约束分布：

#### **Step 1: 暴力标准化 (The Constraint)**

对于一个 Batch 的输入 $x$，在通道 (Channel) 维度上计算均值 $\mu_B$ 和方差 $\sigma^2_B$，然后归一化：

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}$$

这一步保证了无论输入是什么分布，变换后的 $\hat{x}$ 一定满足均值为 0，方差为 1。

#### **Step 2: 再缩放 (Scale and Shift)**

如果仅仅强制为 $\mathcal{N}(0,1)$，可能会破坏网络学到的特征。所以 BN 引入了两个 可学习参数 $\gamma$ (缩放) 和 $\beta$ (平移)：

$$y_i = \gamma \cdot \hat{x}_i + \beta$$

* 如果网络发现保持原始分布最好，它可以学习成 $\gamma = \sqrt{\sigma^2}, \beta = \mu$ (即变回去了)。


**4\. 架构演进与优化**

### **为什么 Transformer (LLM) 不用 BN?**

* **ResNet (CNN) 使用 BN**：CNN 处理的是图像，同一 Batch 内不同样本的**同一通道**具有相似的统计特性。  
* **Transformer (NLP) 使用 Layer Normalization (LN)**：  
  * NLP 序列长度不一，且 Batch 内句子差异巨大，计算 Batch 均值往往没有意义。  
  * LN 是对 **每一个样本自己** 的所有通道 (Embedding 维度) 做归一化，不依赖 Batch Size。

### **融合加速 (Conv-BN Fusion)**

在模型部署 (TensorRT / ONNX) 时，BN 层通常会被“吸”进前面的卷积层里。  
因为 BN 是线性变换：$y = \gamma \frac{Wx+b-\mu}{\sigma} + \beta$，可以把 $\gamma, \sigma, \mu, \beta$ 全部融合进卷积核的权重 $W$ 和偏置 $b$ 中：

$$W_{new} = W \cdot \frac{\gamma}{\sigma}, \quad b_{new} = (b - \mu) \cdot \frac{\gamma}{\sigma} + \beta$$

这样推理时就没有 BN 层了，速度更快。


**5\. 高级视角：ResNet 是浅层网络的集成**

大家知道 ResNet 解决了梯度消失问题，但有研究表明 **ResNet 实际上表现得像浅层网络的集成 (Ensemble of Shallow Networks)**。

**展开公式**：对于一个 3 层的 ResNet，$y = (x + f_1) + f_2 + f_3$，展开后其实包含 $2^n$ 条路径：

* 路径 1: $x \to \text{out}$ (全跳过)  
* 路径 2: $x \to f_1 \to \text{out}$  
* ...  
* 路径 N: $x \to f_1 \to f_2 \to f_3 \to \text{out}$ (最长路径)

**鲁棒性对比**：

* **VGG (串联系统)**：$y = f_3(f_2(f_1(x)))$。删掉 $f_2$，数据流断裂，网络直接瘫痪。  
* **ResNet (并联系统)**：删掉 $f_2$，只是去掉了经过 $f_2$ 的那些路径。由于还存在 $x \to f_1 \to f_3$ 或 $x \to f_3$ 等大量其他路径，网络性能只会平滑下降，而不会崩溃。