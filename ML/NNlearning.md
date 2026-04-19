### 9 神经网络的学习
**正向传播：**
给定输入$x$，一层一层算出每个神经元的$$z^{(l)} \quad\text{和}\quad a^{(l)}$$
直到最后得到预测$$\hat y \quad\text{和}\quad a^{(L)}$$

**反向传播：**
已知真实标签$y$后，倒着算出每一层每个参数对损失$J$的影响，也就是$$\frac{\partial J}{\partial \Theta}$$

#### 记号

对于第$l$层：

- $a^{(l)}$：第$l$层的激活值
- $z^{(l)}$：第$l$层激活前的线性组合
- $\Theta^{(l)}$：从第$l$层连到第$l+1$层的权重矩阵
前向传播是：$$z^{(l+1)}=\Theta^{(l)} a^{(l)}$$
$$a^{(l+1)}=g\!\left(z^{(l+1)}\right)$$

#### 什么是反向传播

反向传播的是：损失函数对中间变量的影响，计算的是$$\frac{\partial J}{\partial z^{(L)}},
\frac{\partial J}{\partial z^{(L-1)}},
\frac{\partial J}{\partial z^{(L-2)}}, \dots$$
也就是每一层“线性输入”$z$对最终损失的贡献。

所以$$\delta^{(l)}$$的含义是：$$\delta^{(l)}=\frac{\partial J}{\partial z^{(l)}}$$可以理解成如果我把这一层这个神经元的$z$稍微拨大一点，最终损失$J$会变大多少？

#### 隐藏层的 $\delta$ 为什么能往回推

假设你看隐藏层第 $l$ 层的某个神经元 $j$，它本身不直接进入损失函数 $J$，它是先影响下一层，再由下一层继续影响输出和损失。
所以用链式法则：
$$\delta_j^{(l)}=\frac{\partial J}{\partial z_j^{(l)}}
=\frac{\partial J}{\partial a_j^{(l)}}\cdot \frac{\partial a_j^{(l)}}{\partial z_j^{(l)}}$$

而$$a_j^{(l)} = g(z_j^{(l)})$$

所以$$\frac{\partial a_j^{(l)}}{\partial z_j^{(l)}} = g'(z_j^{(l)})$$

因为 $a_j^{(l)}$ 会影响下一层所有节点的 $z_i^{(l+1)}$，所以
$$\frac{\partial J}{\partial a_j^{(l)}}=\sum_i\frac{\partial J}{\partial z_i^{(l+1)}}\frac{\partial z_i^{(l+1)}}{\partial a_j^{(l)}}$$

又因为$$z_i^{(l+1)}=\sum_j \Theta_{ij}^{(l)} a_j^{(l)}$$

所以$$\frac{\partial z_i^{(l+1)}}{\partial a_j^{(l)}}=\Theta_{ij}^{(l)}$$

代回去：$$\frac{\partial J}{\partial a_j^{(l)}}=\sum_i \delta_i^{(l+1)}\Theta_{ij}^{(l)}$$

再乘上本层激活函数导数：$$\delta_j^{(l)}=\left(\sum_i \Theta_{ij}^{(l)}\delta_i^{(l+1)}\right) g'(z_j^{(l)})$$

这就是隐藏层反传公式来源。
按向量写：$$\delta^{(l)}=\left((\Theta^{(l)})^T\delta^{(l+1)}\right)\odot g'(z^{(l)})$$

这里的 $\odot$ 表示按元素相乘。
如果激活函数是sigmoid，还可以写成$$g'(z^{(l)}) = a^{(l)} \odot (1-a^{(l)})$$

于是$$\delta^{(l)}=\left((\Theta^{(l)})^T\delta^{(l+1)}\right)\odot a^{(l)}\odot(1-a^{(l)})$$

Stanford 的讲义给出的标准 backprop 公式就是这个结构：输出层先算 $\delta$，隐藏层按 $(W^T\delta)\odot f'(z)$ 往回推。

#### 为什么 bias 单元通常要“去掉”
$$\delta^{(l)} = \left((\Theta^{(l)})^T \delta^{(l+1)}\right)\odot g'(z^{(l)})$$
还要把第0个元素去掉，原因：
- bias 单元 $a_0^{(l)}=1$ 不是由某个 $z_0^{(l)}$ 经过激活函数算出来的
- 所以它没有对应的“普通神经元误差项”
- 反传到上一层时，算出来的那个 bias 位置只是为了矩阵维度方便，最后要去掉

#### 梯度为什么是$\delta \times a$
对某个权重 $\Theta_{ij}^{(l)}$，他只出现在下一层第 $i$ 个节点的线性组合里：$$z_i^{(l+1)}=\sum_j \Theta_{ij}^{(l)} a_j^{(l)}$$

所以：$$\frac{\partial J}{\partial \Theta_{ij}^{(l)}}=\frac{\partial J}{\partial z_i^{(l+1)}}\cdot \frac{\partial z_i^{(l+1)}}{\partial \Theta_{ij}^{(l)}}$$

而$$\frac{\partial J}{\partial z_i^{(l+1)}} = \delta_i^{(l+1)}$$

$$\frac{\partial z_i^{(l+1)}}{\partial \Theta_{ij}^{(l)}} = a_j^{(l)}$$

所以：$$\frac{\partial J}{\partial \Theta_{ij}^{(l)}}=\delta_i^{(l+1)} a_j^{(l)}$$

矩阵形式就是：$$\frac{\partial J}{\partial \Theta^{(l)}}=\delta^{(l+1)}(a^{(l)})^T$$

**某条边的梯度 = 这条边终点的误差 $\times$ 这条边起点的输出**

#### 总之
输出层误差：$$\delta^{(L)} = a^{(L)} - y$$

隐藏层误差怎么回传：$$\delta^{(l)}=\left((\Theta^{(l)})^T\delta^{(l+1)}\right)\odot g'(z^{(l)})$$

误差一旦有了，梯度就有了：$$\frac{\partial J}{\partial \Theta^{(l)}}=\delta^{(l+1)}(a^{(l)})^T$$
