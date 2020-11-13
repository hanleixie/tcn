# 算法详解

## 1、目的

## 2、算法总览

**时间序列算法**

* 1、Arima算法
* 2、X11_arima算法
* 3、Prophet算法
* 4、LSTM算法

**回归算法**

* 1、Lightgbm算法
* 2、GBDT算法
* 3、LSTM算法
* 4、TCN算法

## 3、算法详解

**LSTM**

![lstm动图](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9CblNORWFmaWNGQWIxaDZOQ28wNWF0Qlh2ZHU3UThQaWJwaWFBRmtjUkJFbkVDQ2lhSGhNbEpvSkhQWXRURDN0NmxqYjhmWjVCTWthNHhyZGJ6aWJsdHU0RTN3LzY0MD93eF9mbXQ9Z2lm?x-oss-process=image/format,png)

使用Lstm原因：

* NN会受到短时记忆的影响。如果一个序列足够长，它们很难讲信息从较早的时间步传送到后面的时间步。
* 在反向传播期间，RNN会面对梯度消失的问题。梯度是用于更新神经网络的权重值，消失的梯度问题是当梯度随时间的推移传播时梯度下降，如果剃度值非常小，就不会继续学习。


LSTM 原理，参考下图：

<img src="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9CblNORWFmaWNGQWIxaDZOQ28wNWF0Qlh2ZHU3UThQaWJwNDcxRHI1WVNDTW1pYmtpYTZ4cTdxS3A4T0xHUWNqZTgzeWlhc2VjUFRmSkNvb3VpYkpIR1RRQnlwZy82NDA_d3hfZm10PXBuZw?x-oss-process=image/format,png" width="700">

**核心概念**

LSTM 的核心概念在于细胞状态以及“门”结构。细胞状态相当于信息传输的路径，让信息能在序列连中传递下去。你可以将其看作网络的“记忆”。理论上讲，细胞状态能够将序列处理过程中的相关信息一直传递下去。因此，即使是较早时间步长的信息也能携带到较后时间步长的细胞中来，这克服了短时记忆的影响。信息的添加和移除我们通过“门”结构实现，“门”结构在训练过程中会去学习该保存或遗忘哪些信息。

* 遗忘门：

  遗忘门的功能是决定应丢弃或保留哪些信息。来自前一个隐藏状态的信息和当前输入的信息同时传递到 sigmoid 函数中去，输出值介于 0 和 1 之间，越接近 0 意味着越应该丢弃，越接近 1 意味着越应该保留。

  ![遗忘门](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9CblNORWFmaWNGQWIxaDZOQ28wNWF0Qlh2ZHU3UThQaWJwbURHVmljTHhVRGRVYlhsUmF5Tm9NUzZUU3dsUE5CYW1vNVR1WlBLWlpFVDdoZ2ZYUGJXZ1pQZy82NDA_d3hfZm10PWdpZg?x-oss-process=image/format,png)

* 输入门：

  输入门用于更新细胞状态。首先将前一层隐藏状态的信息和当前输入的信息传递到 sigmoid 函数中去。将值调整到 0~1 之间来决定要更新哪些信息。0 表示不重要，1 表示重要。其次还要将前一层隐藏状态的信息和当前输入的信息传递到 tanh 函数中去，创造一个新的侯选值向量。最后将 sigmoid 的输出值与 tanh 的输出值相乘，sigmoid 的输出值将决定 tanh 的输出值中哪些信息是重要且需要保留下来的。

  ![输入门](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9CblNORWFmaWNGQWIxaDZOQ28wNWF0Qlh2ZHU3UThQaWJwdE1pYkx1NDFGWjhNaWJZU3RiaEg2WU93MlFEUzNPY25aYmtZNkpEa3ZwZXJndXdlVndSMVR2aWFBLzY0MD93eF9mbXQ9Z2lm?x-oss-process=image/format,png)

* 细胞状态：

  下一步，就是计算细胞状态。首先前一层的细胞状态与遗忘向量逐点相乘。如果它乘以接近 0 的值，意味着在新的细胞状态中，这些信息是需要丢弃掉的。然后再将该值与输入门的输出值逐点相加，将神经网络发现的新信息更新到细胞状态中去。至此，就得到了更新后的细胞状态。

  ![细胞状态](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9CblNORWFmaWNGQWIxaDZOQ28wNWF0Qlh2ZHU3UThQaWJwc2V1WXV4R0oyN1hjeHAxRUl1WUxpY2hYaWFRVGs3ZG5wNU1WODZQaWFiSHhRTFA5UWJQaHlGZEJBLzY0MD93eF9mbXQ9Z2lm?x-oss-process=image/format,png)

* 输出门：

  输出门用来确定下一个隐藏状态的值，隐藏状态包含了先前输入的信息。首先，我们将前一个隐藏状态和当前输入传递到 sigmoid 函数中，然后将新得到的细胞状态传递给 tanh 函数。最后将 tanh 的输出与 sigmoid 的输出相乘，以确定隐藏状态应携带的信息。再将隐藏状态作为当前细胞的输出，把新的细胞状态和新的隐藏状态传递到下一个时间步长中去。

  ![输出门](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9CblNORWFmaWNGQWIxaDZOQ28wNWF0Qlh2ZHU3UThQaWJwSW9ZaENOVHp1YVM0WGNQb3Nnb1c2b3dkZ0VlUFJibm5BbzVaaWNpYmMyV1pVcUNLR0RpYmRxUXZ3LzY0MD93eF9mbXQ9Z2lm?x-oss-process=image/format,png)

**输入数据格式**

* input ( seq_len, batch, input_size)

  h0 ( num_layers * num_directions, batch, hidden_size)

  c0 ( num_layers * num_directions, batch, hidden_size)

  假设有24组数据，每组特征包含四个值，既[a1, a2, a3, a4]、维度为[24，4]，其input的维度为[24，1，4]

  标签为[seq_len, batch, 1]，既[24，1，1]

**输出数据格式**

* output(seq_len, batch, hidden_size * num_directions) 

  hn(num_layers * num_directions, batch, hidden_size) 

  cn(num_layers * num_directions, batch, hidden_size)

**Inputs: input, (h_0, c_0)**
- **input** of shape `(seq_len, batch, input_size)`: tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence` or :func:`torch.nn.utils.rnn.pack_sequence` for details. 
- **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor containing the initial hidden state for each element in the batch. If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
- **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor containing the initial cell state for each element in the batch. If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

**Outputs: output, (h_n, c_n)**

- **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor containing the output features `(h_t)` from the last layer of the LSTM, for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output will also be a packed sequence. For the unpacked case, the directions can be separated using ``output.view(seq_len, batch, num_directions, hidden_size)``, with forward and backward being direction `0` and `1` respectively. Similarly, the directions can be separated in the packed case.
- **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor containing the hidden state for `t = seq_len`. Like *output*, the layers can be separated using``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
- **c_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor containing the cell state for `t = seq_len`.

涉及到的主要计算过程为：

$$\begin{array} {ll} \\
    i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
    f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
    g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
    o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
    c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
    h_t = o_t \odot \tanh(c_t)
\end{array}$$

各个步骤的具体计算过程及图解如下所示：

<table>
  <tr>
    <td>
      <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png" border=0>
    </td>
    <td>
      <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png" border=0>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png" border=0>
    </td>
    <td>
      <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png" border=0>
    </td>
  </tr>
</table>

关于 LSTM 的更多内容可参考 colah 的博客[^1]。

BiLSTM

BILSTM 即 双向 LSTM，一般结构如下图所示：

<img src="http://colah.github.io/posts/2015-09-NN-Types-FP/img/RNN-bidirectional.png" width="700">

BiLSTM 实际上只是将两个独立的 LSTM 放在一起。对于一个 LSTM，按正常时序进行处理，另一个 LSTM 则按反向时序处理。输出的结果通常采用的是拼接(concat)，而不是求和。即上图中的 $y=[A|A']$。

需要注意的是，拼接的过程是正向的第一个时序与反向的最后一个时序进行拼接，依次类推。

**TCN**



## 补注：

**决策树模型**

* 1、决策树模型：由有向边和结点组成；结点有两种类型：内部节点（表示一个特征和属性）和叶子结点（表示一个类）。

* 2、决策树学习的本质是从训练集中归纳出一组分类规则。

* 3、决策树学习的损失函数通常是正则化的极大似然函数。

* 4、决策树学习算法包括特征选择、决策树生成和决策树剪枝。

  决策树学习算法通常是一个递归的选择最优特征，并根据该特征对训练集数据进行分割，使得对各个子数据集有一个最好的分类过程。开始，构建根节点，将所有训练数据都放在根节点。选择一个最优特征，按照这一特征将训练数据集分割成子集，使得各个子集有一个当前条件下最好的分类。如果这些子集已经能够被基本正确分类，那么构建叶结点，并将这些子集分到多对应的叶结点中；如果仍有子集不能被正确分类，那么就对这些子集选择新的最优特征，继续对其进行分割，构建相应的结点。如此递归的进行下去，知道所有的子集都被正确分类，或者没有合适的新特征为止。生成的决策树对未知的测试集未必有很好的分类能力，此时需要去掉过于细分的叶结点，使其退回父节点，甚至更高的结点，然后将父结点或者更高的结点改为新的叶结点。

* 5、决策树学习常用的算法有ID3、C4.5和CART。

### 信息增益

* 1、熵：表示随机变量不确定性的度量。

$$
H(p)=-\sum_{i=1}^{n}p_{i}log{p}_{i}
$$

与X的取值无关，只与X的分布有关。

* 2、信息增益表示得知特征X的信息而使得类Y的信息不确定性减少的程度。

$$
g(D,A)=H(D)-H(D|A)
$$

特征A对训练数据集D的信息增益 $$g(D|A)$$ ，定义为集合D的经验熵与特征A给定条件下D的经验条件上 $H(D|A)$ 之差。

* 信息增益使用ID3划分，偏向于多数样本的类别

**根据信息增益准则的特征选择方法是：对训练集（或子集）D，计算其每个特征的信息增益，并比较它们的大小，选择信息增益最大的特征**

### 信息增益比

* 将特征A对训练数据集D的信息增益比 $_{gR}(D,A)$ 定义为信息增益 $g(D,A)$ 与训练数据集D关于特征A的值得熵 $H_{A}(D)$ 之比。

$$
_{gR}(D,A)=\frac{g(D,A)}{H_{A}(D)}
$$

其中，$H_{A}=-\sum_{i=1}^{n}\frac{|D_i}{|D|}log_{2}\frac{D_i}{D}$ ，n是特征A取值的个数。

* 信息增益比使用C4.5划分，偏向少数样本的类别

### 剪枝

* 在决策树学习中将已生成的树进行简化的过程称为剪枝。剪枝从已生成的树上裁掉一些子树或叶结点，并将其根结点或父结点作为新的叶结点，从而简化分类树的模型。

### CART 树

* CART 算法由以下两步组成：

  （1）决策树生成：基于训练数据集生成决策树，生成的决策树要尽量打。

  （2）决策树剪枝：用于验证数据集对已生成的树进行剪枝并选择最优子树，用损							   失函数最小作为剪枝的标准。

  *决策树生成时，对回归树用平方误差最小化原则，对分类树用基尼指数最小化准则*
  
* 使用基尼指数，反应数据集中随机抽取两个样本，其类别标记不一致的概率，偏向样本多的类别。



















#### 数据归一化和归一化

##### 1、归一化（Normalization）

* 定义：将一列数据变化到某个固定区间（范围）中。

$$
归一化（normalization）：\frac{X_{i}-X_{min}}{X_{max}-X_{min}}
$$

##### 2、标准化（standardization）

* 定义：将数据变换为均值为0，标准差为1。

$$
标准化（standardization）：\frac{X_{i}-\mu}{\delta}
$$

##### 3、区别和联系

* 联系：二者本质上都是对数据的线性变换，不会改变原始数据的排列顺序。

* 差异：

  1、Normalization会严格限定变换后数据的范围，Standardization不会限定变换后的范围，其均值为0，标准差为1.

  2、Normalization对数据的缩放比例仅仅和极值相关，Standardization则不然。

##### 4、原因和用途

* 统计建模中，如回归模型，自变量 X 的量纲不一致导致回归系数无法直接解读或者错误解读，需要将 X 统一量纲。
* 机器学习和统计学习任务中有很多地方要用到“距离”的计算，比如PCA、KNN、Kmeans等，假设计算欧式距离，不同维度量纲可能导致距离计算依赖以量纲大的那些特征。
* 参数估计时使用梯度下降，在使用梯度下降的方法求解最优解问题时，归一化/标准化可以加快梯度下降的求解速度，既提升模型的收敛速度。







[^1]: 