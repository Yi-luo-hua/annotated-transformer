# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# %% [markdown]
# **【中文对照 / Chinese Translation】**
# 在过去五年中，Transformer 引起了广大学界与工业界的密切关注。
# 本文呈现了原论文 *Attention Is All You Need* 的逐行代码实现及其详细注解。它重新梳理并整理了原论文的部分章节，并在全文中添加了详细的代码与理论注释。本文档本身就是一个可以交互运行的 Notebook，并且是一份完全可用的 PyTorch 代码实现。
# 代码仓库参见 [这里](https://github.com/harvardnlp/annotated-transformer/)。
#

# %% [markdown]
# <h3> Table of Contents (目录) </h3>
# <ul>
# <li><a href="#prelims">Prelims (准备工作与依赖库)</a></li>
# <li><a href="#background">Background (研究背景与动机)</a></li>
# <li><a href="#part-1-model-architecture">Part 1: Model Architecture (第一部分：模型架构)</a></li>
# <li><a href="#model-architecture">Model Architecture (模型整体架构)</a><ul>
# <li><a href="#encoder-and-decoder-stacks">Encoder and Decoder Stacks (编码器与解码器堆栈)</a></li>
# <li><a href="#position-wise-feed-forward-networks">Position-wise Feed-Forward Networks (逐位置前馈网络)</a></li>
# <li><a href="#embeddings-and-softmax">Embeddings and Softmax (词嵌入与 Softmax 输出)</a></li>
# <li><a href="#positional-encoding">Positional Encoding (位置编码)</a></li>
# <li><a href="#full-model">Full Model (完整 Transformer 模型组装)</a></li>
# <li><a href="#inference">Inference (模型推理/预测过程)</a></li>
# </ul></li>
# <li><a href="#part-2-model-training">Part 2: Model Training (第二部分：模型训练)</a></li>
# <li><a href="#training">Training (训练过程)</a><ul>
# <li><a href="#batches-and-masking">Batches and Masking (批处理与 Mask 掩码机制)</a></li>
# <li><a href="#training-loop">Training Loop (训练循环机制)</a></li>
# <li><a href="#training-data-and-batching">Training Data and Batching (训练数据与动态 Batch 分割)</a></li>
# <li><a href="#hardware-and-schedule">Hardware and Schedule (硬件配置与学习率调度)</a></li>
# <li><a href="#optimizer">Optimizer (优化器配置)</a></li>
# <li><a href="#regularization">Regularization (正则化与标签平滑)</a></li>
# </ul></li>
# <li><a href="#a-first-example">A First Example (基础示例：复制任务)</a><ul>
# <li><a href="#synthetic-data">Synthetic Data (合成数据生成)</a></li>
# <li><a href="#loss-computation">Loss Computation (损失计算)</a></li>
# <li><a href="#greedy-decoding">Greedy Decoding (贪婪解码)</a></li>
# </ul></li>
# <li><a href="#part-3-a-real-world-example">Part 3: A Real World Example (第三部分：真实世界机器翻译示例)</a>
# <ul>
# <li><a href="#data-loading">Data Loading (数据加载与预处理)</a></li>
# <li><a href="#iterators">Iterators (数据迭代器构造)</a></li>
# <li><a href="#training-the-system">Training the System (系统训练)</a></li>
# </ul></li>
# <li><a href="#additional-components-bpe-search-averaging">Additional Components: BPE, Search, Averaging (拓展组件：BPE、束搜索与模型平均)</a></li>
# <li><a href="#results">Results (实验结果与可视化)</a><ul>
# <li><a href="#attention-visualization">Attention Visualization (注意力权重可视化)</a></li>
# <li><a href="#encoder-self-attention">Encoder Self Attention (编码器自注意力可视化)</a></li>
# <li><a href="#decoder-self-attention">Decoder Self Attention (解码器自注意力可视化)</a></li>
# <li><a href="#decoder-src-attention">Decoder Src Attention (解码器-源文本交叉注意力可视化)</a></li>
# </ul></li>
# <li><a href="#conclusion">Conclusion (结语与总结)</a></li>
# </ul>
#

# %% [markdown]
# # Prelims (准备工作与依赖库)
#

# %% [markdown]
# > My comments are blockquoted. The main text is all from the paper itself.
# >
# > **【中文对照 / Chinese Translation】**
# > 引用块（>）中的内容是作者的注解说明，其余主干文本均直接来自原论文 *Attention Is All You Need*。
#

# %% [markdown]
# # Background (研究背景与动机)
#

# %% [markdown]
# The goal of reducing sequential computation also forms the
# foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of
# which use convolutional neural networks as basic building block,
# computing hidden representations in parallel for all input and
# output positions. In these models, the number of operations required
# to relate signals from two arbitrary input or output positions grows
# in the distance between positions, linearly for ConvS2S and
# logarithmically for ByteNet. This makes it more difficult to learn
# dependencies between distant positions. In the Transformer this is
# reduced to a constant number of operations, albeit at the cost of
# reduced effective resolution due to averaging attention-weighted
# positions, an effect we counteract with Multi-Head Attention.
#
# Self-attention, sometimes called intra-attention is an attention
# mechanism relating different positions of a single sequence in order
# to compute a representation of the sequence. Self-attention has been
# used successfully in a variety of tasks including reading
# comprehension, abstractive summarization, textual entailment and
# learning task-independent sentence representations. End-to-end
# memory networks are based on a recurrent attention mechanism instead
# of sequencealigned recurrence and have been shown to perform well on
# simple-language question answering and language modeling tasks.
#
# To the best of our knowledge, however, the Transformer is the first
# transduction model relying entirely on self-attention to compute
# representations of its input and output without using sequence
# aligned RNNs or convolution.
#
# **【中文对照 / Chinese Translation】**
# 减少顺序计算（Sequential Computation）的目标也是 Extended Neural GPU、ByteNet 和 ConvS2S 的构建基础，这些模型均采用卷积神经网络作为基本构建块，能够并行计算所有输入和输出位置的隐藏表示。在这些模型中，关联两个任意输入或输出位置信号所需的计算操作数随位置间距离增长而增加：ConvS2S 呈线性增长，ByteNet 呈对数增长。这使得捕捉长距离位置之间的依赖关系变得极其困难。而在 Transformer 中，这一计算操作数被成功缩减到了常数级别（$O(1)$），虽然代价是由于对注意力权重位置进行平均而降低了有效分辨率，但我们通过多头注意力（Multi-Head Attention）机制成功抵消了这一负面影响。
#
# 自注意力机制（Self-Attention），有时也称为内部注意力（Intra-Attention），是一种将单个序列的不同位置相互关联以计算该序列整体表示的注意力机制。自注意力已成功应用于阅读理解、摘要生成、文本蕴涵和通用句子表示等多种任务。端到端记忆网络（End-to-End Memory Networks）基于循环注意力机制而非序列对齐的循环结构，已被证明在简单语言问答和语言建模任务中表现良好。
#
# 据我们所知，Transformer 是首个完全依赖自注意力机制来计算输入和输出表示、而不使用序列对齐 RNN 或卷积神经网络的转换模型（Transduction Model）。
#

# %% [markdown]
# # Part 1: Model Architecture (第一部分：模型架构与原理)
#

# %% [markdown]
# # Model Architecture (模型整体架构)
#

# %% [markdown]
# Most competitive neural sequence transduction models have an
# encoder-decoder structure
# [(cite)](https://arxiv.org/abs/1409.0473). Here, the encoder maps an
# input sequence of symbol representations $(x_1, ..., x_n)$ to a
# sequence of continuous representations $\mathbf{z} = (z_1, ...,
# z_n)$. Given $\mathbf{z}$, the decoder then generates an output
# sequence $(y_1,...,y_m)$ of symbols one element at a time. At each
# step the model is auto-regressive
# [(cite)](https://arxiv.org/abs/1308.0850), consuming the previously
# generated symbols as additional input when generating the next.
#
# **【中文对照 / Chinese Translation】**
# 目前最具竞争力的神经序列转换模型大多采用编码器-解码器（Encoder-Decoder）结构。在该结构中，编码器将符号表示的输入序列 $(x_1, ..., x_n)$ 映射为连续表示序列 $\mathbf{z} = (z_1, ..., z_n)$。在给定 $\mathbf{z}$ 的情况下，解码器逐个元素生成符号输出序列 $(y_1,...,y_m)$。在生成的每一步中，模型都是自回归的（Auto-Regressive），即将此前生成的符号作为额外的输入来生成下一个符号。
#

# %% [markdown]
# The Transformer follows this overall architecture using stacked
# self-attention and point-wise, fully connected layers for both the
# encoder and decoder, shown in the left and right halves of Figure 1,
# respectively.
#
# **【中文对照 / Chinese Translation】**
# Transformer 整体遵循这一架构，其编码器和解码器均采用了堆叠的自注意力层（Stacked Self-Attention）和逐位置的全连接层（Point-wise Fully Connected Layers），分别展示在论文图 1 的左半部分和右半部分。
#

# %% [markdown]
# ![](images/ModalNet-21.png)
#
#
#

# %% [markdown]
# ## Encoder and Decoder Stacks (编码器与解码器堆栈)
#
# ### Encoder (编码器)
#
# The encoder is composed of a stack of $N=6$ identical layers.
#
# **【中文对照 / Chinese Translation】**
# 编码器由 $N=6$ 个完全相同的层堆叠而成。
#

# %% [markdown]
# We employ a residual connection
# [(cite)](https://arxiv.org/abs/1512.03385) around each of the two
# sub-layers, followed by layer normalization
# [(cite)](https://arxiv.org/abs/1607.06450).
#
# **【中文对照 / Chinese Translation】**
# 我们在两个子层中的每一个周围都采用了残差连接（Residual Connection），随后跟上层归一化（Layer Normalization）。
#

# %% [markdown]
# That is, the output of each sub-layer is $\mathrm{LayerNorm}(x +
# \mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$ is the function
# implemented by the sub-layer itself.  We apply dropout
# [(cite)](http://jmlr.org/papers/v15/srivastava14a.html) to the
# output of each sub-layer, before it is added to the sub-layer input
# and normalized.
#
# To facilitate these residual connections, all sub-layers in the
# model, as well as the embedding layers, produce outputs of dimension
# $d_{	ext{model}}=512$.
#
# **【中文对照 / Chinese Translation】**
# 也就是说，每个子层的输出为 $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$，其中 $\mathrm{Sublayer}(x)$ 是子层自身实现的函数。在将子层的输出与其输入相加并进行归一化之前，我们对子层输出应用 Dropout 随机失活。
#
# 为了便于实现残差连接，模型中所有的子层以及词嵌入层的输出维度统一设定为 $d_{	ext{model}}=512$。
#

# %% [markdown]
# Each layer has two sub-layers. The first is a multi-head
# self-attention mechanism, and the second is a simple, position-wise
# fully connected feed-forward network.
#
# **【中文对照 / Chinese Translation】**
# 编码器的每一层包含两个子层。第一个是多头自注意力机制（Multi-Head Self-Attention），第二个是简单的逐位置全连接前馈网络（Position-wise Fully Connected Feed-Forward Network）。
#

# %% [markdown]
# ### Decoder (解码器)
#
# The decoder is also composed of a stack of $N=6$ identical layers.
#
# **【中文对照 / Chinese Translation】**
# 解码器同样由 $N=6$ 个完全相同的层堆叠而成。
#

# %% [markdown]
# In addition to the two sub-layers in each encoder layer, the decoder
# inserts a third sub-layer, which performs multi-head attention over
# the output of the encoder stack.  Similar to the encoder, we employ
# residual connections around each of the sub-layers, followed by
# layer normalization.
#
# **【中文对照 / Chinese Translation】**
# 除了每个编码器层中的两个子层之外，解码器还插入了第三个子层，该子层对编码器堆栈的输出执行多头注意力计算（即交叉注意力 Cross-Attention）。与编码器类似，我们在每个子层周围都采用了残差连接，随后进行层归一化。
#

# %% [markdown]
# We also modify the self-attention sub-layer in the decoder stack to
# prevent positions from attending to subsequent positions.  This
# masking, combined with fact that the output embeddings are offset by
# one position, ensures that the predictions for position $i$ can
# depend only on the known outputs at positions less than $i$.
#
# **【中文对照 / Chinese Translation】**
# 我们还修改了解码器堆栈中的自注意力子层，以防止当前位置关注到后续位置（即 Mask 掩码机制）。这种掩码结合输出嵌入向右偏移一个位置的操作，确保对位置 $i$ 的预测只能依赖于小于 $i$ 的已知前文输出。
#

# %% [markdown]
# > Below the attention mask shows the position each tgt word (row) is
# > allowed to look at (column). Words are blocked for attending to
# > future words during training.
# >
# > **【中文对照 / Chinese Translation】**
# > 下图中的注意力掩码展示了目标序列中的每个词（行）允许关注的位置（列）。在训练过程中，掩码屏蔽掉了当前词对未来词的注意力。
#

# %% [markdown]
# ### Attention (注意力机制)
#
# An attention function can be described as mapping a query and a set
# of key-value pairs to an output, where the query, keys, values, and
# output are all vectors.  The output is computed as a weighted sum of
# the values, where the weight assigned to each value is computed by a
# compatibility function of the query with the corresponding key.
#
# We call our particular attention "Scaled Dot-Product Attention".
# The input consists of queries and keys of dimension $d_k$, and
# values of dimension $d_v$.  We compute the dot products of the query
# with all keys, divide each by $\sqrt{d_k}$, and apply a softmax
# function to obtain the weights on the values.
#
# **【中文对照 / Chinese Translation】**
# 注意力函数（Attention Function）可以描述为将一个 Query（查询向量）和一组 Key-Value（键-值向量对）映射到一个 Output（输出向量）的过程，其中 Query、Keys、Values 和 Output 均为向量。输出是由 Values 的加权和计算得到的，分配给每个 Value 的权重是由 Query 与相应 Key 的匹配度函数（Compatibility Function）计算得出的。
#
# 我们将我们特有的注意力机制称为“缩放点积注意力”（Scaled Dot-Product Attention）。输入由维度为 $d_k$ 的 Queries 和 Keys，以及维度为 $d_v$ 的 Values 组成。我们计算 Query 与所有 Keys 的点积，将每个点积除以 $\sqrt{d_k}$，然后应用 Softmax 函数以获取分配给 Values 的权重。
#
# ![](images/ModalNet-19.png)
#

# %% [markdown]
# In practice, we compute the attention function on a set of queries
# simultaneously, packed together into a matrix $Q$.  The keys and
# values are also packed together into matrices $K$ and $V$.  We
# compute the matrix of outputs as:
#
# $$
#    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(
# rac{QK^T}{\sqrt{d_k}})V
# $$
#
# **【中文对照 / Chinese Translation】**
# 在实践中，我们同时对一组 Query 进行注意力计算，并将其打包拼接成矩阵 $Q$。Keys 和 Values 也分别打包成矩阵 $K$ 和 $V$。输出矩阵的计算公式为：
# $$
#    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(
# rac{QK^T}{\sqrt{d_k}})V
# $$
#

# %% [markdown]
# The two most commonly used attention functions are additive
# attention [(cite)](https://arxiv.org/abs/1409.0473), and dot-product
# (multiplicative) attention.  Dot-product attention is identical to
# our algorithm, except for the scaling factor of
# $
# rac{1}{\sqrt{d_k}}$. Additive attention computes the
# compatibility function using a feed-forward network with a single
# hidden layer.  While the two are similar in theoretical complexity,
# dot-product attention is much faster and more space-efficient in
# practice, since it can be implemented using highly optimized matrix
# multiplication code.
#
#
# While for small values of $d_k$ the two mechanisms perform
# similarly, additive attention outperforms dot product attention
# without scaling for larger values of $d_k$
# [(cite)](https://arxiv.org/abs/1703.03906). We suspect that for
# large values of $d_k$, the dot products grow large in magnitude,
# pushing the softmax function into regions where it has extremely
# small gradients (To illustrate why the dot products get large,
# assume that the components of $q$ and $k$ are independent random
# variables with mean $0$ and variance $1$.  Then their dot product,
# $q \cdot k = \sum_{i=1}^{d_k} q_ik_i$, has mean $0$ and variance
# $d_k$.). To counteract this effect, we scale the dot products by
# $
# rac{1}{\sqrt{d_k}}$.
#
# **【中文对照 / Chinese Translation】**
# 两种最常用的注意力函数是加性注意力（Additive Attention）和点积/乘性注意力（Dot-Product Attention）。点积注意力除了缩放因子 $
# rac{1}{\sqrt{d_k}}$ 之外与我们的算法完全相同。加性注意力使用带有单个隐层的前馈网络来计算匹配函数。虽然两者在理论复杂度上相似，但在实践中，点积注意力要快得多且更节省空间，因为它可以利用高度优化的矩阵乘法运算来实现。
#
# 虽然对于较小的 $d_k$ 值，这两种机制的表现相似，但在没有缩放因子的情况下，随着 $d_k$ 增大，加性注意力超越了点积注意力。我们怀疑，对于较大的 $d_k$ 值，点积的数值量级会急剧增大，从而将 Softmax 函数推向具有极小梯度的饱和区域（为了解释为什么点积会变大：假设 $q$ 和 $k$ 的各分量是均值为 0、方差为 1 的独立随机变量，则它们的点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 0，方差为 $d_k$）。为了抵消这种副作用，我们将点积乘以缩放因子 $
# rac{1}{\sqrt{d_k}}$。
#

# %% [markdown]
# ![](images/ModalNet-20.png)
#
#
#

# %% [markdown]
# Multi-head attention allows the model to jointly attend to
# information from different representation subspaces at different
# positions. With a single attention head, averaging inhibits this.
#
# $$
# \mathrm{MultiHead}(Q, K, V) =
#     \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \
#     	ext{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
# $$
#
# Where the projections are parameter matrices $W^Q_i \in
# \mathbb{R}^{d_{	ext{model}} 	imes d_k}$, $W^K_i \in
# \mathbb{R}^{d_{	ext{model}} 	imes d_k}$, $W^V_i \in
# \mathbb{R}^{d_{	ext{model}} 	imes d_v}$ and $W^O \in
# \mathbb{R}^{hd_v 	imes d_{	ext{model}}}$.
#
# In this work we employ $h=8$ parallel attention layers, or
# heads. For each of these we use $d_k=d_v=d_{	ext{model}}/h=64$. Due
# to the reduced dimension of each head, the total computational cost
# is similar to that of single-head attention with full
# dimensionality.
#
# **【中文对照 / Chinese Translation】**
# 多头注意力（Multi-Head Attention）允许模型联合关注来自不同位置的不同表示子空间（Subspaces）的信息。而如果只有一个注意力头，对所有位置进行简单平均会抑制这种多角度信息的捕获。
# $$
# \mathrm{MultiHead}(Q, K, V) =
#     \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \
#     	ext{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
# $$
# 其中线性投影是参数矩阵 $W^Q_i \in \mathbb{R}^{d_{	ext{model}} 	imes d_k}$、$W^K_i \in \mathbb{R}^{d_{	ext{model}} 	imes d_k}$、$W^V_i \in \mathbb{R}^{d_{	ext{model}} 	imes d_v}$ 以及 $W^O \in \mathbb{R}^{hd_v 	imes d_{	ext{model}}}$。
#
# 在这项工作中，我们采用 $h=8$ 个平行的注意力层（即 8 个注意力头）。对于每个头，我们设定 $d_k = d_v = d_{	ext{model}}/h = 64$。由于减少了每个头的维度，总计算成本与具有全维度单头注意力的计算成本非常接近。
#

# %% [markdown]
# ### Applications of Attention in our Model (注意力机制在模型中的三大应用)
#
# The Transformer uses multi-head attention in three different ways:
# 1) In "encoder-decoder attention" layers, the queries come from the
# previous decoder layer, and the memory keys and values come from the
# output of the encoder.  This allows every position in the decoder to
# attend over all positions in the input sequence.  This mimics the
# typical encoder-decoder attention mechanisms in sequence-to-sequence
# models such as [(cite)](https://arxiv.org/abs/1609.08144).
#
#
# 2) The encoder contains self-attention layers.  In a self-attention
# layer all of the keys, values and queries come from the same place,
# in this case, the output of the previous layer in the encoder.  Each
# position in the encoder can attend to all positions in the previous
# layer of the encoder.
#
#
# 3) Similarly, self-attention layers in the decoder allow each
# position in the decoder to attend to all positions in the decoder up
# to and including that position.  We need to prevent leftward
# information flow in the decoder to preserve the auto-regressive
# property.  We implement this inside of scaled dot-product attention
# by masking out (setting to $-\infty$) all values in the input of the
# softmax which correspond to illegal connections.
#
# **【中文对照 / Chinese Translation】**
# Transformer 以三种不同的方式应用多头注意力机制：
# 1）在“编码器-解码器交叉注意力”（Encoder-Decoder Attention）层中：Query 来自前一个解码器层，而 Key 和 Value 来自编码器的输出 Memory。这使得解码器中的每个位置都能关注到输入源序列的所有位置，模仿了 Seq2Seq 模型中典型的编码器-解码器注意力机制。
# 2）编码器包含“自注意力”（Self-Attention）层：在自注意力层中，所有的 Key、Value 和 Query 都来自同一个地方（即编码器中前一层的输出）。编码器中的每个位置都可以关注到编码器前一层的所有位置。
# 3）类似地，解码器中的“自注意力层”允许解码器中的每个位置关注到解码器中直到并包括该位置在内的所有位置。我们需要防止解码器中的向左信息流动，以保持自回归特性。我们在缩放点积注意力内部通过掩码机制（将 Softmax 输入中对应于非法连接的所有值设置为 $-\infty$）来实现这一点。
#

# %% [markdown]
# ## Position-wise Feed-Forward Networks (逐位置前馈神经网络)
#
# In addition to attention sub-layers, each of the layers in our
# encoder and decoder contains a fully connected feed-forward network,
# which is applied to each position separately and identically.  This
# consists of two linear transformations with a ReLU activation in
# between.
#
# $$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$
#
# While the linear transformations are the same across different
# positions, they use different parameters from layer to
# layer. Another way of describing this is as two convolutions with
# kernel size 1.  The dimensionality of input and output is
# $d_{	ext{model}}=512$, and the inner-layer has dimensionality
# $d_{ff}=2048$.
#
# **【中文对照 / Chinese Translation】**
# 除了注意力子层之外，编码器和解码器中的每一层都包含一个全连接的前馈网络（FFN），该网络独立且相同地作用于每个位置。它由两个线性变换组成，中间带有 ReLU 激活函数：
# $$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$
# 虽然线性变换在不同位置之间是相同的，但它们在层与层之间使用不同的参数。描述它的另一种方式是将其视为两个核大小为 1 的卷积。输入和输出的维度为 $d_{	ext{model}}=512$，中间隐层的维度为 $d_{ff}=2048$。
#

# %% [markdown]
# ## Embeddings and Softmax (词嵌入层与 Softmax 输出)
#
# Similarly to other sequence transduction models, we use learned
# embeddings to convert the input tokens and output tokens to vectors
# of dimension $d_{	ext{model}}$.  We also use the usual learned
# linear transformation and softmax function to convert the decoder
# output to predicted next-token probabilities.  In our model, we
# share the same weight matrix between the two embedding layers and
# the pre-softmax linear transformation, similar to
# [(cite)](https://arxiv.org/abs/1608.05859). In the embedding layers,
# we multiply those weights by $\sqrt{d_{	ext{model}}}$.
#
# **【中文对照 / Chinese Translation】**
# 与其他序列转换模型类似，我们使用可学习的词嵌入（Learned Embeddings）将输入 Token 和输出 Token 转换为维度为 $d_{	ext{model}}$ 的向量。我们还使用常见的可学习线性变换与 Softmax 函数，将解码器输出转换为预测下一个 Token 的概率。在我们的模型中，源语言嵌入层、目标语言嵌入层以及 Softmax 前的线性变换这三者共享相同的权重矩阵。在嵌入层中，我们将这些权重乘以 $\sqrt{d_{	ext{model}}}$ 进行缩放。
#

# %% [markdown]
# ## Positional Encoding (位置编码)
#
# Since our model contains no recurrence and no convolution, in order
# for the model to make use of the order of the sequence, we must
# inject some information about the relative or absolute position of
# the tokens in the sequence.  To this end, we add "positional
# encodings" to the input embeddings at the bottoms of the encoder and
# decoder stacks.  The positional encodings have the same dimension
# $d_{	ext{model}}$ as the embeddings, so that the two can be summed.
# There are many choices of positional encodings, learned and fixed
# [(cite)](https://arxiv.org/pdf/1705.03122.pdf).
#
# In this work, we use sine and cosine functions of different frequencies:
#
# $$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{	ext{model}}})$$
#
# $$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{	ext{model}}})$$
#
# where $pos$ is the position and $i$ is the dimension.  That is, each
# dimension of the positional encoding corresponds to a sinusoid.  The
# wavelengths form a geometric progression from $2\pi$ to $10000 \cdot
# 2\pi$.  We chose this function because we hypothesized it would
# allow the model to easily learn to attend by relative positions,
# since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a
# linear function of $PE_{pos}$.
#
# In addition, we apply dropout to the sums of the embeddings and the
# positional encodings in both the encoder and decoder stacks.  For
# the base model, we use a rate of $P_{drop}=0.1$.
#
# **【中文对照 / Chinese Translation】**
# 由于我们的模型既不包含循环结构也不包含卷积结构，为了让模型能够利用序列的顺序信息，我们必须引入关于序列中 Token 相对或绝对位置的信息。为此，我们在编码器和解码器堆栈底部的输入词嵌入中相加了“位置编码”（Positional Encodings）。位置编码具有与词嵌入相同的维度 $d_{	ext{model}}$，以便两者可以直接相加。位置编码有许多选择，包括可学习的和固定公式计算的。
#
# 在这项工作中，我们使用了不同频率的正弦和余弦函数：
# $$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{	ext{model}}})$$
# $$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{	ext{model}}})$$
# 其中 $pos$ 为位置，$i$ 为维度。也就是说，位置编码的每个维度都对应一个正弦波。波长形成从 $2\pi$ 到 $10000 \cdot 2\pi$ 的等比数列。我们选择这个函数是因为我们假设它能让模型轻松学会根据相对位置来进行注意力计算，因为对于任何固定的偏移量 $k$，$PE_{pos+k}$ 都可以表示为 $PE_{pos}$ 的线性函数。
#
# 此外，我们在编码器和解码器堆栈中，对词嵌入与位置编码相加后的结果应用了 Dropout 随机失活。对于基础模型，我们设定的失活率为 $P_{drop}=0.1$。
#

# %% [markdown]
#
# > Below the positional encoding will add in a sine wave based on
# > position. The frequency and offset of the wave is different for
# > each dimension.
#
#

# %%
def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


show_example(example_positional)




# %% [markdown]
#
# We also experimented with using learned positional embeddings
# [(cite)](https://arxiv.org/pdf/1705.03122.pdf) instead, and found
# that the two versions produced nearly identical results.  We chose
# the sinusoidal version because it may allow the model to extrapolate
# to sequence lengths longer than the ones encountered during
# training.
#
#

# %% [markdown]
# ## Full Model (完整 Transformer 模型构建)
#
# > Here we define a function from hyperparameters to a full model.
# >
# > **【中文对照 / Chinese Translation】**
# > 这里我们定义一个根据超参数构建完整 Transformer 模型的辅助函数。
#

# %% [markdown]
# ## Inference: (模型推理与前向预测)
#
# > Here we make a forward step to generate a prediction of the
# model. We try to use our transformer to memorize the input. As you
# will see the output is randomly generated due to the fact that the
# model is not trained yet. In the next tutorial we will build the
# training function and try to train our model to memorize the numbers
# from 1 to 10.
#
# **【中文对照 / Chinese Translation】**
# 这里我们执行一步前向传播以生成模型的预测结果。我们尝试让 Transformer 记住输入。正如你将看到的，由于模型尚未训练，输出是随机生成的。在接下来的教程中，我们将构建训练函数，并尝试训练模型记住 1 到 10 的数字序列。
#

# %% [markdown]
# # Part 2: Model Training (第二部分：模型训练)
#

# %% [markdown]
# # Training (训练过程与方案)
#

# %% [markdown]
#
# > We stop for a quick interlude to introduce some of the tools
# > needed to train a standard encoder decoder model. First we define a
# > batch object that holds the src and target sentences for training,
# > as well as constructing the masks.
#
#

# %% [markdown]
# ## Batches and Masking (批数据处理与 Mask 掩码机制)
#

# %% [markdown]
# > Next we create a generic training and scoring function to keep
# > track of loss. We pass in a generic loss compute function that
# > also handles parameter updates.
# >
# > **【中文对照 / Chinese Translation】**
# > 接下来，我们创建一个通用的训练与评分函数来跟踪 Loss 损失。我们传入一个通用的损失计算函数，该函数同时也负责处理参数的梯度更新。
#

# %% [markdown]
# ## Training Loop (训练循环机制)
#

# %% [markdown]
# ## Training Data and Batching (训练数据构建与动态 Batch 分割)
#
# We trained on the standard WMT 2014 English-German dataset
# consisting of about 4.5 million sentence pairs.  Sentences were
# encoded using byte-pair encoding, which has a shared source-target
# vocabulary of about 37000 tokens. For English-French, we used the
# significantly larger WMT 2014 English-French dataset consisting of
# 36M sentences and split tokens into a 32000 word-piece vocabulary.
#
#
# Sentence pairs were batched together by approximate sequence length.
# Each training batch contained a set of sentence pairs containing
# approximately 25000 source tokens and 25000 target tokens.
#
# **【中文对照 / Chinese Translation】**
# 我们在包含约 450 万句对的标准 WMT 2014 德英数据集上进行了训练。句子使用字节对编码（BPE）进行编码，源语言和目标语言共享一个约 37000 个 Token 的词表。对于英法翻译，我们使用了规模大得多的 WMT 2014 英法数据集，包含 3600 万个句子，并将其拆分为 32000 个词碎片（Word-piece）词表。
#
# 句对按近似序列长度组装成 Batch。每个训练 Batch 包含一组句对，总计约包含 25000 个源语言 Token 和 25000 个目标语言 Token。
#

# %% [markdown]
# ## Hardware and Schedule (硬件资源与学习率调度算法)
#
# We trained our models on one machine with 8 NVIDIA P100 GPUs.  For
# our base models using the hyperparameters described throughout the
# paper, each training step took about 0.4 seconds.  We trained the
# base models for a total of 100,000 steps or 12 hours. For our big
# models, step time was 1.0 seconds.  The big models were trained for
# 300,000 steps (3.5 days).
#
# **【中文对照 / Chinese Translation】**
# 我们在一台配备 8 张 NVIDIA P100 GPU 的机器上训练模型。对于采用论文中所述超参数的基础模型，每个训练步（Step）耗时约 0.4 秒。我们对基础模型训练了总计 100,000 步（约 12 小时）。对于大模型（Big models），单步耗时为 1.0 秒，训练了 300,000 步（约 3.5 天）。
#

# %% [markdown]
# ## Optimizer (优化器与学习率 Warmup)
#
# We used the Adam optimizer [(cite)](https://arxiv.org/abs/1412.6980)
# with $eta_1=0.9$, $eta_2=0.98$ and $\epsilon=10^{-9}$.  We
# varied the learning rate over the course of training, according to
# the formula:
#
# $$
# lrate = d_{	ext{model}}^{-0.5} \cdot
#   \min({step\_num}^{-0.5},
#     {step\_num} \cdot {warmup\_steps}^{-1.5})
# $$
#
# This corresponds to increasing the learning rate linearly for the
# first $warmup\_steps$ training steps, and decreasing it thereafter
# proportionally to the inverse square root of the step number.  We
# used $warmup\_steps=4000$.
#
# **【中文对照 / Chinese Translation】**
# 我们使用了 Adam 优化器，超参数设置为 $eta_1=0.9$、$eta_2=0.98$ 以及 $\epsilon=10^{-9}$。在训练过程中，我们根据以下公式动态调整学习率：
# $$
# lrate = d_{	ext{model}}^{-0.5} \cdot
#   \min({step\_num}^{-0.5},
#     {step\_num} \cdot {warmup\_steps}^{-1.5})
# $$
# 这对应于在前 $warmup\_steps$ 个训练步中线性增加学习率，此后按步数的平方根倒数比例降低学习率。我们设置 $warmup\_steps=4000$。
#

# %% [markdown]
# > Note: This part is very important. Need to train with this setup
# > of the model.
# >
# > **【中文对照 / Chinese Translation】**
# > 注意：这部分非常重要，模型训练时必须沿用该优化方案与 Warmup 策略。
#

# %% [markdown]
# > Example of the curves of this model for different model sizes and
# > for optimization hyperparameters.
# >
# > **【中文对照 / Chinese Translation】**
# > 下图展示了针对不同模型规模和优化超参数时的学习率变化曲线示例。
#

# %% [markdown]
# ## Regularization (正则化与标签平滑)
#
# ### Label Smoothing (标签平滑)
#
# During training, we employed label smoothing of value
# $\epsilon_{ls}=0.1$ [(cite)](https://arxiv.org/abs/1512.00567).
# This hurts perplexity, as the model learns to be more unsure, but
# improves accuracy and BLEU score.
#
# **【中文对照 / Chinese Translation】**
# 在训练期间，我们采用了平滑值为 $\epsilon_{ls}=0.1$ 的标签平滑（Label Smoothing）。由于模型学会了不那么“盲目自信”，这会轻微损害困惑度（Perplexity），但显著提升了准确率和 BLEU 得分。
#

# %% [markdown]
# > We implement label smoothing using the KL div loss. Instead of
# > using a one-hot target distribution, we create a distribution that
# > has `confidence` of the correct word and the rest of the
# > `smoothing` mass distributed throughout the vocabulary.
# >
# > **【中文对照 / Chinese Translation】**
# > 我们使用 KL 散度损失来实现标签平滑。与传统的 One-Hot 独热目标分布不同，我们创建了一个具有正确词概率 `confidence`、其余概率质量 `smoothing` 均匀平摊至词表中其他词的平滑分布。
#

# %% [markdown]
# > Here we can see an example of how the mass is distributed to the
# > words based on confidence.
# >
# > **【中文对照 / Chinese Translation】**
# > 下图展示了概率质量如何根据置信度分配到各个词的示例。
#

# %% [markdown]
# > Label smoothing actually starts to penalize the model if it gets
# > very confident about a given choice.
# >
# > **【中文对照 / Chinese Translation】**
# > 如果模型对某个特定预测过于自信，标签平滑实际上会惩罚模型。
#

# %% [markdown]
# # A First Example (基础示例：简单的 Copy 复制任务)
#
# > We can begin by trying out a simple copy-task. Given a random set
# > of input symbols from a small vocabulary, the goal is to generate
# > back those same symbols.
# >
# > **【中文对照 / Chinese Translation】**
# > 我们首先尝试一个简单的复制任务（Copy-Task）。给定从小词表中随机抽取的一组输入符号，目标是原样生成输出这些相同的符号。
#

# %% [markdown]
# ## Synthetic Data (合成数据集生成)
#

# %% [markdown]
# ## Loss Computation (损失计算)
#

# %% [markdown]
# ## Greedy Decoding (贪婪解码算法)
#

# %% [markdown]
# > This code predicts a translation using greedy decoding for simplicity.
# >
# > **【中文对照 / Chinese Translation】**
# > 为简单起见，此代码使用贪婪解码（Greedy Decoding）来预测生成结果。
#

# %% [markdown]
# # Part 3: A Real World Example (第三部分：真实世界机器翻译示例)
#
# > Now we consider a real-world example using the Multi30k
# > German-English Translation task. This task is much smaller than
# > the WMT task considered in the paper, but it illustrates the whole
# > system. We also show how to use multi-gpu processing to make it
# > really fast.
# >
# > **【中文对照 / Chinese Translation】**
# > 现在我们考虑一个使用 Multi30k 德英翻译任务的真实示例。这个任务的规模远小于论文中使用的 WMT 数据集，但它完整地展现了整个系统流程。我们还将演示如何利用多 GPU 并行处理来极大加快训练速度。
#

# %% [markdown]
# ## Data Loading (数据加载与预处理)
#
# > We will load the dataset using torchtext and spacy for
# > tokenization.
# >
# > **【中文对照 / Chinese Translation】**
# > 我们将使用 `torchtext` 加载数据集，并使用 `spacy` 进行分词（Tokenization）。
#

# %% [markdown]
# > Batching matters a ton for speed. We want to have very evenly
# > divided batches, with absolutely minimal padding. To do this we
# > have to hack a bit around the default torchtext batching. This
# > code patches their default batching to make sure we search over
# > enough sentences to find tight batches.
# >
# > **【中文对照 / Chinese Translation】**
# > 批处理对训练速度至关重要。我们希望能获得划分非常均匀且 Padding 填充最少的 Batch。为此，我们需要对 `torchtext` 的默认 Batch 机制进行微调。这段代码对默认批处理进行了补丁扩展，确保在足够数量的句子中搜索以找到紧凑的 Batch 组合。
#

# %% [markdown]
# ## Iterators (数据迭代器构造)
#

# %% [markdown]
# ## Training the System (训练完整系统)
#

# %% [markdown]
# > Once trained we can decode the model to produce a set of
# > translations. Here we simply translate the first sentence in the
# > validation set. This dataset is pretty small so the translations
# > with greedy search are reasonably accurate.
# >
# > **【中文对照 / Chinese Translation】**
# > 训练完成后，我们可以对模型进行解码以生成一系列翻译。这里我们简单翻译验证集中的第一个句子。由于该数据集规模较小，使用贪婪搜索（Greedy Search）的翻译结果已足够准确。
#

# %% [markdown]
# # Additional Components: BPE, Search, Averaging (拓展组件：BPE、束搜索与模型平均)
#
# > So this mostly covers the transformer model itself. There are four
# > aspects that we didn't cover explicitly. We also have all these
# > additional features implemented in
# > [OpenNMT-py](https://github.com/opennmt/opennmt-py).
# >
# > **【中文对照 / Chinese Translation】**
# > 至此，我们基本涵盖了 Transformer 模型本身的构建。不过还有四个方面我们未做详细展开，这些扩展功能在 [OpenNMT-py](https://github.com/opennmt/opennmt-py) 中均有完整实现。
#

# %% [markdown]
#
# > So this mostly covers the transformer model itself. There are four
# > aspects that we didn't cover explicitly. We also have all these
# > additional features implemented in
# > [OpenNMT-py](https://github.com/opennmt/opennmt-py).
#
#
#
#

# %% [markdown]
# > 1) BPE/ Word-piece: We can use a library to first preprocess the
# > data into subword units. See Rico Sennrich's
# > [subword-nmt](https://github.com/rsennrich/subword-nmt)
# > implementation. These models will transform the training data to
# > look like this:
# >
# > **【中文对照 / Chinese Translation】**
# > 1) BPE / Word-piece（子词分词）: 我们可以使用第三方库将数据预处理为子词单元（Subword Units）。参见 Rico Sennrich 的 [subword-nmt](https://github.com/rsennrich/subword-nmt) 实现。这些模型会将训练数据转换为如下形式：
#

# %% [markdown]
# ▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP
# ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden .
#
#

# %% [markdown]
# > 2) Shared Embeddings: When using BPE with shared vocabulary we can
# > share the same weight vectors between the source / target /
# > generator. See the [(cite)](https://arxiv.org/abs/1608.05859) for
# > details. To add this to the model simply do this:
# >
# > **【中文对照 / Chinese Translation】**
# > 2) 共享 Embedding（Shared Embeddings）: 当使用共享词表的 BPE 时，我们可以在源语言、目标语言和 Generator 生成器之间共享相同的权重向量。详见论文 [(cite)](https://arxiv.org/abs/1608.05859)。在模型中添加此特性只需进行相应的权重共享赋值。
#

# %% [markdown]
# > 3) Beam Search: This is a bit too complicated to cover here. See the
# > [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py/)
# > for a pytorch implementation.
# >
# > **【中文对照 / Chinese Translation】**
# > 3) 束搜索（Beam Search）: 这在本文中过于复杂，无法详细展开。其 PyTorch 实现请参考 [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py/)。
#

# %% [markdown]
# > 4) Model Averaging: The paper averages the last k checkpoints to
# > create an ensembling effect. We can do this after the fact if we
# > have a bunch of models:
# >
# > **【中文对照 / Chinese Translation】**
# > 4) 模型平均（Model Averaging）: 原论文通过平均最后 $k$ 个检查点（Checkpoint）的权重来创造集成（Ensemble）效果。如果我们保存了一系列模型，可以在训练结束后进行权重平均。
#

# %% [markdown]
# # Results (实验结果与可视化)
#
# On the WMT 2014 English-to-German translation task, the big
# transformer model (Transformer (big) in Table 2) outperforms the
# best previously reported models (including ensembles) by more than
# 2.0 BLEU, establishing a new state-of-the-art BLEU score of
# 28.4. The configuration of this model is listed in the bottom line
# of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base
# model surpasses all previously published models and ensembles, at a
# fraction of the training cost of any of the competitive models.
#
# On the WMT 2014 English-to-French translation task, our big model
# achieves a BLEU score of 41.0, outperforming all of the previously
# published single models, at less than 1/4 the training cost of the
# previous state-of-the-art model. The Transformer (big) model trained
# for English-to-French used dropout rate Pdrop = 0.1, instead of 0.3.
#
# **【中文对照 / Chinese Translation】**
# 在 WMT 2014 英德翻译任务上，大号 Transformer 模型（表 2 中的 Transformer (big)）比此前报道的最优模型（包括集成模型）提升了超过 2.0 BLEU，创造了 28.4 的全新 SOTA（最先进）BLEU 得分。该模型的配置列在表 3 的最后一行。在 8 张 P100 GPU 上训练耗时 3.5 天。甚至我们的基础模型（Base model）也超越了此前发表的所有单模型和集成模型，而训练成本仅为竞品模型的极小一部分。
#
# 在 WMT 2014 英法翻译任务上，我们的大号模型达到了 41.0 的 BLEU 得分，超越了此前发表的所有单模型，而训练成本还不到此前 SOTA 模型的 1/4。用于英法翻译的 Transformer (big) 模型使用的 Dropout 率为 $P_{drop}=0.1$，而非 0.3。
#

# %% [markdown]
# ![](images/results.png)
#
#

# %% [markdown]
# > With the addtional extensions in the last section, the OpenNMT-py
# > replication gets to 26.9 on EN-DE WMT. Here I have loaded in those
# > parameters to our reimplemenation.
# >
# > **【中文对照 / Chinese Translation】**
# > 结合上一节所述的拓展配置，OpenNMT-py 在 WMT 英德任务上的复现达到了 26.9 的 BLEU 值。这里我已将这些训练好的参数加载到了我们的复现代码中。
#

# %% [markdown]
# ## Attention Visualization (注意力机制权重可视化)
#
# > Even with a greedy decoder the translation looks pretty good. We
# > can further visualize it to see what is happening at each layer of
# > the attention
# >
# > **【中文对照 / Chinese Translation】**
# > 即使只使用贪婪解码器，翻译效果也相当不错。我们可以进一步对其进行可视化，以便直观观察注意力机制在每一层中发生了什么。
#

# %% [markdown]
# ## Encoder Self Attention (编码器自注意力可视化)
#

# %% [markdown]
# ## Decoder Self Attention (解码器自注意力可视化)
#

# %% [markdown]
# ## Decoder Src Attention (解码器-源文本交叉注意力可视化)
#

# %% [markdown]
# # Conclusion (结语与总结)
#
#  Hopefully this code is useful for future research. Please reach
#  out if you have any issues.
#
#  Cheers,
#  Sasha Rush, Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak,
#  Stella Biderman
#
# **【中文对照 / Chinese Translation】**
# 希望这份代码对大家未来的研究有所帮助！如果您遇到任何问题，欢迎随时与我们联系。
#
# 祝好，
# Sasha Rush, Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, Stella Biderman
#
