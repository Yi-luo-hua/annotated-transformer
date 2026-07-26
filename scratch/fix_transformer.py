import os

content = '''# -*- coding: utf-8 -*-
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

# %% [markdown]
# # Prelims (准备工作与依赖库)

# %% [markdown]
# > My comments are blockquoted. The main text is all from the paper itself.
# >
# > **【中文对照 / Chinese Translation】**
# > 引用块（>）中的内容是作者的注解说明，其余主干文本均直接来自原论文 *Attention Is All You Need*。

# %%
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
import warnings

warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import altair as alt
except ImportError:
    pd = None
    alt = None

try:
    import spacy
    import torchtext
    from torchtext.data.functional import to_map_style_dataset
    from torch.utils.data import DataLoader
    from torchtext.vocab import build_vocab_from_iterator
except ImportError:
    spacy = None
    torchtext = None

RUN_EXAMPLES = True


def is_interactive_notebook():
    return hasattr(__builtins__, "__IPYTHON__")


def show_example(fn, args=[]):
    if RUN_EXAMPLES and is_interactive_notebook():
        return fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

# %% [markdown]
# # Background (研究背景与动机)

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

# %% [markdown]
# # Part 1: Model Architecture (第一部分：模型架构与原理)

# %% [markdown]
# # Model Architecture (模型整体架构)

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

# %%
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    
    【洛熙人工解析】
    标准的编码器-解码器架构，是 Transformer 以及许多序列到序列（Seq2Seq）模型的基础。
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences. / 接收并处理带掩码的源序列与目标序列。"
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # 编码阶段：先经过源语言 Embed 嵌入层，再传入 Encoder 堆叠块
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # 解码阶段：接收编码器的输出 memory，结合目标语言 Embed 与掩码传入 Decoder
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# %%
class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    【洛熙人工解析】生成器：将解码器输出的向量维度通过 Linear 线性层映射回词表大小 (vocab size)，并用 log_softmax 输出预测概率。
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

# %% [markdown]
# The Transformer follows this overall architecture using stacked
# self-attention and point-wise, fully connected layers for both the
# encoder and decoder, shown in the left and right halves of Figure 1,
# respectively.
#
# **【中文对照 / Chinese Translation】**
# Transformer 整体遵循这一架构，其编码器和解码器均采用了堆叠的自注意力层（Stacked Self-Attention）和逐位置的全连接层（Point-wise Fully Connected Layers），分别展示在论文图 1 的左半部分和右半部分。

# %% [markdown] id="oredWloYTsqC"
# ![](images/ModalNet-21.png)

# %% [markdown]
# ## Encoder and Decoder Stacks (编码器与解码器堆栈)
#
# ### Encoder (编码器)
#
# The encoder is composed of a stack of $N=6$ identical layers.
#
# **【中文对照 / Chinese Translation】**
# 编码器由 $N=6$ 个完全相同的层堆叠而成。

# %%
def clones(module, N):
    "Produce N identical layers. / 深拷贝产生 N 个完全相同的模块层，避免权重共享。"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# %%
class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    【洛熙人工解析】编码器核心：由 N 个 EncoderLayer 层级联而成，最后追加一层 LayerNorm。
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn. / 将输入 x 与掩码依次传入每一层 EncoderLayer，最后归一化输出。"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# %% [markdown]
# We employ a residual connection
# [(cite)](https://arxiv.org/abs/1512.03385) around each of the two
# sub-layers, followed by layer normalization
# [(cite)](https://arxiv.org/abs/1607.06450).
#
# **【中文对照 / Chinese Translation】**
# 我们在两个子层中的每一个周围都采用了残差连接（Residual Connection），随后跟上层归一化（Layer Normalization）。

# %%
class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See citation for details).
    【洛熙人工解析】层归一化 (Layer Normalization)：对特征维度求均值 mean 与标准差 std 进行标准化，并使用可学习参数 a_2 (gamma) 和 b_2 (beta) 进行缩放与平移。
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

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
# $d_{\text{model}}=512$.
#
# **【中文对照 / Chinese Translation】**
# 也就是说，每个子层的输出为 $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$，其中 $\mathrm{Sublayer}(x)$ 是子层自身实现的函数。在将子层的输出与其输入相加并进行归一化之前，我们对子层输出应用 Dropout 随机失活。
#
# 为了便于实现残差连接，模型中所有的子层以及词嵌入层的输出维度统一设定为 $d_{\text{model}}=512$。

# %%
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    
    【洛熙人工解析】
    残差连接与层归一化模块。
    注意：为了代码简洁性与训练稳定性，这里采用了 Pre-LN（先归一化再过子层），而非 Post-LN。
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size. / 对输入维度相同的任何子层应用 Pre-LN 残差连接。"
        return x + self.dropout(sublayer(self.norm(x)))

# %% [markdown]
# Each layer has two sub-layers. The first is a multi-head
# self-attention mechanism, and the second is a simple, position-wise
# fully connected feed-forward network.
#
# **【中文对照 / Chinese Translation】**
# 编码器的每一层包含两个子层。第一个是多头自注意力机制（Multi-Head Self-Attention），第二个是简单的逐位置全连接前馈网络（Position-wise Fully Connected Feed-Forward Network）。

# %%
class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    【洛熙人工解析】单个编码器层：包含两个核心子层——自注意力机制 (Self-Attention) 和逐位置前馈网络 (Feed-Forward)。
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections. / 对应论文图1（左侧）：先过自注意力子层，再过前馈神经网络子层。"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

# %% [markdown]
# ### Decoder (解码器)
#
# The decoder is also composed of a stack of $N=6$ identical layers.
#
# **【中文对照 / Chinese Translation】**
# 解码器同样由 $N=6$ 个完全相同的层堆叠而成。

# %%
class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    【洛熙人工解析】解码器：由 N 个带有 Mask 掩码机制的 DecoderLayer 堆叠而成。
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# %% [markdown]
# In addition to the two sub-layers in each encoder layer, the decoder
# inserts a third sub-layer, which performs multi-head attention over
# the output of the encoder stack.  Similar to the encoder, we employ
# residual connections around each of the sub-layers, followed by
# layer normalization.
#
# **【中文对照 / Chinese Translation】**
# 除了每个编码器层中的两个子层之外，解码器还插入了第三个子层，该子层对编码器堆栈的输出执行多头注意力计算（即交叉注意力 Cross-Attention）。与编码器类似，我们在每个子层周围都采用了残差连接，随后进行层归一化。

# %%
class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    【洛熙人工解析】单个解码器层：包含三个子层：
    1. 掩码自注意力 (Masked Self-Attention)
    2. 交叉注意力 (Encoder-Decoder Cross-Attention)
    3. 逐位置前馈网络 (Feed-Forward)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections. / 对应论文图1（右侧）：依次执行掩码自注意力 -> 交叉注意力 -> 前馈网络。"
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

# %% [markdown]
# We also modify the self-attention sub-layer in the decoder stack to
# prevent positions from attending to subsequent positions.  This
# masking, combined with fact that the output embeddings are offset by
# one position, ensures that the predictions for position $i$ can
# depend only on the known outputs at positions less than $i$.
#
# **【中文对照 / Chinese Translation】**
# 我们还修改了解码器堆栈中的自注意力子层，以防止当前位置关注到后续位置（即 Mask 掩码机制）。这种掩码结合输出嵌入向右偏移一个位置的操作，确保对位置 $i$ 的预测只能依赖于小于 $i$ 的已知前文输出。

# %%
def subsequent_mask(size):
    """
    Mask out subsequent positions.
    【洛熙人工解析】生成下三角矩阵掩码：掩盖未来未生成的词，阻断从左向右的信息泄露，保持自回归特性。
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

# %% [markdown]
# > Below the attention mask shows the position each tgt word (row) is
# > allowed to look at (column). Words are blocked for attending to
# > future words during training.
# >
# > **【中文对照 / Chinese Translation】**
# > 下图中的注意力掩码展示了目标序列中的每个词（行）允许关注的位置（列）。在训练过程中，掩码屏蔽掉了当前词对未来词的注意力。

# %%
def example_mask():
    if pd is None or alt is None:
        return None
    LS_data = pd.DataFrame(
        {
            "Subsequent Mask": subsequent_mask(20)[0].numpy().flatten(),
            "Window": list(range(20)) * 20,
            "Masking": [i // 20 for i in range(400)],
        }
    )
    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


show_example(example_mask)

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

# %% [markdown]
# In practice, we compute the attention function on a set of queries
# simultaneously, packed together into a matrix $Q$.  The keys and
# values are also packed together into matrices $K$ and $V$.  We
# compute the matrix of outputs as:
#
# $$
#    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
# $$
#
# **【中文对照 / Chinese Translation】**
# 在实践中，我们同时对一组 Query 进行注意力计算，并将其打包拼接成矩阵 $Q$。Keys 和 Values 也分别打包成矩阵 $K$ 和 $V$。输出矩阵的计算公式为：
# $$
#    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
# $$

# %%
def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    【洛熙人工解析】计算“缩放点积注意力”：Formula: Softmax(Q * K^T / sqrt(d_k)) * V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# %% [markdown]
# The two most commonly used attention functions are additive
# attention [(cite)](https://arxiv.org/abs/1409.0473), and dot-product
# (multiplicative) attention.  Dot-product attention is identical to
# our algorithm, except for the scaling factor of
# $\frac{1}{\sqrt{d_k}}$. Additive attention computes the
# compatibility function using a feed-forward network with a single
# hidden layer.  While the two are similar in theoretical complexity,
# dot-product attention is much faster and more space-efficient in
# practice, since it can be implemented using highly optimized matrix
# multiplication code.
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
# $\frac{1}{\sqrt{d_k}}$.
#
# **【中文对照 / Chinese Translation】**
# 两种最常用的注意力函数是加性注意力（Additive Attention）和点积/乘性注意力（Dot-Product Attention）。点积注意力除了缩放因子 $\frac{1}{\sqrt{d_k}}$ 之外与我们的算法完全相同。加性注意力使用带有单个隐层的前馈网络来计算匹配函数。虽然两者在理论复杂度上相似，但在实践中，点积注意力要快得多且更节省空间，因为它可以利用高度优化的矩阵乘法运算来实现。
#
# 虽然对于较小的 $d_k$ 值，这两种机制的表现相似，但在没有缩放因子的情况下，随着 $d_k$ 增大，加性注意力超越了点积注意力。我们怀疑，对于较大的 $d_k$ 值，点积的数值量级会急剧增大，从而将 Softmax 函数推向具有极小梯度的饱和区域（为了解释为什么点积会变大：假设 $q$ 和 $k$ 的各分量是均值为 0、方差为 1 的独立随机变量，则它们的点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 0，方差为 $d_k$）。为了抵消这种副作用，我们将点积乘以缩放因子 $\frac{1}{\sqrt{d_k}}$。

# %% [markdown]
# ![](images/ModalNet-20.png)

# %% [markdown]
# Multi-head attention allows the model to jointly attend to
# information from different representation subspaces at different
# positions. With a single attention head, averaging inhibits this.
#
# $$
# \mathrm{MultiHead}(Q, K, V) =
#     \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \\
#     \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
# $$
#
# Where the projections are parameter matrices $W^Q_i \in
# \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in
# \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in
# \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in
# \mathbb{R}^{hd_v \times d_{\text{model}}}$.
#
# In this work we employ $h=8$ parallel attention layers, or
# heads. For each of these we use $d_k=d_v=d_{\text{model}}/h=64$. Due
# to the reduced dimension of each head, the total computational cost
# is similar to that of single-head attention with full
# dimensionality.
#
# **【中文对照 / Chinese Translation】**
# 多头注意力（Multi-Head Attention）允许模型联合关注来自不同位置的不同表示子空间（Subspaces）的信息。而如果只有一个注意力头，对所有位置进行简单平均会抑制这种多角度信息的捕获。
# $$
# \mathrm{MultiHead}(Q, K, V) =
#     \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \\
#     \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
# $$
# 其中线性投影是参数矩阵 $W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ 以及 $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$。
#
# 在这项工作中，我们采用 $h=8$ 个平行的注意力层（即8个注意力头）。对于每个头，我们设定 $d_k = d_v = d_{\text{model}}/h = 64$。由于减少了每个头的维度，总计算成本与具有全维度单头注意力的计算成本非常接近。

# %%
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention (多头注意力机制)
    【洛熙人工解析】将 Query、Key、Value 投影到 h 个不同的子空间中分别计算注意力，最后拼接输出。
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

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
# 2) The encoder contains self-attention layers.  In a self-attention
# layer all of the keys, values and queries come from the same place,
# in this case, the output of the previous layer in the encoder.  Each
# position in the encoder can attend to all positions in the previous
# layer of the encoder.
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
# 1）在“编码器-解码器交叉注意力”（Encoder-Decoder Attention）层中：Query 来自前一个解码器层，而 Key 和 Value 来自编码器的输出 memory。这使得解码器中的每个位置都能关注到输入源序列的所有位置。
# 2）编码器包含“自注意力”（Self-Attention）层：在自注意力层中，所有的 Key、Value 和 Query 都来自同一个地方（即编码器中前一层的输出）。编码器中的每个位置都可以关注到编码器前一层的所有位置。
# 3）类似地，解码器中的“自注意力层”允许解码器中的每个位置关注到解码器中直到并包括该位置在内的所有位置。我们需要防止解码器中的向左信息流动，以保持自回归特性。我们在缩放点积注意力内部通过掩码机制（将 Softmax 输入中对应于非法连接的所有值设置为 $-\infty$）来实现这一点。

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
# $d_{\text{model}}=512$, and the inner-layer has dimensionality
# $d_{ff}=2048$.
#
# **【中文对照 / Chinese Translation】**
# 除了注意力子层之外，编码器和解码器中的每一层都包含一个全连接的前馈网络，该网络独立且相同地作用于每个位置。它由两个线性变换组成，中间带有 ReLU 激活函数：
# $$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$
# 虽然线性变换在不同位置之间是相同的，但它们在层与层之间使用不同的参数。描述它的另一种方式是将其视为两个核大小为 1 的卷积。输入和输出的维度为 $d_{\text{model}}=512$，中间隐层的维度为 $d_{ff}=2048$。

# %%
class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    【洛熙人工解析】实现 FFN 公式：FFN(x) = max(0, xW_1 + b_1)W_2 + b_2。
    升维到 d_ff (2048) 再降维回 d_model (512)。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

# %% [markdown]
# ## Embeddings and Softmax (词嵌入层与 Softmax 输出)
#
# Similarly to other sequence transduction models, we use learned
# embeddings to convert the input tokens and output tokens to vectors
# of dimension $d_{\text{model}}$.  We also use the usual learned
# linear transformation and softmax function to convert the decoder
# output to predicted next-token probabilities.  In our model, we
# share the same weight matrix between the two embedding layers and
# the pre-softmax linear transformation, similar to
# [(cite)](https://arxiv.org/abs/1608.05859). In the embedding layers,
# we multiply those weights by $\sqrt{d_{\text{model}}}$.
#
# **【中文对照 / Chinese Translation】**
# 与其他序列转换模型类似，我们使用可学习的词嵌入（Learned Embeddings）将输入 Token 和输出 Token 转换为维度为 $d_{\text{model}}$ 的向量。我们还使用常见的可学习线性变换与 Softmax 函数，将解码器输出转换为预测下一个 Token 的概率。在我们的模型中，源语言嵌入层、目标语言嵌入层以及 Softmax 前的线性变换这三者共享相同的权重矩阵。在嵌入层中，我们将这些权重乘以 $\sqrt{d_{\text{model}}}$ 进行缩放。

# %%
class Embeddings(nn.Module):
    """
    【洛熙人工解析】词嵌入层：将输入的 Token ID 映射为 d_model 维度的向量，并乘以 sqrt(d_model) 进行缩放，使嵌入向量的方差与位置编码对齐。
    """

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# %% [markdown]
# ## Positional Encoding (位置编码)
#
# Since our model contains no recurrence and no convolution, in order
# for the model to make use of the order of the sequence, we must
# inject some information about the relative or absolute position of
# the tokens in the sequence.  To this end, we add "positional
# encodings" to the input embeddings at the bottoms of the encoder and
# decoder stacks.  The positional encodings have the same dimension
# $d_{\text{model}}$ as the embeddings, so that the two can be summed.
# There are many choices of positional encodings, learned and fixed
# [(cite)](https://arxiv.org/pdf/1705.03122.pdf).
#
# In this work, we use sine and cosine functions of different frequencies:
#
# $$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
#
# $$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$
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
# 由于我们的模型既不包含循环结构也不包含卷积结构，为了让模型能够利用序列的顺序信息，我们必须引入关于序列中 Token 相对或绝对位置的信息。为此，我们在编码器和解码器堆栈底部的输入词嵌入中相加了“位置编码”（Positional Encodings）。位置编码具有与词嵌入相同的维度 $d_{\text{model}}$，以便两者可以直接相加。位置编码有许多选择，包括可学习的和固定公式计算的。
#
# 在这项工作中，我们使用了不同频率的正弦和余弦函数：
# $$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
# $$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$
# 其中 $pos$ 为位置，$i$ 为维度。也就是说，位置编码的每个维度都对应一个正弦波。波长形成从 $2\pi$ 到 $10000 \cdot 2\pi$ 的等比数列。我们选择这个函数是因为我们假设它能让模型轻松学会根据相对位置来进行注意力计算，因为对于任何固定的偏移量 $k$，$PE_{pos+k}$ 都可以表示为 $PE_{pos}$ 的线性函数。
#
# 此外，我们在编码器和解码器堆栈中，对词嵌入与位置编码相加后的结果应用了 Dropout 随机失活。对于基础模型，我们设定的失活率为 $P_{drop}=0.1$。

# %%
class PositionalEncoding(nn.Module):
    """
    Positional Encoding (位置编码)
    【洛熙人工解析】使用正弦和余弦函数交替编码序列中每个 token 的位置信息。
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

# %% [markdown]
# > Below the positional encoding will add in a sine wave based on
# > position. The frequency and offset of the wave is different for
# > each dimension.
# >
# > **【中文对照 / Chinese Translation】**
# > 下图展示了基于位置添加的正弦波位置编码。每个特征维度对应的正弦波频率和偏移量都是独特的。

# %%
def example_positional():
    if pd is None or alt is None:
        return None
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
# We also experimented with using learned positional embeddings
# [(cite)](https://arxiv.org/pdf/1705.03122.pdf) instead, and found
# that the two versions produced nearly identical results.  We chose
# the sinusoidal version because it may allow the model to extrapolate
# to sequence lengths longer than the ones encountered during
# training.
#
# **【中文对照 / Chinese Translation】**
# 我们还尝试了使用可学习的位置嵌入（Learned Positional Embeddings）来替代固定公式，发现两种版本的实验效果几乎完全相同。我们最终选择正弦公式版本，是因为它可能允许模型外推到比训练期间遇到的序列长度更长的序列。

# %% [markdown]
# ## Full Model (完整 Transformer 模型组装)
#
# > Here we define a function from hyperparameters to a full model.
# >
# > **【中文对照 / Chinese Translation】**
# > 这里我们定义一个根据超参数构建完整 Transformer 模型的辅助函数。

# %%
def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters. / 辅助函数：根据超参数构建完整的 Transformer 模型。"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# %% [markdown]
# ## Inference: (模型推理与前向预测)
#
# > Here we make a forward step to predict a translation using greedy
# > decoding.
# >
# > **【中文对照 / Chinese Translation】**
# > 这里我们使用贪婪解码（Greedy Decoding）进行前向推断预测。

# %%
def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src.data)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Model Output:", ys)


def run_tests():
    for fn in [inference_test]:
        fn()


show_example(run_tests)

# %% [markdown]
# # Part 2: Model Training (第二部分：模型训练流程)

# %% [markdown]
# ## Batches and Masking (批数据处理与 Mask 掩码机制)

# %%
class Batch:
    """
    Object for holding a batch of data with mask during training.
    【洛熙人工解析】Batch 批处理类：用于在训练过程中持有并构造带 Mask 掩码的输入序列与目标序列。
    """

    def __init__(self, src, tgt=None, pad=2):  # 2 表示 padding 的 token id
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words. / 构造掩码：同时隐藏 padding 填充符与未来位置。"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

# %% [markdown]
# ## Training Loop (训练循环机制)

# %%
class TrainState:
    """
    Track number of steps, examples, and tokens processed.
    【洛熙人工解析】训练状态跟踪：记录当前 Step 步数、处理样本数与 Token 总数。
    """

    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """
    Train a single epoch
    【洛熙人工解析】运行单个 Epoch：包含前向传播、损失计算、反向传播与梯度累积机制。
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train" or mode == "train_cum":
            loss_node.backward()
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 0:
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (
                    i,
                    n_accum,
                    loss / batch.ntokens,
                    tokens / elapsed,
                    scheduler.get_last_lr()[0],
                )
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state

# %% [markdown]
# ## Training Data and Batching (训练数据构建与动态 Batch 分割)

# %% [markdown]
# We trained on the standard WMT 2014 English-German dataset consisting
# of about 4.5 million sentence pairs.  Sentences were encoded using
# byte-pair encoding, which has a shared source-target vocabulary of
# about 37000 tokens. For English-French, we used the much larger WMT
# 2014 English-French dataset consisting of 36M sentences and split
# tokens into a 32000 word-piece vocabulary.
#
# Sentence pairs were batched together by approximate sequence
# length. Each training batch contained a set of sentence pairs
# containing approximately 25000 source tokens and 25000 target
# tokens.
#
# **【中文对照 / Chinese Translation】**
# 我们在由约 450 万对句子组成的标准 WMT 2014 英德数据集上进行了训练。句子使用字节对编码（BPE）进行编码，源语言与目标语言共享一个约 37,000 个 Token 的词表。对于英法翻译，我们使用了规模大得多的 WMT 2014 英法数据集（包含 3,600 万个句子），并将 Token 切分为 32,000 个词碎片（Word-piece）词表。
#
# 句对按近似序列长度组合成批（Batch）。每个训练批次包含的句对大约涵盖 25,000 个源语言 Token 和 25,000 个目标语言 Token。

# %%
def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours. / 重新封装 Batch 结构以适应自定义数据迭代。"
    src, tgt = batch.src, batch.tgt
    return Batch(src, tgt, pad_idx)

# %% [markdown]
# ## Hardware and Schedule (硬件资源与学习率调度算法)
#
# We trained our models on one machine with 8 NVIDIA P100 GPUs.  For
# the base models using the hyperparameters described throughout the
# paper, each training step took about 0.4 seconds.  We trained the
# base models for a total of 100,000 steps or 12 hours.  For our big
# models, step time was 1.0 seconds.  The big models were trained for
# 300,000 steps (3.5 days).
#
# **【中文对照 / Chinese Translation】**
# 我们在一台配备 8 张 NVIDIA P100 GPU 的机器上训练模型。对于使用论文中所述超参数的基础模型（Base model），每个训练步耗时约 0.4 秒。我们对基础模型总共训练了 100,000 步（约 12 小时）。对于大号模型（Big model），每步耗时 1.0 秒。大号模型共训练了 300,000 步（约 3.5 天）。

# %%
def rate(step, model_size, factor, warmup):
    """
    we have to supply a model_size, factor, and warmup.
    【洛熙人工解析】根据公式计算动态学习率：lrate = factor * (model_size^(-0.5) * min(step^(-0.5), step * warmup^(-1.5)))
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def example_learning_schedule():
    if pd is None or alt is None:
        return None
    opts = [
        [512, 1, 4000],  # BASE MODEL
        [512, 1, 8000],
        [256, 1, 4000],
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    for opts_val in opts:
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(
                step, opts_val[0], opts_val[1], opts_val[2]
            ),
        )
        lrs = []
        for step in range(20000):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(lrs)

    opts_labels = [f"{v[0]}:{v[1]}:{v[2]}" for v in opts]
    results = pd.DataFrame(
        {
            "Step": list(range(20000)) * len(opts),
            "Learning Rate": [lr for lrs in learning_rates for lr in lrs],
            "Options": [
                label for label in opts_labels for _ in range(20000)
            ],
        }
    )

    return (
        alt.Chart(results)
        .mark_line()
        .properties(width=500)
        .encode(x="Step", y="Learning Rate", color="Options:N")
        .interactive()
    )


show_example(example_learning_schedule)

# %% [markdown]
# ## Optimizer (优化器与学习率 Warmup)
#
# We used the Adam optimizer with $\beta_1=0.9$, $\beta_2=0.98$ and
# $\epsilon=10^{-9}$. We varied the learning rate over the course of
# training, according to the formula:
#
# $$
# lrate = d_{\text{model}}^{-0.5} \cdot
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
# 我们使用了 Adam 优化器，超参数设置为 $\beta_1=0.9$、$\beta_2=0.98$ 以及 $\epsilon=10^{-9}$。在训练过程中，我们根据以下公式动态调整学习率：
# $$
# lrate = d_{\text{model}}^{-0.5} \cdot
#   \min({step\_num}^{-0.5},
#     {step\_num} \cdot {warmup\_steps}^{-1.5})
# $$
# 这对应于在前 $warmup\_steps$ 个训练步中线性增加学习率，此后按步数的平方根倒数比例降低学习率。我们设置 $warmup\_steps=4000$。

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

# %%
class LabelSmoothing(nn.Module):
    "Implement label smoothing. / 实现标签平滑 KL 散度损失。"

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def example_label_smoothing():
    if pd is None or alt is None:
        return None
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.DataFrame(
        {
            "Target probability": crit.true_dist.flatten(),
            "Vocabulary Window": list(range(5)) * 5,
            "Target Cell": [i // 5 for i in range(25)],
        }
    )
    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=200, width=200)
        .encode(
            alt.X("Vocabulary Window:O"),
            alt.Y("Target Cell:O"),
            alt.Color(
                "Target probability:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .interactive()
    )


show_example(example_label_smoothing)

# %% [markdown]
# # A First Example (基础示例：简单的 Copy 复制任务)
#
# > We can begin by trying out a simple copy-task. Given a random set
# > of input symbols from a small vocabulary, the goal is to generate
# > back those same symbols.
# >
# > **【中文对照 / Chinese Translation】**
# > 我们首先尝试一个简单的复制任务（Copy-Task）。给定从小词表中随机抽取的一组输入符号，目标是原样生成输出这些相同的符号。

# %% [markdown]
# ## Synthetic Data (合成数据集生成)

# %%
def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task. / 为复制任务生成随机的输入与目标数据。"
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False)
        tgt = data.requires_grad_(False)
        yield Batch(src, tgt, 0)

# %% [markdown]
# ## Loss Computation (损失计算)

# %%
class SimpleLossCompute:
    "A simple loss compute and train function. / 简单损失计算与训练更新包装器。"

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss

# %% [markdown]
# ## Greedy Decoding (贪婪解码算法)

# %%
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, 30, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, 30, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )

    model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

# %% [markdown]
# # Part 3: A Real World Example (第三部分：真实世界机器翻译示例)
#
# > Now we consider a real-world example using the Multi30k
# > German-English Translation task.
# >
# > **【中文对照 / Chinese Translation】**
# > 现在我们考虑一个使用 Multi30k 德英翻译任务的真实示例。

# %% [markdown]
# ## Data Loading (数据加载与预处理)

# %%
def load_tokenizers():
    if spacy is None:
        return None, None
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

# %% [markdown]
# ## Iterators (数据迭代器构造)

# %%
def build_vocabulary(spacy_de, spacy_en):
    if torchtext is None:
        return None, None
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary...")
    train, val, test = to_map_style_dataset(
        torchtext.datasets.Multi30k(split=("train", "val", "test"))
    )
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train, tokenize_de, index=0),
        min_freq=2,
        specials=["<blank>", "<unk>", "<s>", "</s>"],
    )

    print("Building English Vocabulary...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train, tokenize_en, index=1),
        min_freq=2,
        specials=["<blank>", "<unk>", "<s>", "</s>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt

# %% [markdown]
# ## Training the System (训练完整系统)

# %% [markdown]
# # Additional Components: BPE, Search, Averaging (拓展组件：BPE、束搜索与模型平均)
#
# > 1) BPE / Word-piece (子词分词)
# > 2) Shared Embeddings (共享嵌入权重)
# > 3) Beam Search (束搜索)
# > 4) Model Averaging (模型平均)
# >
# > **【中文对照 / Chinese Translation】**
# > 在实际部署和工业级 Transformer 模型中，通常还会结合 BPE 子词切分、源与目标端 Embedding 权重共享、Beam Search 束搜索解码以及多 Checkpoint 模型权重平均等技术手段。

# %% [markdown]
# # Results (实验结果与可视化)
#
# On the WMT 2014 English-to-German translation task, the big
# transformer model outperforms previously reported models.
#
# **【中文对照 / Chinese Translation】**
# 在 WMT 2014 英德翻译任务上，大号 Transformer 模型展现出了超越此前诸多模型的优秀性能。

# %% [markdown]
# ## Attention Visualization (注意力机制权重可视化)

# %% [markdown]
# # Conclusion (结语与总结)
#
# Hopefully this code is useful for future research.
#
# **【中文对照 / Chinese Translation】**
# 希望这份代码对大家未来的研究与学习有所帮助！
'''

with open(os.path.join(r"e:\GOGOGO!\transformer", "the_annotated_transformer.py"), "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated the_annotated_transformer.py with conditional imports.")
