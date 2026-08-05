**【中文对照 / Chinese Translation】**
在过去五年中，Transformer 引起了广大学界与工业界的密切关注。
本文呈现了原论文 *Attention Is All You Need* 的逐行代码实现及其详细注解。它重新梳理并整理了原论文的部分章节，并在全文中添加了详细的代码与理论注释。本文档本身就是一个可以交互运行的 Notebook，并且是一份完全可用的 PyTorch 代码实现。
代码仓库参见 [这里](https://github.com/harvardnlp/annotated-transformer/)。

<h3> Table of Contents (目录) </h3>
<ul>
<li><a href="#prelims">Prelims (准备工作与依赖库)</a></li>
<li><a href="#background">Background (研究背景与动机)</a></li>
<li><a href="#part-1-model-architecture">Part 1: Model Architecture (第一部分：模型架构)</a></li>
<li><a href="#model-architecture">Model Architecture (模型整体架构)</a><ul>
<li><a href="#encoder-and-decoder-stacks">Encoder and Decoder Stacks (编码器与解码器堆栈)</a></li>
<li><a href="#position-wise-feed-forward-networks">Position-wise Feed-Forward Networks (逐位置前馈网络)</a></li>
<li><a href="#embeddings-and-softmax">Embeddings and Softmax (词嵌入与 Softmax 输出)</a></li>
<li><a href="#positional-encoding">Positional Encoding (位置编码)</a></li>
<li><a href="#full-model">Full Model (完整 Transformer 模型组装)</a></li>
<li><a href="#inference">Inference (模型推理/预测过程)</a></li>
</ul></li>
<li><a href="#part-2-model-training">Part 2: Model Training (第二部分：模型训练)</a></li>
<li><a href="#training">Training (训练过程)</a><ul>
<li><a href="#batches-and-masking">Batches and Masking (批处理与 Mask 掩码机制)</a></li>
<li><a href="#training-loop">Training Loop (训练循环机制)</a></li>
<li><a href="#training-data-and-batching">Training Data and Batching (训练数据与动态 Batch 分割)</a></li>
<li><a href="#hardware-and-schedule">Hardware and Schedule (硬件配置与学习率调度)</a></li>
<li><a href="#optimizer">Optimizer (优化器配置)</a></li>
<li><a href="#regularization">Regularization (正则化与标签平滑)</a></li>
</ul></li>
<li><a href="#a-first-example">A First Example (基础示例：复制任务)</a><ul>
<li><a href="#synthetic-data">Synthetic Data (合成数据生成)</a></li>
<li><a href="#loss-computation">Loss Computation (损失计算)</a></li>
<li><a href="#greedy-decoding">Greedy Decoding (贪婪解码)</a></li>
</ul></li>
<li><a href="#part-3-a-real-world-example">Part 3: A Real World Example (第三部分：真实世界机器翻译示例)</a>
<ul>
<li><a href="#data-loading">Data Loading (数据加载与预处理)</a></li>
<li><a href="#iterators">Iterators (数据迭代器构造)</a></li>
<li><a href="#training-the-system">Training the System (系统训练)</a></li>
</ul></li>
<li><a href="#additional-components-bpe-search-averaging">Additional Components: BPE, Search, Averaging (拓展组件：BPE、束搜索与模型平均)</a></li>
<li><a href="#results">Results (实验结果与可视化)</a><ul>
<li><a href="#attention-visualization">Attention Visualization (注意力权重可视化)</a></li>
<li><a href="#encoder-self-attention">Encoder Self Attention (编码器自注意力可视化)</a></li>
<li><a href="#decoder-self-attention">Decoder Self Attention (解码器自注意力可视化)</a></li>
<li><a href="#decoder-src-attention">Decoder Src Attention (解码器-源文本交叉注意力可视化)</a></li>
</ul></li>
<li><a href="#conclusion">Conclusion (结语与总结)</a></li>
</ul>

# Prelims (准备工作与依赖库)

> My comments are blockquoted. The main text is all from the paper itself.
>
> **【中文对照 / Chinese Translation】**
> 引用块（>）中的内容是作者的注解说明，其余主干文本均直接来自原论文 *Attention Is All You Need*。

```python
# ============================================================================
# 第 0 部分：准备工作与依赖库导入
# ============================================================================
# 通俗理解：就像做饭前需要准备食材和厨具，写深度学习代码前需要导入各种工具库。
# 下面每一行 import 都是在"拿出"一个工具包。

import os                     # 【操作系统接口】用于文件和目录操作，比如检查文件是否存在
from os.path import exists    # 【文件存在检查】exists("file.txt") 返回 True/False
import torch                  # 【PyTorch 深度学习框架核心库】提供张量（Tensor）、自动求导（Autograd）等
import torch.nn as nn         # 【神经网络模块】nn 是 neural network 的缩写，包含各种网络层（Linear、Embedding 等）
                               # "as nn" 是给模块起别名，之后用 nn.Linear 代替 torch.nn.Linear，更简洁
from torch.nn.functional import log_softmax, pad  
                               # 【函数式 API】从 nn 的功能模块中导入两个特定函数：
                               # - log_softmax: 计算 log(softmax(x))，比先算 softmax 再算 log 更数值稳定
                               # - pad: 对张量进行填充（padding），比如在序列末尾补零
import math                    # 【Python 标准数学库】提供 sqrt、log、sin、cos 等数学函数
import copy                    # 【深拷贝模块】copy.deepcopy() 能完全复制一个对象（包括其内部嵌套对象），
                               # 在 clones 函数中用来复制多个结构相同但参数独立的网络层
import time                    # 【时间模块】用于计时，比如计算训练速度（每秒处理多少 tokens）
import warnings                # 【警告控制模块】用于过滤掉一些不重要的警告信息

# 忽略所有警告信息（让输出更干净，专注于关键信息）
# 注意：在实际项目中要谨慎使用，有些警告可能提示潜在 bug
warnings.filterwarnings("ignore")

# ============================================================================
# 可选依赖：pandas 用于数据处理，altair 用于可视化
# ============================================================================
try:
    import pandas as pd        # 【数据分析库】pd.DataFrame 是类似 Excel 表格的数据结构
    import altair as alt       # 【声明式可视化库】用简洁的语法画交互式图表（散点图、折线图等）
except ImportError:
    # 如果用户的 Python 环境中没有安装这些库，就把它们设为 None
    # 这样代码不会报错，只是可视化相关的示例会跳过（show_example 函数中有判断）
    pd = None
    alt = None

# ============================================================================
# 可选依赖：spaCy 用于多语言分词，torchtext 用于 NLP 数据处理
# ============================================================================
try:
    import spacy                # 【自然语言处理库】提供预训练的文本分词器（tokenizer），
                                # 能把 "Hello world" 切分成 ["Hello", "world"]
    import torchtext            # 【PyTorch 文本处理库】提供 NLP 数据集、词表构建、数据迭代器等
    from torchtext.data.functional import to_map_style_dataset  
                                # 【数据集转换函数】将可迭代数据集转为支持索引访问的 Map 风格数据集
    from torch.utils.data import DataLoader  
                                # 【数据加载器】自动将数据分批次（batch）、打乱（shuffle）、
                                # 多线程加载，是 PyTorch 训练循环中的标准组件
    from torchtext.vocab import build_vocab_from_iterator  
                                # 【词表构建函数】从文本迭代器中自动统计词频并构建词汇映射表
                                # 例如 {"<unk>": 0, "the": 1, "cat": 2, ...}
except ImportError:
    # 同样，如果环境不支持，设为 None，代码优雅降级
    spacy = None
    torchtext = None

# ============================================================================
# 全局开关与辅助函数
# ============================================================================

# 【全局开关】控制是否运行示例代码。True 表示在 Jupyter Notebook 中自动运行示例
RUN_EXAMPLES = True


def is_interactive_notebook():
    """【判断当前运行环境是否为 Jupyter Notebook】
    
    原理：Jupyter/Ipython 环境会在 __builtins__ 中注入 __IPYTHON__ 这个特殊变量。
    标准 Python 脚本中没有这个变量，因此返回 False。
    
    Returns:
        bool: True 表示在 Notebook 中运行，False 表示在普通 Python 脚本中运行
    """
    # hasattr(obj, "attr_name") 检查对象是否有某个属性，比 try-except 更简洁
    return hasattr(__builtins__, "__IPYTHON__")


def show_example(fn, args=[]):
    """【条件性执行示例函数】
    
    只在满足两个条件时才执行：
    1. RUN_EXAMPLES 为 True（全局开关打开）
    2. 当前在 Jupyter Notebook 环境（能渲染交互式图表）
    
    这样设计的好处是：纯 Python 脚本中不会尝试渲染图表导致报错。
    
    Args:
        fn: 要执行的示例函数
        args: 传给函数的参数列表（默认空列表）
    
    Returns:
        函数的返回值（如果执行了），否则返回 None
    """
    if RUN_EXAMPLES and is_interactive_notebook():
        return fn(*args)  # *args 是 Python 的解包语法，把列表展开为函数参数
                          # 例如 fn(*[1,2,3]) 等价于 fn(1,2,3)


# ============================================================================
# 占位类：在不需要真正训练时，替代优化器和学习率调度器
# ============================================================================
# 通俗理解：当你只想"跑一遍看结果"（eval 模式）而并不想更新参数时，
# 用这些"假"优化器来防止代码报错，因为它们实现了相同的接口但什么也不做。
# 这是编程中的"空对象模式"（Null Object Pattern）。

class DummyOptimizer(torch.optim.Optimizer):
    """【伪优化器】冒牌优化器，实现必要的接口但所有操作都是空操作。
    
    继承自 torch.optim.Optimizer 是为了在需要 optimizer.step() 的地方
    不报类型错误。在模型评估（eval）时使用，因为评估阶段不需要更新参数。
    """
    def __init__(self):
        # param_groups 是 PyTorch 优化器必须有的属性，是一个字典列表
        # 每个字典包含一组参数和对应的学习率等配置
        # 这里给一个虚拟的 lr=0 的配置，够让代码不报错就行
        self.param_groups = [{"lr": 0}]
        None  # 空语句，什么都不做，仅作为占位

    def step(self):
        """【空操作】正常优化器的 step() 会用梯度更新参数，这里什么都不做"""
        None  # pass 也可以，None 是一种等效写法

    def zero_grad(self, set_to_none=False):
        """【空操作】正常这里是清除累积的梯度，这里什么都不做"""
        None


class DummyScheduler:
    """【伪学习率调度器】冒牌调度器，step() 是空操作。
    
    学习率调度器通常在每个 batch 或 epoch 后调整学习率。
    评估时不需要调整学习率，用这个占位即可。
    注意：这里不需要继承任何父类，只要实现了 step() 方法就行（鸭子类型）。
    """
    def step(self):
        """【空操作】正常调度器会调整学习率，这里什么都不做"""
        None
```

# Background (研究背景与动机)

The goal of reducing sequential computation also forms the
foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of
which use convolutional neural networks as basic building block,
computing hidden representations in parallel for all input and
output positions. In these models, the number of operations required
to relate signals from two arbitrary input or output positions grows
in the distance between positions, linearly for ConvS2S and
logarithmically for ByteNet. This makes it more difficult to learn
dependencies between distant positions. In the Transformer this is
reduced to a constant number of operations, albeit at the cost of
reduced effective resolution due to averaging attention-weighted
positions, an effect we counteract with Multi-Head Attention.

Self-attention, sometimes called intra-attention is an attention
mechanism relating different positions of a single sequence in order
to compute a representation of the sequence. Self-attention has been
used successfully in a variety of tasks including reading
comprehension, abstractive summarization, textual entailment and
learning task-independent sentence representations. End-to-end
memory networks are based on a recurrent attention mechanism instead
of sequencealigned recurrence and have been shown to perform well on
simple-language question answering and language modeling tasks.

To the best of our knowledge, however, the Transformer is the first
transduction model relying entirely on self-attention to compute
representations of its input and output without using sequence
aligned RNNs or convolution.

**【中文对照 / Chinese Translation】**
减少顺序计算（Sequential Computation）的目标也是 Extended Neural GPU、ByteNet 和 ConvS2S 的构建基础，这些模型均采用卷积神经网络作为基本构建块，能够并行计算所有输入和输出位置的隐藏表示。在这些模型中，关联两个任意输入或输出位置信号所需的计算操作数随位置间距离增长而增加：ConvS2S 呈线性增长，ByteNet 呈对数增长。这使得捕捉长距离位置之间的依赖关系变得极其困难。而在 Transformer 中，这一计算操作数被成功缩减到了常数级别（$O(1)$），虽然代价是由于对注意力权重位置进行平均而降低了有效分辨率，但我们通过多头注意力（Multi-Head Attention）机制成功抵消了这一负面影响。

自注意力机制（Self-Attention），有时也称为内部注意力（Intra-Attention），是一种将单个序列的不同位置相互关联以计算该序列整体表示的注意力机制。自注意力已成功应用于阅读理解、摘要生成、文本蕴涵和通用句子表示等多种任务。端到端记忆网络（End-to-End Memory Networks）基于循环注意力机制而非序列对齐的循环结构，已被证明在简单语言问答和语言建模任务中表现良好。

据我们所知，Transformer 是首个完全依赖自注意力机制来计算输入和输出表示、而不使用序列对齐 RNN 或卷积神经网络的转换模型（Transduction Model）。

# Part 1: Model Architecture (第一部分：模型架构与原理)

# Model Architecture (模型整体架构)

Most competitive neural sequence transduction models have an
encoder-decoder structure
[(cite)](https://arxiv.org/abs/1409.0473). Here, the encoder maps an
input sequence of symbol representations $(x_1, ..., x_n)$ to a
sequence of continuous representations $\mathbf{z} = (z_1, ...,
z_n)$. Given $\mathbf{z}$, the decoder then generates an output
sequence $(y_1,...,y_m)$ of symbols one element at a time. At each
step the model is auto-regressive
[(cite)](https://arxiv.org/abs/1308.0850), consuming the previously
generated symbols as additional input when generating the next.

**【中文对照 / Chinese Translation】**
目前最具竞争力的神经序列转换模型大多采用编码器-解码器（Encoder-Decoder）结构。在该结构中，编码器将符号表示的输入序列 $(x_1, ..., x_n)$ 映射为连续表示序列 $\mathbf{z} = (z_1, ..., z_n)$。在给定 $\mathbf{z}$ 的情况下，解码器逐个元素生成符号输出序列 $(y_1,...,y_m)$。在生成的每一步中，模型都是自回归的（Auto-Regressive），即将此前生成的符号作为额外的输入来生成下一个符号。

```python
# ============================================================================
# 第 1 部分：编码器-解码器总体架构（Encoder-Decoder Architecture）
# ============================================================================
# 通俗理解：这就像一个"翻译官"的骨架——左边是"理解原文"的编码器，
# 右边是"产出译文"的解码器，中间通过 Generator 把解码器的输出变成具体单词。

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    
    【洛熙人工解析】
    标准的编码器-解码器架构，是 Transformer 以及许多序列到序列（Seq2Seq）模型的基础。
    
    【新手补充详解】
    整个架构流程：输入句子 -> 源语言词嵌入 -> 编码器 -> 解码器 -> 生成器 -> 输出句子
    
    nn.Module 是什么？
    - PyTorch 中所有神经网络的"祖宗"类。继承它意味着：
      1) 自动追踪所有网络层参数（权重和偏置）
      2) 支持 .train() / .eval() 模式切换
      3) 支持 .to(device) 一键搬到 GPU
      4) 支持参数保存和加载（model.state_dict()）
    - 子类必须实现 __init__ 和 forward 两个方法
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """【构造函数】定义模型的各个子模块
        
        Args:
            encoder: 编码器（Encoder 对象），负责理解源语言句子
            decoder: 解码器（Decoder 对象），负责逐词生成目标语言句子
            src_embed: 源语言词嵌入层（Embeddings + PositionalEncoding），
                       把 token ID 转换为稠密向量
            tgt_embed: 目标语言词嵌入层，同理
            generator: 生成器（Generator 对象），把解码器最后一层的输出
                       向量转换为词表上的概率分布
        """
        # super().__init__() 调用父类 nn.Module 的初始化方法
        # 这一步至关重要：忘记调用会导致参数无法被 PyTorch 追踪！
        super(EncoderDecoder, self).__init__()
        # 将传入的子模块保存为实例属性（self.xxx）
        # PyTorch 会自动检测所有 nn.Module 类型的属性，将它们纳入参数管理
        self.encoder = encoder        # 编码器
        self.decoder = decoder        # 解码器
        self.src_embed = src_embed    # 源语言（source）嵌入
        self.tgt_embed = tgt_embed    # 目标语言（target）嵌入
        self.generator = generator    # 输出生成器

    def forward(self, src, tgt, src_mask, tgt_mask):
        """【前向传播】定义数据如何流经整个模型
        
        这是 PyTorch 中最核心的方法。不需要手动调用，当你写 model(src, tgt, ...)
        时，PyTorch 会自动调用 model.forward(src, tgt, ...)。
        
        参数说明与形状约定（batch 维度在最前面）：
        Args:
            src: 源语言输入序列，形状为 (batch_size, src_seq_len)
                 每个元素是一个 token 的整数 ID，例如 [1, 45, 23, 89, 2]
            tgt: 目标语言输入序列，形状为 (batch_size, tgt_seq_len)
                 注意：这是"已经生成的前文"，不包含最后要预测的那个词
            src_mask: 源语言掩码，形状为 (batch_size, 1, src_seq_len)
                      值为 1 表示"有效位置"，值为 0 表示"填充位置"（需忽略）
            tgt_mask: 目标语言掩码，形状为 (batch_size, tgt_seq_len, tgt_seq_len)
                      除了遮挡填充位置，更重要的是下三角掩码
                      （防止当前位置"偷看"未生成的未来词）
        
        Returns:
            Tensor: 解码器输出，形状为 (batch_size, tgt_seq_len, d_model)
                    后续传给 generator 得到每个位置的预测概率
        
        数据流: src -> encode -> memory -> decode -> output
        """
        "Take in and process masked src and target sequences. / 接收并处理带掩码的源序列与目标序列。"
        # 整条数据流一步写出：编码 -> 解码
        # self.encode(src, src_mask) 返回 memory（编码器的输出）
        # self.decode(memory, ...) 在 memory 的基础上进行解码
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """【编码阶段】将源语言句子压缩为稠密向量表示（memory）
        
        流程：
        1. self.src_embed(src)：将 token ID 转为 d_model 维向量，并加上位置编码
           输入形状: (batch_size, src_seq_len) -> 输出形状: (batch_size, src_seq_len, d_model)
        2. self.encoder(...)：N 层自注意力 + 前馈网络的堆叠编码
           输入形状: (batch_size, src_seq_len, d_model)
           输出形状: (batch_size, src_seq_len, d_model) — 形状不变
        
        Returns:
            Tensor (memory): 编码后的"记忆"表示，解码器后续会反复查阅
        """
        # 编码阶段：先经过源语言 Embed 嵌入层，再传入 Encoder 堆叠块
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """【解码阶段】基于编码结果和前文输出，逐词生成译文
        
        流程：
        1. self.tgt_embed(tgt)：将已生成的目标 token ID 转为向量
        2. self.decoder(...)：N 层自注意力 + 交叉注意力 + 前馈网络的堆叠解码
           其中交叉注意力层会"查阅" memory（编码器输出）
        
        Args:
            memory: 编码器输出，形状 (batch_size, src_seq_len, d_model)
            src_mask: 同上
            tgt: 目标语言已生成序列
            tgt_mask: 目标语言掩码（含下三角掩码）
        
        Returns:
            Tensor: 解码器输出，形状 (batch_size, tgt_seq_len, d_model)
        """
        # 解码阶段：接收编码器的输出 memory，结合目标语言 Embed 与掩码传入 Decoder
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```

```python
# ============================================================================
# 生成器（Generator）：解码器输出 → 词表概率分布
# ============================================================================
# 通俗理解：解码器输出的是一个"高维抽象向量"（512维），
# Generator 的工作就是把这个抽象向量"翻译"成"具体该选哪个词"。
# 输入: (batch_size, d_model=512) → 输出: (batch_size, vocab_size=37000)
# 37000 个词每个都得到一个概率值，取最大的那个就是模型预测的下一个词。

class Generator(nn.Module):
    """
    Define standard linear + log softmax generation step.
    【洛熙人工解析】生成器：将解码器输出的向量维度通过 Linear 线性层映射回词表大小 (vocab size)，并用 log_softmax 输出预测概率。
    
    【新手补充详解】
    为什么用 log_softmax 而不是 softmax？
    1. 数值稳定性：log(softmax(x)) 比先算 softmax 再取 log 更稳定，
       避免了 softmax 中的指数溢出问题
    2. 配合损失函数：NLLLoss（负对数似然损失）恰好需要 log 概率，
       而 log_softmax + NLLLoss = CrossEntropyLoss
    3. dim=-1：沿最后一个维度做 softmax，即对每个位置的词表概率归一化
    """

    def __init__(self, d_model, vocab):
        """
        Args:
            d_model: 模型隐藏层维度（论文中为 512），即解码器输出向量的维度
            vocab: 目标语言词表大小（英德翻译中约 37000）
        """
        super(Generator, self).__init__()
        # 【核心操作】一个线性变换（全连接层）
        # 公式：output = input @ W^T + b
        # 输入形状: (..., d_model) → 输出形状: (..., vocab)
        # 权重矩阵 W 形状: (vocab, d_model)  — 注意 PyTorch 的 Linear 存储的是转置形式
        # 偏置 b 形状: (vocab,)
        # 这个矩阵乘法把 512 维的语义向量"投影"到 37000 维的词表空间
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """前向传播：线性投影 + 对数 softmax
        
        Args:
            x: 解码器最后一层的输出，形状为 (batch_size, seq_len, d_model)
               每个位置都是一个 d_model 维的语义向量
        
        Returns:
            Tensor: log 概率，形状为 (batch_size, seq_len, vocab)
                    每个位置的每个词都有一个 log 概率值
                    值越大（越接近 0），表示模型认为该词越可能是正确输出
        """
        # self.proj(x): (batch_size, seq_len, d_model) → (batch_size, seq_len, vocab)
        # log_softmax(..., dim=-1): 沿 vocab 维度做归一化并取对数
        #   例如 [0.1, 0.5, 0.3, 0.1] 经过 softmax → [0.15, 0.40, 0.30, 0.15]
        #   经过 log_softmax → [-1.90, -0.92, -1.20, -1.90]
        #   log 概率的最大值（-0.92）对应的词就是预测结果
        return log_softmax(self.proj(x), dim=-1)
```

The Transformer follows this overall architecture using stacked
self-attention and point-wise, fully connected layers for both the
encoder and decoder, shown in the left and right halves of Figure 1,
respectively.

**【中文对照 / Chinese Translation】**
Transformer 整体遵循这一架构，其编码器和解码器均采用了堆叠的自注意力层（Stacked Self-Attention）和逐位置的全连接层（Point-wise Fully Connected Layers），分别展示在论文图 1 的左半部分和右半部分。

![](ModalNet-21.png)

## Encoder and Decoder Stacks (编码器与解码器堆栈)

### Encoder (编码器)

The encoder is composed of a stack of $N=6$ identical layers.

**【中文对照 / Chinese Translation】**
编码器由 $N=6$ 个完全相同的层堆叠而成。

```python
# ============================================================================
# clones 辅助函数：快速创建 N 个结构相同但参数独立的模块
# ============================================================================
# 通俗理解：就像复制粘贴——你需要 6 层一模一样的编码器层，
# 但每层要有自己独立的参数（权重不能共享，否则学不到不同层次的特征）。
# copy.deepcopy() 确保每个复制品是完全独立的。

def clones(module, N):
    """深拷贝产生 N 个完全相同的模块层，避免权重共享。
    
    【核心知识点】为什么用 deepcopy 而不是浅拷贝？
    - 浅拷贝（copy.copy）：只复制最外层容器，内部的权重张量还是共享的
      → 如果更新 layer[0] 的权重，layer[1] 也会变（灾难！）
    - 深拷贝（copy.deepcopy）：递归复制所有嵌套对象，包括权重矩阵
      → 每层拥有完全独立的参数，可以各自学习不同的特征模式
    
    Args:
        module: 要复制的 PyTorch 模块（比如一个 EncoderLayer 实例）
        N: 需要复制的份数（比如 6，对应论文中 N=6 层）
    
    Returns:
        nn.ModuleList: 包含 N 个独立副本的列表，可像普通 Python 列表一样索引和遍历
                       注意：ModuleList 是 PyTorch 特殊容器，能正确追踪子模块参数
    
    用法示例：
        layer = EncoderLayer(size=512, self_attn=attn, feed_forward=ff, dropout=0.1)
        encoder_layers = clones(layer, 6)  # 得到 6 个参数独立的编码器层
    """
    # 列表推导式：[表达式 for 变量 in 可迭代对象]
    # 这里：对 range(N) 中的每个数字（0,1,...,5），都执行一次 copy.deepcopy(module)
    # nn.ModuleList 将列表包装为 PyTorch 能识别的模块列表
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

```python
# ============================================================================
# 编码器（Encoder）：理解源语言句子的"大脑"
# ============================================================================
# 通俗理解：编码器就像一个"阅读器"，把输入句子逐词阅读，
# 通过自注意力机制理解词与词之间的关系，最终输出对整个句子的"理解"。

class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    【洛熙人工解析】编码器核心：由 N 个 EncoderLayer 层级联而成，最后追加一层 LayerNorm。
    
    【新手补充详解】
    编码器的数据流（以"我 爱 你"为例）：
    1. 输入三个 token 的嵌入向量，形状 (1, 3, 512)
    2. 经过第 1 层 EncoderLayer：自注意力让"爱"注意到"我"和"你"
    3. 经过第 2~6 层：层层抽象，逐渐理解深层语义
    4. 最后经过 LayerNorm：归一化输出，使数值稳定
    
    编码器不使用因果掩码（causal mask），每个词都可以看到句子中所有其他词。
    因为我们的目标是"理解整个句子"，而不是"预测下一个词"。
    """

    def __init__(self, layer, N):
        """
        Args:
            layer: 一个 EncoderLayer 实例（模板），会被 clones 复制 N 份
            N: 编码器层数（论文中 N=6），每层结构相同但参数独立
        """
        super(Encoder, self).__init__()
        # 【层叠】用 clones 函数复制 N 份 EncoderLayer
        # 这样 6 个层中的每个层都有自己独立的注意力权重和前馈网络参数
        self.layers = clones(layer, N)
        # 【最终归一化】在所有层处理完后，再做一次 LayerNorm
        # layer.size 就是 d_model（512），即归一化的特征维度
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """前向传播：将输入逐层传递，每层的输出作为下一层的输入
        
        【残差连接的体现】：
        虽然代码中看起来只是 x = layer(x, mask)，
        但 EncoderLayer 内部已经通过 SublayerConnection 实现了残差连接。
        所以实际上信息有两条路径：一条通过子层变换，一条直接传递（跳跃连接）。
        
        Args:
            x: 输入嵌入序列，形状 (batch_size, seq_len, d_model)
               已经包含了词嵌入 + 位置编码
            mask: 源语言掩码，形状 (batch_size, 1, seq_len)
                  主要用于遮挡 <pad> 填充位置，防止注意力计算时关注到无意义的填充
        
        Returns:
            Tensor: 编码后的记忆表示（memory），形状 (batch_size, seq_len, d_model)
                    解编码器将在交叉注意力层中查阅这个输出
        """
        "Pass the input (and mask) through each layer in turn. / 将输入 x 与掩码依次传入每一层 EncoderLayer，最后归一化输出。"
        # 逐层处理：layer_1(x) -> layer_2(layer_1(x)) -> ... -> layer_6(...)
        for layer in self.layers:
            x = layer(x, mask)  # 每层内部包含自注意力 + 前馈网络 + 残差连接 + LayerNorm
        # 所有层处理完后的归一化（对应论文架构图中的最顶层 LayerNorm）
        return self.norm(x)
```

We employ a residual connection
[(cite)](https://arxiv.org/abs/1512.03385) around each of the two
sub-layers, followed by layer normalization
[(cite)](https://arxiv.org/abs/1607.06450).

**【中文对照 / Chinese Translation】**
我们在两个子层中的每一个周围都采用了残差连接（Residual Connection），随后跟上层归一化（Layer Normalization）。

```python
# ============================================================================
# 层归一化（Layer Normalization）：让每层的数据分布保持稳定
# ============================================================================
# 通俗理解：深度学习就像传话游戏——信号经过很多层后会"失真"（梯度消失/爆炸）。
# LayerNorm 的作用是在每一层之后把数据"校准"一下，让它保持在合理的数值范围内。
# 与 BatchNorm 的区别：LayerNorm 沿特征维度归一化，不依赖 batch 内其他样本，适合 NLP。

class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See citation for details).
    【洛熙人工解析】层归一化 (Layer Normalization)：对特征维度求均值 mean 与标准差 std 进行标准化，
    并使用可学习参数 a_2 (gamma) 和 b_2 (beta) 进行缩放与平移。
    
    【新手补充详解】
    公式：LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    
    为什么要归一化？
    1. 稳定训练：防止数值爆炸或消失
    2. 加速收敛：让每层输入保持相似的分布
    3. 允许更大学习率：归一化后的梯度更稳定
    
    为什么在 NLP 中用 LayerNorm 而非 BatchNorm？
    - BatchNorm 在 batch 维度做归一化，需要统计 batch 内所有样本的均值和方差
    - NLP 中序列长度不一，padding 位置的统计量会干扰 BatchNorm
    - LayerNorm 对每个样本独立做归一化，不受 batch size 和序列长度影响
    
    可学习参数的意义：
    - gamma（缩放因子）：让网络可以"调整"归一化后的方差
      → 如果 gamma=1 不变，gamma>1 放大，gamma<1 缩小
    - beta（平移因子）：让网络可以"调整"归一化后的均值
      → 如果 beta=0 不变，beta>0 右移，beta<0 左移
    - 这两个参数的存在保证了"归一化"不会丢失表达能力
    """

    def __init__(self, features, eps=1e-6):
        """
        Args:
            features: 归一化的特征维度（即 d_model=512）
            eps: 小常数，防止除以零。1e-6 = 0.000001，很小但足以避免 NaN
        """
        super(LayerNorm, self).__init__()
        # 【可学习参数】gamma（缩放因子）
        # nn.Parameter 将普通的 Tensor 包装为可训练参数
        # torch.ones(features)：初始化为全 1，即初始时不改变方差
        # 形状：(features,) = (512,) — 每个特征维度有一个独立的缩放因子
        self.a_2 = nn.Parameter(torch.ones(features))
        
        # 【可学习参数】beta（平移因子）
        # torch.zeros(features)：初始化为全 0，即初始时不改变均值
        # 形状：(features,) = (512,)
        self.b_2 = nn.Parameter(torch.zeros(features))
        
        # 小常数，避免 std=0 时除以零导致 NaN
        self.eps = eps

    def forward(self, x):
        """前向传播：对输入 x 沿最后一个维度做标准化
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
               例如 (32, 50, 512) 表示 32 个句子，每个最多 50 词，每词 512 维
        
        Returns:
            Tensor: 归一化后的张量，形状不变 (batch_size, seq_len, d_model)
        
        计算步骤详解（以形状 (2, 3, 4) 为例）：
        1. x.mean(-1) → 沿最后一维（dim=3 即 d_model 维）求均值
           keepdim=True 保持维度：(2, 3, 4) → (2, 3, 1) 而非 (2, 3)
           这样广播机制才能正确工作
        2. x.std(-1) → 同样地求标准差，(2, 3, 4) → (2, 3, 1)
        3. (x - mean) / (std + eps) → 标准化为均值 0 方差 1
        4. * gamma + beta → 可学习的仿射变换
        """
        # 沿最后一个维度（d_model 维度）计算均值
        # mean 是每个样本每个位置在 512 个特征上的平均值
        mean = x.mean(-1, keepdim=True)  
        
        # 沿最后一个维度计算标准差（衡量数据的离散程度）
        # std 小 → 特征值都比较接近（数据"紧凑"）
        # std 大 → 特征值差异很大（数据"分散"）
        std = x.std(-1, keepdim=True)
        
        # 三步归一化 + 可学习变换：
        # 1. (x - mean): 中心化，使均值为 0
        # 2. / (std + eps): 缩放，使方差为 1（+eps 防除零）
        # 3. * self.a_2 + self.b_2: 仿射变换，恢复表达能力
        # 广播机制：a_2 形状 (512,) 自动广播到 (batch, seq, 512)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

That is, the output of each sub-layer is $\mathrm{LayerNorm}(x +
\mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$ is the function
implemented by the sub-layer itself.  We apply dropout
[(cite)](http://jmlr.org/papers/v15/srivastava14a.html) to the
output of each sub-layer, before it is added to the sub-layer input
and normalized.

To facilitate these residual connections, all sub-layers in the
model, as well as the embedding layers, produce outputs of dimension
$d_{\text{model}}=512$.

**【中文对照 / Chinese Translation】**
也就是说，每个子层的输出为 $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$，其中 $\mathrm{Sublayer}(x)$ 是子层自身实现的函数。在将子层的输出与其输入相加并进行归一化之前，我们对子层输出应用 Dropout 随机失活。

为了便于实现残差连接，模型中所有的子层以及词嵌入层的输出维度统一设定为 $d_{\text{model}}=512$。

```python
# ============================================================================
# 子层连接（SublayerConnection）：残差连接 + 层归一化 的打包
# ============================================================================
# 通俗理解：残差连接就像是"高速公路"——信息除了正常通过子层处理，
# 还可以直接跳跃过去。这解决了深层网络的"退化"问题。
# 想象你在写作文：每一遍修改（子层）可能让你的文章更好，但也可能改糟。
# 残差连接就是"保留原稿 + 修改意见"，既不会丢失原文，又能吸收改进。

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    
    【洛熙人工解析】
    残差连接与层归一化模块。
    注意：为了代码简洁性与训练稳定性，这里采用了 Pre-LN（先归一化再过子层），而非 Post-LN。
    
    【新手补充详解】
    Pre-LN vs Post-LN（两种残差连接范式）：
    
    - Post-LN（论文原版描述）: x + Sublayer(LayerNorm(x))
      即：先过子层，再加残差，最后归一化
      问题：深层网络训练不稳定，容易发散
    
    - Pre-LN（本代码实现）: x + Sublayer(LayerNorm(x))
      即：先归一化，再过子层，再加残差
      优点：训练更稳定，梯度流动更顺畅
      缺点：理论上表达能力略弱，但实践中影响很小
    
    残差连接的数学本质：
    output = x + f(x)  其中 f 是子层变换
    - 梯度反向传播时: d(output)/dx = 1 + d(f(x))/dx
    - 那个 "1" 保证了即使 d(f(x))/dx 很小，梯度也不会消失
    - 这就是残差网络能训练几百层的原因！
    """

    def __init__(self, size, dropout):
        """
        Args:
            size: 特征维度 d_model（512），用于 LayerNorm
            dropout: Dropout 概率（0.1），训练时随机丢弃一部分神经元防止过拟合
        """
        super(SublayerConnection, self).__init__()
        # 预层归一化（Pre-LN）
        self.norm = LayerNorm(size)
        # Dropout 层：训练时以概率 p=dropout(0.1) 随机将神经元输出置零
        # 为什么要 Dropout？ → 防止过拟合：强迫网络不依赖特定神经元，学习更鲁棒的特征
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """前向传播：Pre-LN 残差连接
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
            sublayer: 子层函数（callable），比如自注意力或前馈网络
                      可以是 Lambda 表达式: lambda x: self.self_attn(x, x, x, mask)
        
        Returns:
            Tensor: x + Dropout(Sublayer(LayerNorm(x)))，形状不变
        
        数据流（Pre-LN）：
        1. LayerNorm(x): 先归一化输入，稳定数值
        2. sublayer(norm_x): 通过子层处理（自注意力 或 前馈网络）
        3. Dropout(sublayer_output): 随机丢弃部分输出
        4. x + dropout_output: 残差连接（跳跃连接）
        
        注意：这里的 x 是"原始输入"，不是归一化后的！
        残差连接确保即使子层什么都没学到（输出接近 0），
        信息仍然可以通过 x 这一路径无损传递。
        """
        "Apply residual connection to any sublayer with the same size. / 对输入维度相同的任何子层应用 Pre-LN 残差连接。"
        # 一行代码实现 Pre-LN 残差连接：
        # self.norm(x): 先归一化
        # sublayer(self.norm(x)): 再通过子层
        # self.dropout(...): 对子层输出做 dropout
        # x + ...: 最后加上原始输入（残差）
        return x + self.dropout(sublayer(self.norm(x)))
```

Each layer has two sub-layers. The first is a multi-head
self-attention mechanism, and the second is a simple, position-wise
fully connected feed-forward network.

**【中文对照 / Chinese Translation】**
编码器的每一层包含两个子层。第一个是多头自注意力机制（Multi-Head Self-Attention），第二个是简单的逐位置全连接前馈网络（Position-wise Fully Connected Feed-Forward Network）。

```python
# ============================================================================
# 编码器层（EncoderLayer）：编码器的基本构建块
# ============================================================================
# 通俗理解：一个 EncoderLayer 就是编码器的"一层"，
# 每层做两件事：1) 自注意力（理解词之间的关系） 2) 前馈网络（独立加工每个词）
# N=6 层堆叠起来，形成深度的语义理解能力。

class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    【洛熙人工解析】单个编码器层：包含两个核心子层——自注意力机制 (Self-Attention) 和逐位置前馈网络 (Feed-Forward)。
    
    【新手补充详解】
    一张编码器层的数据流图：
    
    输入 x (batch, seq_len, 512)
       |
       ├──→ [LayerNorm] → [Self-Attention] → [Dropout] → (+) ──┐
       |                                                ↑        |
       |                                         (残差连接 x)     |
       |                                                         |
       ├──→ [LayerNorm] → [Feed Forward] → [Dropout] → (+) ──┐  |
       |                                                ↑      |  |
       |                                         (残差连接 x)   |  |
       |                                                        |  |
       输出 (batch, seq_len, 512) ←────────────────────────────┘──┘
    
    注意：这是一个 Pre-LN 架构，LayerNorm 在子层之前而非之后。
    
    为什么需要前馈网络（Feed Forward）？
    - 自注意力是"线性+加权求和"操作，表达能力有限
    - 前馈网络中的 ReLU 激活函数引入非线性，让模型能学习更复杂的语义模式
    - 公式：FFN(x) = ReLU(xW1 + b1)W2 + b2（先升维到 2048，再降回 512）
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Args:
            size: 模型维度 d_model = 512
            self_attn: 多头自注意力模块（MultiHeadedAttention 实例）
            feed_forward: 逐位置前馈网络（PositionwiseFeedForward 实例）
            dropout: Dropout 概率，论文中为 0.1
        """
        super(EncoderLayer, self).__init__()
        # 保存子层引用
        self.self_attn = self_attn          # 多头自注意力
        self.feed_forward = feed_forward    # 位置前馈网络
        # 【关键】创建两个完全相同的 SublayerConnection 模块
        # 每个 SublayerConnection 包含一个 LayerNorm 和一个 Dropout
        # clones(... , 2) 产生 2 个独立副本（各有自己的 LayerNorm 参数！）
        # → sublayer[0] 服务于自注意力子层
        # → sublayer[1] 服务于前馈网络子层
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size  # 保存维度信息

    def forward(self, x, mask):
        """前向传播：自注意力 → 前馈网络（每个子层都有残差连接和 LayerNorm）
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
            mask: 源语言掩码，形状 (batch_size, 1, seq_len)
                  用于遮挡 padding 位置
        
        Returns:
            Tensor: 编码后的张量，形状不变 (batch_size, seq_len, d_model)
        
        【关键理解】Lambda 表达式的妙用：
        lambda x: self.self_attn(x, x, x, mask) 创建了一个"匿名函数"，
        它只接受一个参数 x，但在内部调用 self.self_attn 时传了 4 个参数。
        
        为什么需要这样包装？
        因为 SublayerConnection.forward 的设计是：
            def forward(self, x, sublayer):
                return x + self.dropout(sublayer(self.norm(x)))
        它期望 sublayer 是一个"接受单个参数 x 并返回结果"的函数。
        但 self.self_attn.forward(query, key, value, mask) 需要 4 个参数！
        所以用 lambda 把 4 参数包装成 1 参数的形式。
        
        在自注意力中 query=key=value=x（因为是 self-attention，Q、K、V 都来自同一个序列）
        """
        "Follow Figure 1 (left) for connections. / 对应论文图1（左侧）：先过自注意力子层，再过前馈神经网络子层。"
        # 【子层 1：多头自注意力】
        # self.sublayer[0] 是第一个 SublayerConnection
        # lambda x: self.self_attn(x, x, x, mask) 把多头注意力包装为单参数函数
        #   三个 x 分别对应 Query、Key、Value（自注意力中三者相同）
        #   mask 防止对 padding 位置做注意力计算
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        
        # 【子层 2：逐位置前馈网络】
        # self.sublayer[1] 是第二个 SublayerConnection
        # 这里不需要 lambda 包装，因为 self.feed_forward 本身只接受一个参数 x
        return self.sublayer[1](x, self.feed_forward)
```

### Decoder (解码器)

The decoder is also composed of a stack of $N=6$ identical layers.

**【中文对照 / Chinese Translation】**
解码器同样由 $N=6$ 个完全相同的层堆叠而成。

```python
# ============================================================================
# 解码器（Decoder）：逐词生成目标语言句子的"作家"
# ============================================================================
# 通俗理解：解码器就像一个有参考材料的翻译——它一方面看自己已经写出的内容
# （自注意力），一方面查阅编码器的"理解"（交叉注意力），逐词生成译文。

class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    【洛熙人工解析】解码器：由 N 个带有 Mask 掩码机制的 DecoderLayer 堆叠而成。
    
    【新手补充详解】
    解码器与编码器的关键区别：
    1. 解码器使用了"因果掩码"（causal mask）→ 当前位置不能看到未来位置的信息
       这保证了自回归特性：生成第 i 个词时只能依赖前 i-1 个词
    2. 解码器多了一个"交叉注意力"子层 → 解码器可以查阅编码器的输出
       这是编码器-解码器架构的核心：decoder 结合"原文理解"和"已写内容"来生成下一个词
    3. 训练时使用 Teacher Forcing：用真实的前文（而非模型自己生成的前文）来指导训练
    """

    def __init__(self, layer, N):
        """
        Args:
            layer: 一个 DecoderLayer 实例（模板），会被 clones 复制 N 份
            N: 解码器层数（论文中 N=6）
        """
        super(Decoder, self).__init__()
        # 堆叠 N 层 DecoderLayer
        self.layers = clones(layer, N)
        # 最终层归一化
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """前向传播：逐层处理，每层内部包含三个子层
        
        Args:
            x: 目标语言嵌入序列，形状 (batch_size, tgt_seq_len, d_model)
               已经包含了词嵌入 + 位置编码
            memory: 编码器的最终输出，形状 (batch_size, src_seq_len, d_model)
                    这是解码器在交叉注意力层中"查阅"的对象
            src_mask: 源语言掩码，形状 (batch_size, 1, src_seq_len)
                      遮挡源语言的 padding 位置
            tgt_mask: 目标语言掩码，形状 (batch_size, tgt_seq_len, tgt_seq_len)
                      这是下三角矩阵，防止解码器"偷看"未来词
        
        Returns:
            Tensor: 解码后的表示，形状 (batch_size, tgt_seq_len, d_model)
                    后续传给 Generator 转换为词表概率
        """
        # 逐层处理：每层同时接收 x（自己的前文）和 memory（编码器输出）
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

In addition to the two sub-layers in each encoder layer, the decoder
inserts a third sub-layer, which performs multi-head attention over
the output of the encoder stack.  Similar to the encoder, we employ
residual connections around each of the sub-layers, followed by
layer normalization.

**【中文对照 / Chinese Translation】**
除了每个编码器层中的两个子层之外，解码器还插入了第三个子层，该子层对编码器堆栈的输出执行多头注意力计算（即交叉注意力 Cross-Attention）。与编码器类似，我们在每个子层周围都采用了残差连接，随后进行层归一化。

```python
# ============================================================================
# 解码器层（DecoderLayer）：解码器的基本构建块（比编码器层多一个交叉注意力）
# ============================================================================
# 通俗理解：解码器层就像翻译时的思考过程——
# 1) 先看自己已经翻译了什么（掩码自注意力）
# 2) 再看看原文是怎么说的（交叉注意力）
# 3) 最后独立加工每个位置的语义（前馈网络）

class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    【洛熙人工解析】单个解码器层：包含三个子层：
    1. 掩码自注意力 (Masked Self-Attention) — 关注自己已生成的前文
    2. 交叉注意力 (Encoder-Decoder Cross-Attention) — 关注编码器的输出
    3. 逐位置前馈网络 (Feed-Forward) — 独立加工每个位置的表示
    
    【新手补充详解】
    一张解码器层的数据流图：
    
    输入 x (已生成的目标序列)    memory (编码器输出)
       |                            |
       ├──→ [LayerNorm] → [Masked Self-Attn] → [Dropout] → (+)
       |                                              ↑      |
       |                                       (残差 x)       |
       |                                                      |
       ├──→ [LayerNorm] → [Cross-Attn(Q=x, K=memory, V=memory)] → [Dropout] → (+)
       |                                                                     ↑
       |                                                              (残差 x)
       |
       ├──→ [LayerNorm] → [Feed Forward] → [Dropout] → (+)
       |                                          ↑
       |                                   (残差 x)
       输出
    
    三个子层的 Query/Key/Value 来源：
    - 自注意力：Q=K=V=x（都是解码器自己的前文）
    - 交叉注意力：Q=x（解码器状态），K=V=memory（编码器输出）
    - 前馈网络：输入只是 x
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        Args:
            size: 模型维度 d_model = 512
            self_attn: 掩码自注意力模块（同样是 MultiHeadedAttention，但会传入 tgt_mask）
            src_attn: 交叉注意力模块（Query 来自解码器，Key/Value 来自编码器）
            feed_forward: 前馈网络
            dropout: Dropout 概率
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn          # 自注意力（带因果掩码）
        self.src_attn = src_attn            # 交叉注意力（查阅编码器输出）
        self.feed_forward = feed_forward    # 前馈网络
        # 【关键】创建 3 个 SublayerConnection（比 EncoderLayer 多 1 个）
        # sublayer[0] → 自注意力子层
        # sublayer[1] → 交叉注意力子层
        # sublayer[2] → 前馈网络子层
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """前向传播：掩码自注意力 → 交叉注意力 → 前馈网络
        
        Args:
            x: 目标语言表示，形状 (batch_size, tgt_seq_len, d_model)
            memory: 编码器输出，形状 (batch_size, src_seq_len, d_model)
            src_mask: 源语言掩码，形状 (batch_size, 1, src_seq_len)
            tgt_mask: 目标语言掩码（下三角 + padding），形状 (batch_size, tgt_seq_len, tgt_seq_len)
        
        Returns:
            Tensor: 形状不变 (batch_size, tgt_seq_len, d_model)
        """
        "Follow Figure 1 (right) for connections. / 对应论文图1（右侧）：依次执行掩码自注意力 -> 交叉注意力 -> 前馈网络。"
        m = memory  # 简短别名，提高可读性
        
        # 【子层 1：掩码自注意力】
        # Q=K=V=x, mask=tgt_mask（下三角掩码，防止看到未来词）
        # lambda 包装：把 self.self_attn(x, x, x, tgt_mask) 变成单参数函数
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        
        # 【子层 2：编码器-解码器交叉注意力】
        # Q=x（来自解码器，即"我要查什么"）
        # K=m（来自编码器，即"原文的键"）
        # V=m（来自编码器，即"原文的值"）
        # mask=src_mask（只掩掉 padding，不掩未来——因为本来就是看整个原文）
        # 通俗理解：解码器问"这个词和原文的哪些部分最相关？"
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        
        # 【子层 3：逐位置前馈网络】
        # 对每个位置独立做非线性变换（升维→ReLU→降维）
        return self.sublayer[2](x, self.feed_forward)
```

We also modify the self-attention sub-layer in the decoder stack to
prevent positions from attending to subsequent positions.  This
masking, combined with fact that the output embeddings are offset by
one position, ensures that the predictions for position $i$ can
depend only on the known outputs at positions less than $i$.

**【中文对照 / Chinese Translation】**
我们还修改了解码器堆栈中的自注意力子层，以防止当前位置关注到后续位置（即 Mask 掩码机制）。这种掩码结合输出嵌入向右偏移一个位置的操作，确保对位置 $i$ 的预测只能依赖于小于 $i$ 的已知前文输出。

```python
# ============================================================================
# 因果掩码（Subsequent Mask / Causal Mask）：防止"穿越"看未来
# ============================================================================
# 通俗理解：做翻译时，你只能根据已经写出的部分来决定下一个词，
# 不能先看到正确答案再反推。这个掩码就是确保"当前位置只能看到过去，不能看到未来"。

def subsequent_mask(size):
    """生成下三角矩阵掩码：防止解码器在预测位置 i 时"偷看"位置 i+1, i+2, ...
    
    【新手补充详解】
    为什么需要这个掩码？
    训练时，我们一次性把整个目标句子输入给解码器（Teacher Forcing 策略）。
    如果不加掩码，解码器在预测第 3 个词时就能直接"看到"第 4 个词是什么，
    这就等于作弊！训练出来的模型在推理时（只能看到前文）表现会很差。
    
    掩码的形状和含义：
    一个 size=5 的掩码矩阵：
        0  1  2  3  4  (目标位置，key)
    0 [ 1, 0, 0, 0, 0 ]  ← 位置 0 只能看到位置 0
    1 [ 1, 1, 0, 0, 0 ]  ← 位置 1 能看到 0 和 1
    2 [ 1, 1, 1, 0, 0 ]  ← 位置 2 能看到 0,1,2
    3 [ 1, 1, 1, 1, 0 ]  ← ...
    4 [ 1, 1, 1, 1, 1 ]  ← 位置 4 能看到所有前面的位置
    (query)
    
    其中 1 表示"允许关注"，0 表示"禁止关注"（对应位置在 softmax 前设为 -∞）
    
    Args:
        size: 序列长度（目标语言的 token 数量）
    
    Returns:
        Tensor: 布尔型掩码，形状 (1, size, size)
                第一个维度是 1（方便后续广播到 batch 维度）
                True 表示可以关注，False 表示禁止关注
    """
    # 步骤 1：创建注意力形状 (1, size, size)
    attn_shape = (1, size, size)
    
    # 步骤 2：生成上三角矩阵
    # torch.ones(attn_shape)：创建全 1 矩阵
    # torch.triu(..., diagonal=1)：保留对角线上方（不含对角线）的元素，下方置 0
    #   例如 size=3 时：
    #   全 1 矩阵: [[1,1,1],    上三角(diag=1): [[0,1,1],
    #              [1,1,1],  →                  [0,0,1],
    #              [1,1,1]]                      [0,0,0]]
    #   上面 1 的位置就是"未来位置"（需要被掩码掉的位置）
    # .type(torch.uint8)：转为无符号 8 位整数类型（旧版 PyTorch 用，新版本可用 bool）
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    
    # 步骤 3：取反 —— 上三角的 1 变成 0（禁止），下三角的 0 变成 1（允许）
    # subsequent_mask == 0 会产生一个布尔张量：
    #   上三角 → False（被掩码）
    #   下三角（含对角线）→ True（可以关注）
    return subsequent_mask == 0
```

> Below the attention mask shows the position each tgt word (row) is
> allowed to look at (column). Words are blocked for attending to
> future words during training.
>
> **【中文对照 / Chinese Translation】**
> 下图中的注意力掩码展示了目标序列中的每个词（行）允许关注的位置（列）。在训练过程中，掩码屏蔽掉了当前词对未来词的注意力。

```python
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
```

### Attention (注意力机制)

An attention function can be described as mapping a query and a set
of key-value pairs to an output, where the query, keys, values, and
output are all vectors.  The output is computed as a weighted sum of
the values, where the weight assigned to each value is computed by a
compatibility function of the query with the corresponding key.

We call our particular attention "Scaled Dot-Product Attention".
The input consists of queries and keys of dimension $d_k$, and
values of dimension $d_v$.  We compute the dot products of the query
with all keys, divide each by $\sqrt{d_k}$, and apply a softmax
function to obtain the weights on the values.

**【中文对照 / Chinese Translation】**
注意力函数（Attention Function）可以描述为将一个 Query（查询向量）和一组 Key-Value（键-值向量对）映射到一个 Output（输出向量）的过程，其中 Query、Keys、Values 和 Output 均为向量。输出是由 Values 的加权和计算得到的，分配给每个 Value 的权重是由 Query 与相应 Key 的匹配度函数（Compatibility Function）计算得出的。

我们将我们特有的注意力机制称为“缩放点积注意力”（Scaled Dot-Product Attention）。输入由维度为 $d_k$ 的 Queries 和 Keys，以及维度为 $d_v$ 的 Values 组成。我们计算 Query 与所有 Keys 的点积，将每个点积除以 $\sqrt{d_k}$，然后应用 Softmax 函数以获取分配给 Values 的权重。

![](ModalNet-19.png)

In practice, we compute the attention function on a set of queries
simultaneously, packed together into a matrix $Q$.  The keys and
values are also packed together into matrices $K$ and $V$.  We
compute the matrix of outputs as:

$$
   \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(
\frac{QK^T}{\sqrt{d_k}})V
$$

**【中文对照 / Chinese Translation】**
在实践中，我们同时对一组 Query 进行注意力计算，并将其打包拼接成矩阵 $Q$。Keys 和 Values 也分别打包成矩阵 $K$ 和 $V$。输出矩阵的计算公式为：
$$
   \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(
\frac{QK^T}{\sqrt{d_k}})V
$$

```python
# ============================================================================
# 缩放点积注意力（Scaled Dot-Product Attention）：Transformer 的灵魂
# ============================================================================
# 通俗理解：注意力机制就是”带着问题去读书”——
# 你有一个问题（Query），书中的每句话都有”主题标签”（Key）和”内容”（Value），
# 你用问题去匹配主题标签，找到最相关的句子，然后把这些句子的内容按相关程度加权汇总。
#
# 数学公式：Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
#
# 用生活中的例子理解：
# Q (Query): 你要找什么？→ “关于猫的信息”
# K (Key):   这段讲什么？→ [“关于狗”, “关于猫”, “关于鸟”, ...]
# V (Value): 这段的具体内容 → [“狗有四条腿...”, “猫喜欢睡觉...”, ...]
#
# 计算过程：
# 1. Q @ K^T：计算”问题”和每个”标签”的相似度（点积越大越相关）
# 2. / sqrt(d_k)：缩放，防止点积值过大导致 softmax 梯度消失
# 3. softmax：将相似度转换为概率权重（和为 1）
# 4. @ V：用权重对”内容”做加权求和，得到最终答案

def attention(query, key, value, mask=None, dropout=None):
    “””计算缩放点积注意力（Scaled Dot-Product Attention）
    
    这是整个 Transformer 最核心的数学操作。每一层的每一个注意力头
    都在执行这个函数。理解它就理解了 Transformer 的 80%。
    
    【洛熙人工解析】计算”缩放点积注意力”：
    Formula: Softmax(Q * K^T / sqrt(d_k)) * V
    
    Args:
        query:  查询矩阵，形状 (batch, h, seq_len_q, d_k)
                batch: 批量大小，h: 注意力头数
                seq_len_q: query 的序列长度
                d_k: 每个注意力头的维度（512/8=64）
        key:    键矩阵，形状 (batch, h, seq_len_k, d_k)
        value:  值矩阵，形状 (batch, h, seq_len_k, d_k)
                注意：K 和 V 的序列长度相同（都是 seq_len_k），
                但可以和 Q 的序列长度不同（交叉注意力中不同，自注意力中相同）
        mask:   可选的掩码张量，形状会广播到 (batch, 1, seq_len_q, seq_len_k)
                值为 0 的位置会被设为 -1e9（极大负数），softmax 后权重接近 0
        dropout: Dropout 层，对注意力权重做随机失活
    
    Returns:
        tuple: (加权输出, 注意力权重)
            - output: 形状 (batch, h, seq_len_q, d_k)
            - p_attn: 注意力权重矩阵，形状 (batch, h, seq_len_q, seq_len_k)
                      可用于可视化和分析
    “””
    # ========== 步骤 1：获取 d_k ==========
    # query.size(-1)：取 query 张量的最后一个维度的大小
    # 例如 query 形状 (32, 8, 10, 64) → d_k = 64
    d_k = query.size(-1)
    
    # ========== 步骤 2：计算注意力分数 ==========
    # Q @ K^T / sqrt(d_k)
    # 
    # key.transpose(-2, -1)：交换最后两个维度
    #   key 形状: (batch, h, seq_len_k, d_k)
    #   key^T 形状: (batch, h, d_k, seq_len_k)
    #
    # torch.matmul(query, key^T)：矩阵乘法
    #   (batch, h, seq_len_q, d_k) @ (batch, h, d_k, seq_len_k)
    #   = (batch, h, seq_len_q, seq_len_k)
    #   每个元素 scores[b][h][i][j] 表示第 i 个 query 和第 j 个 key 的原始相似度
    #
    # math.sqrt(d_k)：缩放因子 sqrt(64) = 8
    #   为什么要除以 sqrt(d_k)？
    #   假设 q 和 k 的各分量是独立随机的（均值 0, 方差 1），
    #   则点积 q·k 的方差 = d_k（随维度增大而增大）
    #   方差大的 softmax 输出接近 one-hot → 梯度极小 → 训练困难
    #   除以 sqrt(d_k) 将方差归一化回 1，保持梯度健康
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # ========== 步骤 3：应用掩码 ==========
    if mask is not None:
        # masked_fill：将 mask==0 的位置用指定值填充
        # mask 形状需要能广播到 scores 的形状
        # -1e9（负十亿）：一个极大的负数
        #   经过 softmax 后，exp(-1e9) ≈ 0，相当于该位置被”完全忽略”
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # ========== 步骤 4：Softmax 归一化 ==========
    # scores.softmax(dim=-1)：沿最后一个维度（key 维度）做 softmax
    #   softmax 公式：softmax(x_i) = exp(x_i) / sum(exp(x_j))
    #   结果：每个 query 对各个 key 的注意力权重，总和为 1
    #   形状：(batch, h, seq_len_q, seq_len_k)
    p_attn = scores.softmax(dim=-1)
    
    # ========== 步骤 5：可选 Dropout ==========
    if dropout is not None:
        # 对注意力权重做 Dropout：随机将一些权重置零再重新归一化
        # 防止模型过度依赖某些特定的注意力模式
        p_attn = dropout(p_attn)
    
    # ========== 步骤 6：加权求和 ==========
    # 注意力权重 @ Value = 加权汇总
    # (batch, h, seq_len_q, seq_len_k) @ (batch, h, seq_len_k, d_k)
    # = (batch, h, seq_len_q, d_k)
    # 每个 query 位置得到一个 d_k 维的”上下文向量”
    # 返回注意力权重 p_attn 是为了后续可视化和分析
    return torch.matmul(p_attn, value), p_attn
```

The two most commonly used attention functions are additive
attention [(cite)](https://arxiv.org/abs/1409.0473), and dot-product
(multiplicative) attention.  Dot-product attention is identical to
our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_k}}$. Additive attention computes the
compatibility function using a feed-forward network with a single
hidden layer.  While the two are similar in theoretical complexity,
dot-product attention is much faster and more space-efficient in
practice, since it can be implemented using highly optimized matrix
multiplication code.

While for small values of $d_k$ the two mechanisms perform
similarly, additive attention outperforms dot product attention
without scaling for larger values of $d_k$
[(cite)](https://arxiv.org/abs/1703.03906). We suspect that for
large values of $d_k$, the dot products grow large in magnitude,
pushing the softmax function into regions where it has extremely
small gradients (To illustrate why the dot products get large,
assume that the components of $q$ and $k$ are independent random
variables with mean $0$ and variance $1$.  Then their dot product,
$q \cdot k = \sum_{i=1}^{d_k} q_ik_i$, has mean $0$ and variance
$d_k$.). To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.

**【中文对照 / Chinese Translation】**
两种最常用的注意力函数是加性注意力（Additive Attention）和点积/乘性注意力（Dot-Product Attention）。点积注意力除了缩放因子 $\frac{1}{\sqrt{d_k}}$ 之外与我们的算法完全相同。加性注意力使用带有单个隐层的前馈网络来计算匹配函数。虽然两者在理论复杂度上相似，但在实践中，点积注意力要快得多且更节省空间，因为它可以利用高度优化的矩阵乘法运算来实现。

虽然对于较小的 $d_k$ 值，这两种机制的表现相似，但在没有缩放因子的情况下，随着 $d_k$ 增大，加性注意力超越了点积注意力。我们怀疑，对于较大的 $d_k$ 值，点积的数值量级会急剧增大，从而将 Softmax 函数推向具有极小梯度的饱和区域（为了解释为什么点积会变大：假设 $q$ 和 $k$ 的各分量是均值为 0、方差为 1 的独立随机变量，则它们的点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 0，方差为 $d_k$）。为了抵消这种副作用，我们将点积乘以缩放因子 $\frac{1}{\sqrt{d_k}}$。

![](ModalNet-20.png)

Multi-head attention allows the model to jointly attend to
information from different representation subspaces at different
positions. With a single attention head, averaging inhibits this.

$$
\mathrm{MultiHead}(Q, K, V) =
    \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \
    \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

Where the projections are parameter matrices $W^Q_i \in
\mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in
\mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in
\mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in
\mathbb{R}^{hd_v \times d_{\text{model}}}$.

In this work we employ $h=8$ parallel attention layers, or
heads. For each of these we use $d_k=d_v=d_{\text{model}}/h=64$. Due
to the reduced dimension of each head, the total computational cost
is similar to that of single-head attention with full
dimensionality.

**【中文对照 / Chinese Translation】**
多头注意力（Multi-Head Attention）允许模型联合关注来自不同位置的不同表示子空间（Subspaces）的信息。而如果只有一个注意力头，对所有位置进行简单平均会抑制这种多角度信息的捕获。
$$
\mathrm{MultiHead}(Q, K, V) =
    \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \
    \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$
其中线性投影是参数矩阵 $W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ 以及 $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$。

在这项工作中，我们采用 $h=8$ 个平行的注意力层（即8个注意力头）。对于每个头，我们设定 $d_k = d_v = d_{\text{model}}/h = 64$。由于减少了每个头的维度，总计算成本与具有全维度单头注意力的计算成本非常接近。

```python
# ============================================================================
# 多头注意力（MultiHeadedAttention）：让模型从多个角度理解语义
# ============================================================================
# 通俗理解：单一注意力头就像一个人只能从一个角度看问题——
# 而多头注意力就像有 8 个人，每个人从不同角度分析同一段文字，
# 最后把 8 个人的分析结果综合起来。这样就能捕捉到更丰富的语义关系。
#
# 举个例子：在句子 "The cat sat on the mat because it was tired" 中，
# - 头 1 可能关注语法依赖（cat ← sat）
# - 头 2 可能关注指代关系（it → cat）
# - 头 3 可能关注位置关系（on → mat）
# - ...以此类推
#
# 计算过程（h=8 个头，d_model=512，d_k=64）：
# 1. 将 Q、K、V 分别通过线性投影分割为 8 份（每份 64 维）
# 2. 8 个头各自独立计算缩放点积注意力
# 3. 将 8 个头的输出拼接回 512 维
# 4. 最后通过一个线性投影整合多头信息

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention (多头注意力机制)
    【洛熙人工解析】将 Query、Key、Value 投影到 h 个不同的子空间中分别计算注意力，最后拼接输出。
    
    【新手补充详解】
    多头注意力的参数矩阵和维度变换：
    
    有 4 个线性投影矩阵（通过 clones 创建）：
    - linears[0]: W^Q — 将所有头的 Q 投影合并在一起（512 → 512）
    - linears[1]: W^K — 将所有头的 K 投影合并在一起（512 → 512）
    - linears[2]: W^V — 将所有头的 V 投影合并在一起（512 → 512）
    - linears[3]: W^O — 输出投影，将拼接结果映射回 d_model（512 → 512）
    
    Q 的维度变换过程：
    (batch, seq_len, 512) → [W^Q] → (batch, seq_len, 512)
    → view → (batch, seq_len, 8, 64)
    → transpose → (batch, 8, seq_len, 64)  ← 头维度移到前面方便批量计算
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        Args:
            h: 注意力头数（论文中 h=8）
            d_model: 模型总维度（论文中 512），必须能被 h 整除
            dropout: Dropout 概率（论文中 0.1）
        """
        super(MultiHeadedAttention, self).__init__()
        # 断言 d_model 能被 h 整除
        # 原因：每个头处理的维度 d_k = d_model / h
        # 如果 512 不能被 8 整除，就没法均匀分配维度给每个头
        assert d_model % h == 0
        # 【每个头的维度】d_k = 512 / 8 = 64
        # 虽然每个头只处理 64 维，但 8 个头拼起来又是 512 维
        self.d_k = d_model // h
        self.h = h  # 头数
        
        # 【创建 4 个线性投影矩阵】
        # clones(nn.Linear(d_model, d_model), 4) 生成 4 个独立的 512→512 线性层
        # 为什么是 512→512 而不是 512→64？
        # 因为把 8 个头的投影合并在一起计算更高效：
        #   Q_proj = Q @ W^Q（一次性完成 8 个头的投影）
        #   然后通过 reshape 拆分成 8 个头
        # 这样可以利用矩阵乘法的硬件加速
        # linears 索引含义：
        #   [0]=W^Q, [1]=W^K, [2]=W^V, [3]=W^O（输出投影）
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        
        # 保存最近一次计算的注意力权重（用于可视化分析）
        self.attn = None
        # Dropout 层
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """多头注意力前向传播
        
        Args:
            query: 形状 (batch, seq_len_q, d_model)
            key:   形状 (batch, seq_len_k, d_model)
            value: 形状 (batch, seq_len_k, d_model)
                   注意：Q 可以和 K/V 有不同的序列长度
                   自注意力中 seq_len_q == seq_len_k
                   交叉注意力中 seq_len_q (解码器长度) ≠ seq_len_k (编码器长度)
            mask:  形状 (batch, seq_len_q, seq_len_k) 或可广播到该形状
        
        Returns:
            Tensor: 形状 (batch, seq_len_q, d_model)
                    所有头的输出拼接后经过 W^O 投影的结果
        """
        # ========== 步骤 1：掩码维度调整 ==========
        if mask is not None:
            # mask 当前形状可能是 (batch, 1, seq_len) 或 (batch, seq_len, seq_len)
            # 在第 1 维增加一个维度：变为 (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)
            # 这样 mask 自然广播到 (batch, h, seq_len_q, seq_len_k)
            #   因为第 1 维是 h=8，而 mask 的第 1 维是 1 → 自动广播
            mask = mask.unsqueeze(1)
        
        # ========== 步骤 2：获取 batch 大小 ==========
        nbatches = query.size(0)  # 取出第 0 维的大小（batch size）
        
        # ========== 步骤 3：线性投影 + 多头拆分 ==========
        # 这个列表推导式一次性完成 Q、K、V 三者的投影和拆分
        # 对于 (query, key, value) 三个输入分别操作：
        #
        # 以 query 为例（linears[0] 是 W^Q）：
        #   输入: (batch, seq_len_q, 512)
        #   ↓ lin(x) — 线性投影
        #   (batch, seq_len_q, 512)  ← 8 个头的信息混在一起
        #   ↓ .view(nbatches, -1, self.h, self.d_k)
        #   (batch, seq_len_q, 8, 64)  ← 明确分为 8 个头，每个 64 维
        #   ↓ .transpose(1, 2)
        #   (batch, 8, seq_len_q, 64)  ← 把"头数"维度移到前面
        #
        # 为什么要把头维度移到前面？
        # 这样 attention 函数可以直接对 (batch, h, seq, d_k) 做批量矩阵乘法
        # PyTorch 的 matmul 支持 batch 矩阵乘法，会自动处理前两个维度
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # zip(self.linears, (query, key, value)) 将投影矩阵和输入一一配对：
        #   (linears[0]=W^Q, query) → 投影 query
        #   (linears[1]=W^K, key)   → 投影 key
        #   (linears[2]=W^V, value) → 投影 value
        # 注意：linears[3]=W^O 在这里不使用，最后才用

        # ========== 步骤 4：计算缩放点积注意力 ==========
        # 8 个头各自独立计算注意力
        # 输入形状: Q (batch, 8, seq_q, 64), K (batch, 8, seq_k, 64), V (batch, 8, seq_k, 64)
        # x 形状: (batch, 8, seq_q, 64) — 加权后的输出
        # self.attn 形状: (batch, 8, seq_q, seq_k) — 注意力权重矩阵（保存用于可视化）
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # ========== 步骤 5：多头合并 ==========
        # 将 8 个头的输出重新拼接在一起
        #
        # 步骤 5a：transpose(1, 2)
        #   (batch, 8, seq_q, 64) → (batch, seq_q, 8, 64)
        #   把头维度从位置 1 移回位置 2，为拼接做准备
        #
        # 步骤 5b：contiguous()
        #   transpose 只是改变了"视角"（stride），没有真正改变内存布局
        #   contiguous() 确保内存是连续排列的，这样 view() 才能正确工作
        #   （如果跳过这步，view() 可能会报错）
        #
        # 步骤 5c：view(nbatches, -1, self.h * self.d_k)
        #   (batch, seq_q, 8, 64) → (batch, seq_q, 512)
        #   -1 告诉 PyTorch 自动计算这一维的大小（即 seq_q 保持不变）
        #   8*64=512，8 个头的结果拼接回原始维度
        x = (
            x.transpose(1, 2)       # 把头维度移回中间
            .contiguous()            # 确保内存连续
            .view(nbatches, -1, self.h * self.d_k)  # 拼接 8 个头
        )
        
        # ========== 步骤 6：清理中间变量 ==========
        # 释放不再需要的中间张量，节约显存
        # 在训练大型模型时，这种显式的内存管理很重要
        del query
        del key
        del value
        
        # ========== 步骤 7：最终输出投影 ==========
        # self.linears[-1] 即 W^O（输出投影矩阵）: 512 → 512
        # 将拼接后的多头信息通过线性变换融合为一个统一的表示
        # 形状: (batch, seq_q, 512) → (batch, seq_q, 512)
        return self.linears[-1](x)
```

### Applications of Attention in our Model (注意力机制在模型中的三大应用)

The Transformer uses multi-head attention in three different ways:
1) In "encoder-decoder attention" layers, the queries come from the
previous decoder layer, and the memory keys and values come from the
output of the encoder.  This allows every position in the decoder to
attend over all positions in the input sequence.  This mimics the
typical encoder-decoder attention mechanisms in sequence-to-sequence
models such as [(cite)](https://arxiv.org/abs/1609.08144).

2) The encoder contains self-attention layers.  In a self-attention
layer all of the keys, values and queries come from the same place,
in this case, the output of the previous layer in the encoder.  Each
position in the encoder can attend to all positions in the previous
layer of the encoder.

3) Similarly, self-attention layers in the decoder allow each
position in the decoder to attend to all positions in the decoder up
to and including that position.  We need to prevent leftward
information flow in the decoder to preserve the auto-regressive
property.  We implement this inside of scaled dot-product attention
by masking out (setting to $-\infty$) all values in the input of the
softmax which correspond to illegal connections.

**【中文对照 / Chinese Translation】**
Transformer 以三种不同的方式应用多头注意力机制：
1）在“编码器-解码器交叉注意力”（Encoder-Decoder Attention）层中：Query 来自前一个解码器层，而 Key 和 Value 来自编码器的输出 memory。这使得解码器中的每个位置都能关注到输入源序列的所有位置。
2）编码器包含“自注意力”（Self-Attention）层：在自注意力层中，所有的 Key、Value 和 Query 都来自同一个地方（即编码器中前一层的输出）。编码器中的每个位置都可以关注到编码器前一层的所有位置。
3）类似地，解码器中的“自注意力层”允许解码器中的每个位置关注到解码器中直到并包括该位置在内的所有位置。我们需要防止解码器中的向左信息流动，以保持自回归特性。我们在缩放点积注意力内部通过掩码机制（将 Softmax 输入中对应于非法连接的所有值设置为 $-\infty$）来实现这一点。

## Position-wise Feed-Forward Networks (逐位置前馈神经网络)

In addition to attention sub-layers, each of the layers in our
encoder and decoder contains a fully connected feed-forward network,
which is applied to each position separately and identically.  This
consists of two linear transformations with a ReLU activation in
between.

$$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$

While the linear transformations are the same across different
positions, they use different parameters from layer to
layer. Another way of describing this is as two convolutions with
kernel size 1.  The dimensionality of input and output is
$d_{\text{model}}=512$, and the inner-layer has dimensionality
$d_{ff}=2048$.

**【中文对照 / Chinese Translation】**
除了注意力子层之外，编码器和解码器中的每一层都包含一个全连接的前馈网络，该网络独立且相同地作用于每个位置。它由两个线性变换组成，中间带有 ReLU 激活函数：
$$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$
虽然线性变换在不同位置之间是相同的，但它们在层与层之间使用不同的参数。描述它的另一种方式是将其视为两个核大小为 1 的卷积。输入和输出的维度为 $d_{\text{model}}=512$，中间隐层的维度为 $d_{ff}=2048$。

```python
# ============================================================================
# 逐位置前馈网络（Position-wise Feed-Forward Network, FFN）
# ============================================================================
# 通俗理解：自注意力负责"交流"——让每个词看到其他词；
# 前馈网络负责"思考"——每个词独立地加工自己的语义信息。
# 就像开会：大家先讨论交流（自注意力），然后各自回去独立思考总结（FFN）。
#
# 数学公式：FFN(x) = max(0, xW1 + b1)W2 + b2
# 即：线性变换（512→2048）→ ReLU 激活 → 线性变换（2048→512）
# "先升维再降维"的设计让模型有更大的容量来学习复杂的非线性变换。

class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    【洛熙人工解析】实现 FFN 公式：FFN(x) = max(0, xW_1 + b_1)W_2 + b_2。
    升维到 d_ff (2048) 再降维回 d_model (512)。
    
    【新手补充详解】
    为什么"升维再降维"？
    1. 升维到 2048 提供了更大的"思考空间"，让每个 token 的表示可以展开为更丰富的特征
    2. ReLU 激活函数引入非线性：f(x)=max(0,x)，负数变 0，正数不变
       → 没有非线性，多层线性变换等价于单层（能力有限）
    3. 降维回 512 压缩信息，保持与残差连接兼容
    4. 每个位置独立计算（"逐位置"的含义），不同位置使用相同的参数 W1/W2
    
    这个设计的灵感类似于：把一句话展开成一段详细解释（2048维），
    然后再压缩回一个精炼的摘要（512维）。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 输入/输出维度，512（也是整个 Transformer 的统一维度）
            d_ff: 中间隐层维度，2048（论文设定，约为 d_model 的 4 倍）
            dropout: Dropout 概率，0.1
        """
        super(PositionwiseFeedForward, self).__init__()
        # 【第一层线性变换：升维】512 → 2048
        # W1 形状: (2048, 512)，b1 形状: (2048,)
        # 这一步把每个 token 的表示从 512 维"展开"到 2048 维
        self.w_1 = nn.Linear(d_model, d_ff)
        
        # 【第二层线性变换：降维】2048 → 512
        # W2 形状: (512, 2048)，b2 形状: (512,)
        # 这一步把展开后的表示"压缩"回 512 维
        self.w_2 = nn.Linear(d_ff, d_model)
        
        # Dropout 放在两层之间，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """前向传播：升维 → ReLU → Dropout → 降维
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model=512)
               通常是自注意力（或交叉注意力）子层的输出
        
        Returns:
            Tensor: 形状不变 (batch_size, seq_len, d_model=512)
        
        计算过程（逐行分解）：
        1. self.w_1(x)： (batch, seq, 512) @ W1^T → (batch, seq, 2048)
        2. .relu()：对每个元素应用 max(0, x)，负值变 0
        3. self.dropout(...)：随机丢弃部分神经元输出
        4. self.w_2(...)： (batch, seq, 2048) @ W2^T → (batch, seq, 512)
        """
        # 链式调用：Layer1 → ReLU → Dropout → Layer2
        # PyTorch 中 .relu() 是张量的方法，等价于 torch.relu(tensor)
        # 这种链式写法很 Pythonic（流畅），也可以写成：
        #   hidden = self.w_1(x)
        #   hidden = torch.relu(hidden)
        #   hidden = self.dropout(hidden)
        #   output = self.w_2(hidden)
        return self.w_2(self.dropout(self.w_1(x).relu()))
```

## Embeddings and Softmax (词嵌入层与 Softmax 输出)

Similarly to other sequence transduction models, we use learned
embeddings to convert the input tokens and output tokens to vectors
of dimension $d_{\text{model}}$.  We also use the usual learned
linear transformation and softmax function to convert the decoder
output to predicted next-token probabilities.  In our model, we
share the same weight matrix between the two embedding layers and
the pre-softmax linear transformation, similar to
[(cite)](https://arxiv.org/abs/1608.05859). In the embedding layers,
we multiply those weights by $\sqrt{d_{\text{model}}}$.

**【中文对照 / Chinese Translation】**
与其他序列转换模型类似，我们使用可学习的词嵌入（Learned Embeddings）将输入 Token 和输出 Token 转换为维度为 $d_{\text{model}}$ 的向量。我们还使用常见的可学习线性变换与 Softmax 函数，将解码器输出转换为预测下一个 Token 的概率。在我们的模型中，源语言嵌入层、目标语言嵌入层以及 Softmax 前的线性变换这三者共享相同的权重矩阵。在嵌入层中，我们将这些权重乘以 $\sqrt{d_{\text{model}}}$ 进行缩放。

```python
# ============================================================================
# 词嵌入层（Embeddings）：把"单词编号"变成"语义向量"
# ============================================================================
# 通俗理解：计算机看不懂文字，只能理解数字。
# Embedding 就是给每个词分配一个"语义坐标"——意思相近的词坐标也相近。
# 比如 "cat" 和 "kitten" 的嵌入向量会很接近，而和 "car" 的向量相距较远。
#
# 类比：就像给世界上每个城市标注经纬度——
# 输入城市名（token ID）→ 输出经纬度坐标（embedding vector）

class Embeddings(nn.Module):
    """
    【洛熙人工解析】词嵌入层：将输入的 Token ID 映射为 d_model 维度的向量，
    并乘以 sqrt(d_model) 进行缩放，使嵌入向量的方差与位置编码对齐。
    
    【新手补充详解】
    为什么要乘以 sqrt(d_model)？
    1. nn.Embedding 默认使用 N(0,1) 初始化权重，嵌入向量的方差约为 1
    2. 位置编码（Positional Encoding）使用 sin/cos 生成，范围在 [-1, 1]，方差较小
    3. 两者相加后，如果词嵌入和位置编码的数值量级不同，
       位置信息可能被词嵌入"淹没"或"过度放大"
    4. 乘以 sqrt(512)=22.6 增大词嵌入的量级，使其与位置编码合理平衡
    5. 这个技巧也在原论文中被提及
    
    权重共享（Weight Sharing）：
    论文中提到，源语言嵌入、目标语言嵌入和生成器的线性层共享权重矩阵。
    这意味着三个地方使用同一套参数，大幅减少参数量并提高泛化能力。
    """

    def __init__(self, d_model, vocab):
        """
        Args:
            d_model: 嵌入向量的维度，512
            vocab: 词表大小（不同 token 的个数），
                   英德翻译中约 37000（BPE 子词词表）
        """
        super(Embeddings, self).__init__()
        # 【核心】nn.Embedding 是一个大型查找表（Look-Up Table, 所以叫 lut）
        # 内部存储一个形状为 (vocab, d_model) 的权重矩阵
        # 例如 vocab=37000, d_model=512 → 约 1900 万个参数
        # 
        # 工作原理（类比字典）：
        #   输入 token_id=42 → 返回权重矩阵的第 42 行（一个 512 维的向量）
        #   输入 [1, 45, 23] → 返回 3 个 512 维的向量
        #
        # 训练过程中，这些向量会被优化，使得语义相近的词向量也相近
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """前向传播：Token ID → 嵌入向量 → 缩放
        
        Args:
            x: 输入序列，形状 (batch_size, seq_len)
               每个元素是词表中某个 token 的整数 ID
               例如: [[1, 45, 23, 89, 2], [1, 67, 12, 3, 2]]  (2 个句子，每句 5 个词)
        
        Returns:
            Tensor: 形状 (batch_size, seq_len, d_model)
                    每个 token ID 被替换为对应的 d_model 维语义向量
        """
        # 步骤 1：self.lut(x) — 查表，把 ID 变成向量
        #   输入 (batch, seq) → 输出 (batch, seq, d_model)
        # 
        # 步骤 2：* math.sqrt(self.d_model) — 缩放
        #   math.sqrt(512) ≈ 22.627
        #   将嵌入向量的数值范围放大，与位置编码的量级匹配
        return self.lut(x) * math.sqrt(self.d_model)
```

## Positional Encoding (位置编码)

Since our model contains no recurrence and no convolution, in order
for the model to make use of the order of the sequence, we must
inject some information about the relative or absolute position of
the tokens in the sequence.  To this end, we add "positional
encodings" to the input embeddings at the bottoms of the encoder and
decoder stacks.  The positional encodings have the same dimension
$d_{\text{model}}$ as the embeddings, so that the two can be summed.
There are many choices of positional encodings, learned and fixed
[(cite)](https://arxiv.org/pdf/1705.03122.pdf).

In this work, we use sine and cosine functions of different frequencies:

$$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$

$$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$

where $pos$ is the position and $i$ is the dimension.  That is, each
dimension of the positional encoding corresponds to a sinusoid.  The
wavelengths form a geometric progression from $2\pi$ to $10000 \cdot
2\pi$.  We chose this function because we hypothesized it would
allow the model to easily learn to attend by relative positions,
since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a
linear function of $PE_{pos}$.

In addition, we apply dropout to the sums of the embeddings and the
positional encodings in both the encoder and decoder stacks.  For
the base model, we use a rate of $P_{drop}=0.1$.

**【中文对照 / Chinese Translation】**
由于我们的模型既不包含循环结构也不包含卷积结构，为了让模型能够利用序列的顺序信息，我们必须引入关于序列中 Token 相对或绝对位置的信息。为此，我们在编码器和解码器堆栈底部的输入词嵌入中相加了“位置编码”（Positional Encodings）。位置编码具有与词嵌入相同的维度 $d_{\text{model}}$，以便两者可以直接相加。位置编码有许多选择，包括可学习的和固定公式计算的。

在这项工作中，我们使用了不同频率的正弦和余弦函数：
$$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
$$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$
其中 $pos$ 为位置，$i$ 为维度。也就是说，位置编码的每个维度都对应一个正弦波。波长形成从 $2\pi$ 到 $10000 \cdot 2\pi$ 的等比数列。我们选择这个函数是因为我们假设它能让模型轻松学会根据相对位置来进行注意力计算，因为对于任何固定的偏移量 $k$，$PE_{pos+k}$ 都可以表示为 $PE_{pos}$ 的线性函数。

此外，我们在编码器和解码器堆栈中，对词嵌入与位置编码相加后的结果应用了 Dropout 随机失活。对于基础模型，我们设定的失活率为 $P_{drop}=0.1$。

```python
# ============================================================================
# 位置编码（Positional Encoding）：让模型知道"第几个词"
# ============================================================================
# 通俗理解：Transformer 没有 RNN 那样的"顺序处理"机制，它是一次性看整个句子。
# 这带来一个问题：它不知道"我"是第一个词、"爱"是第二个词。
# 位置编码就是给每个词打上"位置标签"——用不同频率的正弦/余弦波来表示位置。
#
# 数学公式（论文原版）：
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))    ← 偶数维度用 sin
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))    ← 奇数维度用 cos
#
# 其中 pos 是词在句子中的位置（0,1,2,...），i 是维度索引（0,1,...,d_model/2-1）
#
# 为什么用正弦/余弦？
# 1. 值域在 [-1, 1]，与嵌入向量数值范围匹配
# 2. 不同频率允许模型学习相对位置关系（因为 sin(a+b) 可以用 sin(a)cos(b)+cos(a)sin(b) 表示）
# 3. 不需要训练（固定的），可以外推到比训练时更长的序列
# 4. 每个位置的编码是唯一的，且相邻位置的编码是连续的

class PositionalEncoding(nn.Module):
    """
    Positional Encoding (位置编码)
    【洛熙人工解析】使用正弦和余弦函数交替编码序列中每个 token 的位置信息。
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    【新手补充详解】
    位置编码的可视化理解：
    - pos=0（第一个词）：所有偶数维度=sin(0)=0，奇数维度=cos(0)=1
    - 随着 pos 增大，低维度索引变化快（高频），高维度索引变化慢（低频）
    - 这就像一个"二进制计数"的连续版本：低维像秒针（变化快），高维像时针（变化慢）
    
    10000^(2i/d_model) 的含义：
    - 当 i=0（最低维度对）：10000^(0/512) = 1（频率最高，变化最快）
    - 当 i=255（最高维度对）：10000^(510/512) ≈ 10000（频率最低，几乎是常数）
    - 这个等比数列从高频到低频覆盖，形成丰富的"位置指纹"
    """

    def __init__(self, d_model, dropout, max_len=5000):
        """
        Args:
            d_model: 嵌入维度，512（位置编码的每个向量也是 512 维）
            dropout: Dropout 概率，0.1
            max_len: 预设的最大序列长度，5000
                     位置编码会预先计算到 5000 个位置，足够覆盖绝大多数句子
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # ========== 预计算位置编码矩阵 ==========
        # 在 __init__ 中就计算好所有位置编码，避免 forward 中重复计算
        
        # 【步骤 1】创建形状 (max_len, d_model) 的全零矩阵
        # 后续会用 sin 和 cos 值填充
        pe = torch.zeros(max_len, d_model)
        
        # 【步骤 2】创建位置索引列向量
        # torch.arange(0, max_len): [0, 1, 2, ..., 4999] 形状 (5000,)
        # .unsqueeze(1): 增加一维 → (5000, 1)，方便后续广播乘法
        # 每一行代表一个位置（pos=0,1,2,...,4999）
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # 【步骤 3】计算除数项 div_term
        # 公式中需要计算: pos / 10000^(2i/d_model)
        # 等价于: pos * 1/10000^(2i/d_model) = pos * exp(-2i * log(10000)/d_model)
        #
        # 分解：
        # torch.arange(0, d_model, 2): [0, 2, 4, ..., 510] 形状 (256,)
        #   取所有偶数索引（对应 sin 要用的维度）
        # math.log(10000.0): 约等于 9.21
        # math.log(10000.0) / d_model: 约等于 9.21/512 ≈ 0.018
        # -(math.log(10000.0) / d_model): ≈ -0.018
        # arange(0, d_model, 2) * -0.018: [0, -0.036, -0.054, ..., -9.18]
        # torch.exp(...): [1, 0.965, 0.931, ..., ~0.0001]
        #   这形成了从 1 到 ~1/10000 的等比数列
        #
        # 最终 div_term 形状: (256,) — 每个元素对应一对 (2i, 2i+1) 维度的除数
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        
        # 【步骤 4a】填充偶数维度（0, 2, 4, ..., 510）用 sin
        # position: (5000, 1), div_term: (256,)
        # position * div_term → 广播 → (5000, 256)
        #   含义：对于位置 pos 的第 2i 维，值 = sin(pos / 10000^(2i/d_model))
        # pe[:, 0::2]: 选取所有行的偶数索引列，形状 (5000, 256)
        pe[:, 0::2] = torch.sin(position * div_term)
        #   0::2 是 Python 切片语法：从 0 开始，步长为 2
        #   等价于 [0, 2, 4, 6, ..., 510]
        
        # 【步骤 4b】填充奇数维度（1, 3, 5, ..., 511）用 cos
        # 1::2 从 1 开始，步长为 2 → [1, 3, 5, ..., 511]
        # 同样的 position * div_term，但用 cos 函数
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 【步骤 5】增加 batch 维度
        # pe 当前形状: (max_len, d_model) = (5000, 512)
        # unsqueeze(0) 在第 0 维前插入一个维度
        # → (1, max_len, d_model) = (1, 5000, 512)
        # 这样在 forward 中可以直接与 (batch, seq_len, d_model) 的嵌入相加
        # 因为第 0 维是 1，会自动广播到 batch_size
        pe = pe.unsqueeze(0)
        
        # 【步骤 6】注册为 buffer（非参数但需保存的张量）
        # register_buffer 和 nn.Parameter 的区别：
        # - nn.Parameter: 会被优化器更新（可训练的参数）
        # - register_buffer:  不会被优化器更新，但会随模型一起保存和移动到 GPU
        #   位置编码不需要训练（是固定的数学函数），所以用 buffer
        # 这个 pe 会保存在 model.state_dict() 中，model.to(device) 时也会自动迁移
        self.register_buffer("pe", pe)

    def forward(self, x):
        """前向传播：将位置编码加到词嵌入上
        
        Args:
            x: 词嵌入输出，形状 (batch_size, seq_len, d_model)
               来自 Embeddings.forward() 的返回值
        
        Returns:
            Tensor: x + PE，形状 (batch_size, seq_len, d_model)
                    词嵌入和位置编码的和经过 Dropout
        
        示例（seq_len=3, d_model=512）：
        x 的形状 (1, 3, 512)，pe 的形状 (1, 5000, 512)
        self.pe[:, :3, :] 取前 3 个位置的编码 → (1, 3, 512)
        x + pe → (1, 3, 512)（逐元素相加）
        """
        # 【关键】将位置编码加到词嵌入上
        # self.pe[:, :x.size(1)]: 切出与当前序列长度匹配的位置编码
        #   x.size(1) = seq_len（当前 batch 的序列长度）
        #   self.pe 形状 (1, 5000, 512)，取前 seq_len 个位置
        #
        # .requires_grad_(False): 明确声明位置编码不需要梯度
        #   因为位置编码是固定的 sin/cos 值，不需要通过反向传播更新
        #   这个调用确保即使在训练模式下也不会为 PE 计算梯度，节省显存
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        
        # 对相加后的结果应用 Dropout
        # 注意：Dropout 只在训练模式下生效（model.train()），评估模式下自动关闭
        return self.dropout(x)
```

> Below the positional encoding will add in a sine wave based on
> position. The frequency and offset of the wave is different for
> each dimension.
>
> **【中文对照 / Chinese Translation】**
> 下图展示了基于位置添加的正弦波位置编码。每个特征维度对应的正弦波频率和偏移量都是独特的。

```python
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
```

We also experimented with using learned positional embeddings
[(cite)](https://arxiv.org/pdf/1705.03122.pdf) instead, and found
that the two versions produced nearly identical results.  We chose
the sinusoidal version because it may allow the model to extrapolate
to sequence lengths longer than the ones encountered during
training.

**【中文对照 / Chinese Translation】**
我们还尝试了使用可学习的位置嵌入（Learned Positional Embeddings）来替代固定公式，发现两种版本的实验效果几乎完全相同。我们最终选择正弦公式版本，是因为它可能允许模型外推到比训练期间遇到的序列长度更长的序列。

## Full Model (完整 Transformer 模型组装)

> Here we define a function from hyperparameters to a full model.
>
> **【中文对照 / Chinese Translation】**
> 这里我们定义一个根据超参数构建完整 Transformer 模型的辅助函数。

```python
# ============================================================================
# make_model：根据超参数一键构建完整 Transformer
# ============================================================================
# 通俗理解：这就是 Transformer 的"组装工厂"——
# 你只需要告诉它词表大小、层数等参数，它就自动把所有零件拼装成一个完整的模型。

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    """辅助函数：根据超参数构建完整的 Transformer 模型。
    
    【新手补充详解】
    超参数说明（对应论文中的 Base Model 配置）：
    - N=6: 编码器和解码器各 6 层（共 12 层）
    - d_model=512: 所有子层和嵌入层的统一输出维度
    - d_ff=2048: 前馈网络中间层维度（约为 d_model 的 4 倍）
    - h=8: 多头注意力的头数（d_model/h = 512/8 = 64，即每头维度）
    - dropout=0.1: 随机失活概率（10% 的神经元被随机置零）
    
    模型结构总览：
    EncoderDecoder(
        ├── Encoder（6 层 EncoderLayer）
        │   └── 每层: Self-Attention → Feed-Forward（都带残差+LayerNorm）
        ├── Decoder（6 层 DecoderLayer）
        │   └── 每层: Self-Attention → Cross-Attention → Feed-Forward
        ├── src_embed: Embeddings → PositionalEncoding
        ├── tgt_embed: Embeddings → PositionalEncoding
        └── Generator: Linear(d_model, vocab) → log_softmax
    )
    
    Args:
        src_vocab: 源语言词表大小
        tgt_vocab: 目标语言词表大小
        N, d_model, d_ff, h, dropout: 超参数
    
    Returns:
        EncoderDecoder: 完整的 Transformer 模型实例
    """
    # 【重要】取 copy.deepcopy 函数的短别名
    # 为什么要用深拷贝？因为编码器的 attention/ff 和解码器的 attention/ff
    # 必须是参数独立的副本。如果共享引用，训练编码器会意外修改解码器的参数。
    c = copy.deepcopy
    
    # 【构建通用子模块】这些模块会被编码器和解码器共享结构模板
    # 实际的权重通过 deepcopy 来复制，保证参数独立
    
    # 多头注意力模块模板（h=8, d_model=512, dropout=0.1）
    attn = MultiHeadedAttention(h, d_model)
    # 前馈网络模板（512→2048→512, dropout=0.1）
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 位置编码模板（d_model=512, dropout=0.1, max_len=5000）
    position = PositionalEncoding(d_model, dropout)
    
    # 【组装完整模型】
    model = EncoderDecoder(
        # ── 编码器部分 ──
        # Encoder 封装了 N=6 层 EncoderLayer
        # c(attn) 和 c(ff) 各产生一个独立副本给编码器
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        
        # ── 解码器部分 ──
        # Decoder 封装了 N=6 层 DecoderLayer
        # 注意：解码器需要两个注意力模块：
        #   c(attn) → 自注意力（带因果掩码）
        #   c(attn) → 交叉注意力（查阅编码器输出）
        #   c(ff)   → 前馈网络
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        
        # ── 源语言嵌入管线 ──
        # nn.Sequential 将多个层串联：先 Embedding 再 PositionalEncoding
        # 输入 token ID → 嵌入向量 → +位置编码 → 输出
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        
        # ── 目标语言嵌入管线 ──
        # 同理，目标语言的嵌入 + 位置编码
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        
        # ── 生成器 ──
        # Linear(d_model, tgt_vocab): 512 维向量 → 词表概率分布
        Generator(d_model, tgt_vocab),
    )

    # 【参数初始化】使用 Xavier/Glorot 均匀分布初始化所有权重矩阵
    # 遍历模型的所有可学习参数
    for p in model.parameters():
        if p.dim() > 1:  # 只对矩阵参数（≥2 维）做初始化，跳过偏置（1 维）
            # Xavier 初始化：使每层输出的方差 ≈ 输入的方差
            # 这对于深层网络的训练稳定性至关重要
            # 如果不初始化或初始化不当，深层网络容易出现梯度消失/爆炸
            nn.init.xavier_uniform_(p)
    return model
```

## Inference: (模型推理与前向预测)

> Here we make a forward step to predict a translation using greedy
> decoding.
>
> **【中文对照 / Chinese Translation】**
> 这里我们使用贪婪解码（Greedy Decoding）进行前向推断预测。

```python
# ============================================================================
# inference_test：贪婪解码推理示例
# ============================================================================
# 通俗理解：这个函数展示了 Transformer 如何"一步接一步"地生成译文。
# 就像写作文——根据已经写好的部分，每次决定下一个最合适的词。

def inference_test():
    """用一个小型 Transformer 演示贪婪解码的完整流程
    
    模型配置：
    - 词表大小：11（只有 token 0~10）
    - 层数 N=2（为了快速演示，而非论文的 N=6）
    - 任务：从输入序列 [1,2,3,...,10] 生成输出序列
    
    推理 vs 训练的关键区别：
    - 训练：一次性看到完整的目标序列（Teacher Forcing），并行计算
    - 推理：逐词生成，每次只多生成一个词，循环直到结束
    """
    # 创建 2 层的小型 Transformer（词表大小 11，仅用于测试）
    test_model = make_model(11, 11, 2)
    # model.eval()：切换到评估模式，关键影响：
    # 1. Dropout 被禁用（所有神经元都参与计算）
    # 2. BatchNorm（如果有）使用全局统计量而非 batch 统计量
    test_model.eval()
    
    # 【构造源语言输入】一个序列，包含 token 1 到 10
    # torch.LongTensor：创建整数类型的张量（token ID 必须是整数）
    # 形状：(1, 10) = (batch_size=1, seq_len=10)
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # 【源语言掩码】全 1，表示所有位置都是有效 token（没有 padding）
    # 形状：(1, 1, 10) = (batch, 1, seq_len)
    # 中间维度 1 会自动广播到注意力头数 h
    src_mask = torch.ones(1, 1, 10)

    # 【编码阶段】一次性处理整个源序列
    # 输入 src (1,10) → 嵌入 → 编码器 → 输出 memory (1,10,512)
    memory = test_model.encode(src, src_mask)
    
    # 【初始化输出序列】仅包含起始符号（token 0 作为 <sos>）
    # 形状：(1, 1) — 一个 batch，一个 token
    ys = torch.zeros(1, 1).type_as(src.data)  # type_as 确保和 src 同设备同类型

    # 【逐词生成循环】最多生成 9 个词（加上起始符号共 10 个）
    for i in range(9):
        # ── 步骤 1：解码 ──
        # 用当前已生成的序列 ys 和编码器输出 memory 进行解码
        # subsequent_mask(ys.size(1)) 生成因果掩码，确保位置 i 看不到 i+1 及之后
        # out 形状: (1, current_len, 512)
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        
        # ── 步骤 2：生成下一个词的概率 ──
        # out[:, -1] 取最后一个位置的输出（因为只需要预测"下一个词"）
        # 形状：(1, 512)
        # generator 将其映射为词表上的 log 概率：(1, 11) — 11 个词各有一个 log 概率
        prob = test_model.generator(out[:, -1])
        
        # ── 步骤 3：贪心选择概率最高的词 ──
        # torch.max(prob, dim=1) 沿词表维度找最大值
        # 返回 (最大值, 最大值索引)
        # _ 是最大值（这里不需要），next_word 是索引（即预测的词 ID）
        _, next_word = torch.max(prob, dim=1)
        # .data[0] 取出标量值（从 GPU tensor 转为 Python 整数）
        next_word = next_word.data[0]
        
        # ── 步骤 4：将新词追加到输出序列 ──
        # torch.cat([ys, new_token], dim=1) 沿序列维度拼接
        # torch.empty(1,1).type_as(...).fill_(next_word)：创建包含新词的 1×1 张量
        # 序列从 (1, t) 变为 (1, t+1)
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    # 打印模型生成的完整序列
    print("Example Model Output:", ys)


def run_tests():
    """运行所有测试函数
    
    当前只有 inference_test，但可以轻松扩展添加更多测试
    """
    for fn in [inference_test]:
        fn()  # 逐个执行测试函数


show_example(run_tests)  # 在 Jupyter 中自动展示示例
```

# Part 2: Model Training (第二部分：模型训练流程)

## Batches and Masking (批数据处理与 Mask 掩码机制)

```python
# ============================================================================
# Batch 类：一次处理的一个小批量数据
# ============================================================================
# 通俗理解：GPU 一次处理一个句子效率太低，所以把多个句子打包成一批（batch）一起处理。
# 这个类不仅存储数据，还自动生成训练所需的掩码（mask）。

class Batch:
    """批处理类：用于在训练过程中持有并构造带 Mask 掩码的输入序列与目标序列。
    
    【洛熙人工解析】Batch 批处理类：用于在训练过程中持有并构造带 Mask 掩码的输入序列与目标序列。
    
    【新手补充详解】
    Batch 中几个核心概念：
    
    1. Teacher Forcing（教师强制）：
       训练时，解码器的输入是"真实的前文"而非"模型自己生成的前文"。
       例如目标序列是 [<s>, 我, 爱, 你, </s>]：
       - tgt  (解码器输入): [<s>, 我, 爱, 你]        （去掉最后一个词）
       - tgt_y (预测目标):    [我, 爱, 你, </s>]      （去掉第一个词）
       模型用 tgt 去预测 tgt_y 的每个位置
    
    2. 两种掩码：
       - Padding Mask: 句子长度不一，短的用 <pad> 填充。掩码确保模型忽略填充位置
       - Causal Mask (因果掩码): 确保位置 i 不能看到位置 i+1, i+2, ...
         目标掩码 = Padding Mask & Causal Mask（两个掩码的交集）
    """

    def __init__(self, src, tgt=None, pad=2):  # 2 表示 padding 的 token id
        """初始化一个 Batch
        
        Args:
            src: 源语言序列，形状 (batch_size, src_seq_len)
                 例如 [[1, 45, 23, 2, 0, 0], ...]（末尾 0 是 padding）
            tgt: 目标语言序列，形状 (batch_size, tgt_seq_len)
                 None 表示只有源语言（如推理时）
            pad: padding token 的 ID，默认 2（论文中 <blank> 的 id）
        """
        self.src = src
        # 【源语言掩码】标记哪些位置不是 padding
        # src != pad: 产生布尔张量，True 表示有效位置，False 表示 padding
        #   (batch, seq_len) 的布尔矩阵
        # .unsqueeze(-2): 在倒数第二维增加一维
        #   (batch, seq_len) → (batch, 1, seq_len)
        #   为喵要在中间加一维？→ 方便广播到 (batch, h, seq_len_q, seq_len_k) 的注意力分数
        self.src_mask = (src != pad).unsqueeze(-2)
        
        # 如果有目标序列（训练时），构造目标相关的各种数据
        if tgt is not None:
            # ── Teacher Forcing 的输入输出拆分 ──
            # tgt 形状: (batch, tgt_seq_len)
            #   例如 [[1, 5, 23, 8, 2]]（1=<s>, 2=</s>）
            # tgt[:, :-1]: 去掉最后一个 token → 解码器输入
            #   [[1, 5, 23, 8]]   (前面部分)
            self.tgt = tgt[:, :-1]
            # tgt[:, 1:]: 去掉第一个 token → 预测目标（正确答案）
            #   [[5, 23, 8, 2]]   (后面部分)
            # 模型的任务就是用 tgt 中的每个位置预测 tgt_y 中对应的位置
            self.tgt_y = tgt[:, 1:]
            
            # ── 目标掩码 = padding 掩码 & 因果掩码 ──
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            
            # ── 统计有效 token 数量（排除 padding） ──
            # 用于后续的损失归一化（总损失 / 有效 token 数）
            # .data.sum() 取纯数值（不参与梯度计算）
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """构造目标语言的标准掩码：同时隐藏 padding 和未来词
        
        这个方法是 @staticmethod（静态方法），意味着不需要实例化 Batch 就能调用。
        
        Args:
            tgt: 目标语言解码器输入，形状 (batch, tgt_seq_len)
            pad: padding token 的 ID
        
        Returns:
            Tensor: 布尔掩码，形状 (batch, tgt_seq_len, tgt_seq_len)
                    True 表示该位置可以关注，False 表示禁止关注
        
        掩码的可视化（假设 tgt 长度为 4，其中位置 3 是 padding）：
        padding 掩码:          因果掩码:          最终掩码（&）:
        [1,1,1,0]  (行)      [1,0,0,0]          [1,0,0,0]
        [1,1,1,0]             [1,1,0,0]          [1,1,0,0]
        [1,1,1,0]             [1,1,1,0]          [1,1,1,0]
        [1,1,1,0]             [1,1,1,1]          [1,1,1,0]
        """
        "Create a mask to hide padding and future words. / 构造掩码：同时隐藏 padding 填充符与未来位置。"
        # 【第一步】Padding 掩码
        # tgt != pad: (batch, seq_len) 布尔矩阵
        # .unsqueeze(-2): (batch, 1, seq_len)
        # 广播后每行相同，标记哪些 key 位置是有效的
        tgt_mask = (tgt != pad).unsqueeze(-2)
        
        # 【第二步】结合因果掩码（下三角矩阵）
        # subsequent_mask(tgt.size(-1)): (1, seq_len, seq_len) 下三角矩阵
        # & 操作：两个都为 True 的位置才为 True（逻辑与）
        # 结果：既不是 padding，又不是未来位置的那些位置才允许关注
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
```

## Training Loop (训练循环机制)

```python
# ============================================================================
# 训练循环（Training Loop）：深度学习训练的"发动机"
# ============================================================================

class TrainState:
    """训练状态跟踪器：用简单的类记录训练进度
    
    【洛熙人工解析】训练状态跟踪：记录当前 Step 步数、处理样本数与 Token 总数。
    
    【新手补充详解】
    这里用类属性（class attribute）而非实例属性（instance attribute）来定义默认值。
    注意这里有个微妙之处：每次 TrainState() 创建新实例时，这些类属性会作为默认值，
    但 Python 中可变默认参数可能有一些陷阱。
    
    各字段含义：
    - step: 当前训练的 step 编号（每个 batch 对应一个 step）
    - accum_step: 实际执行参数更新的次数（考虑了梯度累积）
    - samples: 已处理的样本（句子）总数
    - tokens: 已处理的目标语言 token 总数
    """
    step: int = 0        # 批次步数（每个 batch 计数一次）
    accum_step: int = 0  # 累积步数（每次实际更新参数时计数一次）
    samples: int = 0     # 累计处理的样本总数
    tokens: int = 0      # 累计处理的 token 总数


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
    """运行一个完整的 Epoch（遍历全部训练数据一遍）
    
    【洛熙人工解析】运行单个 Epoch：包含前向传播、损失计算、反向传播与梯度累积机制。
    
    【新手补充详解】
    训练循环的核心步骤（每个 batch 重复一次）：
    1. 前向传播：数据流过模型，计算预测输出
    2. 损失计算：比较预测与真实标签，量化误差
    3. 反向传播：计算损失对每个参数的梯度
    4. 参数更新：用梯度更新模型权重
    5. 学习率调整：根据调度器更新学习率
    
    梯度累积（Gradient Accumulation）：
    当 GPU 显存不够一次处理很大的 batch 时，可以：
    - 用较小的 batch 多次前向传播
    - 每次的梯度累加起来
    - 每 accum_iter 次才真正更新一次参数
    这样等效于用更大的 batch size 训练，但显存占用较小。
    
    Args:
        data_iter: 数据迭代器，每次 yield 一个 Batch 对象
        model: Transformer 模型
        loss_compute: 损失计算函数（如 SimpleLossCompute 实例）
        optimizer: 优化器（如 Adam）
        scheduler: 学习率调度器
        mode: "train"（训练）/ "eval"（评估）/ "train_cum"（训练+累积）
        accum_iter: 梯度累积步数（每隔多少 batch 更新一次参数）
        train_state: 训练状态跟踪器
    
    Returns:
        tuple: (平均损失, 更新后的 train_state)
    """
    start = time.time()      # 记录开始时间（用于计算速度）
    total_tokens = 0          # 累计 token 数（整个 epoch）
    total_loss = 0            # 累计损失（整个 epoch）
    tokens = 0                # 当前 40-step 窗口内的 token 数（用于速度显示）
    n_accum = 0               # 已执行的参数更新次数
    
    # enumerate(data_iter) 返回 (索引, 数据) 对
    # i 从 0 开始递增，batch 是 Batch 对象
    for i, batch in enumerate(data_iter):
        # ──── 步骤 1：前向传播 ────
        # 数据流过整个模型（编码器 → 解码器 → 生成器）
        # out 形状：(batch_size, tgt_seq_len, d_model)
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        
        # ──── 步骤 2：计算损失 ────
        # loss_compute 是一个可调用对象（实现了 __call__ 方法）
        # 内部会先用 generator 将 out 映射为词表概率，再与 tgt_y 比较
        # loss: 纯数值（Python float），用于显示
        # loss_node: 带梯度的张量（PyTorch tensor），用于反向传播
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        
        # ──── 步骤 3 & 4：反向传播与参数更新（仅训练模式）────
        if mode == "train" or mode == "train_cum":
            # 反向传播：计算损失对每个参数的偏导数（梯度）
            # 梯度存储在各参数的 .grad 属性中
            loss_node.backward()
            
            # 更新统计信息
            # batch.src.shape[0] = batch_size（本 batch 中的句子数）
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            
            # 梯度累积：每 accum_iter 个 batch 才真正更新参数
            # i % accum_iter == 0 在 i=0 时也成立，因为 0 % n == 0
            if i % accum_iter == 0:
                # optimizer.step()：根据累积的梯度更新所有参数
                # 对于 Adam 优化器，这包括计算动量、自适应学习率等
                optimizer.step()
                
                # optimizer.zero_grad()：清空所有梯度
                # set_to_none=True：将 .grad 设为 None 而非全零张量
                #   设为 None 比全零张量更省显存，PyTorch 推荐做法
                optimizer.zero_grad(set_to_none=True)
                
                n_accum += 1                    # 更新次数 +1
                train_state.accum_step += 1     # 累积步数 +1
            
            # scheduler.step()：在每个 batch 后调整学习率
            # 注意：论文中的 warmup 策略是在每个 step（而非每个 epoch）调整 lr
            scheduler.step()

        # ──── 步骤 5：累计统计 ────
        total_loss += loss           # 累加总损失（用于 epoch 平均）
        total_tokens += batch.ntokens # 累加总 token 数
        tokens += batch.ntokens       # 累加当前窗口 token 数
        
        # ──── 每 40 个 batch 打印一次进度 ────
        if i % 40 == 0:
            elapsed = time.time() - start  # 距离上次打印的秒数
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (
                    i,                              # 当前 batch 索引
                    n_accum,                        # 已更新次数
                    loss / batch.ntokens,           # 每个 token 的平均损失
                    tokens / elapsed,               # 每秒处理的 token 数（速度）
                    scheduler.get_last_lr()[0],     # 当前学习率
                )
            )
            start = time.time()  # 重置计时
            tokens = 0           # 重置窗口 token 计数
        
        # 显式删除中间变量，释放显存（对大模型很重要）
        del loss
        del loss_node
    
    # 返回整个 epoch 的平均损失（总损失/总 token 数）和训练状态
    return total_loss / total_tokens, train_state
```

## Training Data and Batching (训练数据构建与动态 Batch 分割)

We trained on the standard WMT 2014 English-German dataset consisting
of about 4.5 million sentence pairs.  Sentences were encoded using
byte-pair encoding, which has a shared source-target vocabulary of
about 37000 tokens. For English-French, we used the much larger WMT
2014 English-French dataset consisting of 36M sentences and split
tokens into a 32000 word-piece vocabulary.

Sentence pairs were batched together by approximate sequence
length. Each training batch contained a set of sentence pairs
containing approximately 25000 source tokens and 25000 target
tokens.

**【中文对照 / Chinese Translation】**
我们在由约 450 万对句子组成的标准 WMT 2014 英德数据集上进行了训练。句子使用字节对编码（BPE）进行编码，源语言与目标语言共享一个约 37,000 个 Token 的词表。对于英法翻译，我们使用了规模大得多的 WMT 2014 英法数据集（包含 3,600 万个句子），并将 Token 切分为 32,000 个词碎片（Word-piece）词表。

句对按近似序列长度组合成批（Batch）。每个训练批次包含的句对大约涵盖 25,000 个源语言 Token 和 25,000 个目标语言 Token。

```python
def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours. / 重新封装 Batch 结构以适应自定义数据迭代。"
    src, tgt = batch.src, batch.tgt
    return Batch(src, tgt, pad_idx)
```

## Hardware and Schedule (硬件资源与学习率调度算法)

We trained our models on one machine with 8 NVIDIA P100 GPUs.  For
the base models using the hyperparameters described throughout the
paper, each training step took about 0.4 seconds.  We trained the
base models for a total of 100,000 steps or 12 hours.  For our big
models, step time was 1.0 seconds.  The big models were trained for
300,000 steps (3.5 days).

**【中文对照 / Chinese Translation】**
我们在一台配备 8 张 NVIDIA P100 GPU 的机器上训练模型。对于使用论文中所述超参数的基础模型（Base model），每个训练步耗时约 0.4 秒。我们对基础模型总共训练了 100,000 步（约 12 小时）。对于大号模型（Big model），每步耗时 1.0 秒。大号模型共训练了 300,000 步（约 3.5 天）。

```python
# ============================================================================
# rate：Transformer 特有的学习率预热+衰减策略
# ============================================================================
# 通俗理解：学习率就像"学习速度"——太快会学飞（不收敛），太慢学不动（收敛慢）。
# Transformer 的策略是：先慢慢加速（warmup），再慢慢减速（衰减）。
# 就像开车：起步慢踩油门，速度上来后慢慢松油门滑行。
#
# 公式（论文原版）：
# lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
#
# 图像是一个先上升后下降的曲线，峰值在 warmup_steps 处。

def rate(step, model_size, factor, warmup):
    """根据论文公式计算当前 step 的动态学习率
    
    【洛熙人工解析】根据公式计算动态学习率：
    lrate = factor * (model_size^(-0.5) * min(step^(-0.5), step * warmup^(-1.5)))
    
    【新手补充详解】
    学习率曲线的两个阶段（以 warmup=4000 为例）：
    
    阶段 1：预热期（step < warmup，即前 4000 步）
    - 学习率线性增长：lrate ∝ step * warmup^(-1.5)
    - 从接近 0 开始，逐渐增大到峰值
    - 为什么需要预热？模型初始参数是随机的，梯度方向不稳定，
      用小学习率让模型先"站稳脚跟"，避免一开始就走错方向
    
    阶段 2：衰减期（step >= warmup，即 4000 步之后）
    - 学习率按 step^(-0.5) 衰减（平方根倒数衰减）
    - 越往后学习率越小，逐步精细调优
    - 为什么需要衰减？接近最优解时需要小步微调，大步可能跳过最优解
    
    factor 和 model_size 的作用：
    - model_size^(-0.5) = 1/sqrt(512) ≈ 0.044：根据模型大小缩放学习率
      大模型梯度方差更大，需要更小的学习率
    - factor：用户可调的额外缩放因子（论文中 factor=1）
    
    Args:
        step: 当前训练步数（第几个 batch）
        model_size: 模型维度 d_model（512）
        factor: 学习率缩放因子（通常为 1）
        warmup: 预热步数（论文中为 4000）
    
    Returns:
        float: 当前步数的学习率
    """
    # 防止 step=0 时除以零（0^(-0.5) = 1/0 = 无穷大）
    # 将 step=0 视为 step=1，学习率从非常小的正值开始
    if step == 0:
        step = 1
    
    # 核心公式分步解析：
    # model_size ** (-0.5) = 512^(-0.5) = 1/sqrt(512) ≈ 0.0442
    #   → 基础缩放因子，模型越大学习率越小
    # 
    # min(step^(-0.5), step * warmup^(-1.5))：
    #   - step^(-0.5)：衰减项，step 越大值越小
    #   - step * warmup^(-1.5)：预热项，step 越大值越大
    #   - min(...)：两者取小，形成"先升后降"的曲线
    #     前 warmup 步：预热项 < 衰减项 → 学习率上升
    #     后 warmup 步：衰减项 < 预热项 → 学习率下降
    # 
    # factor * ...：最终缩放（默认 factor=1，不改变）
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
```

## Optimizer (优化器与学习率 Warmup)

We used the Adam optimizer with $\beta_1=0.9$, $\beta_2=0.98$ and
$\epsilon=10^{-9}$. We varied the learning rate over the course of
training, according to the formula:

$$
lrate = d_{\text{model}}^{-0.5} \cdot
  \min({step\_num}^{-0.5},
    {step\_num} \cdot {warmup\_steps}^{-1.5})
$$

This corresponds to increasing the learning rate linearly for the
first $warmup\_steps$ training steps, and decreasing it thereafter
proportionally to the inverse square root of the step number.  We
used $warmup\_steps=4000$.

**【中文对照 / Chinese Translation】**
我们使用了 Adam 优化器，超参数设置为 $\beta_1=0.9$、$\beta_2=0.98$ 以及 $\epsilon=10^{-9}$。在训练过程中，我们根据以下公式动态调整学习率：
$$
lrate = d_{\text{model}}^{-0.5} \cdot
  \min({step\_num}^{-0.5},
    {step\_num} \cdot {warmup\_steps}^{-1.5})
$$
这对应于在前 $warmup\_steps$ 个训练步中线性增加学习率，此后按步数的平方根倒数比例降低学习率。我们设置 $warmup\_steps=4000$。

## Regularization (正则化与标签平滑)

### Label Smoothing (标签平滑)

During training, we employed label smoothing of value
$\epsilon_{ls}=0.1$ [(cite)](https://arxiv.org/abs/1512.00567).
This hurts perplexity, as the model learns to be more unsure, but
improves accuracy and BLEU score.

**【中文对照 / Chinese Translation】**
在训练期间，我们采用了平滑值为 $\epsilon_{ls}=0.1$ 的标签平滑（Label Smoothing）。由于模型学会了不那么“盲目自信”，这会轻微损害困惑度（Perplexity），但显著提升了准确率和 BLEU 得分。

```python
# ============================================================================
# 标签平滑（Label Smoothing）：让模型不要"太自信"
# ============================================================================
# 通俗理解：标准训练中，模型被告知"第 5 个词 100% 是正确的，其他词 0% 正确"。
# 标签平滑说："第 5 个词 90% 正确，其他词平分剩下的 10%"。
# 这样模型不会过度自信，对未见过的数据泛化更好（提升 BLEU 分数）。
#
# 为什么有效？
# 1. 防止过拟合：模型不会把概率质量全压在单个词上
# 2. 更真实的分布：真实语言中，一个位置可能有多个合理的词
# 3. 更平滑的梯度：softmax + 交叉熵在极端分布时梯度很小

class LabelSmoothing(nn.Module):
    "Implement label smoothing. / 实现标签平滑 KL 散度损失。"
    
    """标签平滑损失——使用 KL 散度度量预测分布与平滑后真实分布的距离
    
    【新手补充详解】
    两种损失函数的对比：
    
    1. 标准交叉熵（无标签平滑）：
       真实分布: [0, 0, 1, 0, 0]  ← 只有正确词是 1（one-hot）
       预测分布: [0.1, 0.05, 0.7, 0.1, 0.05]
       损失鼓励预测集中在正确词上 → 容易过拟合
    
    2. 标签平滑（smoothing=0.1）：
       真实分布: [0.025, 0.025, 0.9, 0.025, 0.025]
                ↑ 每个非正确词分到 smoothing/(vocab-2) = 0.1/4 = 0.025
                ↑ 正确词得到 confidence = 1 - smoothing = 0.9
       预测分布: 同上
       模型学会了"不确定"，在测试集上表现更好
    
    为什么用 KL 散度而非交叉熵？
    - KLDivLoss 要求输入是 log 概率（配合 log_softmax 使用）
    - 等价于带有平滑标签的交叉熵，但数值实现更稳定
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        Args:
            size: 词表大小（目标语言）
            padding_idx: padding token 的 ID
                         该位置的标签被设为零概率（因为 padding 不需要预测）
            smoothing: 平滑系数 ε，论文中为 0.1
                       0 表示不做平滑（退化为标准交叉熵）
        """
        super(LabelSmoothing, self).__init__()
        # 【核心】KL 散度损失
        # reduction="sum"：返回 batch 中所有元素的 KL 散度之和（而非均值）
        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * log P) - sum(P * log Q)
        # 其中第二项就是带权重的交叉熵
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx    # padding 的 token ID
        self.confidence = 1.0 - smoothing  # 正确标签的概率（如 0.9）
        self.smoothing = smoothing         # 平滑量（如 0.1）
        self.size = size                   # 词表大小
        self.true_dist = None              # 缓存最近一次构建的真实分布（用于可视化）

    def forward(self, x, target):
        """计算标签平滑 KL 散度损失
        
        Args:
            x: 模型的 log 概率输出（来自 Generator 的 log_softmax）
               形状：(batch_size * seq_len, vocab_size)
               注意：通常会被 view 成 (总token数, 词表大小)
            target: 真实标签（正确的 token ID）
                    形状：(batch_size * seq_len,)
        
        Returns:
            Tensor: KL 散度损失（标量，但内部是 sum 模式的张量）
        
        核心逻辑——构建平滑后的"真实"分布：
        1. 所有位置先均匀分配 smoothing/(vocab-2) 的概率
        2. 正确位置额外加上 confidence 的概率
        3. padding 位置设为 0（这些位置不需要预测）
        """
        # 断言：预测的词汇维度必须等于词表大小
        # x.size(1) 是 vocab_size 维度
        assert x.size(1) == self.size
        
        # ── 步骤 1：初始化均匀分布 ──
        # x.data.clone()：复制 x 的数据部分（不复制梯度），只作为模板
        # 形状与 x 相同：(总token数, vocab_size)
        true_dist = x.data.clone()
        
        # fill_：将所有位置填充为 smoothing / (vocab - 2)
        # 为什么是 vocab-2 而不是 vocab？
        # 排除两个特殊 token：padding_idx 和 正确标签的位置
        # 但这里先均匀填，后续再修正正确位置和 padding 位置
        true_dist.fill_(self.smoothing / (self.size - 2))
        
        # ── 步骤 2：设置正确标签的概率 ──
        # scatter_(dim, index, value)：将 value 写入指定位置
        # dim=1 表示沿词汇维度操作
        # target.data.unsqueeze(1)：将 (N,) 变为 (N,1)，每行一个正确标签索引
        # self.confidence (如 0.9) 被写入每个正确标签的位置
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        # ── 步骤 3：处理 padding 位置 ──
        # padding 位置的概率应全为 0（模型不需要预测 padding）
        # 将整个 padding_idx 列设为 0
        true_dist[:, self.padding_idx] = 0
        
        # 处理 target 本身就是 padding 的情况
        # 如果某个位置的正确标签恰好是 padding（句子结束后的填充），
        # 该位置的全部概率都应设为 0（表示"不用预测"）
        mask = torch.nonzero(target.data == self.padding_idx)
        # mask.dim() > 0：确实存在 target 为 padding 的位置
        if mask.dim() > 0:
            # index_fill_(dim, index, val)：将指定行所有列设为 val
            # 把那些 target 为 padding 的行的所有概率设为 0
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        
        # 保存构建的分布（用于可视化）
        self.true_dist = true_dist
        
        # ── 步骤 4：计算 KL 散度 ──
        # self.criterion(x, true_dist.clone().detach())
        # x: 模型的 log 概率（有梯度）
        # true_dist: 平滑后的真实分布（.detach() 切断梯度，不反向传播到标签）
        # KLDivLoss 计算：loss = sum(y_true * (log(y_true) - y_pred))
        #   由于输入 x 已经是 log 概率，所以实际上计算的是加权交叉熵
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
```

# A First Example (基础示例：简单的 Copy 复制任务)

> We can begin by trying out a simple copy-task. Given a random set
> of input symbols from a small vocabulary, the goal is to generate
> back those same symbols.
>
> **【中文对照 / Chinese Translation】**
> 我们首先尝试一个简单的复制任务（Copy-Task）。给定从小词表中随机抽取的一组输入符号，目标是原样生成输出这些相同的符号。

## Synthetic Data (合成数据集生成)

```python
# ============================================================================
# data_gen：为复制任务生成合成数据
# ============================================================================
# 通俗理解：这个函数是一个"数据工厂"，每次生产一批训练数据。
# 复制任务（copy task）是最简单的序列到序列任务：输入什么就输出什么。
# 比如输入 [1,5,3,8,...] → 输出 [1,5,3,8,...]（完全复制）
# 这是一个很好的测试用例——如果模型连复制都学不会，那更复杂的翻译也学不会。

def data_gen(V, batch_size, nbatches):
    """为复制任务（src-tgt copy task）生成随机的输入与目标数据。
    
    【新手补充详解】
    这个生成器的设计特点：
    1. 使用 Python 生成器（yield 关键字），每次调用产生一个 Batch
       → 不需要一次性把所有数据加载到内存，节省显存
    2. 源序列和目标序列完全相同（复制任务）
    3. 第一个 token 固定为 1（模拟起始符号 <s>）
    4. 其余 token 从 [1, V) 中随机选取
    
    Args:
        V: 词表大小（token 的取值范围是 [1, V-1]）
        batch_size: 每个 batch 包含的句子数
        nbatches: 总共生成多少个 batch
    
    Yields:
        Batch: 包含源序列和目标序列（相同）的批处理对象
    """
    for i in range(nbatches):
        # torch.randint(low, high, size)：生成 [low, high) 范围内的随机整数
        # 这里生成 (batch_size, 10) 的矩阵
        # 10 表示每个句子固定 10 个 token
        # 范围 [1, V)：排除 0（通常 0 是 padding 标记）
        data = torch.randint(1, V, size=(batch_size, 10))
        
        # 将每行的第一个 token 统一设为 1（模拟起始符号 <s>）
        # data[:, 0] 表示所有行的第 0 列（Python 索引从 0 开始）
        data[:, 0] = 1
        
        # requires_grad_(False)：确保这些数据不需要计算梯度
        # 输入数据不是模型参数，不需要反向传播
        # 这可以节省内存（PyTorch 不会为不需要梯度的张量构建计算图）
        src = data.requires_grad_(False)  # 源语言 = 原始数据
        tgt = data.requires_grad_(False)  # 目标语言 = 原始数据（复制任务，所以相同）
        
        # yield：生成器语法，返回一个 Batch 对象后暂停，
        # 下次调用时从 yield 之后继续执行
        # pad=0 表示用 0 作为 padding token ID
        yield Batch(src, tgt, 0)
```

## Loss Computation (损失计算)

```python
# ============================================================================
# SimpleLossCompute：损失计算包装器
# ============================================================================
# 通俗理解：把"生成概率"和"计算损失"两步打包在一起，简化训练循环的代码。
# 实现了 __call__ 方法，所以它的实例可以像函数一样被调用。

class SimpleLossCompute:
    """简单损失计算与训练更新包装器
    
    【新手补充详解】
    为什么要把 generator 和 criterion 打包？
    1. 训练循环中的损失计算步骤是固定的：generator → criterion
    2. 封装后，run_epoch 只需要调用 loss_compute(out, tgt_y, ntokens)
    3. 这个设计遵循"单一职责"原则——每个类只做一件事
    
    实现了 __call__ 方法意味着什么？
    - Python 中，obj() 会调用 obj.__call__(...)
    - 这让 SimpleLossCompute 实例可以当函数用
    - 在 run_epoch 中：loss, loss_node = loss_compute(out, ...)
    """

    def __init__(self, generator, criterion):
        """
        Args:
            generator: Generator 实例（线性层 + log_softmax）
            criterion: 损失函数（如 LabelSmoothing 或 nn.NLLLoss）
        """
        self.generator = generator      # 概率生成器
        self.criterion = criterion      # 损失计算标准

    def __call__(self, x, y, norm):
        """计算归一化后的损失
        
        Args:
            x: 解码器输出，形状 (batch_size, tgt_seq_len, d_model)
            y: 真实标签，形状 (batch_size, tgt_seq_len)
               每个元素是正确 token 的 ID
            norm: 归一化因子（通常是 batch 中的有效 token 数量）
                  损失除以 norm 后得到"每个 token 的平均损失"
        
        Returns:
            tuple: (显示用的损失值, 反向传播用的损失节点)
                - 第一个元素：Python 标量（float），用于打印和统计
                  sloss.data * norm 还原为未归一化的总损失
                - 第二个元素：PyTorch 张量，用于 loss.backward()
                  sloss 是归一化后的损失，保留计算图
        
        为什么返回两个损失？
        - sloss（带梯度）：用于反向传播，计算梯度
        - sloss.data * norm（无梯度标量）：用于显示总损失
          "总损失"比"每 token 平均损失"更直观地反映模型表现
        """
        # ── 步骤 1：生成概率分布 ──
        # self.generator(x)：将解码器输出映射为 log 概率
        # x 形状变化：(batch, tgt_seq_len, d_model) → (batch, tgt_seq_len, vocab_size)
        x = self.generator(x)
        
        # ── 步骤 2：计算损失 ──
        # x.contiguous().view(-1, x.size(-1)):
        #   将 (batch, tgt_seq_len, vocab_size) 展平为 (batch*tgt_seq_len, vocab_size)
        #   .contiguous() 确保内存连续（view 操作的前提条件）
        #   -1 告诉 PyTorch 自动计算这一维的大小
        #
        # y.contiguous().view(-1):
        #   将 (batch, tgt_seq_len) 展平为 (batch*tgt_seq_len,)
        #   每个元素是一个正确的 token ID
        #
        # 为什么需要展平？
        # 损失函数期望输入形状为 (N, C) 和 (N,)
        # N=总 token 数，C=词表大小
        # 展平后每个 token 独立计算损失，不区分属于哪个句子
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)),  # 预测：(总token数, 词表大小)
                y.contiguous().view(-1)                # 目标：(总token数,)
            )
            / norm  # 除以有效 token 数，得到平均损失
        )
        
        # ── 返回两个值 ──
        # sloss.data：取张量的纯数据部分（不含计算图），Python 标量
        # sloss.data * norm：还原为未归一化的总损失
        # sloss：保留了梯度信息，供后续 .backward() 使用
        return sloss.data * norm, sloss
```

## Greedy Decoding (贪婪解码算法)

```python
# ============================================================================
# greedy_decode：贪心解码——推理时逐词生成译文
# ============================================================================
# 通俗理解：贪心解码就是"每次都选当前最可能的词"。
# 就像一个小朋友写作文：每写一个字，都选此刻觉得最合适的那个字。
# 优点：简单快速；缺点：可能错过全局最优（局部最优≠全局最优）

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """使用贪心策略从模型生成输出序列
    
    算法流程：
    1. 编码源序列一次（编码器只运行一次）
    2. 从起始符号开始，循环以下步骤：
       a. 用当前已生成的序列解码
       b. 取最后一个位置的输出，通过 generator 得到词表概率
       c. 选概率最大的词
       d. 将这个词追加到输出序列
    3. 重复直到达到最大长度
    
    Args:
        model: 完整的 EncoderDecoder 模型
        src: 源语言输入，形状 (1, src_seq_len)  # batch_size 为 1
        src_mask: 源语言掩码，形状 (1, 1, src_seq_len)
        max_len: 最大生成长度
        start_symbol: 起始符号的 token ID（用于初始化输出序列）
    
    Returns:
        Tensor: 生成的完整序列，形状 (1, max_len)
    """
    # ── 步骤 1：编码源序列 ──
    # 编码器只运行一次（不管要生成多少个目标词）
    # memory 形状：(1, src_seq_len, 512)
    memory = model.encode(src, src_mask)
    
    # ── 步骤 2：初始化输出序列 ──
    # 从起始符号开始（如 <s>），形状 (1, 1)
    # fill_(start_symbol)：将所有元素填充为起始符号
    # type_as(src.data)：确保与 src 在同一设备（CPU/GPU）且同类型
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    
    # ── 步骤 3：自回归循环生成 ──
    # max_len - 1：因为初始序列已有一个 token
    for i in range(max_len - 1):
        # 3a. 解码：用当前已生成的序列和编码器输出进行解码
        # subsequent_mask(ys.size(1))：生成因果掩码，确保不出轨（不看未来）
        # out 形状：(1, current_len, 512)
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        
        # 3b. 生成下一个词的概率
        # out[:, -1]：只取最后一个位置的输出（因为我们只需要预测"下一个词"）
        # 形状变化：(1, current_len, 512) → (1, 512)
        # generator 输出：(1, vocab_size) — 每个词的 log 概率
        prob = model.generator(out[:, -1])
        
        # 3c. 贪心选择：取 log 概率最大的词
        # torch.max(prob, dim=1)：沿词表维度（dim=1）找最大值
        # 返回 (max_values, max_indices)
        # _ 是最大值（不需要），next_word 是最大值的索引（即预测的词 ID）
        _, next_word = torch.max(prob, dim=1)
        # .data[0]：将 GPU 上的 1 元素张量转为 Python 整数
        next_word = next_word.data[0]
        
        # 3d. 追加到输出序列
        # torch.cat([ys, new_token], dim=1)：沿序列维度拼接
        # 序列从 (1, t) 变为 (1, t+1)
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    
    # 返回生成的完整序列（可能包含起始符号和后续生成的所有词）
    return ys


# ============================================================================
# example_simple_model：完整的训练+评估示例（复制任务）
# ============================================================================
# 通俗理解：这是整个 Transformer 的"烟感测试"（smoke test）。
# 用一个极小规模的任务（V=11 个词）跑一遍完整的训练流程，
# 验证模型是否能学会最简单的"复制"任务。

def example_simple_model():
    """小型复制任务的端到端训练示例
    
    配置：
    - 词表大小 V=11（只有 token 0~10）
    - 编码器/解码器各 N=2 层（而非论文的 6 层）
    - 训练 20 个 epoch，每个 epoch 20 个 batch，每 batch 30 个句子
    - 目标：输入 [1,2,3,...,10] → 输出 [1,2,3,...,10]
    """
    V = 11  # 词表大小（很小，仅用于测试）
    
    # ── 损失函数：无平滑的标签平滑（即标准交叉熵）──
    # smoothing=0.0 意味着不做标签平滑
    # padding_idx=0 表示 0 号 token 是 padding 标记
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    
    # ── 构建 2 层的小型 Transformer ──
    # make_model 的参数：(源词表, 目标词表, 层数, d_model, d_ff, 头数, dropout)
    # N=2 意味着编码器和解码器各只有 2 层（快速测试用）
    model = make_model(V, V, N=2)
    
    # ── Adam 优化器（论文配置）──
    # lr=0.5：基础学习率（会被 scheduler 动态调整）
    # betas=(0.9, 0.98)：Adam 的动量参数（论文设定）
    #   β1=0.9：一阶动量衰减率（近期梯度的权重）
    #   β2=0.98：二阶动量衰减率（近期梯度平方的权重，论文特殊值）
    # eps=1e-9：防止除以零的小常数
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    
    # ── 学习率调度器（warmup + 衰减）──
    # LambdaLR：用自定义函数控制学习率
    # lr_lambda 是一个函数，输入 step 输出缩放因子
    # model.src_embed[0] 是 nn.Sequential 的第一个模块（即 Embeddings）
    # model.src_embed[0].d_model = 512
    # warmup=400：前 400 步学习率线性增长（比论文的 4000 少，因为任务简单）
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    # ── 训练 20 个 epoch ──
    for epoch in range(20):
        # model.train()：切换到训练模式
        #   影响：Dropout 生效，BatchNorm（如有）使用 batch 统计量
        model.train()
        # data_gen(V, 30, 20)：生成 20 个 batch，每个 30 个句子
        # mode="train"：执行反向传播和参数更新
        run_epoch(
            data_gen(V, 30, 20),              # 训练数据生成器
            model,                              # 模型
            SimpleLossCompute(model.generator, criterion),  # 损失计算
            optimizer,                          # 优化器（用于更新参数）
            lr_scheduler,                       # 学习率调度器
            mode="train",                       # 训练模式
        )
        
        # model.eval()：切换到评估模式
        #   影响：Dropout 关闭，所有神经元参与计算
        model.eval()
        # 评估阶段：用 DummyOptimizer 和 DummyScheduler 占位
        # 因为评估时不需要更新参数和调整学习率
        run_epoch(
            data_gen(V, 30, 5),                # 5 个 batch 用于评估
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),                   # 伪优化器（不做更新）
            DummyScheduler(),                   # 伪调度器（不调学习率）
            mode="eval",                        # 评估模式（不做反向传播）
        )

    # ── 最终测试：用学到的模型做一次推理 ──
    model.eval()
    # 输入序列 [1,2,3,4,5,6,7,8,9,10]
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # 全 1 掩码（没有 padding）
    src_mask = torch.ones(1, 1, 10)
    # 贪心解码生成输出，看看模型学会了复制没有
    # 如果模型训练成功，输出应该接近 [1,2,3,...,10]
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
```

# Part 3: A Real World Example (第三部分：真实世界机器翻译示例)

> Now we consider a real-world example using the Multi30k
> German-English Translation task.
>
> **【中文对照 / Chinese Translation】**
> 现在我们考虑一个使用 Multi30k 德英翻译任务的真实示例。

## Data Loading (数据加载与预处理)

```python
# ============================================================================
# 数据加载函数：准备真实翻译任务所需的工具和数据
# ============================================================================

def load_tokenizers():
    """加载德语和英语的 spaCy 分词器
    
    【新手补充详解】
    什么是分词（Tokenization）？
    将文本切分为最小处理单元（token）的过程。
    例如："I love cats" → ["I", "love", "cats"]
    
    spaCy 是一个工业级的自然语言处理库，提供了：
    1. 分词（tokenization）：将句子切分为词
    2. 词性标注、命名实体识别等高级功能
    3. 多语言支持（本代码用了德语 de 和英语 en 模型）
    
    模型的命名规则：
    - de_core_news_sm：德语小型模型（sm = small），基于新闻语料
    - en_core_web_sm：英语小型模型，基于网络语料
    
    错误处理：
    如果模型未安装，自动通过 pip 下载后再加载
    （spaCy 模型是独立的 Python 包，需要单独安装）
    
    Returns:
        tuple: (德语分词器, 英语分词器) 或 (None, None) 如果 spaCy 不可用
    """
    if spacy is None:
        return None, None  # spaCy 未安装，优雅降级
    
    # ── 加载德语分词器 ──
    try:
        # spacy.load("模型名")：加载预训练的语言模型
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        # 如果模型文件不存在，通过 pip 自动下载
        # os.system() 在终端执行命令（这在生产代码中不推荐，这里仅用于演示）
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    # ── 加载英语分词器 ──
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    """使用指定分词器对文本进行分词
    
    Args:
        text: 原始文本字符串，如 "Ein Mann steht auf einer Leiter."
        tokenizer: spaCy 分词器（Language 对象）
    
    Returns:
        list[str]: 分词后的 token 列表
                   例如 ["Ein", "Mann", "steht", "auf", "einer", "Leiter", "."]
    
    工作原理：
    tokenizer(text) 返回一个 spaCy Doc 对象，包含所有 token
    tok.text 提取每个 token 的原始文本
    列表推导式 [tok.text for tok in ...] 收集所有 token 文本
    """
    return [tok.text for tok in tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    """从数据迭代器中逐句产出分好词后的 token 列表（用于构建词表）
    
    这是一个 Python 生成器（使用了 yield 关键字），
    每次只产出一个句子的 token 列表，而非一次性把所有数据加载到内存。
    
    Args:
        data_iter: 数据迭代器，每个元素是 (德语, 英语) 的元组
        tokenizer: 分词函数（如 tokenize_de 或 tokenize_en）
        index: 取元组中的哪个元素（0=德语，1=英语）
    
    Yields:
        list[str]: 一个句子的 token 列表
    
    示例：
        如果 data_iter 的第一个元素是 ("Ein Mann.", "A man.")
        tokenizer=tokenize_de, index=0
        → yield ["Ein", "Mann", "."]
    """
    for from_to_tuple in data_iter:
        # from_to_tuple[index] 取源语言或目标语言句子
        # tokenizer(...) 对这个句子分词
        yield tokenizer(from_to_tuple[index])
```

## Iterators (数据迭代器构造)

```python
# ============================================================================
# build_vocabulary：构建源语言和目标语言的词表
# ============================================================================
# 通俗理解：词表（Vocabulary）就是把"单词"映射到"数字 ID"的字典。
# 模型只能处理数字，所以需要把每个词（或子词）分配一个唯一的编号。
# 同时还需要一些特殊标记来处理边界情况。

def build_vocabulary(spacy_de, spacy_en):
    """从 Multi30k 训练数据中构建德语和英语词表
    
    【新手补充详解】
    词表中的特殊标记（Special Tokens）：
    - <blank> (ID=0): 空白/填充标记，也用作句子起始
    - <unk>  (ID=1): 未知词标记（Unknown），遇到词表中没有的词时用它代替
    - <s>    (ID=2): 句子起始标记（Start of Sentence）
    - </s>   (ID=3): 句子结束标记（End of Sentence）
    
    这些特殊标记在序列转换任务中扮演重要角色：
    - <s> 和 </s> 告诉模型句子的边界
    - <unk> 处理词表外的词（OOV, Out-Of-Vocabulary）
    - <blank> 用于批处理时对齐不同长度的句子
    
    Multi30k 数据集：
    一个包含约 30,000 对德英平行句子的多模态翻译数据集。
    每对包含德语原文和对应的英语译文。
    
    Args:
        spacy_de: 德语 spaCy 分词器
        spacy_en: 英语 spaCy 分词器
    
    Returns:
        tuple: (德语词表, 英语词表) 或 (None, None) 如果 torchtext 不可用
    """
    if torchtext is None:
        return None, None  # torchtext 未安装，优雅降级
    
    # ── 定义语言特定的分词函数（闭包）──
    # 这两个函数"捕获"了外部变量 spacy_de/spacy_en
    # 这样传给 build_vocab_from_iterator 时就很简洁
    def tokenize_de(text):
        """德语分词器包装"""
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        """英语分词器包装"""
        return tokenize(text, spacy_en)

    # ── 加载 Multi30k 数据集 ──
    print("Building German Vocabulary...")
    # Multi30k(split=("train", "val", "test"))：加载训练/验证/测试三个子集
    # to_map_style_dataset()：将迭代式数据集转为支持索引访问的形式
    #   Map 风格数据集可以多次遍历，支持随机访问
    #   train 包含约 29,000 个训练样本
    train, val, test = to_map_style_dataset(
        torchtext.datasets.Multi30k(split=("train", "val", "test"))
    )
    
    # ── 构建德语词表 ──
    # build_vocab_from_iterator 的原理：
    # 1. 遍历 yield_tokens 产生的所有 token 列表
    # 2. 统计每个 token 出现的频率
    # 3. 只保留出现频率 ≥ min_freq 的 token
    # 4. 加上特殊标记（放在最前面，ID 从 0 开始）
    # 5. 按频率降序排列普通 token
    #
    # 结果示例：
    #   {"<blank>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "the": 4, "a": 5, ...}
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train, tokenize_de, index=0),  # 德语（index=0）的所有 token
        min_freq=2,  # 只保留至少出现 2 次的词（过滤掉生僻词）
        specials=["<blank>", "<unk>", "<s>", "</s>"],  # 特殊标记，固定在最前面
    )

    # ── 构建英语词表（同理）──
    print("Building English Vocabulary...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train, tokenize_en, index=1),  # 英语（index=1）的所有 token
        min_freq=2,
        specials=["<blank>", "<unk>", "<s>", "</s>"],
    )

    # ── 设置默认索引 ──
    # 当查询一个不在词表中的词时，返回 <unk> 的索引
    # 例如：vocab_src["some_unknown_word"] → 返回 1（即 <unk> 的 ID）
    # 这样即使遇到训练时没见过的词，模型也不会崩溃
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt
```

## Training the System (训练完整系统)

# Additional Components: BPE, Search, Averaging (拓展组件：BPE、束搜索与模型平均)

> 1) BPE / Word-piece (子词分词)
> 2) Shared Embeddings (共享嵌入权重)
> 3) Beam Search (束搜索)
> 4) Model Averaging (模型平均)
>
> **【中文对照 / Chinese Translation】**
> 在实际部署和工业级 Transformer 模型中，通常还会结合 BPE 子词切分、源与目标端 Embedding 权重共享、Beam Search 束搜索解码以及多 Checkpoint 模型权重平均等技术手段。

# Results (实验结果与可视化)

On the WMT 2014 English-to-German translation task, the big
transformer model outperforms previously reported models.

**【中文对照 / Chinese Translation】**
在 WMT 2014 英德翻译任务上，大号 Transformer 模型展现出了超越此前诸多模型的优秀性能。

## Attention Visualization (注意力机制权重可视化)

# Conclusion (结语与总结)

Hopefully this code is useful for future research.

**【中文对照 / Chinese Translation】**
希望这份代码对大家未来的研究与学习有所帮助！
