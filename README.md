# The Annotated Transformer：中英文对照与代码详解

本仓库是在 Harvard NLP 的 [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) 基础上整理的独立学习版本，保留英文原文，并补充中文对照、术语解释和更细致的代码解析，方便中文读者系统理解 Transformer 的模型结构、训练流程与推理实现。

This is an independent bilingual study edition based on Harvard NLP's [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). It preserves the original English text while adding Chinese translations, terminology notes, and detailed code explanations.

## 主要内容 / Highlights

- 英文原文与中文翻译逐段对照
- Transformer 核心模块的详细代码注释
- 模型架构、训练、损失计算、贪心解码和注意力可视化解析
- 可直接阅读的 Notebook、Python 源码与导出的 HTML

## 阅读入口 / Start Here

- [`AnnotatedTransformer.ipynb`](AnnotatedTransformer.ipynb)：推荐使用的交互式 Notebook
- [`AnnotatedTransformer.html`](AnnotatedTransformer.html)：无需配置环境即可阅读的静态页面
- [`AnnotatedTransformer.py`](AnnotatedTransformer.py)：带详细解析的 Python 版本
- [`the_annotated_transformer.py`](the_annotated_transformer.py)：与 Jupytext 工作流兼容的源码

## 来源与许可 / Attribution

原项目由 Harvard NLP 发布：[harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer)。本仓库保留原作者署名，并在原项目 MIT License 条款下发布；中文翻译、术语说明与新增代码解析由本仓库维护。

The original project is [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer). Original attribution is retained and this repository remains under the MIT License. The Chinese translations, terminology notes, and added code explanations are maintained here.

---

## Original Project Instructions

Code for The Annotated Transformer blog post:

http://nlp.seas.harvard.edu/annotated-transformer/

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yi-luo-hua/annotated-transformer/blob/master/AnnotatedTransformer.ipynb)

![image](https://user-images.githubusercontent.com/35882/166251887-9da909a9-660b-45a9-ae72-0aae89fb38d4.png)




# Package Dependencies

Use `requirements.txt` to install library dependencies with pip:

```
pip install -r requirements.txt
```


# Notebook Setup

The Annotated Transformer is created using [jupytext](https://github.com/mwouts/jupytext).

Regular notebooks pose problems for source control - cell outputs end up in the repo history and diffs between commits are difficult to examine. Using jupytext, there is a python script (`.py` file) that is automatically kept in sync with the notebook file by the jupytext plugin.

The committed Python script contains all cell content and can be used to generate the notebook file. The Python script is regular source code; Markdown sections use a standard comment convention, and outputs are not saved. This bilingual edition also commits the generated notebook and HTML files so they can be read directly.

Prior to using this repo, make sure jupytext is installed by following the [installation instructions here](https://github.com/mwouts/jupytext/blob/main/docs/install.md).

To produce the `.ipynb` notebook file using the markdown source, run (under the hood, the `notebook` build target simply runs `jupytext --to ipynb the_annotated_transformer.py`):

```
make notebook
```

To produce the html version of the notebook, run:

```
make html
```

`make html` is just a shortcut for for generating the notebook with `jupytext --to ipynb the_annotated_transformer.py` followed by using the jupyter nbconvert command to produce html using `jupyter nbconvert --to html the_annotated_transformer.ipynb`                             
 

# Formatting and Linting

To keep the code formatting clean, the annotated transformer git repo has a git action to check that the code conforms to [PEP8 coding standards](https://www.python.org/dev/peps/pep-0008/).

To make this easier, there are two `Makefile` build targets to run automatic code formatting with black and flake8.

Be sure to [install black](https://github.com/psf/black#installation) and [flake8](https://flake8.pycqa.org/en/latest/).

You can then run:

```
make black
```

(or alternatively manually call black `black --line-length 79 the_annotated_transformer.py`) to format code automatically using black and:

```
make flake
```

(or manually call flake8 `flake8 --show-source the_annotated_transformer.py) to check for PEP8 violations.

It's recommended to run these two commands and fix any flake8 errors that arise, when submitting a PR, otherwise the github actions CI will report an error.
