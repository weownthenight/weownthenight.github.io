---
layout: wiki

title: LaTeX基本用法

---

参考：一份不太简短的$\LaTeX$介绍

## 文字

### 斜体

斜体加上\emph就可以：

```latex
\emph{Social Network Fake Account Dataset}
```

### 加粗

加粗用\bf:

```latex
\bf{0.9}
```

### 链接

用\url:

```latex
\url{https://colab.research.google.com/}
```

### 脚注

用\footnote

```latex
\footnote{\url{https://colab.research.google.com/}}
```

## 数学公式

公式可以参考之前写过的[Tex数学公式常用打印命令](https://weownthenight.github.io/wiki/TeX%E6%95%B0%E5%AD%A6%E5%85%AC%E5%BC%8F%E5%B8%B8%E7%94%A8%E6%89%93%E5%8D%B0%E5%91%BD%E4%BB%A4/)。写法如下：

```latex
\begin{equation}
x_{rcnn-out} = MaxPool(x_{intermediate}) \in R^h
\end{equation}
```

## 表格

在科研论文排版中广泛应用的表格形式是三线表，由booktabs宏包支持，提供了\toprule, \midrule, \bottomrule命令用以排版三线表的三条线，以及\cmidrule用于绘制跨越部分单元格的横线。除此之外，最好不要用其他横线以及竖线。例子如下：

```latex
\begin{table}
  \caption{Confusion matrix}
  \centering
  \begin{tabular}{llll}
    \toprule
    && \multicolumn{2}{l}{Predicted}\\
            && Fake  &Legitimate \\
    \midrule
    \multirow{2}{*}{Actual}
    & Fake    &TP &FN \\
    & Legitimate  &FP &TN \\
    \bottomrule
  \end{tabular}
\end{table}
```

显示效果如下：

![image-20211027152852617](images/posts/2021102801.png)

其中：&表示空格，\multirow需要包含multirow宏包支持。

一个更复杂的例子：

```latex
\begin{table}
  \caption{Experiment results}
  \centering
  \begin{tabular}{llllll}
    \toprule
    &Model   & Accuracy  & Precision  & Recall    &F1-score \\
    \midrule
    \multirow{4}{*}{Dataset A}
    &BERT   &0.9173 &0.0000 &0.0000 &0.0000 \\
    &BERT + RNN &0.9173 &0.0000 &0.0000 &0.0000 \\
    &BERT + CNN &0.9094 &0.4699 &{\bf 0.7449} &0.5762 \\
    &BERT + RCNN    &{\bf 0.9446} &{\bf 0.6578} &0.6880 &{\bf 0.6726} \\
    \midrule
    \multirow{4}{*}{Dataset B}
    &BERT   &0.9544 &{\bf 0.7781} &0.6726 &0.6948 \\
    &BERT + RNN &0.9173 &0.0000 &0.0000 &0.0000 \\
    &BERT + CNN &0.9544 &0.7654 &0.6469 &0.7012 \\
    &BERT + RCNN    &{\bf 0.9555} &0.7500 &{\bf 0.6929} &{\bf 0.7203} \\
    \midrule
    \multirow{4}{*}{Dataset C}
    &BERT   &{\bf 0.9592} &0.7682 &0.7255 &{\bf 0.7463} \\
    &BERT + RNN &0.9564 &{\bf 0.7871} &0.6481 &0.7109 \\
    &BERT + CNN &0.9562 &0.7254 &{\bf 0.7570} &0.7408 \\
    &BERT + RCNN    &0.9565 &0.7707 &0.6747 &0.7195 \\
    \midrule
    \multirow{4}{*}{Dataset D}
    &BERT    &0.9595 & {\bf 0.8067*} & 0.6711  & 0.7327  \\
    &BERT + RNN    &0.9594   & 0.7688    &0.7279 &0.7478 \\
    &BERT + CNN    &{\bf 0.9617*}   & 0.7898    &0.7316 &0.7596 \\
    &BERT + RCNN   &0.9613   &0.7576    &{\bf 0.7823*} &{\bf 0.7698*} \\
    \bottomrule
  \end{tabular}
\end{table}
```

显示效果如下：

![image-20211027153822839](images/posts/2021102802.png)

## 插入图片

想要插入图片需要添加宏包graphicx。例子如下：

```latex
\begin{figure}
    \centering
    \includegraphics[scale=0.9]{fig4.png}
    \caption{Overview of neural network model architecture }
\end{figure}
```

当图片过大或过小时，可以设置scale来调整缩放比例。







