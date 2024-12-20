---

layout: wiki
title: Tex数学公式常用打印命令
categories: 备忘 Tex
description: LaTeX数学公式打印

---

### 1、分数表示，根号表示：

| 表达式（行内公式需要加上$$） | 表示形式            |
| :--------------------------- | ------------------- |
| \frac{分子}{分母}            | $\frac{分子}{分母}$ |
| \sqrt{X}                     | $\sqrt{X}$          |
| \sqrt[3]{X}                  | $\sqrt[3]{X}$       |

### 2、不等号

| 表达式（行内公式需要加上$$） | 表示形式 |
| ---------------------------- | -------- |
| \leq                         | $\leq$   |
| \geq                         | $\geq$   |
| \approx                      | $\approx$ |

### 3、希腊字母和其他特殊字母

| 表达式（行内公式需要加上$$） | 表示形式   |
| ---------------------------- | ---------- |
| \alpha                       | $\alpha$   |
| \beta                        | $\beta$    |
| \gamma                       | $\gamma$   |
| \epsilon                     | $\epsilon$ |
| \sigma                       | $\sigma$   |
| \Sigma                       | $\Sigma$   |
| \tau                         | $\tau$     |
| \delta                       | $\delta$   |
| \Delta                       | $\Delta$   |
| \pi                          | $\pi$      |
| \varphi                      | $\varphi$  |
| \theta                       | $\theta$   |
| \lambda                      | $\lambda$  |
| \rho                         | $\rho$     |
| \Lambda                      | $\Lambda$  |
| \Omega                       | $\Omega$   |
| \mu                          | $\mu$      |
| \eta                         | $\eta$     |
| \kappa                       | $\kappa$   |
| \mathbb{E}                   | $\mathbb{E}$ |

### 4、上标和下标

| 表达式（行内公式需要加上$$） | 表示形式 |
| ---------------------------- | -------- |
| a^b                          | $a^b$    |
| a_b                          | $a_b$    |

### 5、求和公式与连乘公式

| 表达式（行内公式需要加上$$）                                 | 表示形式                       |
| ------------------------------------------------------------ | ------------------------------ |
| 连加：\sum_{下标}^{上标}                                     | $\sum_{下标}^{上标}$           |
| 连乘：\prod_{下标}^{上标}                                    | $\prod_{下标}^{上标}$          |
| 想要与手写完全相同，在行内公式需要加上\displaystyle: <br />例如：\displaystyle\sum{i=1}^{n}i | $\displaystyle\sum_{i=1}^{n}i$ |
| \nabla                       | $\nabla$                                               |

### 6、微积分符号

| 表达式（行内公式需要加上$$）                                 | 表示形式                                                 |
| ------------------------------------------------------------ | -------------------------------------------------------- |
| \infty                                                       | $\infty$                                                 |
| \to                                                          | $\to$                                                    |
| \lim                                                         | $\lim$                                                   |
| 需要 $n\to\infty$ 在lim之下时需要加上\limits：<br />例如：\lim\limits_{n\to\infty} | $\lim\limits_{n\to\infty}$                               |
| \f'(x)                                                       | $f'(x)$                                                  |
| \sim                                                         | $\sim$                                                   |
| \int                                                         | $\int$                                                   |
| \lim\limits_{\substack{x\to x_0\\\y\to y_0}}f(x,y)=A         | $\lim\limits_{\substack{x\to x_0 \\\\y\to y_0}}f(x,y)=A$ |
| \partial                                                     | $\partial$                                               |
| \iint                                                        | $\iint$                                                  |

### 7、矩阵与行列式

| 表达式（行内公式需要加上$$）            | 表示形式                                  |
| --------------------------------------- | ----------------------------------------- |
| \\begin{vmatrix}1&1\\\1&1\end {vmatrix} | $\begin{vmatrix}1&1 \\\\1&1\end{vmatrix}$ |
| \\begin{pmatrix}1&0\\\0&1\end {pmatrix} | $\begin{pmatrix}1&0 \\\\0&1\end{pmatrix}$ |

### 8、括号

| 表达式（行内公式需要加上$$）              | 表示形式                                   |
| ----------------------------------------- | ------------------------------------------ |
| \\begin{cases}1&x>0\\\\-1&x<0\end {cases} | $\begin{cases}1&x>0 \\\\-1&x<0\end{cases}$ |
|  \left[   \right]                         | $\left[   \right]$                         |

### 9、集合与逻辑符号

| 表达式（行内公式需要加上$$） | 表示形式       |
| ---------------------------- | -------------- |
| \because                     | $\because$     |
| \therefore                   | $\therefore$   |
| \iff                         | $\iff$         |
| \cap                         | $\cap$         |
| \cup                         | $\cup$         |
| \Rightarrow                  | $\Rightarrow$  |
| \nRightarrow                 | $\nRightarrow$ |
| \varnothing                  | $\varnothing$  |
| \in                          | $\in$          |
| \forall                      | $\forall$      |

### 10、其他符号

| 表达式（行内公式需要加上$$） | 表示形式        |
| ---------------------------- | --------------- |
| \cdot                        | $\cdot$         |
| \pm                          | $\pm$           |
| \quad（空格）                | $\quad$         |
| \times                       | $\times$        |
| \neq                         | $\neq$          |
| \dots                        | $\dots$         |
| \vdots                       | $\vdots$        |
| \ddots                       | $\ddots$        |
| \overline{x}                 | $\overline{x}$  |
| \underline{x}                | $\underline{x}$ |
| \circ                        | $\circ$         |
| \tag{1}（给公式编号）          | $\tag{1}$       |
| \pm                          | $\pm$           |

### 11、复杂表达式应用举例

| **表达式（行内公式需要加上$$）**                             | 表示形式                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| \displaystyle\frac{\partial(\frac{\partial z}{\partial x})}{\partial x}=\frac{\partial ^2z}{\partial x^2} | $\displaystyle\frac{\partial(\frac{\partial z}{\partial x})}{\partial x}=\frac{\partial ^2z}{\partial x^2}$ |
| V=\iint\limits_{D}f(x,y)d\sigma                              | $V=\iint\limits_{D}f(x,y)d\sigma$                            |
| \\begin{vmatrix}........\\\a_{i1}+ b_{i1} &a_{i2}+b_{i2},...,a_{in}+b_{in}\\\\......\end {vmatrix} | $\begin{vmatrix}........ \\\\a_{i1}+b_{i1} &a_{i2}+b_{i2},...,a_{in}+b_{in} \\\\......\end {vmatrix}$ |
| \\begin{vmatrix}a_{11}\\\a_{21}&a_{22}\\\ \vdots&\vdots&\ddots \\\a_{n1}&a_{n2}&\dots&a_{nn}\end {vmatrix} | $\begin{vmatrix}a_{11} \\\\a_{21}&a_{22} \\\\\vdots&\vdots&\ddots  \\\\a_{n1}&a_{n2}&\dots&a_{nn}\end{vmatrix}$ |

