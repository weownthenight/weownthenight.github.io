---
layout: post

title:  The Missing Semester of CS：The Shell

categories: shell bash

description: shell编程

---

这门MIT课很实用地介绍了CS学习过程中实用的工具用法等等。🔗:[课程网站](https://missing.csail.mit.edu/)。我把这其中平时会经常用到但是我不太会的部分按照课程专题Topic记录下来，可以随时查阅：

与各种图形界面相对应，纯文字的interface就叫作shell。我们以bash(Bourne Again SHell)为例来介绍shell的用法。

## Basic

![image-20220830141513137](/images/posts/image-20220830141513137.png)

- 参数写法：

  比如一个名叫"My Photos"的文件夹名，可以写为:

  - 'My Photos'
  - "My Photos"
  - My\ Photos

- PATH

  shell在运行命令的时候会寻找相应的PATH，按照顺序优先选择对应的PATH：

  ![image-20220830141913478](/images/posts/image-20220830141913478.png)

- 文件权限

  ![image-20220830142110274](/images/posts/image-20220830142110274.png)

  - `d`：directory，表示是一个文件夹。
  - 剩下的权限三个字母一组，分别表示 the owner of the file(`missing`)，the owning group(`users`)和所有其他用户的权限。
  - `r`: read.对于目录，有read权限才能`ls`。
  - `w`：write.包括move, delete, rename......
  - `x`：execute. 对于目录，有execute权限才能`cd`。

- streams

  - `cat`：cat指con**cat**enate，it prints contents from its input stream to its output stream. 默认output是屏幕。

  - `<`: rewrite input; `>`: rewrite output; `>>`: append to a file.

    ![image-20220830143135676](/images/posts/image-20220830143135676.png)

  - pipes

    `|`："chain" programs such that the output of one is the input of another.

## 进阶

从这些命令再进阶，就是写shell脚本啦！接下来以bash scripting为例：

### 变量

‼️bash脚本中的空格是用来分隔参数的，不要随便空格！比如，当你写`foo = bar`时表示`foo`是一个程序名（类似`ls`），`=`是传入`foo`程序的第一个参数，`bar`是传入`foo`程序的第二个参数。

‼️单引号和双引号完全不同！单引号表示的是string，双引号既能表示string，也能代入变量。

变量可以用`$foo`或者`"$foo"`表示。

![image-20220830144911494](/images/posts/image-20220830144911494.png)

### 写文件

将命令写到函数里，放在文件里：

![image-20220830150437475](/images/posts/image-20220830150437475.png)

执行：

![image-20220830150933288](/images/posts/image-20220830150933288.png)

### 参数

bash中有很多预留的参数，可以通过🔗[Special Characters](https://tldp.org/LDP/abs/html/special-chars.html)查看。比较实用的：

- `$0`: name of the script.

- `$1`到`$9`：参数

- `!!`：last command，用法比如（`sudo !!`)：

  ![image-20220830151525389](/images/posts/image-20220830151525389.png)

- `$_`: last argument

- `$?`: return code of the previous command. 如果没有错误，return就是0。例子如下：

  在mcd.sh中没有foobar，所以return code是1。true的error code是0，false的error code是1。

  ![image-20220830151913475](/images/posts/image-20220830151913475.png)

### 布尔运算 

分号隔绝了两个command，所以不管前一个命令是true和false，后面都一样执行。

![image-20220830152106420](/images/posts/image-20220830152106420.png)

### command substitution

将命令执行的结果作为输入。比如：

![image-20220831100317178](/images/posts/image-20220831100317178.png)

### process substitution

比如：

![image-20220831100935344](/images/posts/image-20220831100935344.png)

执行的效果就是concatenate `ls`和`ls ..`的命令结果。以`<(ls)`为例，讲解process substition做了什么：

- 执行`ls`
- 将`ls`的结果写入一个temporary file
- 将`<(ls)`替换为temporary file的文件名

### 一个综合例子

![image-20220831102939949](/images/posts/image-20220831102939949.png)

- bash中的比较可以查看`man test`，比如上面程序中的`-ne`。为了保证bash能和`sh`兼容，比较的时候用`[[ ]]`而不是`[ ]`。

- 执行上述脚本：

  ![image-20220831103054676](/images/posts/image-20220831103054676.png)

### globbing

![image-20220901192350074](/images/posts/image-20220901192350074.png)

### Other scripts

- 🔗[shellcheck](https://github.com/koalaman/shellcheck)。可以用来找出你的sh/bash脚本的错误。

- Python script

  ![image-20220901193047240](/images/posts/image-20220901193047240.png)

## Shell Tools

### Finding how to use commands

1. `man`

2. 🔗[TLDR pages](https://tldr.sh). 比`man`更具体，会有例子说明。

### Finding files

1. `find`

   ![image-20220901194146401](/images/posts/image-20220901194146401.png)

   `-name`是区分大小写的，`-iname`不区分大小写。

   ![image-20220901194204816](/images/posts/image-20220901194204816.png)

2. `fd`。可以用正则表达式，比find更简洁，直接`fd PATTERN`就行。🔗[fd](https://github.com/sharkdp/fd)

3. `locate`。`locate`属于Unix System原生的设计，使用了一个数据库build index，在进行查找时更快，数据库更新需要`updatedb`。

### Finding code

当比起文件，我们更在乎文件内容时，我们可以用`grep`。  

1. `grep`

   ```bash
   grep foobar mcd.sh
   grep -R foobar
   # C表示content, grep -C 5在查找返回时保留上下5行内容
   grep -C 5
   # v表示inverse，这里是反向查找，查找不包含pattern的内容
   grep -v pattern
   ```

2. `rg`（ripgrep)

   🔗[rg](https://github.com/BurntSushi/ripgrep)

   跟`grep`相比，有颜色显示，unicode支持，比`grep`更快。

   ```bash
   # 找到~/scratch下所有包含"important requests"的python文件
   rg "important requests" -t py ~/scratch
   # 找到后要上下5行的context
   rg "important requests" -t py -C 5 ~/scratch
   # 找到所有不以#!(shebang line)开头的sh文件，-u表示不要忽略隐藏文件
   rg -u --files-without-match "^#\!" -t sh
   # --stats会给出match的数据，有多少行match，多少文件match等等
   rg "important request" -t py -C 5 --stats ~/scratch
   # -A指的是following lines
   rg foo -A 5
   ```

3. `ack`

   🔗[ack](https://github.com/beyondgrep/ack3)

4. `ag`

   🔗[ag](https://github.com/ggreer/the_silver_searcher)

### Finding shell commands

在历史中找到自己曾经输入的命令。

1. `history`

   ```bash
   # 在zsh只能print部分history
   history
   # print所有history
   history l
   # 找到所有find命令
   history l | grep find
   ```

2. `CRTL+R`

3. `fzf`(fuzzy finder)

   🔗[fzf](https://github.com/junegunn/fzf)。跟`grep`相比，你不需要写正则表达式，这种模糊查找很简单易用，可以跟`CRTL+R` binding，操作简捷。

   ```bash
   cat example.sh | fzf
   ```

4. history-based autosuggestions

   可以进行命令自动补全，在zsh上设置：🔗[zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions)

### Directory Navigation

```bash
# list some directory structure
ls -R
# 比-R更友好
tree
broot
# more os like, interactive input
nnn
```

1. `broot`

   🔗[broot](https://github.com/Canop/broot)

2. `nnn`

   🔗[nnn](https://missing.csail.mit.edu/2020/shell-tools/)

我们可以用`cd`切换目录，但是也可以选择更快捷的方式去我们经常去的目录下。

1. `autojump`

   🔗[autojump](https://github.com/wting/autojump)

2. `fasd`

   🔗[fasd](https://github.com/clvv/fasd)