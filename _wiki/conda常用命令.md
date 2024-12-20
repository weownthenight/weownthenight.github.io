---
layout: wiki

title: conda常用命令

---

参考：[Conda常用命令整理](https://blog.csdn.net/menc15/article/details/71477949)

## help！

查看帮助：

```bash
conda --help
```

查看某个命令的帮助（比如conda remove）：

```bash
conda remove --help
```

查看环境管理的全部命令帮助：

```bash
conda env --help
```

## 环境

列举所有环境：

```bash
conda env list
```

进入某个环境（比如local_nmt）：

```bash
conda activate local_nmt
```

退出当前环境：

```bash
conda deactivate
```

创建新的环境（比如mp1）：

```bash
conda create --name mp1
```

删除某个环境（比如tf1.12）：

```bash
conda remove --name tf1.12 --all
```

## 包

列举当前活跃环境下的所有包：

```bash
conda list
```

列举指定环境下的所有包：

```bash
conda list -n your_env_name
```

这两个也可以不用指令查看，而是直接在PyCharm的菜单中看到。

在当前环境下安装某个包：

```bash
conda install package_name
```

在指定环境下安装某个包：

```bash
conda instll -n env_name package_name
```

有很多包不能从conda安装，可以用pip直接安装。