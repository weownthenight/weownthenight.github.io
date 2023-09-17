---
layout: post
title: torch tensor的各种常见操作
categories: PyTorch
description: torch tensor的各种常见操作
---
## 创建tensor

### `torch.zeros`

### `torch.arange(start, end, step)`

```python
>>> torch.arange(5)
tensor([ 0,  1,  2,  3,  4])
>>> torch.arange(1, 4)
tensor([ 1,  2,  3])
>>> torch.arange(1, 2.5, 0.5)
tensor([ 1.0000,  1.5000,  2.0000])
```



## 减小维度

### `torch.squeeze()`

将维度为1的部分删掉。比如:

```python
>>> x = torch.zeros(2, 1, 2, 1, 2)
>>> x.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x)
>>> y.size()
torch.Size([2, 2, 2])
>>> y = torch.squeeze(x, 0)
>>> y.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x, 1)
>>> y.size()
torch.Size([2, 2, 1, 2])
>>> y = torch.squeeze(x, (1, 2, 3))
torch.Size([2, 2, 2])
```

## 增加维度

### `torch.unsqueeze(input, dim)`

和`torch.squeeze`为反操作，在dim维度加上一个维度：

```python
# x.shape: (4)
>>> x = torch.tensor([1, 2, 3, 4])
# after unsqueeze on dimension 0: (1, 4)
>>> torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
# after unsquezze on dimension 1: (4, 1)
>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])
```

### `torch.stack(tensors, dim=0, *, out=None)`

Concatenates a sequence of tensors along a new dimension. **All tensors need to be of the same size.**对于size不同的tensor可以使用cat。

## 改变维度

### `torch.cat(tensors, dim)`

```python
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
# after cat: (6,3)
>>> torch.cat((x, x, x), 0)
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497],
        [ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
# after cat: (2, 9)
>>> torch.cat((x, x, x), 1)
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
         -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
         -0.5790,  0.1497]])
```

### `torch.transpose(input, dim0, dim1)`

互换dim0和dim1的维度。

```python
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 1.0028, -0.9893,  0.5809],
        [-0.1669,  0.7299,  0.4942]])
# after transpose: (3, 2)
>>> torch.transpose(x, 0, 1)
tensor([[ 1.0028, -0.1669],
        [-0.9893,  0.7299],
        [ 0.5809,  0.4942]])
```

### `Tensor.expand(*sizes)`
把tensor的维度进行扩展。如果设为-1，表示这个维度不变。这个括号里的size是变化后的形状size。扩展就是按照维度来复制。举例如下：

```python
>>> x = torch.tensor([[1], [2], [3]])
>>> x.size()
torch.Size([3, 1])
>>> x.expand(3, 4)  # expand后形状从[3,1]变为了[3,4]
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
>>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]]) # expand后形状从[3,1]变为了[3,4]
```
也可以用expand来增加维度，增加的维度会放在最前面。举例如下：

```python
>>> x = torch.tensor([[1], [2], [3]])
>>> x.size()
torch.Size([3, 1])
>>> y = x.expand(2, 3, 4)
>>> y
tensor([[[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3]],

        [[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3]]])
>>> y.size()
torch.Size([2, 3, 4])
```

expand不会增加内存，只是在原先的tensor加了view。所以对expand后的tensor做操作需要小心，因为它本质上没有clone。

## 其他

### `torch.split(tensor, split_size, dim)`

将tensor拆成多个tensors，返回拆分后得到的tensor列表。split_size可以是一个列表，它会根据列表给出的size拆分。

```python
>>> a = torch.arange(10).reshape(5, 2)
# a.shape: (5, 2)
>>> a
tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]])
# split后得到列表，默认在dim=0分裂，得到的tensor有3个
>>> torch.split(a, 2)
(tensor([[0, 1],
         [2, 3]]),
 tensor([[4, 5],
         [6, 7]]),
 tensor([[8, 9]]))
# 得到的tensor列表有2个，分别是(1,2)和(4,2)
>>> torch.split(a, [1, 4])
(tensor([[0, 1]]),
 tensor([[2, 3],
         [4, 5],
         [6, 7],
         [8, 9]]))
```

### `torch.tile(input, dims)`

Constructs a tensor by repeating the elements of `input`. 可以用于重复tensor中的元素。dims来指定重复的次数和对应维度。

```python
>>> x = torch.tensor([1, 2, 3])
# dims必须是tuple，如果只是在一个维度上重复，参考这种写法，不要写(2)，会报错X
>>> x.tile((2,))
tensor([1, 2, 3, 1, 2, 3])
# y.shape: (2,2)
>>> y = torch.tensor([[1, 2], [3, 4]])
>>> torch.tile(y, (2, 2))
# y.shape: (4,4)
tensor([[1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 2],
        [3, 4, 3, 4]])
```

当dim的维度比input的维度要大的时候：

```python
# prompt.shape: (2,3,4)
>>> prompt = torch.zeros((2,3,4))
# prompt扩充为(1,2,3,4)，然后tile
>>> y = torch.tile(prompt, (1,10,1,1))
>>> y.shape
>>> (1,20,3,4)
```

当input的维度比dim的维度要大的时候：

```python
# prompt.shape: (2,3,4)
>>> prompt = torch.zeros((2,3,4))
# dims扩充为(1,10,1)，然后tile
>>> y = torch.tile(prompt, (10,1))
>>> y.shape
>>> (2,30,4)
```

