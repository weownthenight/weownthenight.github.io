---
layout: post

title: Pytorch总结：view, reshape, permute

categories: PyTorch

---

这些问题是我在写CS224N的Assignment 5中遇到的，主要涉及到Tensor的定义和形状问题。

## Tensor的定义

当我们自己定义一个tensor时需要注意的问题：

### device

在大部分情况下，你的程序都会在GPU上执行，那么在定义tensor时一定要加上device！

### 数据类型

参考：[TENSOR ATTRIBUTES](https://pytorch.org/docs/stable/tensor_attributes.html#tensor-attributes-doc)

A floating point scalar operand has dtype torch.get_default_dtype() and an integral non-boolean scalar operand has dtype torch.int64. 

获取Tensor的默认数据类型：（这个方法指的是浮点数的默认类型，整数的默认类型就是```torch.int64```）。


```python
import torch 
torch.get_default_dtype()
```




    torch.float32



在Assignment 5作业中句子需要用\<pad\>来padding达到max word length以及相同的句子长度，padding过后的tensor，就需要用torch.long来表示了：


```python
sents = torch.tensor(sents_padded, dtype=torch.long, device=device).contiguous()
```

PyTorch的Tensor数据类型如下：
（参考：[TORCH.TENSOR](https://pytorch.org/docs/stable/tensors.html)

|Data type|dtype|CPU tensor|GPU tensor |
|:------|:------|:------|:------|
|32-bit floating point|``torch.float32`` or ``torch.float``|:class:`torch.FloatTensor`|:class:`torch.cuda.FloatTensor`|
|64-bit floating point|``torch.float64`` or ``torch.double``|:class:`torch.DoubleTensor`|:class:`torch.cuda.DoubleTensor`|
|16-bit floating point|``torch.float16`` or ``torch.half``|:class:`torch.HalfTensor`|:class:`torch.cuda.HalfTensor`|
|16-bit floating point|``torch.bfloat16``|:class:`torch.BFloat16Tensor`|:class:`torch.cuda.BFloat16Tensor`
|32-bit complex|``torch.complex32``|||
|64-bit complex|``torch.complex64``|||
|128-bit complex|``torch.complex128`` or ``torch.cdouble``|||
|8-bit integer (unsigned)|``torch.uint8``|:class:`torch.ByteTensor`|:class:`torch.cuda.ByteTensor`|
|8-bit integer (signed)|``torch.int8``|:class:`torch.CharTensor`| |:class:`torch.cuda.CharTensor`|
|16-bit integer (signed)|``torch.int16`` or ``torch.short``|:class:`torch.ShortTensor`|:class:`torch.cuda.ShortTensor`|
|32-bit integer (signed)|``torch.int32`` or ``torch.int``|:class:`torch.IntTensor`|:class:`torch.cuda.IntTensor`|
|64-bit integer (signed)|``torch.int64`` or ``torch.long``|:class:`torch.LongTensor`|:class:`torch.cuda.LongTensor`|
|Boolean|``torch.bool``|:class:`torch.BoolTensor`|:class:`torch.cuda.BoolTensor`|
|quantized 8-bit integer (unsigned)|``torch.quint8``|:class:`torch.ByteTensor`|/|
|quantized 8-bit integer (signed)|``torch.qint8``|:class:`torch.CharTensor`|/|
|quantized 32-bit integer (signed)|``torch.qfint32``|:class:`torch.IntTensor`|/|
|quantized 4-bit integer (unsigned)|``torch.quint4x2``|:class:`torch.ByteTensor`|/|     


其中```torch.long```和```torch.int64```等价，```torch.float32```也是```torch.float```。

## Tensor的变形

为了让我们的输入输出符合形状，我们总是会在程序中遇到需要变换Tensor形状的情况。

### view

官方文档：[torch.view](https://pytorch.org/docs/stable/tensor_view.html)

```view```并不生成新的tensor，而是在现有tensor上进行indexing，这样可以更快。只有满足连续性条件才能进行view，否则只能用reshape新建一个tensor。所以在使用view之前我们通常都有一个```contiguous()```的操作，为了使tensor满足连续性条件：


```python
# 还是作业里的例子：
sents = torch.tensor(sents_padded, dtype=torch.long, device=device).contiguous()
```

### reshape

官方文档：[torch.reshape](https://pytorch.org/docs/stable/generated/torch.reshape.html?highlight=reshape#torch.reshape)

```reshape```很直接，并且也很强大，将输入转换为指定的输出形状。变换时是根据行一个个取出元素再放入新的形状之中。```view```是```reshape```的子集。下面举个例子来说明：


```python
# 我们定义一个tensor，想象它是一个立方体，有六个面，每个面都是一个二维矩阵：
# 为了一目了然，我特地设置每个向量各不相同
cube = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[2, 3, 4], [5, 6, 7], [8, 9, 10]],
                    [[3, 4, 5], [6, 7, 8], [9, 10, 11]],
                    [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                    [[8, 7, 6], [5, 4, 3], [2, 1, 0]],
                    [[7, 6, 5], [4, 3, 2], [1, 0, -1]]])
```

先看一下cube的形状和数据类型：


```python
print(cube.shape)
print(cube.dtype)
```

    torch.Size([6, 3, 3])
    torch.int64


接下来，我们将shape转换为(3, 6, 3)会怎样？是我们想象中的那样吗？


```python
torch.reshape(cube, (3, 6, 3))
```




    tensor([[[ 1,  2,  3],
             [ 4,  5,  6],
             [ 7,  8,  9],
             [ 2,  3,  4],
             [ 5,  6,  7],
             [ 8,  9, 10]],
    
            [[ 3,  4,  5],
             [ 6,  7,  8],
             [ 9, 10, 11],
             [ 9,  8,  7],
             [ 6,  5,  4],
             [ 3,  2,  1]],
    
            [[ 8,  7,  6],
             [ 5,  4,  3],
             [ 2,  1,  0],
             [ 7,  6,  5],
             [ 4,  3,  2],
             [ 1,  0, -1]]])



可以看到```reshape```就是按照顺序把数字填入了设定的形状中。

参考：[PyTorch：view() 与 reshape() 区别详解](http://www.360doc.com/content/21/0317/09/7669533_967384667.shtml)

在官网上关于```reshape```的说明指明，如果满足连续性条件，那么```reshape```和```view```等价，也就是说```reshape```不会改变现有tensor，也不会新建tensor；如果不满足连续性条件，```reshape```会新建一个tensor。

### permute

```permute```与```reshape```和```view```产生的结果完全不同。```permute```是转换维度，并不是根据形状放入元素，还是以cube举例：


```python
cube
```




    tensor([[[ 1,  2,  3],
             [ 4,  5,  6],
             [ 7,  8,  9]],
    
            [[ 2,  3,  4],
             [ 5,  6,  7],
             [ 8,  9, 10]],
    
            [[ 3,  4,  5],
             [ 6,  7,  8],
             [ 9, 10, 11]],
    
            [[ 9,  8,  7],
             [ 6,  5,  4],
             [ 3,  2,  1]],
    
            [[ 8,  7,  6],
             [ 5,  4,  3],
             [ 2,  1,  0]],
    
            [[ 7,  6,  5],
             [ 4,  3,  2],
             [ 1,  0, -1]]])




```python
# 如果使用reshape：
torch.reshape(cube, (3, 6, 3))
```




    tensor([[[ 1,  2,  3],
             [ 4,  5,  6],
             [ 7,  8,  9],
             [ 2,  3,  4],
             [ 5,  6,  7],
             [ 8,  9, 10]],
    
            [[ 3,  4,  5],
             [ 6,  7,  8],
             [ 9, 10, 11],
             [ 9,  8,  7],
             [ 6,  5,  4],
             [ 3,  2,  1]],
    
            [[ 8,  7,  6],
             [ 5,  4,  3],
             [ 2,  1,  0],
             [ 7,  6,  5],
             [ 4,  3,  2],
             [ 1,  0, -1]]])




```python
# 如果使用permute
cube.permute((1, 0, 2))
```




    tensor([[[ 1,  2,  3],
             [ 2,  3,  4],
             [ 3,  4,  5],
             [ 9,  8,  7],
             [ 8,  7,  6],
             [ 7,  6,  5]],
    
            [[ 4,  5,  6],
             [ 5,  6,  7],
             [ 6,  7,  8],
             [ 6,  5,  4],
             [ 5,  4,  3],
             [ 4,  3,  2]],
    
            [[ 7,  8,  9],
             [ 8,  9, 10],
             [ 9, 10, 11],
             [ 3,  2,  1],
             [ 2,  1,  0],
             [ 1,  0, -1]]])



可以看到，虽然两者的形状一样，但是值是不同的。```reshape```还原了cube，而```permute```则是变换了维度。所以千万不要以为两者等价混用。
