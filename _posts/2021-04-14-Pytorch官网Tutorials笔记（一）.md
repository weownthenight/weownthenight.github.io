# Pytorch官网教程笔记

🔗：https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

## 1. 基础知识

## 1.1 张量（Tensors）

Tensors与NumPy的ndarrays类似，只是tensors还可以在GPU和其他硬件加速器上运行。两者经常共享一样的底层内存，不需要复制。


```python
import torch
```


```python
import numpy as np
```

### 1.1.1 初始化Tensor

- 可以直接赋值


```python
data = [[1,2],[3,4]]
```


```python
x_data = torch.tensor(data)
```


```python
x_data
```




    tensor([[1, 2],
            [3, 4]])



- 可以从Numpy array赋值


```python
np_array = np.array(data)
```


```python
x_np = torch.from_numpy(np_array)
```


```python
x_np
```




    tensor([[1, 2],
            [3, 4]])



- 可以从另一个Tensor赋值


```python
x_ones = torch.ones_like(x_data)
```


```python
print(f"One Tensor: \n {x_ones} \n")
```

    One Tensor: 
     tensor([[1, 1],
            [1, 1]]) 
    



```python
x_rand = torch.rand_like(x_data, dtype = torch.float)
```


```python
print(f"Random Tensor: \n {x_rand} \n")
```

    Random Tensor: 
     tensor([[0.1883, 0.9876],
            [0.9896, 0.8167]]) 
    


- With random or constant values


```python
shape = (2,3,) #tuple
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

    Random Tensor: 
     tensor([[0.3669, 0.5733, 0.1213],
            [0.3788, 0.0999, 0.7571]]) 
    
    Ones Tensor: 
     tensor([[1., 1., 1.],
            [1., 1., 1.]]) 
    
    Zeros Tensor: 
     tensor([[0., 0., 0.],
            [0., 0., 0.]])


### 1.1.2 Tensor的属性


```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

    Shape of tensor: torch.Size([3, 4])
    Datatype of tensor: torch.float32
    Device tensor is stored on: cpu


### 1.1.3 Tensor的操作

🔗: https://pytorch.org/docs/stable/torch.html

将本地CPU的tensor移到GPU：


```python
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
```

#### indexing and slicing


```python
tensor = torch.ones(4,4)
print('First row: ',tensor[0])
print('First column: ',tensor[:, 0])
print('Last cokumn:',tensor[...,-1])
tensor[:,1]=0
print(tensor)
```

    First row:  tensor([1., 1., 1., 1.])
    First column:  tensor([1., 1., 1., 1.])
    Last cokumn: tensor([1., 1., 1., 1.])
    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])


#### Joining tensors


```python
# torch.cat
t1 = torch.cat([tensor, tensor, tensor], dim = 1) #dim=1表示？
print(t1)
```

    tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])



```python
#torch.stack 查看https://pytorch.org/docs/stable/generated/torch.stack.html
```

#### Arithmetic operations


```python
# 矩阵乘法，y1,y2,y3得到的是同一个值
y1 = tensor @ tensor.T #tensor.T表示tensor的转置？
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor) #rand_like?
torch.matmul(tensor, tensor.T, out=y3)

#点乘(element-wise product)
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```




    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])



#### Single-element tensors

可以将只有一个元素的tensor转换为Python的整数、浮点数类型


```python
agg = tensor.sum()
agg_item = agg.item() #item()可以转换类型
print(agg_item, type(agg_item))
```

    12.0 <class 'float'>


#### In-place operations

将结果按照位置操作后赋值到本身的操作叫in place，和一般操作比起来只需要加上后缀_就可以


```python
print(tensor, "\n")
tensor.t_()
print(tensor)
```

    tensor([[6., 6., 6., 6.],
            [5., 5., 5., 5.],
            [6., 6., 6., 6.],
            [6., 6., 6., 6.]]) 
    
    tensor([[6., 5., 6., 6.],
            [6., 5., 6., 6.],
            [6., 5., 6., 6.],
            [6., 5., 6., 6.]])



```python
print(tensor, "\n")
tensor.add_(5)
print(tensor)
```

    tensor([[6., 5., 6., 6.],
            [6., 5., 6., 6.],
            [6., 5., 6., 6.],
            [6., 5., 6., 6.]]) 
    
    tensor([[11., 10., 11., 11.],
            [11., 10., 11., 11.],
            [11., 10., 11., 11.],
            [11., 10., 11., 11.]])



```python
print(tensor, "\n")
tensor.t() #不加后缀则tensor本身不变
print(tensor)
```

    tensor([[11., 10., 11., 11.],
            [11., 10., 11., 11.],
            [11., 10., 11., 11.],
            [11., 10., 11., 11.]]) 
    
    tensor([[11., 10., 11., 11.],
            [11., 10., 11., 11.],
            [11., 10., 11., 11.],
            [11., 10., 11., 11.]])


因为in place操作会改变变量本身的值，所以不提倡

### 1.1.4 与Numpy的衔接

#### Tensor到Numpy array


```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()   #用numpy()就好了
print(f"n: {n}")
```

    t: tensor([1., 1., 1., 1., 1.])
    n: [1. 1. 1. 1. 1.]



```python
#对tensor的操作也会影响到numpy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

    t: tensor([2., 2., 2., 2., 2.])
    n: [2. 2. 2. 2. 2.]


#### Numpy array到Tensor


```python
n = np.ones(5)
t = torch.from_numpy(n) #用from_numpy()就行
```


```python
#对numpy array的操作同样也影响到tensor
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

    t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
    n: [2. 2. 2. 2. 2.]


## 1.2 导入数据集（Datasets & Dataloaders）

Further reading: [torch.utils.data API](https://pytorch.org/docs/stable/data.html)

Pytorch中有两个函数：
- `torch.utils.data.DataLoader`
- `torch.utils.data.Dataset`

Pytorch的domain libraries中提供了已经加载的数据集以及对该数据集处理的具体函数，可以用于测试自己写的model，具体可以在这些地方找到：
- [Image Datasets](https://pytorch.org/image/stable/datasets.html)
- [Text Datasets](https://pytorch.org/text/stable/datasets.html)
- [Audio Datasets](https://pytorch.org/audio/stable/datasets.html)

### 1.2.1 加载数据集

以[Fashion-MNIST](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/)为例导入数据集


```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt      #画图

training_data = datasets.FashionMNIST(
    root="data",                      #root指定data的目录
    train=True,                       #是否训练，区分训练集和测试集
    download=True,                    #在root中没有data就在互联网上下载
    transform=ToTensor()              #数据有时不适合直接训练，需要改变格式叫做transform
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

### 1.2.2 访问和可视化数据集（Iterating and Visualizing the Dataset）


```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8,8))   #figsize?
cols, rows = 3, 3
for i in range(1, cols*rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()   #randint表示[a,b]中的随机数
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```


    
![png](output_60_0.png)
    


### 1.2.3 Creating a Custom Dataset for your files

以 FashionMNIST 为例，iamges存在了`img_dir`,labels存在了CSV：`annotations_file`


```python
import os
import pandas as pd 
from torchvision.io import read_image

class CustomImageDataset(Dataset):   #transform表示转换特征格式，target_transform改变标签格式
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label":label}
        return sample
```

A custom Dataset class must implement three functions:

- `__init__`

- `__len__`：返回数据集中的样本数

- `__getitem__`

The `__getitem__` function loads and returns a sample from the dataset at the given index `idx`. Based on the index, it identifies the image's location on disk, converts that to a tensor using `read_image`, retrieves the corresponding label from the csv data `self.img_labels`, calls the transform functions on them(if applicable), and returns the tensor image and corresponding label in a Python dict.

### 1.2.4 DataLoaders

在实际训练模型时，需要能够reshuffle the data，加速data retrieval。


```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)  #batch_size?
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

For finer-grained control over the data loading order, take a look at [Samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)


```python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

    Feature batch shape: torch.Size([64, 1, 28, 28])
    Labels batch shape: torch.Size([64])



    
![png](output_73_1.png)
    


    Label: 2


## 1.3 Transforms

The [torchvision.transforms]() module offers several commonly-used transforms out of the box.

需要将数据和标签转换为适合训练的格式，就是transform。分别对应transform和target_transform。

The FashionMNIST features are in PIL Image format, and the labels are integers.For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors. To make these transformations, we use `ToTensor` and `Lambda`

scatter用法可看：[scatter_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.scatter_)


```python
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    '''
    torch.zeros create a zero tensor of size 10(数据集中共有10种标签) and calls scatter_ which assigns 
    a value=1 on the index as given the lable y.
    '''
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=10))
)
```

### ToTensor

[ToTensor]() converts a PIL image or NumPy `ndarray` into a `FloatTensor`, and scales the image's pixel intensity values in the range[0.,1.]

### Lambda Transforms

Lambda transforms apply any user-defined lambda function.


```python

```
