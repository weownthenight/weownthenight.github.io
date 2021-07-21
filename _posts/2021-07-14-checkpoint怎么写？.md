---
layout: post

title: checkpoint怎么写？

categories: PyTorch

---

在运行深度学习的项目时，我们总是希望能够保存运行期间的模型，在断点后能接着上次的结果继续运行。为了实现这样的功能，就需要在程序中加入checkpoint。下面介绍不同的深度学习框架下如何使用checkpoint。

## PyTorch

参考：[Saving and loading a general checkpoint in PyTorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)

checkpoint需要存储：
1. model的参数
2. optimizer的参数

需要完成的步骤如下：
1. Import all necessary libraries for loading our data
2. Define and intialize the neural network
3. Initialize the optimizer
4. Save the general checkpoint
5. Load the general checkpoint


```python
# 如果你是在已有的项目上更改代码，大概率这些libraries已经加载了
import torch 
import torch.nn as nn
import torch.optim as optim

# define and initiate neural network.(定义和初始化模型)，下面是一个简单例子
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)
```

    Net(
      (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=400, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )



```python
# initialize the optimizer.
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```


```python
# save the general checkpoint.
# Additional information
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4

# 具体要存储哪些参数根据自己的程序可以自己决定
torch.save({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
```


```python
# load the general checkpoint
model = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```

## TensorFlow 1.X

参考：[checkpoints](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/checkpoints.md)

因为TensorFlow的两个版本不兼容，所以这里分开写了两个。虽然平时主要使用的深度学习框架是PyTorch，但是也有很多开源项目使用TensorFlow，如果我们想要运行这些项目，有可能会需要自己写checkpoint。

使用estimator会自动写入这些进磁盘：
1. checkpoints
2. event files

只要指定目录model_dir即可。


```python
# 以DNNClassifier为例：
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir='models/iris')
```

默认使用checkpoint的设置如下：
- Writes a checkpoint every 10 minutes (600 seconds).
- Writes a checkpoint when the train method starts (first iteration) and completes (final iteration).
- Retains only the 5 most recent checkpoints in the directory.

可以通过```tf.estimator.RunConfig```来修改上面的设置。


```python
my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 20*60,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
)

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir='models/iris',
    # pass the config
    config=my_checkpointing_config)
```

## TensorFlow 2.X

参考：[Training checkpoints](https://www.tensorflow.org/guide/checkpoint)

checkpoint只存储参数值，并不存储描述类信息，必须在有源码的基础上使用。而SavedModel除了存储参数值，是存储描述计算的信息的，独立于源码，可以部署到其他地方。

最简单的用法就是```tf.keras```(TensorFlow的一个high level API)，使用```save_weights```可以自动跟踪和存储checkpoints。可见：[save weights](https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights)


```python
import tensorflow as tf

class Net(tf.keras.Model):
  """A simple linear model."""

  def __init__(self):
    super(Net, self).__init__()
    self.l1 = tf.keras.layers.Dense(5)

  def call(self, x):
    return self.l1(x)

net = Net()

# 使用save_weights
net.save_weights('easy_checkpoint')
```
