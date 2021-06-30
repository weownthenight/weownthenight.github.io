---
layout: post

title: Google Colab:怎么不花钱用GPU？

categories: 深度学习

---

这篇笔记解决一个问题：需要GPU训练模型（模型本身不大）又没卡怎么办？

我也是初入这个坑，使用Google Colab两次，一次是做CS224N的Assignment 4——一个机器翻译模型，tutorial上用的是Azure的虚拟机，估计4h左右。本来我也想步后尘，毕竟Azure有新人优惠，注册成功后什么也没做，就被扣了1美元（忘了说注册需要有信用卡，印象里还得是双币的），并且发现新人优惠的范围里根本没有要用的机器，因此我怒而转投Google Colab。我用Google Colab很幸运地提前meet了requiements，2h多结束了训练。第二次是这个例子，跑了一个Github上关系-实体抽取的模型（BERT+微调），因为本地测试的数据集很小，所以也没有花很长时间。

Colab实际使用中会有断线的情况，可以刷新，看着程序还在跑，只要不是相隔时间太久，应该都没有问题。

因为我也是新手，可能有说的不对的地方，很多设置最初参考：[实验室一块GPU都没有怎么做深度学习？](https://www.zhihu.com/question/299434830/answer/1329278982)。（其实我们实验室不是没有GPU，只是我还是研0，实在是不熟......自己用自己的舒服)

按照这篇文章设置好后，可以先看看分配了什么GPU：


```python
!nvidia-smi
```

    Tue Jun 22 08:53:57 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 465.27       Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   35C    P8    26W / 149W |      0MiB / 11441MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+


接下来，更改目录，把当前目录更改为云盘上的目录：


```python
import os
os.chdir('drive/MyDrive/Entity-Relation-Extraction')
```

查看Google Colab预装的Python, Tensorflow版本号：


```python
!python --version
```

    Python 3.7.10



```python
import tensorflow as tf
tf.__version__
```




    '2.5.0'



可以看到tensorflow版本是2.5.0，python版本是3.7.10。

而这个项目支持：python 3.6+，Tensorflow 1.12.0+。起初我在本地用Tensorflow 2.X跑，报错后将旧API更改为新API，发现太多地方要改，如果直接把```import tensorflow as tf```改为```import tensorflow.compat.v1 as tf```后仍然还有报错，TF2.X和TF1.X根本不兼容。鉴于这样做太麻烦，我还是考虑安装历史版本。也难怪现在大家都用Pytorch，Tensorflow这点真是可以让不懂编程的人都大呼离谱的程度，我还不知道有不向下兼容的产品呢。

下一段代码可以安装历史版本：


```python
%tensorflow_version 1.x
import tensorflow as tf
tf.__version__
```

    TensorFlow 1.x selected.





    '1.15.2'



这段代码必须重启后运行才有效，所以必须重启kernel，然后运行这段代码，这时不要再运行上一段看tensorflow版本的代码了（说的就是一键重启+运行所有的我）。

在Google Colab上运行终端命令，只要在语句前加```!```就可以。


```python
!python run_predicate_classification.py \
--task_name=SKE_2019 \
--do_train=true \
--do_eval=false \
--data_dir=bin/predicate_classifiction/classification_data \
--vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=6.0 \
--output_dir=./output/predicate_classification_model/epochs6/
```


```python
!python bin/subject_object_labeling/sequence_labeling_data_manager.py
```


```python
!python run_sequnce_labeling.py \
--task_name=SKE_2019 \
--do_train=true \
--do_eval=false \
--data_dir=bin/subject_object_labeling/sequence_labeling_data \
--vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=9.0 \
--output_dir=./output/sequnce_labeling_model/epochs9/
```

今天心情好，废话比较多。感慨一下这段时间入门的感受：看不懂很正常，一方面可能是没有前置知识，另一方面可能是对方没说人话。（看一些论文，感觉就是换个名字故弄玄虚；论文作者都会假定你已经在这个方面有一些基础，很多事情都会省略不讲，所以看不懂很正常，我是觉得不要一上来就看论文，先把基础补好，把CS224N认真看完做完作业，达到一个门槛再看论文才知道他们在说什么；另外，还有教材的问题我也想吐槽很久了，cs是实践出真知，不是靠背，很多本科教材就是概念方法的堆砌，无聊质量也低。不如认真做一个实际应用学到的多，记的牢，我已经打算把本科的课程跟着国外带实现项目的课程过一遍+做作业，把基础知识打牢。）

不要质疑自己的能力，自信起来！不要害怕请教的问题有多小白，有些人确实对他们眼里的简单问题不耐烦（这次也让我反思了自己，高中的时候虽然总是第一，但大家都不爱问我问题，我真的无意识地态度不好，因为这些问题我真的觉得没有必要问），也有的大神真的是让人如沐春风，有问必答，超级温柔（点名李沐老师，他的[动手深度学习课程](https://courses.d2l.ai/zh-v2/)入门的同学都去学，真的很照顾我们）。本科时真的感受到人外有人，天外有天，一度因为自己跟不上怀疑人生，最后放弃治疗。他们的看法和成就不代表什么，也不需要比较，只有自己的进步才是实打实的，只要每天都努力进步了，就不要感到羞愧和自卑，加油往前冲就可以了！

还有一点感悟，现在的资源真的比我读本科的时候多很多，也有很多人乐于分享他们成长的经验。中文资料比当时感觉要全很多，一些公众号和博客对入门帮助很大，因为很多教材和论文不说人话，博客和公众号这种说人话的地方就很不错！我觉得关注一下很有必要！

说这么多也是希望大家少走弯路，也是记录下勉励自己吧。
