---
layout: post

title: Hugging Face Transformers教程笔记(6)：Processing the data

categories: NLP
description: 数据集
---

[Processing the data](https://huggingface.co/course/chapter3/2?fw=pt)

接下来的几篇教程会演示如何fine tune一个transformer。这篇教程主要讲的是如何处理数据集。

## 获取数据集

在这一章我们以MRPC（Microsoft Research Paraphrase Corpus）为例，The dataset consists of 5,801 pairs of sentences, with a label indicating if they are paraphrases or not (i.e., if both sentences mean the same thing).MRPC是GLUE benchmark的十个数据集之一，GLUE用来衡量模型在文本分类任务的表现。

Hugging Face除了有transformer model以外，也有[数据集](https://huggingface.co/datasets)可以很方便的load进来：


```python
!pip install datasets transformers[sentencepiece]
```


```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```


    Downloading:   0%|          | 0.00/7.78k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/4.47k [00:00<?, ?B/s]


    Downloading and preparing dataset glue/mrpc (download: 1.43 MiB, generated: 1.43 MiB, post-processed: Unknown size, total: 2.85 MiB) to /root/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad...



    Downloading: 0.00B [00:00, ?B/s]



    Downloading: 0.00B [00:00, ?B/s]



    Downloading: 0.00B [00:00, ?B/s]



    0 examples [00:00, ? examples/s]



    0 examples [00:00, ? examples/s]



    0 examples [00:00, ? examples/s]


    Dataset glue downloaded and prepared to /root/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad. Subsequent calls will reuse this data.





    DatasetDict({
        train: Dataset({
            features: ['sentence1', 'sentence2', 'label', 'idx'],
            num_rows: 3668
        })
        validation: Dataset({
            features: ['sentence1', 'sentence2', 'label', 'idx'],
            num_rows: 408
        })
        test: Dataset({
            features: ['sentence1', 'sentence2', 'label', 'idx'],
            num_rows: 1725
        })
    })



从输出可以看出，MRPC数据集分为训练集（3668 pairs）、验证集（408 pairs）、测试集（1725 pairs）。格式为(sentence1, sentence2, label, idx)。

和model类似，数据集下载后缓存在*~/.cache/huggingface/dataset*，可以通过设置环境变量**HF_HOME**来改变。

取数据：


```python
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
```




    {'idx': 0,
     'label': 1,
     'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
     'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .'}



这里可以看到```label```已经是数字了，我们如果想要知道不同的数字对应的是什么意义可以：


```python
raw_train_dataset.features
```




    {'idx': Value(dtype='int32', id=None),
     'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
     'sentence1': Value(dtype='string', id=None),
     'sentence2': Value(dtype='string', id=None)}



## 数据集预处理


```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
```


    Downloading:   0%|          | 0.00/28.0 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/570 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/232k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/466k [00:00<?, ?B/s]


我们不能直接把tokenized_sentence_1和tokenized_sentence_2作为输入放入模型，需要组合成pair才可以，而tokenizer本身也可以将句子组合成pair：


```python
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs
```




    {'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}



```token_type_ids```标明两句话的位置。


```python
tokenizer.convert_ids_to_tokens(inputs["input_ids"])
```




    ['[CLS]',
     'this',
     'is',
     'the',
     'first',
     'sentence',
     '.',
     '[SEP]',
     'this',
     'is',
     'the',
     'second',
     'one',
     '.',
     '[SEP]']



可以看到格式为： **[CLS] sentence1 [SEP] sentence2 [SEP]**


```python
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
[      0,      0,    0,     0,       0,          0,   0,       0,      1,    1,     1,        1,     1,   1,       1]
```




    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]



tokenizer是否会返回```token_type_ids```还是要看预训练模型是怎么处理的，比如DistillBERT就是没有的。

我们可以这样处理训练集：


```python
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```

上述的做法有几个不好的地方：

1. 返回的结果是字典，包含有key：input_ids, attention_mask, token_type_ids, 值为list of lists。
2. 在运行时需要有足够的RAM。
3. 这样做只能分别处理训练集，验证集和测试集。

为了解决这些问题，我们可以使用```Dataset.map```方法。The map method works by applying a function on each element of the dataset.下面我们先定义这个函数：


```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```

其中```example```是一个dictionary（它的key是features），tokenize_function返回了一个新的dictionary（将attention_mask, input_ids, token_type_ids加入features）。可以回忆一下raw_datasets，和如下的tokenized_datasets作为对比：


```python
raw_datasets
```




    DatasetDict({
        train: Dataset({
            features: ['sentence1', 'sentence2', 'label', 'idx'],
            num_rows: 3668
        })
        validation: Dataset({
            features: ['sentence1', 'sentence2', 'label', 'idx'],
            num_rows: 408
        })
        test: Dataset({
            features: ['sentence1', 'sentence2', 'label', 'idx'],
            num_rows: 1725
        })
    })




```python
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```


      0%|          | 0/4 [00:00<?, ?ba/s]



      0%|          | 0/1 [00:00<?, ?ba/s]



      0%|          | 0/2 [00:00<?, ?ba/s]





    DatasetDict({
        train: Dataset({
            features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
            num_rows: 3668
        })
        validation: Dataset({
            features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
            num_rows: 408
        })
        test: Dataset({
            features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
            num_rows: 1725
        })
    })



```tokenized_function```不止可以对一个element生效，还可以对整个batch生效，只需要加上```batched=True```，这样可以加快tokenizer的速度。另外，我们没有在```tokenized_function```里设置padding，是因为：如果对每一个sample padding到max length是非常低效的，我们可以以batch为单位来padding，这样max length可能也不会那么大，速度也会快很多。

除了batch可以提高速度外，也可以指定多线程来提高效率，指定```num_proc```参数就可以。 

We didn’t do this here because the 🤗 Tokenizers library already uses multiple threads to tokenize our samples faster, but if you are not using a fast tokenizer backed by this library, this could speed up your preprocessing.

tokenizer除了添加了三项特征外，也可以在函数中改变原有的特征。

接下来要做的就是之前讲过的padding，因为按每个batch来padding，所以也叫dynamic padding。

## Dynamic padding

In PyTorch, the function that is responsible for putting together samples inside a batch is called a **collate function**. 

上文中提到的按照batch来padding的做法不适合TPU，TPU需要相同形状的tensor。

collate function可以在```DataLoader```处按照参数传入定义，默认的collate办法就是将样本转换为tensor，recursively if your elements are lists, tuples, or dictionaries.

在这个例子中，我们不能使用默认的collate function，因为我们需要按照batch来padding，Hugging Face Transformers库中提供了```DataCollatorWithPadding```可供使用：


```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

这是在padding之前的情况：（取前8个samples（一个batch），看一下长度）


```python
samples = tokenized_datasets["train"][:8]
# string无法转换为tensor，所以这里去掉了idx, sentence1, sentence2三个特征
samples = {
    k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]
}
[len(x) for x in samples["input_ids"]]
```




    [50, 59, 47, 67, 59, 50, 62, 32]



进行dynamic padding后：


```python
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
```




    {'attention_mask': torch.Size([8, 67]),
     'input_ids': torch.Size([8, 67]),
     'labels': torch.Size([8]),
     'token_type_ids': torch.Size([8, 67])}



可以看到都变为了67，也就是这个batch中的最大长度。
