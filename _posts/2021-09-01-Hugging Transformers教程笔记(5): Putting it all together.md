---
layout: post

title: Hugging Face Transformers教程笔记(5)：Putting it all together

categories: NLP

---

[Putting it all together](https://huggingface.co/course/chapter2/6?fw=pt)

在之前的几篇教程中我们hand by hand实现了一些tokenizer和model的功能。但实际上我们可以更抽象的直接调用transformer：


```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=48.0, style=ProgressStyle(description_w…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=629.0, style=ProgressStyle(description_…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=231508.0, style=ProgressStyle(descripti…


    


这里的```model_inputs```包含了model需要的所有输入：对于DistillBERT则是input IDs和attention masks。tokenizer can be very powerful!

It can process one sequence:


```python
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

It also handles multiple sequences at a time, with no change in the API:


```python
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

model_inputs = tokenizer(sequences)
```

It can pad according to several objectives:


```python
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

It can also truncate sequences:


```python
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

The tokenizer object can handle the conversion to specific framework tensors, which can then be directly sent to the model. For example, in the following code sample we are prompting the tokenizer to return tensors from the different frameworks — "pt" returns PyTorch tensors, "tf" returns TensorFlow tensors, and "np" returns NumPy arrays:


```python
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

## Special tokens

如果我们把我们分步做的结果（先tokenize，再转换为ID）和tokenizer直接得到的结果作为对比：


```python
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
```

    [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
    [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]


可以发现结果不同，tokenizer得到的结果比我们自己convert的结果多了两个数字，就是在首部的101和尾部的102。我们来decode看看这是为什么：


```python
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))
```

    [CLS] i've been waiting for a huggingface course my whole life. [SEP]
    i've been waiting for a huggingface course my whole life.


可以发现，这是因为多了两个特殊的token：```[CLS]```和```[SEP]```。这是因为DistillBERT预训练时是加上了这两个token的，为了保持一致，我们应当也加上这两个token。但是不同的预训练模型的处理方式可能不同，直接使用tokenizer可以保证tokenizer和model的处理方式一致。

## Wrapping up: From tokenizer to model


```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
  "I've been waiting for a HuggingFace course my whole life.",
  "So have I!"
]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=267844284.0, style=ProgressStyle(descri…


    

