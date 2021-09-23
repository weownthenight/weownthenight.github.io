---
layout: post

title: Hugging Face Transformers教程笔记(8)：A full training

categories: NLP

---

上一篇讲的是怎么用**Trainer** API来进行训练的，这一篇讲怎么用PyTorch写训练：


```python
!pip install datasets transformers[sentencepiece]
```


```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
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



    Downloading:   0%|          | 0.00/28.0 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/570 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/232k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/466k [00:00<?, ?B/s]



      0%|          | 0/4 [00:00<?, ?ba/s]



      0%|          | 0/1 [00:00<?, ?ba/s]



      0%|          | 0/2 [00:00<?, ?ba/s]


## Prepare for training

Postprocessing:

- Remove the columns corresponding to values the model does not expect (like the **sentence1** and **sentence2** columns).
- Rename the column **label** to **labels** (because the model expects the argument to be named **labels**).
- Set the format of the datasets so they return PyTorch tensors instead of lists.


```python
# remove columns
tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence1", "sentence2", "idx"]
)
# rename label
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# turn into tensors
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names
```




    ['attention_mask', 'input_ids', 'labels', 'token_type_ids']



接下来就可以定义```DataLoader```类了：


```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```

看一下一个batch：


```python
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
```




    {'attention_mask': torch.Size([8, 65]),
     'input_ids': torch.Size([8, 65]),
     'labels': torch.Size([8]),
     'token_type_ids': torch.Size([8, 65])}



数据处理彻底结束了，下面定义模型：


```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```


    Downloading:   0%|          | 0.00/440M [00:00<?, ?B/s]


    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForSequenceClassification: ['cls.predictions.decoder.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.bias', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.weight']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


为了保证到现在为止没有出错，我们放入一个batch看看情况：


```python
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
```

    tensor(0.7381, grad_fn=<NllLossBackward>) torch.Size([8, 2])


All 🤗 Transformers models will return the loss when labels are provided, and we also get the logits (two for each input in our batch, so a tensor of size 8 x 2).

接下来还需要一个**optimizer**和一个**learning rate scheduler**（learning rate如何调整的机制）：


```python
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

learning rate默认的调整方法是随着training steps线性降低到0，所以我们需要计算training steps:


```python
from transformers import get_scheduler

num_epochs = 3
# len(train_dataloader) == number of training batches
# 默认情况下取num_epochs为3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
print(num_training_steps)
```

    1377


## The training loop

最后一件事就是保证能在GPU上运行，需要定义**device**:


```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device
```




    device(type='cuda')



现在可以开始训练啦！为了可视化进度，我们用**tqdm**定义bar：


```python
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```


      0%|          | 0/1377 [00:00<?, ?it/s]


## The evaluation loop

上面的训练没有加上evaluation loop，导致我们没有办法看到训练过程中loss，accuracy和F1的变化。现在加上evaluation loop：


```python
from datasets import load_metric

metric= load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    # 用add_batch可以累积之前所有batch的结果
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```


    Downloading:   0%|          | 0.00/1.86k [00:00<?, ?B/s]





    {'accuracy': 0.8382352941176471, 'f1': 0.8896321070234114}



## Superchange your training loop with 🤗 Accelerate

之前的做法都是在单CPU或单GPU上训练，如果想要在多GPU上进行分布式训练，可以用Hugging Face Accelerate库：


```python
# 导入Accelerator
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

# Accelerator类本身可以知道你的device，不需要自己定义
- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

# 主体调用部分
+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
# 将loss replace为accelerator
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

将这段代码放入```train.py```中：


```python
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

然后执行：


```python
!accelerate config
```

会让你回答一些问题，以此生成config file。然后就可以运行：


```python
!accelerate launch train.py
```

如果你想在notebook上运行，可以将代码放入函数```training_function```，然后：


```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```
