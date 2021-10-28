---
layout: post

title: Hugging Face Transformers教程笔记(7)：Fine-tuning a pretrained model with the Trainer API

categories: NLP
description: Trainer API，好像用的不多
---


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


🤗 Transformers provides a **Trainer** class to help you fine-tune any of the pretrained models it provides on your dataset.

## Training

定义**Trainer**之前需要先定义**TrainingArguments**，我们只需要指定fine-tune的模型存储的位置（也是checkpoints存储的位置），其他的设定暂且默认：


```python
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
```

接着需要定义模型：


```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```


    Downloading:   0%|          | 0.00/440M [00:00<?, ?B/s]


    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForSequenceClassification: ['cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.bias', 'cls.predictions.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.weight', 'cls.predictions.decoder.weight']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


结果中包含警告，这是因为：

This is because BERT has not been pretrained on classifying pairs of sentences, so the head of the pretrained model has been discarded and a new head suitable for sequence classification has been added instead. The warnings indicate that some weights were not used (the ones corresponding to the dropped pretraining head) and that some others were randomly initialized (the ones for the new head). It concludes by encouraging you to train the model, which is exactly what we are going to do now.

Once we have our model, we can define a **Trainer** by passing it all the objects constructed up to now — the **model**, the **training_args**, the training and validation datasets, our **data_collator**, and our **tokenizer**:


```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    # 可以省略，默认的data_collator就是DataCollatorWithPadding
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```


To fine-tune the model on our dataset, we just have to call the train method of our Trainer:


```python
trainer.train()
```

    The following columns in the training set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence1, idx, sentence2.
    ***** Running training *****
      Num examples = 3668
      Num Epochs = 3
      Instantaneous batch size per device = 8
      Total train batch size (w. parallel, distributed & accumulation) = 8
      Gradient Accumulation steps = 1
      Total optimization steps = 1377




    <div>

      <progress value='1377' max='1377' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [1377/1377 06:32, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>500</td>
      <td>0.557700</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>0.359800</td>
    </tr>
  </tbody>
</table><p>


    Saving model checkpoint to test-trainer/checkpoint-500
    Configuration saved in test-trainer/checkpoint-500/config.json
    Model weights saved in test-trainer/checkpoint-500/pytorch_model.bin
    tokenizer config file saved in test-trainer/checkpoint-500/tokenizer_config.json
    Special tokens file saved in test-trainer/checkpoint-500/special_tokens_map.json
    Saving model checkpoint to test-trainer/checkpoint-1000
    Configuration saved in test-trainer/checkpoint-1000/config.json
    Model weights saved in test-trainer/checkpoint-1000/pytorch_model.bin
    tokenizer config file saved in test-trainer/checkpoint-1000/tokenizer_config.json
    Special tokens file saved in test-trainer/checkpoint-1000/special_tokens_map.json
    
    
    Training completed. Do not forget to share your model on huggingface.co/models =)
    
    





    TrainOutput(global_step=1377, training_loss=0.38740395495401575, metrics={'train_runtime': 393.1934, 'train_samples_per_second': 27.986, 'train_steps_per_second': 3.502, 'total_flos': 405470580750720.0, 'train_loss': 0.38740395495401575, 'epoch': 3.0})



This will start the fine-tuning (which should take a couple of minutes on a GPU) and report the training loss every 500 steps. It won’t, however, tell you how well (or badly) your model is performing. This is because:

1. We didn’t tell the Trainer to evaluate during training by setting **evaluation_strategy** to either "steps" (evaluate every **eval_steps**) or "epoch" (evaluate at the end of each **epoch**).
2. We didn’t provide the Trainer with a **compute_metrics** function to calculate a metric during said evaluation (otherwise the evaluation would just have printed the loss, which is not a very intuitive number).

## Evaluation

Let’s see how we can build a useful **compute_metrics** function and use it the next time we train. The function must take an **EvalPrediction** object (which is a named tuple with a **predictions** field and a **label_ids** field) and will return a dictionary mapping strings to floats (the strings being the names of the metrics returned, and the floats their values). To get some predictions from our model, we can use the **Trainer.predict** command:


```python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```

    The following columns in the test set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence1, idx, sentence2.
    ***** Running Prediction *****
      Num examples = 408
      Batch size = 8




<div>

  <progress value='51' max='51' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [51/51 00:03]
</div>



    (408, 2) (408,)


The output of the predict method is another named tuple with three fields: **predictions**, **label_ids**, and **metrics**. 

其中**predictions**得到的是logits，共有408个pairs，每个pair有两个类别。我们看看第一对pair的prediction：


```python
print(predictions.predictions[0])
```

    [-3.1320443  2.756775 ]


因为2.756775 > -3.1320443，所以预测的类别应该是1，即预测的类别取参数值更大的类别：


```python
import numpy as np
preds = np.argmax(predictions.predictions, axis=-1)
```


```python
print(preds)
```

    [1 0 0 1 0 1 1 1 1 1 1 0 0 1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 0
     0 1 1 0 1 0 0 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0 1 1 1 1 1
     1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 0 0 1 1
     1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 0 1 0 1 1 1 0 1 1 1 0 0 1 0 1 1 1
     0 1 0 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 0 1 1 0 1 1 1 1
     1 0 1 0 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 0 0 1 1 0 1 1 1 1 0 1 1 1
     1 0 1 1 1 1 1 1 1 1 0 0 0 0 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1
     0 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 1 1 1 1 0 1 1 0 0 0 1 1 0 1 0 1 0 1 1 1 0
     0 1 1 1 1 1 1 0 1 1 0 1 0 1 1 1 1 0 1 0 1 1 1 0 0 0 0 1 0 1 1 1 1 1 1 1 1
     1 1 1 1 1 0 1 1 1 0 0 1 1 1 1 1 1 0 1 0 1 0 0 1 1 1 1 0 1 1 0 1 1 1 1 0 0
     1 1 1 1 0 0 0 0 1 1 1 1 0 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 0 0 1 0
     1]


**label_ids**是gold label：


```python
print(predictions.label_ids)
```

    [1 0 0 1 0 1 0 1 1 1 1 0 0 1 1 1 1 0 1 0 0 1 0 1 1 1 0 1 1 1 0 1 1 1 1 0 0
     0 1 1 0 1 0 0 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 0 1 1 1 1 1
     1 1 0 1 1 1 0 1 1 0 1 0 1 0 1 1 0 1 1 0 1 1 0 1 1 1 1 1 1 1 0 1 1 0 0 1 1
     1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 0 1 0 1 1 0 0 1 0 1 0 0 1 0 0 1 1
     1 1 0 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 0 1 1 0 1 1 0 0 1 1 0
     1 0 1 0 1 1 0 1 1 0 1 1 0 1 1 1 0 1 1 1 0 1 1 1 0 0 1 1 0 1 1 1 1 0 1 1 1
     1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 1 0 1 1 1 1 1 0 1 1 1 0 1 0 1 0 1 1 1
     0 1 0 1 1 0 1 1 0 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 0
     0 0 1 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 1 0 1 1 0 0 0 0 0 1 0 1 0 1 1 1 1 1 0
     1 1 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 0 1 0 1 0 0 1 1 1 1 0 1 1 0 1 1 0 1 0 0
     1 1 1 1 0 0 0 0 1 1 1 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 0
     1]


**metrics**包括loss和其他的一些信息：


```python
print(predictions.metrics)
```

    {'test_loss': 0.5279752612113953, 'test_runtime': 4.1279, 'test_samples_per_second': 98.839, 'test_steps_per_second': 12.355}


接下来我们可以做evaluation：


```python
from datasets import load_metric

metric = load_metric("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```


    Downloading:   0%|          | 0.00/1.86k [00:00<?, ?B/s]





    {'accuracy': 0.8725490196078431, 'f1': 0.9100346020761245}



将上述的步骤合在一起定义```compute_metrics```函数：


```python
def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

将```compute_metrics```函数放入**Trainer**中，实现在训练时看到在验证集上的效果：


```python
# evaluate_strategy: 每个epoch结束report metrics
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

    PyTorch: setting up devices
    The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).
    loading configuration file https://huggingface.co/bert-base-uncased/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/3c61d016573b14f7f008c02c4e51a366c67ab274726fe2910691e2a761acf43e.37395cee442ab11005bcd270f3c34464dc1704b715b5d7d52b1a461abe3b9e4e
    Model config BertConfig {
      "architectures": [
        "BertForMaskedLM"
      ],
      "attention_probs_dropout_prob": 0.1,
      "gradient_checkpointing": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "model_type": "bert",
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "pad_token_id": 0,
      "position_embedding_type": "absolute",
      "transformers_version": "4.9.1",
      "type_vocab_size": 2,
      "use_cache": true,
      "vocab_size": 30522
    }
    
    loading weights file https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin from cache at /root/.cache/huggingface/transformers/a8041bf617d7f94ea26d15e218abd04afc2004805632abc0ed2066aa16d50d04.faf6ea826ae9c5867d12b22257f9877e6b8367890837bd60f7c54a29633f7f2f
    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForSequenceClassification: ['cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.bias', 'cls.predictions.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.weight', 'cls.predictions.decoder.weight']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
trainer.train()
```

    The following columns in the training set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence1, idx, sentence2.
    ***** Running training *****
      Num examples = 3668
      Num Epochs = 3
      Instantaneous batch size per device = 8
      Total train batch size (w. parallel, distributed & accumulation) = 8
      Gradient Accumulation steps = 1
      Total optimization steps = 1377




    <div>

      <progress value='1377' max='1377' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [1377/1377 06:49, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>0.557327</td>
      <td>0.806373</td>
      <td>0.872375</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.552700</td>
      <td>0.458040</td>
      <td>0.862745</td>
      <td>0.903448</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.333900</td>
      <td>0.560826</td>
      <td>0.867647</td>
      <td>0.907850</td>
    </tr>
  </tbody>
</table><p>


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence1, idx, sentence2.
    ***** Running Evaluation *****
      Num examples = 408
      Batch size = 8
    Saving model checkpoint to test-trainer/checkpoint-500
    Configuration saved in test-trainer/checkpoint-500/config.json
    Model weights saved in test-trainer/checkpoint-500/pytorch_model.bin
    tokenizer config file saved in test-trainer/checkpoint-500/tokenizer_config.json
    Special tokens file saved in test-trainer/checkpoint-500/special_tokens_map.json
    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence1, idx, sentence2.
    ***** Running Evaluation *****
      Num examples = 408
      Batch size = 8
    Saving model checkpoint to test-trainer/checkpoint-1000
    Configuration saved in test-trainer/checkpoint-1000/config.json
    Model weights saved in test-trainer/checkpoint-1000/pytorch_model.bin
    tokenizer config file saved in test-trainer/checkpoint-1000/tokenizer_config.json
    Special tokens file saved in test-trainer/checkpoint-1000/special_tokens_map.json
    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence1, idx, sentence2.
    ***** Running Evaluation *****
      Num examples = 408
      Batch size = 8
    
    
    Training completed. Do not forget to share your model on huggingface.co/models =)
    
    





    TrainOutput(global_step=1377, training_loss=0.37862846690325436, metrics={'train_runtime': 409.6456, 'train_samples_per_second': 26.862, 'train_steps_per_second': 3.361, 'total_flos': 405470580750720.0, 'train_loss': 0.37862846690325436, 'epoch': 3.0})


