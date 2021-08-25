---
layout: post

title: Hugging Face Transformers教程笔记(4)：Handling Multiple Sequences

categories: NLP

---

解决以下几个问题：
- How do we handle multiple sequences?
- How do we handle multiple sequences of different lengths?
- Are vocabulary indices the only inputs that allow a model to work well?
- Is there such a thing as too long a sequence?

## Models expect a batch of inputs


```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
# 转换为tensor
input_ids = torch.tensor(ids)
# This line will fail.
model(input_ids)
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=48.0, style=ProgressStyle(description_w…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=629.0, style=ProgressStyle(description_…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=231508.0, style=ProgressStyle(descripti…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=267844284.0, style=ProgressStyle(descri…


    



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-4-a35fd37e6170> in <module>()
         13 input_ids = torch.tensor(ids)
         14 # This line will fail.
    ---> 15 model(input_ids)
    

    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1050                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1051             return forward_call(*input, **kwargs)
       1052         # Do not call functions when jit is used
       1053         full_backward_hooks, non_full_backward_hooks = [], []


    /usr/local/lib/python3.7/dist-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, attention_mask, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        630             output_attentions=output_attentions,
        631             output_hidden_states=output_hidden_states,
    --> 632             return_dict=return_dict,
        633         )
        634         hidden_state = distilbert_output[0]  # (bs, seq_len, dim)


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1050                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1051             return forward_call(*input, **kwargs)
       1052         # Do not call functions when jit is used
       1053         full_backward_hooks, non_full_backward_hooks = [], []


    /usr/local/lib/python3.7/dist-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, attention_mask, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)
        486 
        487         if inputs_embeds is None:
    --> 488             inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)
        489         return self.transformer(
        490             x=inputs_embeds,


    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1050                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1051             return forward_call(*input, **kwargs)
       1052         # Do not call functions when jit is used
       1053         full_backward_hooks, non_full_backward_hooks = [], []


    /usr/local/lib/python3.7/dist-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids)
        111         embeddings)
        112         """
    --> 113         seq_length = input_ids.size(1)
        114         position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        115         position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)


    IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)


报错了：dimension out of range.

因为在默认情况下，model的input是多个sequence，而我们只有一句话。

可以看一下tokenizer是怎么处理的：


```python
tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])
```

    tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
              2607,  2026,  2878,  2166,  1012,   102]])



```python
input_ids
```




    tensor([ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607,
             2026,  2878,  2166,  1012])



可以看到，多了一对方括号，也就是多了一维。


```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

# 我们自己加了一维
input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
```

    Input IDs: tensor([[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607,
              2026,  2878,  2166,  1012]])
    Logits: tensor([[-2.7276,  2.8789]], grad_fn=<AddmmBackward>)


## Padding the inputs

Padding makes sure all our sentences have the same length by adding a special word called the padding token to the sentences with fewer values. 


```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
# the padding token ID: tokenizer.pad_token_id
batched_ids = [[200, 200, 200], [200, 200, tokenizer.pad_token_id]]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
```

    tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
    tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
    tensor([[ 1.5694, -1.3895],
            [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)


观察结果可以发现，第二句话在padding前和padding后得到的tensor不同。这是因为transformer会将padding token和句子中所有的token全部attend，要想两者的结果相同，我们需要transformer忽略padding token，对它不计算attention，这一点可以通过attention mask来实现。

## Attention masks

Attention masks are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s: 1s indicate the corresponding tokens should be attended to, and 0s indicate the corresponding tokens should not be attended to (i.e., they should be ignored by the attention layers of the model).

还是以上述例子举例，我们来看如何利用attention mask让第二句话在padding前和padding后的tensor结果相同：


```python
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id]
]

attention_mask = [
  [1, 1, 1],
  [1, 1, 0]
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
```

    tensor([[ 1.5694, -1.3895],
            [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)


## Longer sequences

transformer model对句子的最大长度通常有限制，一般是512或1024 tokens。对于较长的句子，有两种处理方式：
- Use a model with a longer supported sequence length.
- Truncate your sequences.

不同的model有不同的处理方式，这个需要由具体哪种model的实现来决定。

你也可以使用truncate：


```python
sequence = sequence[:max_sequence_length]
```
