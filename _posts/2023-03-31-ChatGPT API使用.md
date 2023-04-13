---
layout: post
title: ChatGPT API使用
categories: AI工具
description: ChatGPT API使用
---

## 为什么要用API？

网页端的ChatGPT回复速度和时不时的验证真的要把我逼疯了，用API可以保证回复的速度，另外现在熟悉使用API对之后可能的科研先做一个铺垫。现在AI的论文只能站在OpenAI的肩膀上来做了，如果是这样，API的使用是绕不开的。

## 收费标准

ChatGPT的API不是免费的，并且和plus分开收费。plus只能使用网页端。ChatGPT的API的rate limit如下：

> **What's the rate limits for the ChatGPT API?**
>
> Free trial users: 20 RPM 40000 TPM
> Pay-as-you-go users (first 48 hours): 60 RPM 60000 TPM
> Pay-as-you-go users (after 48 hours): 3500 RPM 90000 TPM
>
>  
>
> RPM = requests per minute
>
> TPM = tokens per minute

具体tokens的计算可以参考：🔗[How to count tokens with tiktoken.ipynb](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)。简单来说就是`import tiktoken`, 然后调用一下函数。我其实不是太所谓这个tokens，毕竟它的价格很低，后续要收费时我再研究一下怎么交钱。

官网上写的通过API的数据是不用于训练的。前3个月的使用是免费的，送了5刀。剩下是pay as you go: $0.002 / 1K tokens。ChatGPT的max_seq_len是4096 tokens，GPT-4是8192 tokens。

## 设置API

1. 首先要拿到API key：

   🔗:[api-keys](https://platform.openai.com/account/api-keys)

   需要注意：

   - API key生成是一次性的，注意保存。
   - 注意保密

2. 以python为例，首先需要配置环境，可以用conda新建一个openai环境和文件夹，`pip install openai`就行了。我建议将文件组织为以下的结构：

   ![image-20230331155606912](/images/posts/image-20230331155606912.png)

   之后把配置放到config/下，历史对话保存在history/下。

3. 接下来，我总结改动了一下代码，可以在终端用ChatGPT：

   ```python
   import openai
    import json

    def multiline_input(prompt):
        print(prompt)
        result = []
        newline_cnt = 0
        while newline_cnt < 3:
            line = input()
            if line == "":
                newline_cnt += 1
            else:
                newline_cnt = 0
            result.append(line)
        print("End!")
        return "\n".join(result[:-3])

    if __name__ == "__main__":
        openai.organization = "YOUR_ORG_ID"
        openai.api_key = "YOUR API KEY"

        with open('config/engineer.json', 'r') as f:
            config = json.load(f)

        while True:
            content = multiline_input("User: ")
            if content == "结束" or content == "exit":
                break
            config['messages'].append({"role": "user", "content": content})

            completion = openai.ChatCompletion.create(
                model = config['model'],
                messages = config['messages']
            )
            chat_response = completion.choices[0].message.content
            print(f'ChatGPT: {chat_response}')
            config['messages'].append({"role": "assistant", "content": chat_response})

        file_name = config['messages'][1]['content'][:10]

        qa = []
        for reply in config['messages']:
            qa.append(reply['role']+": "+reply['content']+"\n\n")

        with open("history/" + file_name+".md", 'w') as f:
            f.writelines(qa)
   ```

   - 设置角色：

     类似网上现在很多prompt其实就是在设置模型的角色，这个可以直接在system role里定义。具体我用json文件的格式存在config/下，比如这里的engineer.json:

     ```json
     {
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "system", "content": "You are an expert in coding on the topic of nlp, cv and multi-modal algorithms. You can speak both simplified Chinese and English."}]
     }
     ```

     这里的content可以换成任何你需要的prompt。这样不同的任务我们可以设置不同的角色设置。

   - 保存对话历史

     用`config['messages'].append({"role": "assistant", "content": chat_response})`这个方法可以将ChatGPT的历史回答都加入到模型输入中，实现多轮的对话。

     在对话结束后，我将所有历史都保存到了history/下。可以方便事后查看。
   - 退出对话

     我设置了输入结束或者exit结束循环。

4. 接下来只要在终端运行`chatgpt.py`就可以了，用API调用的话比网页端稳定很多，不需要经常验证。除了上面很基本的配置，这个函数还有一些参数可以设置，用的时候需要再说。

