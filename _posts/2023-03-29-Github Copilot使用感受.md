---
layout: post
title: Github Copilot使用教程
categories: AI工具
description: Github Copilot使用教程
---

## 学生认证

学生认证可以白嫖copilot，如果毕业了只能10刀/月或100刀/年。趁着还在学校，赶紧用上学生认证。经过了一次的学生认证失败，我总结下：

1. 使用手机认证，因为申请中途需要位置信息，位置信息应该跟学校所在地一样。我用手机不用翻墙，位置信息也是正确的，如果用电脑验证可能会需要选Google Map的位置，这样位置信息就不对了。并且用手机认证拍照比较方便。入口：🔗[GitHub Education](https://education.github.com/)
2. Github个人账号信息要完整一些。最好上传自己的照片，提供真实姓名，所在的位置和时区。
3. 使用edu邮箱认证。
4. 申请证明的材料要有日期、学校名、你的名字。最好是英文的，我直接用的学校打印的英文成绩单。如果你是中文的材料，可以考虑用翻译软件处理一下，全中文的可能不支持通过。

如果你申请失败了，不要急，他返回的邮件里会指出你的申请问题，你可以根据邮件的建议进行修改。我再次申请后好像没有半小时就收到通过邮件了。

## VScode插件

在VScode上搜索Github Copilot插件，如果提示vscode版本不匹配，记得去官网下载最新版本的vscode。记得在笔试或者公司有规定的时候disable一下插件，disable的按钮在下方右侧：

![IMG_8070](/images/posts/IMG_8070.png)

听说有些公司是不允许员工使用copilot的。

安装好插件以后会有登录Github的提醒跳出，去登录就好了。在授权时可以把"Allow GitHub to use my code snippets for product improvements"取消勾选，至少理论上你的代码不会被他们用到后续训练。他们现在的结果是基于OpenAI的Codex。在Suggestions matching public code上我选了Allow。最后的成功界面：

![image-20230328221126920](/images/posts/image-20230328221126920.png)

可以在VS，VScode，JetBrains，Neovim中利用插件使用copilot，现在只要重启一下VScode就可以了。

## Copilot的使用

官方文档：🔗[Getting started with GitHub Copilot](https://docs.github.com/en/copilot/getting-started-with-github-copilot?tool=vscodet)

### accept suggestions: `Tab`

### reject all suggenstions: `Esc`

### see alternative suggestions:

| OS      | See next suggestion     | See previous suggestion |
| :------ | :---------------------- | :---------------------- |
| macOS   | `Option (⌥) or Alt`+`]` | `Option (⌥) or Alt`+`[` |
| Windows | `Alt`+`]`               | `Alt`+`[`               |
| Linux   | `Alt`+`]`               | `Alt`+`[`               |

### Seeing multiple suggestions in a new tab

- open a new tab with multiple additional options: `Ctrl` + `Enter`
- accept a suggestion: above the suggestion, click **Accept Solution**.
- reject all suggestions: close the tab
