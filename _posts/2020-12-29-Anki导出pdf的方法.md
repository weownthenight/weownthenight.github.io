---
layout: post
title: Anki导出pdf的方法
categories: 备忘
tags:  备忘
description: 复习可以用到的技巧
---

# Anki导出pdf

考研期间我将自己容易出错的地方做成了最简易的闪卡，用于每日复习。到了要考试的时候，就需要将这些闪卡都打印出来，可以在考前方便地浏览纸质版，而不是看手机。为此就有了将Anki卡片导出成pdf的需求。我的方法主要参考https://zhuanlan.zhihu.com/p/137769105，在他提供的方法上进行了改进（或者说是适应）。

## 第一步：安装插件

在Anki上安装插件最方便的办法还是打开Anki后，打开工具-附加组件，然后获取插件，填入插件代码。

![image-20210115111253062](/images/posts/image-20210115111253062.png)

Export deck to HTML这个插件的代码是1897277426。这个插件的主要作用就是把Anki中的数据集导出为HTML。

## 第二步：设置导出格式

选择工具-Export deck to HTML，选择你要导出的牌组，在CSS栏输入：

```html
table {
    font-family: verdana,arial,sans-serif;
    font-size:11px;
    color:#333333;
    border-width: 1px;
    border-color: #666666;
    border-collapse: collapse;
}
th {
    border-width: 1px;
    padding: 8px;
    border-style: solid;
    border-color: #666666;
    background-color: #dedede;
}
td {
    border-width: 1px;
    padding: 8px;
    border-style: solid;
    border-color: #666666;
    background-color: #ffffff;
}


img {
    max-width: 50%;
    height: auto;
} 
```

在HTML栏输入：

```html
<tr>
<td>{{Front}}</td>
<td>{{Back}}</td>
</tr>
```

这里的Front和Back主要取决于你的卡片是如何设置的，我的卡片是默认的格式，需要和卡片一一对应：

![image-20210118114558818](/images/posts/image-20210118114558818.png)

然后Save，用于保存默认样式。接着Export，生成一个HTML文件。

## 第三步：后处理

用xcode打开文件（如果你也是用Mac的话），commod+F找到<body>标签，在下一行加入<table>，对应地，在</body>后加入</table>

如果你的卡片用到了mathjax格式的公式，在HTML文件的最后加上：

```html
<script type="text/x-mathjax-config">
MathJax.Hub.processSectionDelay = 0;
MathJax.Hub.Config({
messageStyle:"none",
showProcessingMessages:false,
tex2jax:{
inlineMath: [['[$]','[/$]']],
processEscapes:true
}
});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_SVG-full"></script>
```

这里inlineMath这一行取决于你是用什么格式，比如我的mathjax格式就是用[\$] 和 [/$]包含起来的，所以这一行设置如上。如果你的具体用法不一样，需要更改。

编辑好HTML文件后保存。

## 第四步：打印为pdf

接下来用chrome或者其他浏览器打开HTML文件，可以选择打印。打印页面有可以保存为pdf的选项，到这一步就完成了。我的pdf打印预览如下：

![image-20210118115842721](/images/posts/image-20210118115842721.png)