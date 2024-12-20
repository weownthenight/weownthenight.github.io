---
layout: wiki

title: baby HTML
---

这是我为了写博客学习的一点点很简单的HTML。🔗：[HTML教程](https://www.w3school.com.cn/h.asp)

## 环境问题

Mac想要用TextEditor编辑html文件，需要提前设置才能正确显示，参考办法：[Mac如何创建HTML文件](https://www.jianshu.com/p/f8b21918ba36)

![image-20210525164812734](/images/posts/2021052501.png)

## 设置背景块

写博客的时候，遇到一些情况，比如Tips等等，想要显示出一个有背景颜色的文字块。这个时候可以这样写：

<pre>
&lt;table&gt; 
    &lt;tr&gt;
        &lt;td style="background-color:Black;color:white"&gt;
这是一个文本块。 
​		&lt;/td&gt;
​    &lt;/tr&gt; 
&lt;/table&gt;
</pre>

显示效果如下：

<table>
    <tr>
        <td style="background-color:Black;color:white">
这是一个文本块。
        </td>
    </tr>
</table>

下面解释一下：

- \<table>：想要显示文本块，需要将这个块看作一个表格中的一格，所以这里用table来实现
- \<tr>：tr表示表格中的一行
- \<td>：表示表格中的单元
- style = ...：新版HTML废弃了之前的用法，用style更好，可以设置字体、颜色、背景色。
- 注意一下，写文字内容时换行需要加上```<br />```

## 附录

🔗：[标签参考手册](https://www.w3school.com.cn/tags/index.asp)

🔗：[HTML颜色名](https://www.w3school.com.cn/html/html_colornames.asp)