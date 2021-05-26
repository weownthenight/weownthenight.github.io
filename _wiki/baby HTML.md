# baby HTML

这是我为了写博客学习的一点点很简单的HTML。🔗：[HTML教程](https://www.w3school.com.cn/h.asp)

## 环境问题

Mac想要用TextEditor编辑html文件，需要提前设置才能正确显示，参考办法：[Mac如何创建HTML文件](https://www.jianshu.com/p/f8b21918ba36)

![image-20210525164812734](/Users/alexandreaswiftie/Library/Application Support/typora-user-images/image-20210525164812734.png)

## 设置背景块

写博客的时候，遇到一些情况，比如Tips等等，想要显示出一个有背景颜色的文字块。这个时候可以这样写：

\<table>
    \<tr>
        \<td style="background-color:Gray;color:white">

​			\<pre>

这是一个文本块。
            \</pre>

​		\</td>
​    \</tr>
\</table>

显示效果如下：

<table>
    <tr>
        <td style="background-color:Gray;color:white">
            <pre>
这是一个文本块。
            </pre>
        </td>
    </tr>
</table>

下面解释一下：

- \<table>：想要显示文本块，需要将这个块看作一个表格中的一格，所以这里用table来实现
- \<tr>：tr表示表格中的一行
- \<td>：表示表格中的单元
- style = ...：新版HTML废弃了之前的用法，用style更好，可以设置字体、颜色、背景色。
- \<pre>：pre表示预文本，使用pre可以保证文字内容部分按照你打字的样式保留，也就是说不需要换行加上\<br />，直接换行就可以，我觉得这样设置更方便。

## 附录

🔗：[标签参考手册](https://www.w3school.com.cn/tags/index.asp)

🔗：[HTML颜色名](https://www.w3school.com.cn/html/html_colornames.asp)