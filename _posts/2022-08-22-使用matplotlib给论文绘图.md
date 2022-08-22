---
layout: post

title:  使用matplotlib给论文绘图

categories: 学术 Python

description: 科研绘图

---

最近开始水一篇论文，好像还是第一次自己给论文做图，我把方法记录下来：

1. 最省事的是直接看matplotlib官方的cheatsheet，这应该是最快的速成方法了吧：🔗[Matplotlib cheatsheets and handouts](https://matplotlib.org/cheatsheets/)

2. 感觉很多第三方包用于论文、出版美化，试了好几个包我都不太喜欢，暂时没有发现特别好用的，感觉matplotlib本身是够用的。

3. 想要图好看，可以参考其他漂亮论文（可以选一些大厂出品的论文，审美都在线，什么Google，Facebook之类）的配色，我就依葫芦画瓢选的配色，下面的几个图我都觉得挺好看的（出自论文Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation):

   | ![image-20220822135454893](/images/posts/image-20220822135454893.png) | ![image-20220822135835052](/images/posts/image-20220822135835052.png) |
   | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | 三个颜色的搭配：红绿蓝；圆、三角形和正方形                   | 两个颜色的搭配：红蓝；两类：实线虚线；圆、正方形             |
   | ![image-20220822135941877](/images/posts/image-20220822135941877.png) | ![image-20220822140040524](/images/posts/image-20220822140040524.png) |
   | 红色和绿色搭配也很好看！                                     | 红、黄、蓝一样有眼前一亮的感觉                               |

4. 分析我们得到的数据类型，如果已经是csv, txt这样结构化的数据，直接用pandas做；json应该也很简单，不要绕远路！

下面是我这次做图的代码，我的数据是训练完后得到的`result.csv`


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
# 如果你已经明确了用哪些列做图，可以不用导入全部，用use_col指定特定的列就好
results = pd.read_csv("results/results.csv")
results.head()
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epoch</th>
      <th>train/box_loss</th>
      <th>train/obj_loss</th>
      <th>train/cls_loss</th>
      <th>metrics/precision</th>
      <th>metrics/recall</th>
      <th>metrics/mAP_0.5</th>
      <th>metrics/mAP_0.5:0.95</th>
      <th>val/box_loss</th>
      <th>val/obj_loss</th>
      <th>val/cls_loss</th>
      <th>x/lr0</th>
      <th>x/lr1</th>
      <th>x/lr2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.033319</td>
      <td>0.025466</td>
      <td>0</td>
      <td>0.93297</td>
      <td>0.81964</td>
      <td>0.91421</td>
      <td>0.63031</td>
      <td>0.017351</td>
      <td>0.012417</td>
      <td>0</td>
      <td>0.070291</td>
      <td>0.003301</td>
      <td>0.003301</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.027304</td>
      <td>0.017404</td>
      <td>0</td>
      <td>0.89533</td>
      <td>0.80561</td>
      <td>0.90397</td>
      <td>0.56975</td>
      <td>0.019814</td>
      <td>0.014309</td>
      <td>0</td>
      <td>0.040269</td>
      <td>0.006612</td>
      <td>0.006612</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.025737</td>
      <td>0.017176</td>
      <td>0</td>
      <td>0.92179</td>
      <td>0.80310</td>
      <td>0.90728</td>
      <td>0.61065</td>
      <td>0.020163</td>
      <td>0.013901</td>
      <td>0</td>
      <td>0.010225</td>
      <td>0.009902</td>
      <td>0.009902</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.026707</td>
      <td>0.018136</td>
      <td>0</td>
      <td>0.64618</td>
      <td>0.39679</td>
      <td>0.47659</td>
      <td>0.21491</td>
      <td>0.031275</td>
      <td>0.027192</td>
      <td>0</td>
      <td>0.009901</td>
      <td>0.009901</td>
      <td>0.009901</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.027679</td>
      <td>0.016580</td>
      <td>0</td>
      <td>0.78914</td>
      <td>0.67735</td>
      <td>0.73899</td>
      <td>0.42854</td>
      <td>0.026546</td>
      <td>0.018784</td>
      <td>0</td>
      <td>0.009901</td>
      <td>0.009901</td>
      <td>0.009901</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 这里的列名还是很烦的，确认一下列名到底是怎样的
results.columns
```




    Index(['               epoch', '      train/box_loss', '      train/obj_loss',
           '      train/cls_loss', '   metrics/precision', '      metrics/recall',
           '     metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', '        val/box_loss',
           '        val/obj_loss', '        val/cls_loss', '               x/lr0',
           '               x/lr1', '               x/lr2'],
          dtype='object')




```python
# 把列名全改了！
results.rename(columns = {'               epoch':'epoch', '      train/box_loss':'train_box_loss', 
                          '      train/obj_loss':'train_obj_loss', '   metrics/precision':'precision', 
                          '      metrics/recall':'recall', '     metrics/mAP_0.5':'mAP_0.5', 
                          'metrics/mAP_0.5:0.95':'mAP_0.5:0.95','        val/box_loss':'val_box_loss',
                         '        val/obj_loss':'val_obj_loss'}, inplace = True)
```


```python
# 读入另外一个模型的数据，做比较
results_anno = pd.read_csv("results/results_anno.csv")
results_anno.rename(columns = {'               epoch':'epoch', '      train/box_loss':'train_box_loss', 
                          '      train/obj_loss':'train_obj_loss', '   metrics/precision':'precision', 
                          '      metrics/recall':'recall', '     metrics/mAP_0.5':'mAP_0.5', 
                          'metrics/mAP_0.5:0.95':'mAP_0.5:0.95','        val/box_loss':'val_box_loss',
                         '        val/obj_loss':'val_obj_loss'}, inplace = True)
```


```python
# 合并loss
results["loss"] = results["train_obj_loss"] + results["train_box_loss"]
results_anno["loss"] = results_anno["train_obj_loss"] + results_anno["train_box_loss"]
```


```python
# plt.subplots()可以做到多个小图，不过这里不需要
fig, ax = plt.subplots()
# 'o-'确定了画的是实线，点是实心圆，markevery确定多少个epoch做一次圆圈标记，color可以自己去选，我在cheatsheet里找的
ax.plot(results['epoch'], results['loss'],"o-", color="C3", markevery=50, 
         label='With Copy-Paste')
# 's-'确定了画的是实线，点是方形
ax.plot(results_anno['epoch'], results_anno['loss'], 's-',  color="C0", markevery=50,
        label='Without Copy-Paste')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
# autoscale()可以自动确定画出的图x轴和y轴的取值范围
plt.autoscale()
ax.legend()
# grid就是画方格线
ax.grid()
plt.show()
# 放论文的图存到pdf文件，再将pdf插入latex里
fig.savefig("loss.pdf")
```

最后的效果：

![image-20220822140953404](/images/posts/image-20220822140953404.png)

​    

