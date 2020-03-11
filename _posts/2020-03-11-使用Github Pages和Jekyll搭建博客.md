---

layout: post

title: 使用GitHub Pages和Jekyll搭建博客

categories: 备忘

---

用GitHub Pages来建博客可能是最折腾的事情之一。几年前我第一次接触到后，用Hexo搭建了博客，成功后只上传了几次，换过笔记本重新搭建本地环境，经常报错，让我放弃了Hexo。时至今日，我又重新搭建好了博客，处理各种问题，两天时间才搭建好。这件事情不像一步步跟着教程做那么简单，也没有几年前我认为的那样难。

## 1、找一个你喜欢的theme

如果你还没有GitHub账号，需要注册一个，用户名最好不要含有大写字母，曾经我有一个含大写字母的账号，不能和GitHub Pages上对应，所以搭建博客总是失败。

在GitHub上所有Jekyll博客主题中选一个喜欢的主题，比较推荐的是Jekyll官网上提供的几个网站和GitHub官方给的一个地址：

> [jamstackthemes.dev](https://jamstackthemes.dev/ssg/jekyll/)
>
> [jekyllthemes.org](http://jekyllthemes.org/)
>
> [https://github.com/topics/jekyll-theme](https://github.com/topics/jekyll-theme)



选主题时要比较注意几点：

- 尽量选择近期有更新维护处理issues的主题，如果你的能力有限（和我一样），你不希望你的主题有什么bug导致接下来的工作进行不下去，如果出现问题，有维护的主题相比之下更好。
- 如果你需要用中文写博客，尽量找支持中文的模版，有些模版英文很好看，中文就不怎么好看了。

## 2、现在就能看看你自己的博客了

找到你满意的theme以后，进入到对应的GitHub repository，Fork一下，这个仓库就到了你的账户下。这个时候，你需要做几件事情：

- 更改仓库名。

  取决于你想要这个主题部署在哪个网址上。如果你想要在 username.github.io 上看到，那么仓库名就要改成 username.github.io，就像这样：

  ![image-1](/images/posts/weownthenight.png)

  其中 weownthenight 要改成自己的用户名。

  如果你想要这个主题部署到 用户名.github.ui/blog 上，就请你把你的仓库名命名为blog，以此类推。

- 更改branch。

  观察一下 fork 来的仓库有几个 branch，默认 branch 是哪个？一般会有两个 branch，一个master，另一个gh-pages。如果你是部署到 username.github.io 上，那么只要留下 master 并把它设为默认就可以了。如果是要部署到 username.github.io/blog 上，那么你需要删掉 gh-pages后再新建一个 gh-pages ，并把新建的 gh-pages 设为默认。

做完这些，你就可以浏览一下网址了，此时你的网站应该和模版网站一模一样。

如果这时候发现有问题，通常是你的网站只有内容一样，效果都没有。通常只要改一下 _config.yml。找到baseurl或者url，将地址改为你自己的博客地址就可以了。

## 3、在本地安装Jekyll

我在安装Jekyll的过程中出现了很多问题，强烈建议按照Jekyll官网的教程一步步做：

> https://jekyllrb.com/docs/

安装好Jekyll后，就可以将 GitHub 上的仓库克隆到本地了：

```
git clone 仓库地址
```

复制到本地后，转换到文件夹，运行 jekyll serve，就可以在本地看到博客了。

## 4、修改模版、上传博客

相比之下修改模版比较简单，即使不熟悉Jekyll也能猜个大概。每次修改文件后，运行 :

```
git add .
git commit -m "update"
git push origin master
```

如果你用的是 gh-pages，就将 master 改为 gh-pages即可。

如果以上还有疑问，可以参考：

https://medium.com/20percentwork/creating-your-blog-for-free-using-jekyll-github-pages-dba37272730a

（大体步骤都写得很详细）

https://jekyllrb.com/docs/

（Jekyll官方文档，基本可以解决安装过程的问题）

https://blog.webjeda.com/

（这个哥们的博客可以解决Jekyll博客搭建过程中普遍的错误，他还有 Youtube 视频可以解惑）

https://help.github.com/en/github/working-with-github-pages/about-github-pages-and-jekyll

（GitHub Pages提供的官方文档，我觉得有点多，看完需要耐心，有疑问的话，可以找一下）