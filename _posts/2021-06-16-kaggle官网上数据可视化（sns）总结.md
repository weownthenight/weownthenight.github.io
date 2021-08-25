---
layout: post

title: Kaggle官网上数据可视化（sns）总结

categories: Python

---

🔗：[https://www.kaggle.com/alexisbcook/choosing-plot-types-and-custom-styles](https://www.kaggle.com/alexisbcook/choosing-plot-types-and-custom-styles)

<img src="https://imgur.com/2VmgDnF.png" height="500" width="1000" usemap="#plottingmap" />
<map name="plottingmap">
  <area shape="rect" coords="262,342,402,476" href="https://www.kaggle.com/alexisbcook/hello-seaborn" title="EXAMPLE: sns.lineplot(data=my_data)">
  <area shape="rect" coords="8,75,154,200" href="https://www.kaggle.com/alexisbcook/bar-charts-and-heatmaps" title="EXAMPLE: sns.swarmplot(x=my_data['Column 1'], y=my_data['Column 2'])">
   <area shape="rect" coords="8,200,154,350" href="https://www.kaggle.com/alexisbcook/bar-charts-and-heatmaps" title="EXAMPLE: sns.regplot(x=my_data['Column 1'], y=my_data['Column 2'])">
   <area shape="rect" coords="8,350,154,500" href="https://www.kaggle.com/alexisbcook/bar-charts-and-heatmaps" title='EXAMPLE: sns.lmplot(x="Column 1", y="Column 2", hue="Column 3", data=my_data)'>
      <area shape="rect" coords="229,10,393,160" href="https://www.kaggle.com/alexisbcook/bar-charts-and-heatmaps" title="EXAMPLE: sns.scatterplot(x=my_data['Column 1'], y=my_data['Column 2'], hue=my_data['Column 3'])">
     <area shape="rect" coords="397,10,566,160" href="https://www.kaggle.com/alexisbcook/line-charts" title="EXAMPLE: sns.heatmap(data=my_data)">
     <area shape="rect" coords="565,10,711,160" href="https://www.kaggle.com/alexisbcook/line-charts" title="EXAMPLE: sns.barplot(x=my_data.index, y=my_data['Column'])">
     <area shape="rect" coords="780,55,940,210" href="https://www.kaggle.com/alexisbcook/scatter-plots" title="EXAMPLE: sns.jointplot(x=my_data['Column 1'], y=my_data['Column 2'], kind='kde')">
     <area shape="rect" coords="780,210,940,350" href="https://www.kaggle.com/alexisbcook/scatter-plots" title="EXAMPLE: sns.kdeplot(data=my_data['Column'], shade=True)">
   <area shape="rect" coords="780,360,1000,500" href="https://www.kaggle.com/alexisbcook/scatter-plots" title="EXAMPLE: sns.distplot(a=my_data['Column'], kde=False)">
</map>

Since it's not always easy to decide how to best tell the story behind your data, we've broken the chart types into three broad categories to help with this.
- **Trends** - A trend is defined as a pattern of change.
    - `sns.lineplot` - **Line charts** are best to show trends over a period of time, and multiple lines can be used to show trends in more than one group.
- **Relationship** - There are many different chart types that you can use to understand relationships between variables in your data.
    - `sns.barplot` - **Bar charts** are useful for comparing quantities corresponding to different groups.
    - `sns.heatmap` - **Heatmaps** can be used to find color-coded patterns in tables of numbers.
    - `sns.scatterplot` - **Scatter plots** show the relationship between two continuous variables; if color-coded, we can also show the relationship with a third [categorical variable](https://en.wikipedia.org/wiki/Categorical_variable).
    - `sns.regplot` - Including a **regression line** in the scatter plot makes it easier to see any linear relationship between two variables.
    - `sns.lmplot` - This command is useful for drawing multiple regression lines, if the scatter plot contains multiple, color-coded groups.
    - `sns.swarmplot` - **Categorical scatter plots** show the relationship between a continuous variable and a categorical variable.
- **Distribution** - We visualize distributions to show the possible values that we can expect to see in a variable, along with how likely they are.
    - `sns.distplot` - **Histograms** show the distribution of a single numerical variable.
    - `sns.kdeplot` - **KDE plots** (or **2D KDE plots**) show an estimated, smooth distribution of a single numerical variable (or two numerical variables).
    - `sns.jointplot` - This command is useful for simultaneously displaying a 2D KDE plot with the corresponding KDE plots for each individual variable.
    
