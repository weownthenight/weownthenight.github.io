---

layout: post
title: VSCode+Pytest:简单的Python单元测试实践
categories: Python
description: pytest
---

参考：[Python testing in Visual Studio Code](https://code.visualstudio.com/docs/python/testing)

[pytest](https://docs.pytest.org/en/7.2.x/)

## Configure tests

VSCode配置首先需要满足：在extension中安装Python扩展，如果是远程环境，远程上的Python扩展也要安装。安装好以后，用 `cmd` + `shift`  + `P`（Mac）或`ctrl` + `shift` + `P` （Windows）输入 `Python:Configure Tests`选择好配置的框架，在这里我们选择`pytest`框架。配置好以后，vscode的侧栏会有一个test beaker icon，如下图：

![Configure Python Tests button displayed in the Test Explorer when tests haven't been configured.](https://code.visualstudio.com/assets/docs/python/testing/test-explorer-no-tests.png)

## pytest写法

首先文件名要符合之前配置的文件名格式，比如`test_xxx.py`以使框架能发现这些测试代码。

在文件中写法以下文为例：

```python
from pathlib import Path

import cv2
import pytest

from app.table.extractor import BorderedCellExtractor


@pytest.fixture
def image_file():
    image_file = "image/IMG_7150.png"
    return image_file


class TestBorderedCellExtractor:
    def test_extract(self, tmp_path: Path, image_file):
        extractor = BorderedCellExtractor()
        image = cv2.imread(image_file)
        cells = extractor.get_cells(image)
        print("cells:", len(cells))
        canvas = extractor.draw_cells(image, cells)
        cv2.imwrite(str(tmp_path / "cells.png"), canvas)
        print("cells saved to", tmp_path / "cells.png")
```

想要看到print的信息，可以打开Debug Console，执行debug，在pytest中tmp_path默认就是`tmp/`。