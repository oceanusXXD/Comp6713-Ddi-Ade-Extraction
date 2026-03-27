# DDI 迁移评测包说明

这个目录收集和 DDI 迁移评测相关的公开数据。

## 目录里的内容

- `DDIExtraction2013/raw/DDICorpus-2013.zip`
  `DDIExtraction 2013` 原始压缩包。
- `DDIExtraction2013/extracted/`
  `DDIExtraction 2013` 解压后的目录。
- `TAC2018_DDI/raw/trainingFiles.zip`
  `TAC 2018 DDI` 训练集压缩包。
- `TAC2018_DDI/raw/test1Files.zip`
  `TAC 2018 DDI` 测试集 1 压缩包。
- `TAC2018_DDI/raw/test2Files.zip`
  `TAC 2018 DDI` 测试集 2 压缩包。

## 这个包适合什么时候用

- 你想测模型在公开 DDI 语料上的迁移表现
- 你想把内部任务和经典 DDI 数据集做横向比较

## 说明

- 这个目录主要保存原始包和解压结果，不会自动转成当前仓库的 ChatML 训练格式。
- 如果后续需要统一格式，一般会在实验脚本里另做适配，而不是直接改动这里的原始文件。
