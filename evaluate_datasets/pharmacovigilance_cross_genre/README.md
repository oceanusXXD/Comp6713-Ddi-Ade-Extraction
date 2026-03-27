# 跨体裁药物警戒评测包说明

这个目录收集和药物警戒跨体裁迁移相关的公开数据。

## 目录里的内容

### `TAC2017_ADR/`

- `TAC2017_ADR/raw/train_xml.tar.gz`
  训练集压缩包。
- `TAC2017_ADR/raw/gold_xml.tar.gz`
  gold 标注压缩包。
- `TAC2017_ADR/raw/unannotated_xml.tar.gz`
  未标注数据压缩包。
- `TAC2017_ADR/extracted/train_xml/`
  解压后的训练集目录。
- `TAC2017_ADR/extracted/gold_xml/`
  解压后的 gold 标注目录。
- `TAC2017_ADR/extracted/unannotated_xml/`
  解压后的未标注目录。

### `CADEC/`

- `CADEC/raw/CADEC.v2.zip`
  官方 `CADEC v2` 压缩包。
- `CADEC/extracted/`
  `CADEC v2` 解压结果。

## 这个包适合什么时候用

- 你想测模型在不同体裁药物警戒文本上的稳定性
- 你想看模型对论坛、报告、病例类文本的跨域泛化能力
