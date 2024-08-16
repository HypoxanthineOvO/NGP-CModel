# Snapshots
## 命名法介绍
存储格式：Hash_{HashMapSize}_{QuantizeType}
- 关于 HashMapSize
  - Hardware 仿真一般使用 16
  - 默认配置为 19
  - 最大为 22
- 关于 QuantizeType
  - Float 是直接使用浮点模块训练的
  - QAT 是训练时量化的版本
- 没有过滤的原始数据会单独标注 `_WOF`

值得注意的是，这里的数据都仅在训练集上进行了训练；后续一部分数据会在全部数据集上进行训练，这些数据以 T 开头。

## 常见 Snapshot 平均质量
- Hash16 Float:     31.6119
- Hash16 QAT:       30.8449
- Hash19 Float:     30.9257