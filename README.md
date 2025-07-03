# 数据挖掘课程 - 序列预测模型比较

本项目实现了两种用于序列预测的神经网络模型：MaskedLinear（带掩码的线性层）和OptimizedLSTM（优化的LSTM模型）。

## 环境设置

```bash
# 创建并激活虚拟环境
conda create -n env_data_mining python=3.9
conda activate env_data_mining

# 安装依赖
pip install -r requirements.txt
```

## 项目结构

- `model.py`: 包含MaskedLinear和OptimizedLSTM模型的定义
- `dataset.py`: 包含数据集加载和预处理函数
- `train.py`: 实现MaskedLinear模型的训练过程
- `train_cuda.py`: 使用CUDA加速MaskedLinear模型的训练
- `train_lstm.py`: 实现LSTM模型的训练过程
- `compare_models.py`: 同时训练并比较两种模型的性能
- `datasets/`: 包含CSV和Excel数据文件的目录
- `requirements.txt`: 所需的Python包

## 模型架构

### MaskedLinear模型
MaskedLinear模型使用带掩码的线性层，确保时间序列中的因果关系（未来的时间步不能影响过去的时间步）。
- 输入：形状为[batch_size, seq_len, input_dim]的时间序列数据
- 特点：通过显式的掩码机制处理时间依赖关系
- 优势：参数较少，模型简单直观

### OptimizedLSTM模型
优化的LSTM模型采用多种技术防止过拟合：
- 使用层归一化提高稳定性
- 应用dropout正则化
- 采用正交初始化提升梯度流
- 使用学习率调度器动态调整学习率
- 应用梯度裁剪防止梯度爆炸

## 使用方法

### 训练MaskedLinear模型

```bash
python train.py
```

### 训练LSTM模型

```bash
python train_lstm.py
```

### 同时训练并比较两种模型

```bash
python compare_models.py
```

此命令将：
1. 加载数据集
2. 创建并训练两种模型
3. 保存训练好的模型
4. 生成比较图表
5. 生成模型分析报告

## 模型比较

两种模型的主要区别：

| 特性 | MaskedLinear | OptimizedLSTM |
|------|------------|--------------|
| 参数量 | 较少 | 较多 |
| 记忆机制 | 显式掩码 | 门控单元 |
| 防过拟合 | 简单结构 | 多种正则化技术 |
| 计算复杂度 | 低 | 中等 |
| 表达能力 | 中等 | 较强 |

## 自定义参数

您可以在各个训练脚本中修改以下参数：
- `seq_len`: 序列长度（默认：12）
- `input_dim`: 输入维度（默认：3）
- `hidden_dim`: 隐藏层维度（MaskedLinear默认：5，LSTM默认：4）
- `output_dim`: 输出维度（默认：1）
- `num_epochs`: 训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率

## 可视化

训练后将生成以下可视化结果：
- 训练曲线（显示验证损失）
- 预测误差对比图（按月份）
- 模型性能比较图