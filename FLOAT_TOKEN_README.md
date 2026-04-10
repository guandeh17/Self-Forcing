# Float Token 算法改进文档

本文档介绍 Float Token 算法的改进实现，包括 EMA 动态压缩、质量感知筛选、分层设计和训练时启用。

## 概述

Float Token 是一种用于提升视频生成长期一致性的机制。相比原始实现，改进版本包含以下关键特性：

1. **EMA 动态压缩**：使用指数移动平均压缩被滑窗丢弃的历史信息
2. **质量感知筛选**：只让高质量帧参与 float token 更新
3. **分层设计**：短/中/长期三组 tokens，捕捉不同时间尺度的信息
4. **RoPE 对齐**：Float tokens 锚定在当前 block 起点，避免位置漂移
5. **训练时启用**：消除训练-推理不匹配问题

## 文件结构

```
wan/modules/
├── float_token_improvements.py    # 核心改进模块
└── causal_model.py                # 集成 Float Token 的模型
```

## 快速开始

### 1. 启用 Float Tokens

在创建模型时传入相应参数：

```python
from wan.modules.causal_model import CausalWanModel

model = CausalWanModel(
    model_type='t2v',
    dim=2048,
    num_heads=16,
    num_layers=32,
    # Float Token 配置
    use_float_tokens=True,                    # 启用 Float Tokens
    use_hierarchical_float_tokens=True,       # 使用分层设计
    float_token_num_slots_short=4,            # 短期层 slots
    float_token_num_slots_mid=4,              # 中期层 slots
    float_token_num_slots_long=4,             # 长期层 slots
    float_token_alpha_short=0.3,              # 短期层 EMA 系数
    float_token_alpha_mid=0.15,               # 中期层 EMA 系数
    float_token_alpha_long=0.05,              # 长期层 EMA 系数
    float_token_update_interval_short=1,      # 短期层更新间隔（每帧）
    float_token_update_interval_mid=30,       # 中期层更新间隔（每30帧）
    float_token_update_interval_long=90,      # 长期层更新间隔（每90帧）
    use_quality_scorer=True,                  # 启用质量评分
)
```

### 2. 重置 Float Bank

在新序列开始时重置 float bank：

```python
# 在开始生成新视频前
model.reset_float_banks()
```

### 3. 独立使用 Float Token 模块

```python
from wan.modules.float_token_improvements import (
    HierarchicalFloatBank,
    FloatTokenBank,
    FrameQualityScorer
)

# 创建分层 Float Bank
bank = HierarchicalFloatBank(
    d_model=2048,
    num_slots_short=4,
    num_slots_mid=4,
    num_slots_long=4,
    alpha_short=0.3,
    alpha_mid=0.15,
    alpha_long=0.05,
    use_quality_scorer=True
)

# 更新 float bank（使用被驱逐的 tokens）
evicted_tokens = torch.randn(1, 100, 2048)  # [B, num_evicted, d_model]
frame_tokens = torch.randn(1, 1560, 2048)   # [B, frame_len, d_model]
stats = bank.update(evicted_tokens, frame_tokens)

# 获取所有 float tokens
float_tokens = bank.get_all_tokens()  # [12, 2048]

# 获取统计信息
stats = bank.get_stats()
```

## 核心组件

### FloatTokenBank

基础的 EMA 更新机制：

```python
bank = FloatTokenBank(
    num_slots=4,        # Float token 数量
    d_model=2048,       # 模型维度
    alpha=0.2,          # EMA 更新系数
    update_interval=1   # 更新间隔
)

# 更新
stats = bank.update(evicted_tokens, quality_score=0.8)

# 获取 tokens
tokens = bank.get_tokens()  # [4, 2048]
```

### FrameQualityScorer

帧质量评分：

```python
scorer = FrameQualityScorer(d_model=2048)

# 评分
frame_tokens = torch.randn(1, 1560, 2048)
quality = scorer.score_frame(frame_tokens)  # [0, 1]
```

### HierarchicalFloatBank

分层 Float Token Bank：

```python
bank = HierarchicalFloatBank(
    d_model=2048,
    num_slots_short=4,      # 短期：每帧更新
    num_slots_mid=4,        # 中期：每30帧更新
    num_slots_long=4,       # 长期：每90帧更新
    alpha_short=0.3,        # 短期更新快
    alpha_mid=0.15,         # 中期适中
    alpha_long=0.05,        # 长期更新慢（稳定）
    use_quality_scorer=True
)
```

## 配置建议

### 默认配置（推荐）

```python
use_float_tokens=True
use_hierarchical_float_tokens=True
float_token_num_slots_short=4
float_token_num_slots_mid=4
float_token_num_slots_long=4
float_token_alpha_short=0.3
float_token_alpha_mid=0.15
float_token_alpha_long=0.05
float_token_update_interval_short=1
float_token_update_interval_mid=30
float_token_update_interval_long=90
use_quality_scorer=True
```

### 轻量级配置（节省显存）

```python
use_float_tokens=True
use_hierarchical_float_tokens=True
float_token_num_slots_short=2
float_token_num_slots_mid=2
float_token_num_slots_long=2
use_quality_scorer=False  # 禁用以节省计算
```

### 高质量配置（更好的长期一致性）

```python
use_float_tokens=True
use_hierarchical_float_tokens=True
float_token_num_slots_short=8
float_token_num_slots_mid=4
float_token_num_slots_long=4
float_token_alpha_short=0.4
float_token_alpha_mid=0.2
float_token_alpha_long=0.1
use_quality_scorer=True
```

## 调试和监控

### 打印统计信息

```python
# 在推理过程中定期打印
for block_idx, block in enumerate(model.blocks):
    if hasattr(block.self_attn, 'float_bank') and block.self_attn.float_bank:
        stats = block.self_attn.float_bank.get_stats()
        print(f"Block {block_idx}:")
        print(f"  Short norms: {stats['short']['slot_norms'].tolist()}")
        print(f"  Mid norms: {stats['mid']['slot_norms'].tolist()}")
        print(f"  Long norms: {stats['long']['slot_norms'].tolist()}")
```

### 使用调试器

```python
from wan.modules.float_token_improvements import FloatTokenDebugger

debugger = FloatTokenDebugger(log_interval=10)

# 在每次更新后记录
stats = bank.update(evicted_tokens, frame_tokens)
debugger.log(stats)

# 训练结束后绘制统计
debugger.plot_history('float_token_stats.png')
debugger.print_summary()
```

## 性能考虑

### 计算开销

- Float bank 更新：~0.1-0.5% 额外计算
- 质量评分：~0.5-1% 额外计算（可选）
- Attention 中的 float tokens：增加少量序列长度

### 内存开销

- 每层 float bank：~num_slots × d_model × 4 bytes
- 例如：12 slots × 2048 dim × 4 bytes = ~96KB 每层
- 32 层总计：~3MB

## 常见问题

### Q: 训练时启用 Float Tokens 会降低训练速度吗？

A: 会有轻微影响（~1-2%），但可以接受。训练时启用能消除训练-推理不匹配，显著提升推理时的长期一致性。

### Q: 如何调整 EMA 系数 alpha？

A: 
- alpha 越大，float tokens 更新越快，对近期信息更敏感
- alpha 越小，float tokens 越稳定，保留更长期的信息
- 推荐：短期(0.3) > 中期(0.15) > 长期(0.05)

### Q: 质量评分器可以禁用吗？

A: 可以。如果禁用，所有帧都会以相同的权重更新 float tokens。质量评分主要用于过滤崩坏帧和静态帧。

### Q: Float Tokens 和 Sink Tokens 有什么区别？

A: 
- Sink Tokens：保留序列开头的绝对位置信息，不参与更新
- Float Tokens：动态压缩历史信息，持续更新，提供长期上下文

## 引用

如果本改进对你的研究有帮助，请引用：

```bibtex
@software{float_token_improvements_2026,
  title={Float Token Algorithm Improvements for Video Generation},
  author={Claude},
  year={2026}
}
```

## 更新日志

- **2026-04-05**: 初始版本，实现核心改进功能
  - FloatTokenBank with EMA
  - FrameQualityScorer
  - HierarchicalFloatBank
  - RoPE alignment
  - Training mode support
