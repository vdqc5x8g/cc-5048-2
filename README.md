# shfield — 球谐全景声场回放工具

把已知方向的测试声源编码为**明确阶次**的球谐（Ambisonics）声道，按听者旋转
变换声场系数，再解码到耳机或给定扬声器布局。所有空间运算都作用在球谐系数上，
不是交换左右声道；声道数量更多也不会在固定播放布局上自动带来更好的空间分辨率。

## 声明的约定（全项目统一）

- 坐标系：右手系，**+x 前、+y 左、+z 上**。
  - `theta` 极角，从 +z（上方）量起，`[0, π]`；
  - `phi` 方位角，从 +x（前方）量起，向左为正，`[-π, π]`。
- 声道顺序：**ACN**，索引 `n = l·l + l + m`（l 阶、m 度）。
- 归一化：
  - **SN3D**（ambix，内部规范）：W = 1/√2，一阶轴分量在对应单位轴上取 1；
  - **N3D**：球面正交归一，W = 1/√(4π)；
  - **FuMa**（max-N，Gerzon B 格式）：W = 1/2，仅支持 1/2/3 阶（4/9/16 声道）。
- 内部统一转换到 **ACN/SN3D** 再计算。导入时读取元数据识别约定并显式转换，
  **绝不按声道编号把所有多声道文件套进同一个矩阵**。

## 安装

```bash
pip install -e .            # 依赖 numpy / scipy / matplotlib
pip install -e .[test]      # 含 pytest
```

## 启动空间音画视窗（浏览器界面）

```bash
python -m shfield.webapp --port 8000
# 然后浏览器打开 http://localhost:8000
```

界面控件全部由 `shfield` Python 内核驱动（浏览器只做绘制和 Web Audio 播放，
没有第二套数学实现）：

1. **测试声源**：脉冲（看方向响应）或互不相关白噪声（可听定位），可增删方向。
2. **球谐阶数 N**：滑块 0–6，实时显示声道数 (N+1)²。
3. **听者旋转**：yaw/pitch/roll 三个轴，旋转矩阵作用于声场系数。
4. **解码目标**：命名扬声器布局或双耳耳机（合成 HRTF，ITD + 头部声影）。

结果区呈现：

- Canvas 等距柱状声场能量图（热区）+ 听者坐标系中的声源三角标记 + 扬声器圆点；
- 旋转矩阵正交误差、各阶能量增益、旋转前后球面能量与相对漂移；
- 每个输出通道的 RMS 与声源的听者相对方向；
- “试听结果”按钮用 Web Audio 播放内核渲染的 WAV。

任何内核失败（错误阶次、缺声道、布局承载不了该阶数）都会以 JSON 错误返回，
并以红色显示在结果区，而不是静默产生错误声场。

## 命令行（离线、可复核）

```bash
# 1) 编码一个 2 阶脉冲场景（三声源），写成 ACN/SN3D WAV + 侧车 JSON
python -m shfield.cli scene --scene impulse --order 2 \
  --source 90,0 --source 90,90 --source 45,-45 \
  --output field_ambix.wav

# 同样声场导出为 ACN/N3D 或 FuMa（数值不同，矩阵不可混用）
python -m shfield.cli scene --scene impulse --order 2 \
  --source 90,0 --ordering ACN --normalization N3D --output field_n3d.wav
python -m shfield.cli scene --scene impulse --order 2 \
  --source 90,0 --ordering FuMa --normalization FuMa --output field_fuma.wav

# 2) 听者 yaw +45°，双耳解码
python -m shfield.cli render field_ambix.wav --yaw 45 --binaural --output ears.wav

# 3) 解码到 9 扬声器布局（俯仰 20°）
python -m shfield.cli render field_ambix.wav --layout enneagon \
  --pitch 20 --output speakers.wav

# 4) 复核每个 ACN 声道内容与各阶能量
python -m shfield.cli inspect field_ambix.wav

# 5) 阶次/声道诊断（失败时退出码 2，信息明确）
python -m shfield.cli diag field_ambix.wav --expected-order 3

# 6) 逐段替换变换诊断：前上方脉冲 + 斜后方窄带声，两轴旋转→逆序还原
python -m shfield.cli diagnose --order 3 --yaw 40 --pitch 25
python -m shfield.cli diagnose --order 3 --bypass-rotation   # 单独旁路某一环节

# 7) 写出可重建的工程样例（两份声场 + 生成它们的各阶系数）
python -m shfield.cli sample examples/diagnostics --order 3
```

每个 SH WAV 都带一个 `<name>.wav.sh.json` 侧车，声明 `order / ordering /
normalization / channels / sample_rate / coordinate_system`。没有侧车且声道数
是合法平方数时才默认按 ambix（ACN/SN3D）处理并标注该假设；声道数非平方数或
元数据不完整会直接报错。

## 关键设计与可复现验证

- **旋转矩阵作用于系数**：用球面 Gauss-Legendre 求积把旋转投影到实球谐基，
  得到分阶块的实 Wigner-D 矩阵；其正交性在测试中以 `max|D Dᵀ-I| < 1e-11` 校验，
  而不是假定正确。
- **听者旋转 ↔ 声源反向移动**：听者旋转 R 时系数变换为 D(Rᵀ)=D(R)ᵀ，
  数值上等价于把每个声源在世界系中按 R 反向移动，三轴（z/y/x）误差均 ~1e-15。
- **能量守恒**：旋转前后球面能量相对漂移 ~1e-15，各阶 RMS 增益为 1。
- **不是左右交换**：D(R) 在每个阶次块是稠密矩阵，测试断言它不是置换矩阵。
- **布局容量硬约束**：N 阶需要至少 (N+1)² 个扬声器；不足时拒绝解码
  （双耳渲染改用更密的虚拟扬声器环，而不是偷偷降阶）。
- **FuMa 导入以独立表校验**：l≥2 的 FuMa 字母顺序与逐声道 max-N 增益
  （m=0 为 1，其余 m 小于 1 的 √(3)/2、√(32/45)、√(5/9)、√(5/8)）由
  `diagnostics.py` 中的参考表独立复核；修复前 2/3 阶 FuMa 文件导入会有
  可测偏差，修复后标准 FuMa 文件往返误差 ~1e-12。

## 诊断视图：逐段替换变换

`shfield.diagnostics` 与浏览器第二个面板共用一个内核函数 `run_diagnostic`。
它用一份**物理定义的独立参照**（不是回放器自己的往返）来定位失败环节：

- **非对称测试场**：前上方脉冲（theta 55°, phi 0°）+ 斜后方窄带声
  （theta 115°, phi 150°, 1500 Hz）。单轴旋转下通道顺序颠倒与手性反转听起来
  相近，所以运动先绕 **+z** 再绕 **+y**（两个不同轴），再**按逆序**还原
  （`R₂ᵀ` 后 `R₁ᵀ`）。
- **可分别旁路**输入映射、归一化、旋转，观察哪一段把偏差带进来。
- **首个偏离阶次被钉在图上**：按阶块 `l` 计算相对偏差，第一个超过容差
  (`DEV_TOL = 1e-2`) 的 l 即指纹 —— 手性/极性反转落在 l=1，高阶通道编号
  错误落在 l≥2，缺少整体增益落在 l=0。
- **独立参照**：旋转对拍“把点源按同一旋转移动后重新编码”，导入对拍
  `encode_fuma_reference`（从物理方向直接生成 FuMa 文件），都不是自洽往返。
- **缩减仍失败的姿态对**：`reduce_poses` 给出在同序还原下仍失败的最小姿态集
  （只有不同轴、非交换的两轴姿态会失败；逆序还原全部回起点）。
- **零阶负对照**：全向零阶场旋转不应改变听到的能量。若它也失败，说明问题不在
  高阶通道编号，而归因会据此改写。

浏览器面板展示环节表（每阶相对偏差）、运动表（逆序 vs 同序）、姿态缩减结果、
零阶能量漂移，以及一张按阶分组的柱状图（红线为容差）。

`python -m shfield.cli sample` 写出可重建的工程样例（默认
`examples/diagnostics/`）：

| 文件 | 内容 |
| --- | --- |
| `asymmetric_field.wav` | 内部 ACN/SN3D 的非对称测试场 |
| `asymmetric_field_fuma.wav` | 由独立参照生成的 FuMa/max-N 副本（检验导入） |
| `omni_field.wav` | 零阶（全向）负对照场 |
| `field_coefficients.json` | 每声源各阶编码系数 `Y_lm(u)`、各阶峰值快照、信号配方 |

`rebuild_field(manifest, name)` 可仅凭 JSON 与信号配方逐样本重建声场并与
原系数逐点对齐（测试中 `atol=1e-12`）。FuMa 副本可被 `read_sh_wav` 导入并
与参照方向对齐。

运行测试与端到端演示：

```bash
pytest                       # 68 个测试
python examples/demo.py      # 打印并断言上述全部数值事实
```

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `shfield/harmonics.py` | 实球谐基、ACN 索引、N3D/SN3D、球面求积 |
| `shfield/conventions.py` | ACN/FuMa 顺序与 SN3D/N3D/FuMa 归一化识别、转换、诊断 |
| `shfield/rotation.py` | 轴/欧拉旋转、实 Wigner-D、系数旋转、能量误差 |
| `shfield/encoding.py` | 平面波编码、脉冲、互不相关噪声、粉噪声 |
| `shfield/decode.py` | 布局、Tikhonov+rE 解码、合成 HRTF 双耳渲染 |
| `shfield/analysis.py` | 方向响应、波束图、各阶/球面能量 |
| `shfield/io_wav.py` | WAV + 侧车 JSON 读写、声道/阶次校验 |
| `shfield/pipeline.py` | 编码→旋转→解码串联 |
| `shfield/diagnostics.py` | 逐段替换诊断、独立参照、姿态缩减、可重建工程样例 |
| `shfield/cli.py` | 命令行 |
| `shfield/webapp.py` + `web/` | 内核驱动的浏览器空间音画视窗 |

## 关于 HRTF

双耳通路使用基于 Woodworth 路径差和一极点头部声影的**合成** HRIR，方向连续、
随球谐旋转移动，足以验证完整编码/旋转/解码链路；它不替代实测 HRTF（如 KEMAR/
SOFA）。`decode.synthetic_hrir` 是唯一需要替换成实测数据的接入点。
