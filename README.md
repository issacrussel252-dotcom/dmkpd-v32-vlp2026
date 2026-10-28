# DMKPD V32 — 2026 VLP 挑战赛参赛作品

赛道 E：DM 码四角关键点检测。公开版本对应 V32 B、seed20261031、epoch3。

实际模型为 Keypoint R-CNN（ResNet-50/FPN）加质量评分和角点残差修正，不是 CenterNet 的复现。

## 发布范围与状态

本仓库提供独立的 CUDA 推理入口、模型结构、几何处理、训练代码和历史训练配置。

`research/` 保留研究阶段训练实现，仍需迁移路径及重建缓存，**不是一键从头训练入口**。

三个正式权重已整理为 `.pth` 文件，全部可学习网络已转换为七个 ONNX 文件。模型权重和 ONNX 文件通过百度网盘共享，下载地址及提取码见下文。

本地模型包已完成导出与分阶段验证，但 ONNX 路径的整图输出与正式 PyTorch 路径仍存在差异。因此，不能宣称 ONNX 路径已完全复现历史官网成绩。具体限制见“ONNX 转换和运行”章节及 [ONNX 说明](docs/ONNX.md)。

本仓库不包含比赛图像、标注、开发集缓存、模型权重或个人账号信息。

## 方法

1. V9 检测器，FrozenBatchNorm2d，输入短边 768、长边上限 1024，候选数 100。
2. V10 质量头对原始四角特征评分；保留 FP16 往返特征数值路径。
3. V28 亚像素解码，然后应用 V32 学习式残差。
4. 保留原点索引对应关系的合法性保护；polygon NMS IoU=0.30。
5. 每图最多 100 个输出，质量分数下限 0.30。

质量分为排序分数，不是校准概率。无 TTA、模型融合或自动改标。

## 环境

已核对环境：Python 3.12，PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、NumPy 2.5.2、Pillow 12.3.0。需要支持 CUDA 的 NVIDIA GPU。

请先按硬件安装匹配的 PyTorch/torchvision CUDA 构建，再安装依赖：

```shell
python -m pip install -r requirements.txt
python infer.py --help
```

CPU 路径和不同软件版本的数值一致性尚未验证。

## 数据与权重

数据须通过比赛官方渠道自行获取；比赛附件禁止二次分发，因此不要把图像和标注提交到本仓库。

推理仅需输入图像，不读取标注。

### 模型权重下载

三个正式 `.pth` 权重文件及 ONNX 模型文件通过百度网盘共享。下载后，请核对文件名称和 SHA256，并按照下方说明放置模型文件。

- **百度网盘下载链接：** https://pan.baidu.com/s/1wq5qwDaWqIyOs7F41acGjg?pwd=hj5g
- **提取码：** `hj5g`
- **共享内容：** 三个正式 `.pth` 权重文件、七个 ONNX 模型文件及相关配置和验证文件。
- **链接有效期：** 请以百度网盘页面显示为准，并确保提交时分享链接有效期不少于 90 天。

需要以下三个原始权重文件。可保存为下列 `.pth` 文件名，但不得改变文件内容：

| 文件 | 身份 | SHA256 |
|---|---|---|
| `weights/detector.pth` | V9 A epoch12 | `89cfc3b044db37c4232cbc6f9e839b0b40bbe3fe2beb51a2e250cc3e47633a88` |
| `weights/quality.pth` | V10 geometry epoch10 | `5fa3bc7156c419799db5c7592955a28e05ca8602e9a4cfb7fdb2b1b755c67abf` |
| `weights/residual.pth` | V32 B seed20261031 epoch3 独立残差头 | `8bbcdcaa0edf7c44f681ca9c7eae0a182a392e708a0d13e06eeb455ea5b48a72` |

模型包中的 `weights/` 包含三个文件；原始文件按字节复制为 `.pth`，内容与上述哈希一致。加载入口校验上述 SHA256。

下载后，请使用 SHA256 校验值核对文件完整性。不要将未经验证的权重文件用于正式推理。

## 推理与比赛结果生成

在仓库根目录执行，参数可使用绝对路径，输出目录须为空：

```shell
python infer.py --images data/test --detector weights/detector.pth --quality weights/quality.pth --residual weights/residual.pth --output outputs/v32
```

快速检查可追加 `--limit 3`，正式比赛提交必须去掉此参数并核对图片总数为 300。

输出 `submission.zip`，内部仅包含 `map_out/<图像名>.txt`。

每行格式：

```text
DPM score x1 y1 x2 y2 x3 y3 x4 y4
```

分数保留 8 位小数，坐标保留 9 位小数；空预测图保留空 TXT。

同时输出 `predictions.json` 与 `summary.json` 便于核对。

2026-10-28 在本机完成 300 张图回归核对：共 463 个预测，每图预测数量和序列化分数均与历史 V32 提交一致；278 个 TXT 逐字节一致，22 个存在微小坐标差异，最大绝对差约 0.000046351 像素。因此不宣称本次跨运行结果逐位相同。

检查记录见 [docs/VALIDATION.json](docs/VALIDATION.json)。

## ONNX 转换和运行

全部可学习网络已转换为七个 ONNX 文件，转换边界和限制见 [ONNX 说明](docs/ONNX.md)。

ONNX 模型文件随百度网盘模型包提供。安装 `requirements-onnx.txt` 后，可从正式权重重新导出：

```shell
python export_onnx.py --detector weights/detector.pth --quality weights/quality.pth --residual weights/residual.pth --output onnx_new
python validate_onnx.py --detector weights/detector.pth --quality weights/quality.pth --residual weights/residual.pth --onnx-dir onnx_new --output onnx_validation.json
python infer_onnx.py --images data/test --onnx-dir onnx --output outputs/onnx_v32
```

ONNX 路径无需加载 `.pth`，所有可学习层通过 ONNX Runtime 执行。

当前实现使用 ONNX Runtime CPU 执行网络，PyTorch CUDA/torchvision 执行非学习算子，仍需要 CUDA GPU。

19 组分阶段动态输入测试已通过；ONNX 与正式 PyTorch 路径的整图输出存在差异。完整 300 张图分别输出 462 与 463 个预测，其中 3 张图的预测数量不同，不能声称 ONNX 路径复现了历史官网分数。

正式成绩复现请使用上面的 `infer.py` 与三个 `.pth` 文件。ONNX 路径的具体差异见 [ONNX 说明](docs/ONNX.md) 和模型包验证报告。

## 训练源码

- `src/v9_train.py`：基础检测器训练，支持 `--manifest`，以 `--help` 查看参数。
- `src/v10_rank.py`：冻结检测器的质量特征收集、目标构造和评分头训练。
- `research/v31/`：残差训练缓存构造及监督定义。
- `research/v32/`：采样、梯度尺度校准、初始化及 V32 配对训练。
- `research/*/RUN_CONFIG.json`：历史配置；`<PROJECT_ROOT>` 是原机器路径的替代标记。

详细依赖和已知限制见 [训练说明](docs/TRAINING.md)。

历史源码用于公开算法与实验过程，不应直接执行路径尚未迁移的脚本。

## 结果与限制

项目历史记录的 V32 官网 AP_50_95 / AP_50 / AP_75 / AR_50_95 为：

97.9707 / 98.0855 / 98.0855 / 99.4684

这些是历史记录，本次代码整理未重新向官网提交。

本地 dev 被多轮用于开发，后续审计发现部分 train_fit/dev 图像具有同源关系，因此不能把该 dev 当作独立泛化验证。

新环境、重新训练或 ONNX 导出不保证完全相同分数。

## 许可与来源

本项目原创代码采用 MIT 许可证，见 [LICENSE](LICENSE)。外部依赖及其许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

代码许可证不授予比赛数据的再分发权限。

`SOURCE_PROVENANCE.json` 记录源文件哈希和整理修改范围。

本项目用于 2026 VLP 挑战赛参赛及相关技术复现。请遵守赛事规则、数据集使用许可及各项第三方软件许可证。
