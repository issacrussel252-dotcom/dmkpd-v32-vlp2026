# 训练源码和复现边界

最终预测依赖三个训练阶段，不只是 V32 残差头。仓库提供所有主要训练阶段的源码，
但目前只将最终权重推理整理为独立命令；完整从零训练流水线尚未在新目录验证。

## 基础检测器

`src/v9_train.py` 包含 prepare/train/infer/verify 等命令，默认训练 12 轮，
种子 20261009。使用 `python src/v9_train.py train --help` 查看参数。
manifest 为 CSV，至少包含 stem、image、annotation、split；
image/annotation 指向用户自行取得的图像和 LabelMe 标注，split 使用 train_fit/dev。
禁止根据预测结果修改比赛标注。

## 质量评分头

`src/v10_rank.py` 提供 collect/targets/fit/emit。
环境变量 DMKPD_DETECTOR 指定检测器，DMKPD_MANIFEST 指定划分文件，
DMKPD_QUALITY_RUN 指定输出目录。geometry 分支使用种子 20261010。
正式 V10 权重为 epoch10。模型与特征维数必须与推理代码一致。
冻结检测器哈希校验针对历史权重；重新训练的权重不能冒充原权重。

## V32 残差头

历史实现依次依赖 V31 的 train_fit 候选缓存、缓存完整性验证、
V32 prepare 生成的校准/采样/初始化文件、冻结配置、最后 train_v32。
前景目标与匹配定义见 research/v31/collect_v31.py；
make_order 和梯度尺度校准见 research/v32/prepare_v32.py。
V32 B 使用 beta=0.01、alpha=0.07782134721914898，背景权重 0.1，
AdamW，学习率和 weight decay 均为 1e-4，microbatch 256、累积 2 次、3 轮。

research 中的源码保留了实验间依赖，<PROJECT_ROOT> 需迁移为自己的项目根目录，
并按导入关系放置 src 中的 source_snapshot 模块和 V28 解码器。
历史采样顺序、缓存、冻结配置的哈希校验不能通过伪造文件绕过。
尚未提供通用缓存重建/目录迁移入口，不能声称只运行一个命令就能重现训练。
重新训练结果应作为新实验验证，不能保证历史权重或官网成绩逐位一致。

## 尚需完成的比赛材料

独立的权重下载链接（有效期不少于 90 天）、完整 ONNX 转换与运行验证仍待发布。
本轮代码整理没有启动训练、修改数据标签或上传数据集。
