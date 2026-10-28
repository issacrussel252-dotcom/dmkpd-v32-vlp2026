from __future__ import annotations
import json, hashlib, os, platform, shutil, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(r"<PROJECT_ROOT>\DMKPD-V31-V28学习式角点残差")
PROJECT=Path(r"<PROJECT_ROOT>")
MANIFEST=PROJECT/"07_第5版_R6正式复跑_20261006"/"inputs"/"split_manifest_r2_internal.csv"
V28=PROJECT/"DMKPD-V28-V10亚像素解码-配对复核"
DETECTOR=PROJECT/"DMKPD-V9-20261009"/"train"/"A_main"/"epoch_12.pt"
QUALITY=PROJECT/"DMKPD-V10-20261010"/"train"/"geometry_seed20261010"/"epoch_10.pt"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()
def dump(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def main():
    if (ROOT/"RUN_CONFIG.json").exists(): raise FileExistsError("RUN_CONFIG already frozen")
    alpha=json.loads((ROOT/"ALPHA_CALIBRATION.json").read_text(encoding="utf-8"))
    cache=json.loads((ROOT/"CACHE_VALIDATION.json").read_text(encoding="utf-8"))
    init=json.loads((ROOT/"INITIALIZATION_HASHES.json").read_text(encoding="utf-8"))
    sampling=json.loads((ROOT/"SAMPLING_MANIFEST.json").read_text(encoding="utf-8"))
    pyfiles=sorted(p for p in (ROOT/"code").rglob("*.py") if "__pycache__" not in p.parts)
    code_hashes={str(p.relative_to(ROOT)):sha(p) for p in pyfiles}
    dump(ROOT/"CODE_HASHES.json",code_hashes)
    key_inputs=[DETECTOR,QUALITY,MANIFEST,SOURCE/"FINAL_REPORT.md",SOURCE/"PLAN.md",SOURCE/"RUN_CONFIG.json",
       SOURCE/"PROVENANCE.json",SOURCE/"code"/"train_v31.py",SOURCE/"code"/"v31_core.py",
       SOURCE/"code"/"evaluate_v31.py",SOURCE/"code"/"checks_v31.py",
       SOURCE/"diagnostics"/"CACHE_INTEGRITY.json",SOURCE/"DATA_INPUT_HASHES.json",
       V28/"RESULTS.json",V28/"results"/"B_predictions.json"]
    source_hashes={str(p):sha(p) for p in key_inputs}
    env={"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,
         "torch_cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),
         "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         "cudnn":torch.backends.cudnn.version(),"matmul_tf32":torch.backends.cuda.matmul.allow_tf32,
         "cudnn_tf32":torch.backends.cudnn.allow_tf32,"cudnn_deterministic":torch.backends.cudnn.deterministic,
         "cudnn_benchmark":torch.backends.cudnn.benchmark}
    config={"experiment":"DMKPD V32 foreground residual SmoothL1 scale paired comparison",
      "frozen_before_training_and_B_evaluation":True,"fixed_main_epoch":3,
      "seeds":[20261031,20261101],"arms":{"A":{"foreground":"mean SmoothL1(pred-target,beta=1.0)","alpha":1.0},
      "B":{"foreground":"alpha * mean SmoothL1(pred-target,beta=0.01)","alpha":alpha["alpha"]},
      "both":{"background":"0.1 * mean SmoothL1(pred,beta=1.0)","sampling":"V31 fixed per-image 1:1 cap; same saved order within seed"}},
      "alpha_calibration":alpha,"training":{"epochs":3,"optimizer":"AdamW","lr":1e-4,"weight_decay":1e-4,
        "microbatch":256,"gradient_accumulation_microbatches":2,"effective_batch_max":512,
        "selected_rows_per_epoch":{str(s):{str(e):sampling[str(s)][str(e)]["num_rows"] for e in (1,2,3)} for s in (20261031,20261101)},
        "augmentation":"none","head_parameters":534216,"output":"zero initialized","residual_limit":0.05},
      "model_inputs":{"detector_path":str(DETECTOR),"detector_sha256":sha(DETECTOR),"quality_head_path":str(QUALITY),
        "quality_head_sha256":sha(QUALITY),"normalization":"FrozenBatchNorm2d","manifest":str(MANIFEST),
        "manifest_sha256":sha(MANIFEST),"train_fit_cache_root":str(SOURCE/"cache"/"train_fit"),
        "dev_cache_root":str(SOURCE/"cache"/"dev"),"cache_input_hashes":cache["input_hashes"],
        "dev_partition":{"images":859,"gt":1981},"feature_reextraction":False},
      "initialization":init,"sampling_manifest":sampling,"diagnostic_batch":json.loads((ROOT/"diagnostics"/"FIXED_DIAGNOSTIC_BATCH.json").read_text(encoding="utf-8")),
      "bootstrap":{"unit":"image","replicates":1000,"seed":20261102,"same_precomputed_draws_across_seeds":True,
        "duplicate_sample_ids":"unique bootstrap copy ordinal"},
      "promotion_gate":{"both_seeds_both_AP_non_decreasing":True,
        "at_least_same_AP_improves_ge_0_002_each_seed_and_each_CI_low_gt_0":True,
        "both_seeds_AP75_non_decreasing":True,"TP_at_FP100_and_200_non_decreasing":True,
        "floor030_small_tiny_boundary_non_decreasing":True,"common_GT_normalized_mean_P95_P99_non_increasing":True,
        "final_illegal_geometry_non_increasing":True},
      "resource":{"gpu_active_hours_hard_cap":4,"no_detector_training":True},
      "forbidden":{"test300_images_predictions_gt":True,"history_validation_or_locked_confirm":True,
        "submission_or_upload":True,"additional_beta_or_hyperparameter_search":True},
      "source_hashes":source_hashes,"code_hashes":code_hashes,"environment":env}
    dump(ROOT/"RUN_CONFIG.json",config)
    plan=f'''# DMKPD V32 实验预注册计划\n\n冻结时间（本机）：{__import__('datetime').datetime.now().astimezone().isoformat()}。本文件与 `RUN_CONFIG.json` 在四臂训练及本轮 B 评价前生成。\n\n## 唯一假设与实验矩阵\n\n仅改变前景残差 SmoothL1 转折点：每个种子各跑 A/B 三轮。A 使用 V31 的 beta=1.0；B 使用 beta=0.01，前景损失乘固定 alpha={alpha['alpha']:.17g}。背景项固定为 0.1×beta=1.0，其他模型、数据、采样、优化器、训练轮次和推理保护保持 V31。种子为 20261031、20261101；A/B 在每种子下共用同一零输出初始化和逐轮缓存采样顺序。epoch3 是固定主结果。\n\n## alpha 与数据统计\n\nalpha 仅由全部 train_fit 前景坐标分量在预测为零时的输出梯度 RMS 比计算：{alpha['beta1_output_gradient_rms']:.12g}/{alpha['beta0_01_output_gradient_rms_unscaled']:.12g}={alpha['alpha']:.17g}。train_fit 共 {alpha['foreground_coordinate_components']:,} 个前景坐标分量；|target| 中位数 {alpha['abs_target_median']:.12g}，超过 0.05 的比例 {alpha['abs_target_fraction_gt_0_05']:.6%}，绝对目标最大 1% 的分量占 beta=1 输出梯度绝对值总和 {alpha['top_1pct_abs_error_gradient_share']:.6%}。这些是候选坐标分量统计，不是独立 GT 数或参数梯度贡献。RMS 校准只匹配零残差时的前景输出梯度，不意味着参数梯度或 AdamW 更新等价。\n\n## 初始化、采样与数据\n\n复用 V31 train_fit 特征缓存，不重新提取。seed 20261031 的重建采样序列已与原 V31 三轮保存顺序逐项一致；20261101 按冻结 V31 采样代码和该 seed 预生成。初始化文件、每轮顺序、缓存与源码哈希见 `RUN_CONFIG.json`、`INITIALIZATION_HASHES.json`、`SAMPLING_MANIFEST.json`。原 V31 未保存可核对的训练前头部 checkpoint；20261031 初始状态按原入口“seed 后构造零输出头”的随机调用顺序重建，因此不能证明与已丢失/未保存的历史初始张量逐位一致。\n\n训练监督仅使用 train_fit；dev859/1981 GT 用于每轮开发评价和筛查。历史 validation、locked-confirm 和 test300 不读取。\n\n## 评价与继续条件\n\n每轮固定评价冻结 V28 dev 候选缓存，polygon NMS=0.30、最终 cap=100；主 AP 无 score floor，local OKS 明确为本地近似。epoch3 每种子分别执行 1000 次按图像配对 bootstrap，seed=20261102，两种子共用同一 859×1000 重采样索引，重复图像使用独立 copy ID。主因果对照为同种子的 B−A。\n\n只有 `RUN_CONFIG.json` 中所有门槛同时成立，才会对 seed20261031 epoch3 A/B 执行全量在线 dev 确认。否则停止本支线，不扫描 beta、不增加训练、不追加种子。bootstrap 不能消除 dev 被用于开发筛查的偏差。\n\n## 资源与边界\n\n总 GPU active time 上限 4 小时，最多四次、每次三轮残差头训练。不训练检测器，不生成提交包，不上传。不依据官网总分或 dev 中途表现改变配置。\n'''
    (ROOT/"PLAN.md").write_text(plan,encoding="utf-8")
    prov={"experiment_root":str(ROOT),"source_experiment":str(SOURCE),"frozen":True,
      "source_hashes":source_hashes,"code_hashes":code_hashes,"cache_validation_sha256":sha(ROOT/"CACHE_VALIDATION.json"),
      "alpha_calibration_sha256":sha(ROOT/"ALPHA_CALIBRATION.json"),"sampling_manifest_sha256":sha(ROOT/"SAMPLING_MANIFEST.json"),
      "initialization_hashes_sha256":sha(ROOT/"INITIALIZATION_HASHES.json"),"run_config_sha256":sha(ROOT/"RUN_CONFIG.json"),
      "test300_read":False,"history_validation_read":False,"submission_created":False,"upload":False}
    dump(ROOT/"PROVENANCE.json",prov)
    for seed in (20261031,20261101):
      for arm in ("A","B"):
        d=ROOT/"runs"/f"seed_{seed}_{arm}"
        for sub in ("checkpoints","logs","results","results/epochs","results/pr","figures","diagnostics"):
          (d/sub).mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/"RUN_CONFIG.json",d/"RUN_CONFIG.json")
    print(json.dumps({"status":"frozen","run_config_sha256":sha(ROOT/"RUN_CONFIG.json"),"code_hashes":code_hashes,"gpu":env["gpu"]},ensure_ascii=False),flush=True)
if __name__=="__main__": main()
