from __future__ import annotations
import hashlib,json,platform,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SRC=Path(r"<PROJECT_ROOT>\DMKPD-V31-V28学习式角点残差")
PROJ=Path(r"<PROJECT_ROOT>")
DET=PROJ/"DMKPD-V9-20261009"/"train"/"A_main"/"epoch_12.pt"
QH=PROJ/"DMKPD-V10-20261010"/"train"/"geometry_seed20261010"/"epoch_10.pt"
SEEDS=(20261031,20261101)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def dump(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def main():
 cfg=load(ROOT/"RUN_CONFIG.json"); res=load(ROOT/"results"/"RESULTS.json")
 checks=load(ROOT/"diagnostics"/"IMPLEMENTATION_CHECKS.json")
 if not checks.get("all_pass"):raise RuntimeError("implementation checks failed")
 if sha(DET)!=cfg["model_inputs"]["detector_sha256"] or sha(QH)!=cfg["model_inputs"]["quality_head_sha256"]:raise RuntimeError("frozen weights changed")
 draws=np.load(ROOT/"results"/"bootstrap_draw_indices_seed20261102.npy")
 if draws.shape!=(1000,859) or sha(ROOT/"results"/"bootstrap_draw_indices_seed20261102.npy")!=res["bootstrap_draw_index_sha256"]:raise RuntimeError("bootstrap draw validation failed")
 total_train_gpu=0.;eval_wall=0.;runs={}
 for seed in SEEDS:
  for arm in ("A","B"):
   name=f"seed_{seed}_{arm}";run=ROOT/"runs"/name;state=load(run/"logs"/"training_complete.json")
   if (state["status"],state["final_epoch"],state["global_step"])!=("complete",3,4020):raise RuntimeError(f"incomplete run {name}")
   if sha(run/"RUN_CONFIG.json")!=sha(ROOT/"RUN_CONFIG.json"):raise RuntimeError("arm config mismatch")
   total_train_gpu+=float(state["GPU_active_seconds_total"]);epochs=[]
   for e in (1,2,3):
    m=load(run/"results"/"epochs"/f"epoch_{e:02d}_metrics.json")
    s=load(run/"logs"/f"epoch_{e:02d}_training_summary.json")
    st=load(run/"results"/"epochs"/f"epoch_{e:02d}_evaluation_status.json")
    if not st.get("complete") or s["optimizer_updates"]!=1340:raise RuntimeError(f"incomplete epoch {name}/{e}")
    eval_wall+=float(m.get("elapsed_cached_evaluation_seconds",0))
    cp=run/"checkpoints"/f"epoch_{e:02d}.pt"
    epochs.append({"epoch":e,"checkpoint":str(cp.resolve()),"sha256":sha(cp),
      "residual_head_sha256":sha(run/"checkpoints"/f"residual_head_epoch_{e:02d}.pt"),
      "updates":s["optimizer_updates"],"gpu_active_seconds":s["GPU_active_seconds_epoch"],
      "candidate_trace_sha256":sha(run/"results"/"epochs"/f"epoch_{e:02d}_candidate_trace.csv.gz"),
      "prediction_sha256":sha(run/"results"/"epochs"/f"epoch_{e:02d}_B_predictions.json")})
   runs[name]={"status":state["status"],"global_step":state["global_step"],
      "GPU_active_seconds_total":state["GPU_active_seconds_total"],"epochs":epochs}
 gate=res["promotion_gate"]
 if gate["proceed_to_online_dev_confirmation"]:raise RuntimeError("frozen online gate unexpectedly passed")
 oldres=load(ROOT/"diagnostics"/"RESULTS_BEFORE_REPORTING_FIX.json")
 summary_unchanged=all(res["seeds"][str(s)]["delta"]==oldres["seeds"][str(s)]["delta"] and
   res["seeds"][str(s)]["bootstrap_summary"]==oldres["seeds"][str(s)]["bootstrap_summary"] for s in SEEDS)
 if not summary_unchanged:raise RuntimeError("primary metrics/bootstrap changed during reporting correction")
 baseline_pass=all(load(ROOT/"runs"/f"seed_{s}_{a}"/"results"/"A_V28_metrics.json")["baseline_checks"]["candidate_output_match"] and
   load(ROOT/"runs"/f"seed_{s}_{a}"/"results"/"A_V28_metrics.json")["baseline_checks"]["floor030_exact"] for s in SEEDS for a in ("A","B"))
 if not baseline_pass:raise RuntimeError("V28 cached baseline reproduction failed in one or more runs")
 codefiles=sorted(p for p in (ROOT/"code").rglob("*.py") if "__pycache__" not in p.parts)
 codehash={str(p.relative_to(ROOT)):sha(p) for p in codefiles}
 dump(ROOT/"CODE_HASHES_FINAL.json",codehash)
 oldhash=cfg["code_hashes"].get(r"code\analyze_v32.py");newhash=codehash.get(r"code\analyze_v32.py")
 repair={"status":"post-training reporting correction","frozen_analyzer_sha256":oldhash,"final_analyzer_sha256":newhash,
  "issue":"The first reporting pass shadowed the common-GT set with a dictionary and reported its key count; it also omitted GTs unmatched by both arms from the status file.",
  "repair":"Keep common_keys separate, enumerate all 1,981 GT status rows including neither, and record pairwise final-NMS candidate-set changes.",
  "impact":"No training, predictions, matching, AP, PR, bootstrap, or decision threshold changed. The complete analysis was rerun; main metrics and bootstrap summaries are exact matches, and the bootstrap index matrix hash is unchanged.",
  "preserved_before_repair":str((ROOT/"diagnostics"/"RESULTS_BEFORE_REPORTING_FIX.json").resolve()),
  "bootstrap_draw_indices_sha256":res["bootstrap_draw_index_sha256"]}
 dump(ROOT/"logs"/"POST_TRAIN_ANALYSIS_REPAIR.json",repair)
 resource={"hard_cap_gpu_active_seconds":14400,"training_gpu_active_seconds_cuda_event_sum":total_train_gpu,
  "cached_dev_evaluation_wall_seconds_upper_bound":eval_wall,
  "conservative_total_upper_bound_seconds":total_train_gpu+eval_wall+8.0,
  "conservative_total_upper_bound_hours":(total_train_gpu+eval_wall+8.0)/3600,
  "within_4_hour_cap":total_train_gpu+eval_wall+8.0<14400,
  "note":"Cached evaluation time is wall time including CPU, and is conservatively added to CUDA-event training active time; isolated CUDA events were not recorded for the cached evaluation.",
  "bootstrap_cpu_only":True,"detector_training":False}
 dump(ROOT/"RESOURCE_REPORT.json",resource)
 v28=load(ROOT/"runs"/"seed_20261031_A"/"results"/"A_V28_metrics.json")
 lines=["# DMKPD V32 最终报告","","## 结论",
  "两个种子的 B 都提高了 local OKS AP50:95 与 polygon AP50:95。polygon AP 增益在两个种子均超过 0.002，且配对 bootstrap 95% 区间下界大于 0。TP@FP100/200 与 score≥0.30 小目标检出均上升，极小/边界检出持平。",
  "B 未通过预注册继续条件：seed 20261101 的 polygon AP75 下降 0.00003697；该种子的共同 GT 归一化角点误差 P99 上升 0.00839251。依照冻结规则，本轮不做在线 dev 确认，不启动新训练，不读取 test300，不制作提交包。结果显示主要 AP 的稳定正向信号伴随严格定位尾部权衡；停止本损失尺度分支，保留 V31/V28 历史方案，不把 B 称为已确认替代版本。","","## 冻结数据与配置",
  f"- RUN_CONFIG SHA256：{sha(ROOT/'RUN_CONFIG.json')}；源码及输入校验见 CODE_HASHES_FINAL.json、PROVENANCE_FINAL.json。",
  f"- 固定 alpha={load(ROOT/'ALPHA_CALIBRATION.json')['alpha']:.17g}。train_fit 前景坐标分量 3,842,408 个；绝对目标残差中位数 0.00846460043；超过 0.05 的比例 6.5164%；最大 1% 分量贡献 beta=1 输出梯度绝对值总和的 23.3345%。这些是候选坐标分量统计，不是 GT 数或参数梯度贡献。",
  "- alpha 只匹配预测为零时前景输出梯度 RMS，不能推出参数梯度或 AdamW 更新等价。",
  "- 完整复用 V31 train_fit 特征缓存，状态 pass，特征形状 7731×100×256×4×4 float32；未重新提取。",
  f"- 检测器 SHA256：{sha(DET)}；原 V10 质量头 SHA256：{sha(QH)}；最终复核哈希与冻结值相同。",
  "- dev859/1981GT 用于开发评价，监督只来自 train_fit。没有读取 test300、历史 validation 或 locked-confirm，没有提交或上传。","","## 必要实现检查",
  "diagnostics/IMPLEMENTATION_CHECKS.json 的检查全部通过：A beta=1 与 V31 loss 精确一致；B 解析导数与 autograd 完全一致；alpha 初始输出梯度 RMS 相对差为0；空前景/背景有限；真实 train_fit 前反向与更新通过；冻结权重不变；尾批379按256+123；保存/加载恢复后下一批模型参数及 optimizer 状态逐项相同；图像推理函数不接收 GT。",
  "seed 20261031 的三轮采样顺序逐项等于原 V31 保存顺序。原 V31 未保存训练前初始化头，故按原始 seed→构造头的随机调用顺序重建；本轮 seed 20261031 A epoch3 主 AP 和工作点与原 V31 结果精确匹配。",
  "四臂都从对应种子的同一零输出初始化文件开始，A/B 共享每轮保存的顺序，更新次数、批次、优化器均相同。每轮685,947行、1,340更新，每臂4,020更新。","","## 原 V31 收益与第二种子",
  "| seed / arm | local OKS AP50:95 | polygon AP50:95 | 相对 V28 Δlocal / Δpolygon | score≥0.30 TP/FP/FN | small / tiny / boundary |",
  "|---|---:|---:|---:|---:|---:|"]
 for seed in SEEDS:
  m=load(ROOT/"runs"/f"seed_{seed}_A"/"results"/"epochs"/"epoch_03_metrics.json");f=m["score_0.30"]
  dl=m["local_oks_ap50_95_post_polygon_nms_no_floor"]-v28["local_oks_ap50_95_post_polygon_nms_no_floor"]
  dp=m["polygon_ap50_95_post_polygon_nms_no_floor"]-v28["polygon_ap50_95_post_polygon_nms_no_floor"]
  lines.append(f"| {seed} A | {m['local_oks_ap50_95_post_polygon_nms_no_floor']:.9f} | {m['polygon_ap50_95_post_polygon_nms_no_floor']:.9f} | {dl:+.6f} / {dp:+.6f} | {f['tp']}/{f['fp']}/{f['fn']} | {f['small_detected']}/699 / {f['tiny_detected']}/31 / {f['boundary_detected']}/26 |")
 lines += ["","seed 20261031 A 与原 V31 epoch3 主指标逐位匹配。seed 20261101 A 相对 V28 的两项主 AP 也为正（local +0.00602649、polygon +0.00461734），但 TP 少2、小目标少1；主 AP 的方向复现，不代表所有检出指标复现。","","## epoch3 配对结果",
  "主 AP 为 polygon NMS=0.30、每图cap100之后、无 score floor；local OKS 是本地近似。按相同 dev 图像配对，不将两个种子合并成双倍样本。",
  "| seed | 指标 | A | B | ΔB−A | 配对bootstrap 95%区间 |","|---|---|---:|---:|---:|---:|"]
 for seed in SEEDS:
  x=res["seeds"][str(seed)]
  for metric,label,key in (("local_oks","local OKS AP50:95（本地近似）","local_oks_ap50_95_post_polygon_nms_no_floor"),("polygon","polygon AP50:95","polygon_ap50_95_post_polygon_nms_no_floor")):
   ci=next(v for v in x["bootstrap_summary"] if v["metric"]==metric);a=x["A_metrics"][key];b=x["B_metrics"][key]
   lines.append(f"| {seed} | {label} | {a:.9f} | {b:.9f} | {b-a:+.6f} | [{ci['ci95_low']:+.6f}, {ci['ci95_high']:+.6f}] |")
  lines.append(f"| {seed} | local / polygon AP75 Δ | — | — | {x['delta']['local_oks_AP75']:+.8f} / {x['delta']['polygon_AP75']:+.8f} | — |")
  lines.append(f"| {seed} | polygon AP50 / AP95 Δ | — | — | {x['B_metrics']['polygon_ap_levels'][0]-x['A_metrics']['polygon_ap_levels'][0]:+.6f} / {x['B_metrics']['polygon_ap_levels'][9]-x['A_metrics']['polygon_ap_levels'][9]:+.6f} | — |")
 lines += ["","1000次按图像配对 bootstrap，seed=20261102；两种子共用同一1000×859索引矩阵，SHA256 "+res["bootstrap_draw_index_sha256"]+"。重复图像使用独立副本ID。区间无法消除 dev 被反复用于研发筛选的偏差。完整 draw 表保存在 results/seed_*/paired_bootstrap_draws.csv.gz。","","### 工作点和共同 GT 定位",
  "| seed | floor | TP/FP/FN A→B | small A→B | tiny A→B | boundary A→B | dense A→B | 固定召回 FP，0.85/0.88/0.89 A→B | TP@FP100 / TP@FP200 A→B |","|---|---:|---|---|---|---|---|---|---|"]
 for seed in SEEDS:
  x=res["seeds"][str(seed)]
  for fl in ("0.3","0.45"):
   a=x["A_floor_metrics"][fl];b=x["B_floor_metrics"][fl]
   oa=x["A_metrics"]["operating_points"];ob=x["B_metrics"]["operating_points"]
   recalls=" / ".join(f"{oa[f'recall_{r}']['fp']}→{ob[f'recall_{r}']['fp']}" for r in ("0.85","0.88","0.89"))
   budgets=f"{x['operating_point_fp_budgets']['100']['A']['tp']}→{x['operating_point_fp_budgets']['100']['B']['tp']} / {x['operating_point_fp_budgets']['200']['A']['tp']}→{x['operating_point_fp_budgets']['200']['B']['tp']}"
   lines.append(f"| {seed} | {fl} | {a['tp']}/{a['fp']}/{a['fn']}→{b['tp']}/{b['fp']}/{b['fn']} | {a['small_detected']}/{a['small_targets']}→{b['small_detected']}/{b['small_targets']} | {a['tiny_detected']}/{a['tiny_targets']}→{b['tiny_detected']}/{b['tiny_targets']} | {a['boundary_detected']}/{a['boundary_targets']}→{b['boundary_detected']}/{b['boundary_targets']} | {a['dense_detected']}/{a['dense_targets']}→{b['dense_detected']}/{b['dense_targets']} | {recalls} | {budgets} |")
 lines.append("")
 for seed in SEEDS:
  x=res["seeds"][str(seed)]
  c=x["common_GT_normalized_error"];a=c["A_normalized_corner_error"];b=c["B_normalized_corner_error"]
  lines.append(f"seed {seed}：共同匹配GT {c['GT_count']} 个；A独有 {c['A_only']}，B独有 {c['B_only']}。共同GT归一化角点误差 mean/P95/P99：{a['mean']:.6f}/{a['p95']:.6f}/{a['p99']:.6f} → {b['mean']:.6f}/{b['p95']:.6f}/{b['p99']:.6f}。固定 A 候选+GT 关联覆盖{x['fixed_A_association_geometry']['count']}个，B−A均值差{x['fixed_A_association_geometry']['delta_mean']:+.6f}；改善/持平/下降{x['fixed_A_association_geometry']['improved']}/{x['fixed_A_association_geometry']['tied']}/{x['fixed_A_association_geometry']['worsened']}。")
  d=x["detection_status_counts"]
  ga=load(ROOT/"runs"/f"seed_{seed}_A"/"results"/"epochs"/"epoch_03_guard_summary.json")
  gb=load(ROOT/"runs"/f"seed_{seed}_B"/"results"/"epochs"/"epoch_03_guard_summary.json")
  lines.append(f"最终候选集合在{d['nms_candidate_set_changed_images_A_vs_B']}张图发生变化，候选ID移除{d['nms_candidate_ids_removed_from_A']}个、加入{d['nms_candidate_ids_added_to_B']}个。进入几何过滤前的原始非法候选为 A/B={ga['raw_invalid']}/{gb['raw_invalid']}；保护并过滤后进入最终输出的非法几何为{ga['final_invalid_after_filter']}/{gb['final_invalid_after_filter']}。非法候选统计、全量PR、AP各阈值、逐图/候选trace和GT匹配记录见 results/ 与 runs/seed_*_*/results/。")
 lines += ["","### 训练误差解释",
  "alpha校准在初始零输出时匹配前景输出梯度 RMS，但没有匹配训练中的参数梯度。训练全量候选坐标统计显示，两种子里 B 对 |target|≤0.05 的平均绝对误差约0.01146降至0.00959；对 |target|>0.05 的分量则由 A 约0.1495降至 B 约0.1519，略差。这支持“误差较小分量改善、较大残差尾部有权衡”的解释，但不证明它是所有 dev 几何变化的唯一机制。",
  "固定 train_fit 梯度诊断批次为256前景+256背景，其中 small前景69个、tiny前景0个，故这批诊断不支持 tiny 梯度判断。四臂每轮层梯度、输出更新、残差饱和和背景位移记录见各 run 的 logs；dev小/极小/边界计数见上表。",
  "",
  "## 预注册门槛",
  "| 条件 | 结果 |","|---|---|"]
 for k,v in gate["tests"].items():lines.append(f"| {k} | {v} |")
 lines += ["","不通过项是：seed20261101 polygon AP75 −0.00003697；seed20261101 共同 GT 归一化误差 P99 +0.00839251。虽然变化幅度小，预注册阈值没有豁免，故不执行全量 online dev 确认，停止这一损失尺度方案。","","## 资源与产物",
  f"- 四臂训练 CUDA-event 活跃时间合计 {total_train_gpu:.2f} 秒；12次缓存 dev 评价 wall 时间合计 {eval_wall:.2f} 秒，保守地全部视为 GPU 活跃时间后约 {(total_train_gpu+eval_wall+8)/3600:.3f} 小时，低于4小时上限。评价 wall 含 CPU 操作，实际 GPU 活跃时间更低；独立 CUDA 时间未记录。",
  "- 每臂保存 latest 与 epoch01–03 raw residual head、optimizer、RNG、epoch和采样位置；运行状态均为 complete。",
  "- 一键复现入口：<PROJECT_ROOT>\\DMKPD-V32-V31残差损失尺度\\REPRODUCE.ps1。输出包含 PLAN.md、RUN_CONFIG.json、ALPHA_CALIBRATION.json、采样顺序和哈希、四臂完整日志/checkpoint、每轮PR与trace、1000次配对bootstrap、逐GT/候选归因、诊断图像与HTML索引。",
  "- 所有 local OKS / polygon AP 阈值 0.50–0.95 的 A/B/差值在 results/PAIRED_AP_LEVEL_DELTAS.csv；全量逐轮 AP 在 results/EPOCH_METRICS.csv。",
  "- 首轮结果汇总有一处共同 GT 计数显示错误，后已完整重跑并保存修复前结果；不影响训练或 AP/PR/bootstrap。证据见 logs/POST_TRAIN_ANALYSIS_REPAIR.json 与 diagnostics/RESULTS_BEFORE_REPORTING_FIX.json。","","## 最终判断",
  "降低 SmoothL1 beta 的方案在两种子主 AP 上均有显著正向开发集差值，但 polygon AP75 和共同GT P99 的跨种子保护条件未全部满足。它显示可重复收益和可定位的尾部权衡，不满足预注册继续规则。没有官网提分保证；不追加训练、不扫描beta、不更改V31/V28历史结论。"]
 (ROOT/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 # Root browser page for representative comparisons.
 (ROOT/"figures"/"index.html").write_text("<!doctype html><meta charset='utf-8'><h1>V32 dev diagnostic examples</h1><ul><li><a href='seed_20261031/index.html'>seed 20261031</a></li><li><a href='seed_20261101/index.html'>seed 20261101</a></li></ul>",encoding="utf-8")
 # Direct paired AP by each frozen similarity threshold, preserving A and B values.
 levels=[]
 for seed in SEEDS:
  x=res["seeds"][str(seed)]
  for metric,field in (("local_oks","local_oks_ap_levels"),("polygon","polygon_ap_levels")):
   for i,(a,b) in enumerate(zip(x["A_metrics"][field],x["B_metrics"][field])):
    levels.append({"seed":seed,"similarity":metric,"threshold":round(.50+.05*i,2),"A_AP":a,"B_AP":b,"delta_B_minus_A":b-a})
 with (ROOT/"results"/"PAIRED_AP_LEVEL_DELTAS.csv").open("w",encoding="utf-8-sig",newline="") as f:
  import csv
  w=csv.DictWriter(f,fieldnames=list(levels[0]));w.writeheader();w.writerows(levels)
 validation={"status":"pass","four_arms_complete":True,"three_epochs_per_arm":True,"updates_per_arm":4020,
  "paired_bootstrap_draws":list(draws.shape),"same_draws_across_seeds":res["bootstrap_same_draws_across_seeds"],
  "bootstrap_draw_sha256":res["bootstrap_draw_index_sha256"],"v28_baseline_exact":True,
  "implementation_checks_pass":True,"v28_baseline_exact_reproduced_all_runs":baseline_pass,
  "primary_metrics_and_bootstrap_unchanged_after_reporting_fix":summary_unchanged,
  "promotion_gate_pass":gate["all_pass"],"online_confirmation_run":False,
  "test300_read":False,"submission_created":False,"upload":False,"detector_sha256_unchanged":sha(DET)==cfg["model_inputs"]["detector_sha256"],
  "quality_head_sha256_unchanged":sha(QH)==cfg["model_inputs"]["quality_head_sha256"],"resource":resource}
 dump(ROOT/"FINAL_VALIDATION.json",validation)
 prov={"frozen_provenance":load(ROOT/"PROVENANCE.json"),"code_hashes_final":codehash,"post_training_analysis_repair":repair,
  "run_config_sha256":sha(ROOT/"RUN_CONFIG.json"),"plan_sha256":sha(ROOT/"PLAN.md"),
  "cache_validation_sha256":sha(ROOT/"CACHE_VALIDATION.json"),"alpha_calibration_sha256":sha(ROOT/"ALPHA_CALIBRATION.json"),
  "source_weight_sha256":{"detector":sha(DET),"quality_head":sha(QH)},"runs":runs,
  "results_sha256":sha(ROOT/"results"/"RESULTS.json"),"epoch_metrics_sha256":sha(ROOT/"results"/"EPOCH_METRICS.csv"),
  "grouped_metrics_sha256":sha(ROOT/"results"/"GROUPED_METRICS.csv"),"paired_AP_level_deltas_sha256":sha(ROOT/"results"/"PAIRED_AP_LEVEL_DELTAS.csv"),
  "bootstrap_draw_sha256":sha(ROOT/"results"/"bootstrap_draw_indices_seed20261102.npy"),
  "final_report_sha256":sha(ROOT/"FINAL_REPORT.md"),"final_validation_sha256":sha(ROOT/"FINAL_VALIDATION.json"),
  "test300_read":False,"history_validation_read":False,"submission_created":False,"upload":False}
 dump(ROOT/"PROVENANCE_FINAL.json",prov)
 print(json.dumps({"status":"finalized","resource":resource,"promotion_gate":gate,"report_sha256":sha(ROOT/"FINAL_REPORT.md")},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
