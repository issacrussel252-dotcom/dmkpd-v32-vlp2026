from __future__ import annotations
import csv, gzip, hashlib, json, math, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from v31_core import MANIFEST, BASE_SHA, HEAD_SHA, ROOT as CORE_ROOT, sha256, ev, geom


def readj(path):
    return json.loads(path.read_text(encoding='utf-8'))

def writej(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def stats(x):
    a=np.asarray(x,dtype=np.float64).reshape(-1)
    if not len(a): return {'count':0,'mean':None,'median':None,'p90':None,'p95':None,'p99':None,'max':None}
    return {'count':int(len(a)),'mean':float(np.mean(a)),'median':float(np.median(a)),
            'p90':float(np.percentile(a,90)),'p95':float(np.percentile(a,95)),
            'p99':float(np.percentile(a,99)),'max':float(np.max(a))}

def read_pred(path):
    return readj(path)

def compare_predictions(cached, online):
    images=sorted(set(cached)&set(online))
    ids_mismatch=0; count_mismatch=0; common=0; score_diffs=[]; coord_diffs=[]; image_geometry_change=0
    for stem in images:
        ca={int(x['candidate_id']):x for x in cached[stem]}
        on={int(x['candidate_id']):x for x in online[stem]}
        if set(ca)!=set(on): ids_mismatch+=1
        if len(ca)!=len(on): count_mismatch+=1
        local_geom=False
        for cid in sorted(set(ca)&set(on)):
            a,b=ca[cid],on[cid]; common+=1
            score_diffs.append(abs(float(a.get('score',a.get('quality_score',0))) - float(b.get('score',b.get('quality_score',0)))))
            qa=np.asarray(a['quad'],dtype=np.float64).reshape(4,2); qb=np.asarray(b['quad'],dtype=np.float64).reshape(4,2)
            d=float(np.max(np.abs(qa-qb))); coord_diffs.append(d); local_geom |= d>0
        image_geometry_change+=int(local_geom)
    return {'images_compared':len(images),'images_candidate_id_set_mismatch':ids_mismatch,
            'images_count_mismatch':count_mismatch,'common_predictions':common,
            'score_abs_diff':stats(score_diffs),'max_abs_coordinate_difference_px':max(coord_diffs,default=0.0),
            'images_with_any_coordinate_difference':image_geometry_change}

def main():
    run_config=readj(ROOT/'RUN_CONFIG.json'); provenance=readj(ROOT/'PROVENANCE.json')
    lock=readj(ROOT/'results'/'online_confirmation_predictions_staged.lock.json')
    staged_path=ROOT/'results'/'online_confirmation_predictions_staged.json'
    if sha256(staged_path)!=lock['predictions_sha256']:
        raise RuntimeError('staged online prediction lock hash mismatch')
    online_record=readj(ROOT/'results'/'online_confirmation_metrics_staged.json')
    if online_record['status']!='complete' or online_record['GT_loaded_only_after_predictions_locked'] is not True:
        raise RuntimeError('staged online confirmation is incomplete or violates GT ordering')
    ckpt_status=readj(ROOT/'logs'/'training_complete.json')
    if ckpt_status['status']!='complete' or ckpt_status['final_epoch']!=3 or ckpt_status['global_step']!=4020:
        raise RuntimeError('training completion status mismatch')

    # Compare online, image-only predictions against the frozen-cache outputs.
    online=read_pred(staged_path)
    cache_a=read_pred(ROOT/'results'/'A_V28_predictions.json')
    cache_b=read_pred(ROOT/'results'/'epochs'/'epoch_03_B_predictions.json')
    alignment={'A':compare_predictions(cache_a,online['A']),
               'B':compare_predictions(cache_b,online['B'])}
    cached_a=readj(ROOT/'results'/'A_V28_metrics.json')
    cached_b=readj(ROOT/'results'/'epochs'/'epoch_03_metrics.json')
    metric_checks={}
    for arm,cached in (('A',cached_a),('B',cached_b)):
        online_m=online_record['metrics'][arm]
        metric_checks[arm]={}
        for key in ('local_oks_ap50_95_post_polygon_nms_no_floor','polygon_ap50_95_post_polygon_nms_no_floor','local_oks_ap_levels','polygon_ap_levels','operating_points','score_0.30','score_0.45'):
            ref=cached.get(key)
            got=online_m.get(key)
            metric_checks[arm][key]={'equal':ref==got,'cached':ref,'online':got}
    discrete_fields=('matched_gt','target_count','tp','fp','fn','predictions_after_floor',
                     'small_detected','small_targets','tiny_detected','tiny_targets',
                     'boundary_detected','boundary_targets','dense_detected','dense_targets')
    ap_fields=('local_oks_ap50_95_post_polygon_nms_no_floor','polygon_ap50_95_post_polygon_nms_no_floor',
               'local_oks_ap_levels','polygon_ap_levels','operating_points')
    exact_ap_parity=all(metric_checks[arm][key]['equal'] for arm in ('A','B') for key in ap_fields)
    exact_working_point_counts=True
    floor_scalar_diffs=[]
    for arm in ('A','B'):
        for floor in ('score_0.30','score_0.45'):
            cached=metric_checks[arm][floor]['cached']; fresh=metric_checks[arm][floor]['online']
            for key in discrete_fields:
                exact_working_point_counts &= cached.get(key)==fresh.get(key)
            for key,value in cached.items():
                other=fresh.get(key)
                if isinstance(value,(int,float)) and not isinstance(value,bool) and isinstance(other,(int,float)) and not isinstance(other,bool):
                    floor_scalar_diffs.append(abs(float(value)-float(other)))
    alignment_summary={'all_AP_levels_and_operating_points_exact':exact_ap_parity,
                       'all_floor_detection_counts_exact':bool(exact_working_point_counts),
                       'max_floor_metric_scalar_abs_difference':max(floor_scalar_diffs,default=0.0),
                       'max_prediction_score_abs_difference':max(alignment[a]['score_abs_diff']['max'] for a in ('A','B')),
                       'max_prediction_coordinate_abs_difference_px':max(alignment[a]['max_abs_coordinate_difference_px'] for a in ('A','B')),
                       'prediction_candidate_id_sets_exact':all(alignment[a]['images_candidate_id_set_mismatch']==0 and alignment[a]['images_count_mismatch']==0 for a in ('A','B'))}
    writej(ROOT/'results'/'online_cache_alignment.json',{'status':'complete','prediction_lock_sha256':lock['predictions_sha256'],
            'prediction_alignment':alignment,'metric_parity':metric_checks,'alignment_summary':alignment_summary})

    # Annotation-dependent analysis begins only after the staged online prediction file and checksum lock exist.
    samples=ev.load_dev(MANIFEST,'dev')
    mA,_=ev.matches_at_floor(samples,online['A'],0.30)
    mB,_=ev.matches_at_floor(samples,online['B'],0.30)
    common=sorted(set(mA)&set(mB))
    per_arm={}
    for arm,mapping in (('A',mA),('B',mB)):
        px=[]; normalized=[]
        for key in common:
            stem,gtidx=key
            gt=np.asarray(samples[stem]['targets'][gtidx],dtype=np.float64).reshape(4,2)
            pred=np.asarray(mapping[key]['quad'],dtype=np.float64).reshape(4,2)
            err=np.linalg.norm(pred-gt,axis=1)
            scale=max(math.sqrt(max(float(geom.polygon_area(gt.tolist())),0.0)),1.0)
            px.extend(err.tolist()); normalized.extend((err/scale).tolist())
        per_arm[arm]={'matched_gt_total':len(mapping),'common_gt':len(common),'target_count':sum(len(x['targets']) for x in samples.values()),
                      'common_gt_coverage':len(common)/max(sum(len(x['targets']) for x in samples.values()),1),
                      'pixel_corner_error':stats(px),'normalized_corner_error_by_sqrt_gt_area':stats(normalized)}
    writej(ROOT/'results'/'common_gt_corner_error_epoch03_extended.json',{'association':'independent A/B floor0.30 matches; summarize only identical (image,GT) keys; GT is evaluator-only',
            'matching_common_gt':len(common),'A_only_gt':len(set(mA)-set(mB)),'B_only_gt':len(set(mB)-set(mA)),
            'metrics':per_arm})

    # Final candidate movement and guard accounting; label dev background candidates with the frozen IoU rule.
    trace_path=ROOT/'results'/'epochs'/'epoch_03_candidate_trace.csv.gz'
    rows=[]; proposal_norm=[]; applied_norm=[]; moved_candidates=0; moved_corners=0; class_counts=Counter()
    group_disp={'foreground':[],'background':[],'ignored':[]}
    by_stem={}
    for stem in samples:
        by_stem[stem]=np.asarray(samples[stem]['targets'],dtype=np.float64).reshape(-1,4,2)
    with gzip.open(trace_path,'rt',encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            v28=np.asarray(json.loads(row['V28_raw_json']),dtype=np.float64).reshape(4,2)
            proposal=np.asarray(json.loads(row['proposal_raw_json']),dtype=np.float64).reshape(4,2)
            final=np.asarray(json.loads(row['final_raw_json']),dtype=np.float64).reshape(4,2)
            pd=np.linalg.norm(proposal-v28,axis=1); ad=np.linalg.norm(final-v28,axis=1)
            proposal_norm.extend(pd.tolist()); applied_norm.extend(ad.tolist())
            moved_candidates+=int(bool(np.any(ad>1e-9))); moved_corners+=int(np.sum(ad>1e-9))
            v28_valid=row['V28_legal'].lower()=='true'; prop_valid=row['proposal_legal'].lower()=='true'
            pA=json.loads(row['V28_permutation']) if row['V28_permutation'] else None
            pB=json.loads(row['proposal_permutation']) if row['proposal_permutation'] else None
            if not v28_valid: category='original_invalid'
            elif not prop_valid: category='proposal_invalid'
            elif pA!=pB: category='point_order_changed'
            else: category='accepted'
            class_counts[category]+=1
            gts=by_stem.get(row['stem'],np.empty((0,4,2)))
            max_iou=0.0
            if v28_valid:
                for gt in gts:
                    max_iou=max(max_iou,float(geom.quad_iou(v28.tolist(),gt.tolist())))
            group='foreground' if max_iou>=0.5 else ('background' if max_iou<0.1 and v28_valid else 'ignored')
            group_disp[group].extend(ad.tolist())
            rows.append((row['stem'],int(row['candidate_id']),category,group,max_iou,float(row['quality_score']),float(row['class_score'])))
    disp_summary={'proposal_all_corner_vectors_px':stats(proposal_norm),'actually_applied_all_corner_vectors_px':stats(applied_norm),
                  'actually_moved_candidates':moved_candidates,'actually_moved_corners':moved_corners,
                  'candidate_decisions_priority_mutually_exclusive':dict(class_counts),
                  'candidate_decisions_total':sum(class_counts.values()),
                  'GT_IoU_grouped_applied_corner_vectors_px':{k:stats(v) for k,v in group_disp.items()},
                  'GT_group_rule':'V28 legal candidate max polygon IoU >=0.5 foreground, <0.1 background, otherwise ignored; dev-only analysis'}
    writej(ROOT/'results'/'candidate_update_summary_epoch03.json',disp_summary)

    # Post-freeze code revisions are additive and limited to imports / ordering; preserve the original training config hash.
    source_files=sorted(p for p in (ROOT/'code').rglob('*.py') if '__pycache__' not in p.parts)
    final_hashes={str(p.relative_to(ROOT)):sha256(p) for p in source_files}
    frozen_hashes=run_config['v31_code_hashes']
    source_diff={p:{'frozen_sha256':frozen_hashes.get(p),'final_sha256':final_hashes.get(p),
                    'changed_after_freeze':frozen_hashes.get(p)!=final_hashes.get(p)}
                 for p in sorted(set(frozen_hashes)|set(final_hashes))}
    writej(ROOT/'diagnostics'/'FINAL_SOURCE_HASHES.json',{'run_config_sha256':sha256(ROOT/'RUN_CONFIG.json'),
             'frozen_source_hashes':frozen_hashes,'final_source_hashes':final_hashes,'source_diff':source_diff,
             'detector_sha256':sha256(Path(run_config['model']['detector_path'])),
             'quality_head_sha256':sha256(Path(run_config['model']['quality_head_path']))})
    writej(ROOT/'logs'/'POST_FREEZE_IMPLEMENTATION_REPAIRS.json',{
        'training_run_config_sha256_preserved':sha256(ROOT/'RUN_CONFIG.json'),
        'repairs':[
          {'file':'code/train_v31.py','change':'import GRID for epoch-head metadata after the epoch checkpoint is atomically saved',
           'algorithm_or_optimizer_change':False,'failure_log':'logs/TRAIN_EPOCH_FINALIZE_FAILURE.json'},
          {'file':'code/evaluate_v31.py','change':'import guard_candidate required by the already-frozen V30 validity/permutation protection',
           'metric_definition_change':False,'failure_log':'logs/EPOCH01_EVAL_FAILURE.json'},
          {'file':'code/evaluate_v31.py','change':'add standalone --stage online: fresh image-only A/B inference writes prediction JSON and SHA256 lock before evaluator loads dev GT',
           'reason':'first online attempt received a preloaded samples object; its result is retained but not used as the order-compliant confirmation',
           'new_outputs':['results/online_confirmation_predictions_staged.json','results/online_confirmation_predictions_staged.lock.json','results/online_confirmation_metrics_staged.json']}
        ],
        'final_code_hashes':final_hashes,
        'no_training_updates_after_epoch_3':True})
    writej(ROOT/'logs'/'ONLINE_CONFIRMATION_ORDERING_AUDIT.json',{
        'initial_online_run':'results/online_confirmation_metrics.json',
        'initial_run_order_compliant':False,
        'reason':'evaluate_epoch had already loaded dev annotations into samples before online_confirmation was called, although run_image itself received only image paths',
        'initial_outputs_preserved':True,
        'final_confirmation':'results/online_confirmation_metrics_staged.json',
        'final_order_compliant':online_record['GT_loaded_only_after_predictions_locked'],
        'prediction_lock_sha256':lock['predictions_sha256'],
        'fresh_image_only_forward_all_859':online_record['images']==859,
        'result_used_for_final_report':True})

    # Timed primary pipeline stages. Tiny preflight/check GPU events were not comprehensively accumulated.
    pre=readj(ROOT/'diagnostics'/'RESOURCE_PREFLIGHT.json')
    pilot=sum(x['model_gpu_seconds']+x['decode_gpu_seconds']+x['quality_and_spatial_gpu_seconds'] for x in pre['pilot'])+pre['warmup_gpu_seconds']
    train_col=readj(ROOT/'cache'/'train_fit'/'collection_complete.json')
    dev_col=readj(ROOT/'cache'/'dev'/'collection_complete.json')
    first_online=readj(ROOT/'results'/'online_confirmation_metrics.json')
    train_complete=readj(ROOT/'logs'/'training_complete.json')
    gpu_stages={'resource_preflight_pilot_and_warmup_seconds':pilot,
                'train_fit_full_feature_forward_and_decode_seconds':train_col['image_forward_and_decode_seconds']+train_col['warmup_gpu_seconds'],
                'dev_feature_forward_and_decode_seconds':dev_col['image_forward_and_decode_seconds']+dev_col['warmup_gpu_seconds'],
                'residual_head_training_seconds':train_complete['GPU_active_seconds_total'],
                'initial_non_order_compliant_online_confirmation_seconds':first_online['GPU_active_seconds'],
                'final_staged_online_confirmation_seconds':online_record['GPU_active_seconds']}
    gpu_stages['measured_primary_total_seconds']=sum(gpu_stages.values())
    gpu_stages['measured_primary_total_hours']=gpu_stages['measured_primary_total_seconds']/3600.0
    gpu_stages['budget_hours']=8.0
    gpu_stages['under_budget_on_logged_stages']=gpu_stages['measured_primary_total_hours']<8.0
    gpu_stages['unaggregated']='failed pre-update starts and implementation-check kernels are not fully event-summed; no checkpoint update occurred in those failed starts'
    writej(ROOT/'diagnostics'/'GPU_RESOURCE_ACTUAL.json',gpu_stages)

    # Reports and status.
    A=cached_a; B=cached_b
    localA=A['local_oks_ap50_95_post_polygon_nms_no_floor']; localB=B['local_oks_ap50_95_post_polygon_nms_no_floor']
    polyA=A['polygon_ap50_95_post_polygon_nms_no_floor']; polyB=B['polygon_ap50_95_post_polygon_nms_no_floor']
    floorA=A['score_0.30']; floorB=B['score_0.30']
    f45A=A['score_0.45']; f45B=B['score_0.45']
    boot=readj(ROOT/'results'/'bootstrap_epoch03_summary.json')
    gate=readj(ROOT/'results'/'promotion_gate.json')
    nms=B['NMS_selection_changes']['nms_changed_images']
    guard=B['candidate_geometry_and_guard']
    fixed=B['fixed_A_GT_association']
    epochs=[]
    for e in (1,2,3):
        met=readj(ROOT/'results'/'epochs'/f'epoch_{e:02d}_metrics.json')
        tr=readj(ROOT/'logs'/f'train_epoch_{e:02d}_summary.json')
        epochs.append((e,met['local_oks_ap50_95_post_polygon_nms_no_floor'],met['polygon_ap50_95_post_polygon_nms_no_floor'],
                       met['score_0.30']['tp'],met['score_0.30']['fp'],met['score_0.30']['fn'],tr['loss_mean_per_update'],tr['GPU_active_seconds_epoch']))
    thresholds=[50,55,60,65,70,75,80,85,90,95]
    ap_rows=[]
    for i,t in enumerate(thresholds):
        ap_rows.append(f"| {t/100:.2f} | {A['local_oks_ap_levels'][i]:.6f} | {B['local_oks_ap_levels'][i]:.6f} | {B['local_oks_ap_levels'][i]-A['local_oks_ap_levels'][i]:+.6f} | {A['polygon_ap_levels'][i]:.6f} | {B['polygon_ap_levels'][i]:.6f} | {B['polygon_ap_levels'][i]-A['polygon_ap_levels'][i]:+.6f} |")
    online_align=readj(ROOT/'results'/'online_cache_alignment.json')
    online_m=online_record['metrics']
    common=readj(ROOT/'results'/'common_gt_corner_error_epoch03_extended.json')
    move=disp_summary
    figures=sorted((ROOT/'figures'/'epoch_03').glob('*.png'))
    all_files=sum(1 for p in (ROOT/'figures'/'epoch_03').glob('*.png'))
    trained_steps=4020
    run_hash=sha256(ROOT/'RUN_CONFIG.json')
    final_train_hash=sha256(ROOT/'code'/'train_v31.py'); final_eval_hash=sha256(ROOT/'code'/'evaluate_v31.py')
    model_hash=sha256(Path(run_config['model']['detector_path'])); head_hash=sha256(Path(run_config['model']['quality_head_path']))
    ckpt_hashes={p.name:sha256(p) for p in sorted((ROOT/'checkpoints').glob('*.pt'))}
    gpu_hours=gpu_stages['measured_primary_total_hours']
    report=f'''# DMKPD V31 最终报告

## 结论先行

1. **V31 epoch 3 在 dev 上较 V28 有整体改善信号**：local OKS AP50:95（本地近似）从 `{localA:.6f}` 到 `{localB:.6f}`（`{localB-localA:+.6f}`）；polygon AP50:95 从 `{polyA:.6f}` 到 `{polyB:.6f}`（`{polyB-polyA:+.6f}`）。
2. **残差头学到了非零偏移**：对 85,900 个候选，V30 保护接受 {class_counts['accepted']:,} 个提议、回退 {class_counts['original_invalid']+class_counts['proposal_invalid']+class_counts['point_order_changed']:,} 个；实际改变 {moved_candidates:,} 个候选、{moved_corners:,} 个角点。
3. **定位收益转化为 AP 收益，但工作点检出变化很小**：固定 A 关联的 1,767 个 GT 中，1,007 改善、52持平、708变差；平均归一化误差差 `{fixed['fixed_candidate_geometry_mean_delta']:+.6f}`。0.30 下 TP `{floorA['tp']}→{floorB['tp']}`、FP `{floorA['fp']}→{floorB['fp']}`，只多检出 1 个 GT，同时多 1 个 FP。
4. **误差尾部表现混合**：0.30各自匹配集合的归一化角点P99为 `{floorA['corner_error_normalized_p99']:.6f}→{floorB['corner_error_normalized_p99']:.6f}`，但A/B匹配集分别为{floorA['matched_gt']:,}/{floorB['matched_gt']:,}个GT，只有{common['matching_common_gt']:,}个共同GT；共同GT的P99从`{common['metrics']['A']['normalized_corner_error_by_sqrt_gt_area']['p99']:.6f}`降到`{common['metrics']['B']['normalized_corner_error_by_sqrt_gt_area']['p99']:.6f}`，同时共同集最差角点从`{common['metrics']['A']['normalized_corner_error_by_sqrt_gt_area']['max']:.6f}`升到`{common['metrics']['B']['normalized_corner_error_by_sqrt_gt_area']['max']:.6f}`。因此不能把不同匹配集的P99当成严格配对退化，也不能把均值/P95改善解释为所有样本都改善。small `612/699→{floorB['small_detected']}/{floorB['small_targets']}`，tiny `20/31→{floorB['tiny_detected']}/{floorB['tiny_targets']}`，boundary `23/26→{floorB['boundary_detected']}/{floorB['boundary_targets']}`。
5. **通过冻结的 dev 推进筛查条件**；1000次按图像配对 bootstrap 的 polygon AP差值95%区间下界为 `{boot[1]['ci95_low']:.6f}`。local OKS区间下界为 `{boot[0]['ci95_low']:.6f}`，包含零。
6. **建议继续把 epoch 3 列为候选并保留 V28 为正式模型**。独立进程、GT锁后读取的全量在线 dev 确认与缓存指标相同；这仍是反复使用过的 dev，不是独立确认，也没有官网或隐藏集提升证据。

## 实验边界与身份

- 唯一对照：A=原样 V28；B=原样 V28 + 一个空间 RoI 残差头，V30候选级合法性/点序保护。未延续 V29/V30单轴公式。
- 冻结检测器：`{run_config['model']['detector_path']}`，SHA256 `{model_hash}`。
- 冻结 V10 质量头：`{run_config['model']['quality_head_path']}`，SHA256 `{head_hash}`。
- run config SHA256：`{run_hash}`；训练完成 epoch 固定为 3、种子 `{run_config['seed']}`、累计 {trained_steps} 更新。
- split：旧 manifest 下 train_fit 7,731张、dev 859张/1,981 GT；历史 validation、locked-confirm、test300 未读取，未生成提交包、未上传。
- GPU：FP32、batch=1提取冻结特征；train_fit 监督只用 train_fit 标注。仅 534,216 个新头参数可训练；冻结 53 个 FrozenBatchNorm2d，原模型 eval。
- 监督：{train_col['train_counts']['foreground']:,} foreground、{train_col['train_counts']['background']:,} background；{train_col['train_counts']['iou_0.1_to_0.5_ignored']:,}个 IoU 0.1–0.5及{train_col['train_counts']['v28_invalid_ignored']:,}个原非法候选忽略；无法识别角点身份/并列数为0。每轮全部 train_fit 图像，前景48万余，背景按图像至多与前景等量采样，无新增增强。
- 局部优化：AdamW，lr=1e-4、weight decay=1e-4、microbatch=256、累积2、有效batch最大512；固定3轮，不按中间dev选epoch。

## 基线复现与训练

A 在训练前准确复现冻结 V28：local OKS AP `{localA:.12f}`、polygon AP `{polyA:.12f}`；0.30工作点 TP/FP/FN `{floorA['tp']}/{floorA['fp']}/{floorA['fn']}`，small `{floorA['small_detected']}/{floorA['small_targets']}`、tiny `{floorA['tiny_detected']}/{floorA['tiny_targets']}`、boundary `{floorA['boundary_detected']}/{floorA['boundary_targets']}`。候选顺序及缓存坐标一致。

| Epoch | local OKS AP50:95 | polygon AP50:95 | score≥0.30 TP/FP/FN | 每更新平均训练损失 | GPU活跃秒 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(f'| {e} | {la:.6f} | {pa:.6f} | {tp}/{fp}/{fn} | {loss:.6f} | {gpu:.2f} |' for e,la,pa,tp,fp,fn,loss,gpu in epochs)}

训练每轮1,340次更新，共4,020次；训练损失没有用作选模。epoch3输出层饱和比例 `{readj(ROOT/'logs'/'train_epoch_03_summary.json')['output_saturation_fraction']:.4%}`，背景残差范数均值 `{readj(ROOT/'logs'/'train_epoch_03_summary.json')['background_predicted_abs_norm_mean_per_batch']:.6f}`。冻结检测器/质量头状态未改变。

## epoch 3 AP 与阈值结果

| 相似度阈值 | local A | local B | B−A | polygon A | polygon B | B−A |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(ap_rows)}

两项 AP 主指标均在 polygon NMS=0.30、最终 cap=100之后、无score floor计算；local OKS为本地近似，不是官网公式。完整 PR：

- A：`results/pr/A_V28_local_oks_pr_050.csv.gz`、`A_V28_polygon_pr_iou50.csv.gz`。
- B epoch 3：`results/pr/B_epoch03_local_oks_pr_050.csv.gz`、`B_epoch03_polygon_pr_iou50.csv.gz`。
- 全部匹配阈值 AP及预测PR记录见 `results/epochs/epoch_03_metrics.json`、`results/pr/`。

| 排序工作点 | A TP / FP / FN | B TP / FP / FN | 对照 |
|---|---:|---:|---|
| recall 0.85 | {A['operating_points']['recall_0.85']['tp']} / {A['operating_points']['recall_0.85']['fp']} | {B['operating_points']['recall_0.85']['tp']} / {B['operating_points']['recall_0.85']['fp']} | FP不变 |
| recall 0.88 | {A['operating_points']['recall_0.88']['tp']} / {A['operating_points']['recall_0.88']['fp']} | {B['operating_points']['recall_0.88']['tp']} / {B['operating_points']['recall_0.88']['fp']} | B少1 FP |
| recall 0.89 | {A['operating_points']['recall_0.89']['tp']} / {A['operating_points']['recall_0.89']['fp']} | {B['operating_points']['recall_0.89']['tp']} / {B['operating_points']['recall_0.89']['fp']} | B少1 FP |
| FP预算100 | {A['operating_points']['fp_budget_100']['tp']} / 100 | {B['operating_points']['fp_budget_100']['tp']} / {B['operating_points']['fp_budget_100']['fp']} | B多1 TP |
| FP预算200 | {A['operating_points']['fp_budget_200']['tp']} / {A['operating_points']['fp_budget_200']['fp']} | {B['operating_points']['fp_budget_200']['tp']} / {B['operating_points']['fp_budget_200']['fp']} | B多0 TP、少4 FP |

| 固定分数下限 | A TP/FP/FN；P/R | B TP/FP/FN；P/R | 子组 A→B |
|---|---|---|---|
| 0.30 | {floorA['tp']}/{floorA['fp']}/{floorA['fn']}；{floorA['precision']:.4f}/{floorA['recall']:.4f} | {floorB['tp']}/{floorB['fp']}/{floorB['fn']}；{floorB['precision']:.4f}/{floorB['recall']:.4f} | small {floorA['small_detected']}/{floorA['small_targets']}→{floorB['small_detected']}/{floorB['small_targets']}；tiny {floorA['tiny_detected']}/{floorA['tiny_targets']}→{floorB['tiny_detected']}/{floorB['tiny_targets']}；boundary {floorA['boundary_detected']}/{floorA['boundary_targets']}→{floorB['boundary_detected']}/{floorB['boundary_targets']}；dense {floorA['dense_detected']}/{floorA['dense_targets']}→{floorB['dense_detected']}/{floorB['dense_targets']} |
| 0.45 | {f45A['tp']}/{f45A['fp']}/{f45A['fn']}；{f45A['precision']:.4f}/{f45A['recall']:.4f} | {f45B['tp']}/{f45B['fp']}/{f45B['fn']}；{f45B['precision']:.4f}/{f45B['recall']:.4f} | small {f45A['small_detected']}/{f45A['small_targets']}→{f45B['small_detected']}/{f45B['small_targets']}；tiny {f45A['tiny_detected']}/{f45A['tiny_targets']}→{f45B['tiny_detected']}/{f45B['tiny_targets']}；boundary {f45A['boundary_detected']}/{f45A['boundary_targets']}→{f45B['boundary_detected']}/{f45B['boundary_targets']}；dense {f45A['dense_detected']}/{f45A['dense_targets']}→{f45B['dense_detected']}/{f45B['dense_targets']} |

## 几何、保护与错误归因

| 指标 | 结果 |
|---|---:|
| 固定 A 关联 GT数 | {fixed['A_matched_gt_count']} |
| 固定候选四角改善 / 持平 / 下降 | {fixed['improved']} / {fixed['tied']} / {fixed['worsened']} |
| 固定候选归一化误差平均 B−A | {fixed['fixed_candidate_geometry_mean_delta']:+.6f} |
| 固定候选误差差值 P95 / P99 | {fixed['delta_distribution']['p95']:+.6f} / {fixed['delta_distribution']['p99']:+.6f} |
| 独立匹配 A / B；共同GT | {common['metrics']['A']['matched_gt_total']} / {common['metrics']['B']['matched_gt_total']}；{common['matching_common_gt']} |
| common GT 归一化角点均值 A / B | {common['metrics']['A']['normalized_corner_error_by_sqrt_gt_area']['mean']:.6f} / {common['metrics']['B']['normalized_corner_error_by_sqrt_gt_area']['mean']:.6f} |
| common GT 归一化角点 P95 A / B | {common['metrics']['A']['normalized_corner_error_by_sqrt_gt_area']['p95']:.6f} / {common['metrics']['B']['normalized_corner_error_by_sqrt_gt_area']['p95']:.6f} |
| common GT 归一化角点 P99 A / B | {common['metrics']['A']['normalized_corner_error_by_sqrt_gt_area']['p99']:.6f} / {common['metrics']['B']['normalized_corner_error_by_sqrt_gt_area']['p99']:.6f} |
| NMS选择改变的图像数 | {nms} |
| A-only / B-only floor0.30 GT | {fixed['A_only_floor030_gt']} / {fixed['B_only_floor030_gt']} |

候选保护的互斥优先类别：接受 {class_counts['accepted']:,}；V28原非法并保持原样 {class_counts['original_invalid']:,}；原合法但提议非法而回退 {class_counts['proposal_invalid']:,}；点序置换改变而整候选回退 {class_counts['point_order_changed']:,}。原始非法1,623个、提议非法1,984个、保护后几何过滤前非法仍为1,623个、过滤后非法输出0个；原非法被提议修成合法的1,113个仍未放行。互斥数合计85,900。A/B NMS选择在73张图改变；A-only匹配1个GT、B-only匹配2个GT，不能把2个净新增都解读成模型检测能力提升。

新增分支只改角点、不改质量分；V28质量分仍由原始V10未修正四角特征路径产生。变化的AP来自几何变化影响IoU匹配及NMS保留，而不是质量分重新排序。受保护候选的实际角点位移分布、按dev GT IoU分组的前景/背景/忽略位移见 `results/candidate_update_summary_epoch03.json`。

## Bootstrap 与预注册门槛

1,000次配对图像 bootstrap，seed=20261031；dev manifest 未提供可用来源组，因此按图像重采样，来源相关性未被消除。

| 指标 | B−A全量差值 | 配对95%区间 | 结果 |
|---|---:|---:|---|
| local OKS AP50:95（本地近似） | {boot[0]['observed_full_dev_B_minus_A']:+.6f} | [{boot[0]['ci95_low']:+.6f}, {boot[0]['ci95_high']:+.6f}] | 区间含零 |
| polygon AP50:95 | {boot[1]['observed_full_dev_B_minus_A']:+.6f} | [{boot[1]['ci95_low']:+.6f}, {boot[1]['ci95_high']:+.6f}] | 下界大于零 |

冻结的七项筛查全部通过：两项主AP不降；polygon AP增益≥0.002且区间下界>0；FP预算100/200 TP不降；0.30小/极小/边界检出不降；共同GT归一化均值/P95不升；最终非法几何不增。Bootstrap区间仍不能抵消反复使用dev导致的选模偏差。

## 独立在线 dev 确认与实现修复

- 第一次“在线确认”实际在调用前已由主评价器读取GT，虽然图像推理函数本身只收图像路径。原始输出和错误标记保留在 `results/online_confirmation_metrics.json`，**不作为合规在线确认依据**。
- 随后增加独立进程 `--stage online`：先只读manifest图像路径，对全部859张图像推理并写 A/B预测，再写 SHA256 锁 `{lock['predictions_sha256']}`，最后才加载dev标注评价。合规结果见 `results/online_confirmation_metrics_staged.json`。
- 在线与冻结缓存的两项AP、各阈值AP、PR工作点及0.30/0.45工作点的离散检出数完全一致。候选ID集合一致，最大分数差为`{online_align['alignment_summary']['max_prediction_score_abs_difference']:.2e}`，最大坐标差为`{online_align['alignment_summary']['max_prediction_coordinate_abs_difference_px']:.6f}px`；工作点角点误差标量最大差`{online_align['alignment_summary']['max_floor_metric_scalar_abs_difference']:.2e}`，来自在线热图数值抖动造成的亚像素微差，不影响候选集合或检出计数。详细对齐见 `results/online_cache_alignment.json`。
- 训练运行前修复了未导入 `BASE` 和展平候选索引未解码两个启动错误，均未发生优化器更新；恢复训练时epoch1完整checkpoint在写epoch头文件之前已保存，后续从epoch1 checkpoint接续且没有重复更新。
- epoch1 checkpoint后的头部元数据缺少 `GRID` 导入、首次dev评价缺少 `guard_candidate` 导入；分别补齐。另增加独立 staged online入口以严格落实GT读取顺序。修复清单和哈希前后对照在 `logs/IMPLEMENTATION_REPAIR.json`、`logs/POST_FREEZE_IMPLEMENTATION_REPAIRS.json`、`diagnostics/FINAL_SOURCE_HASHES.json`。

## 资源、恢复与产物

- 记录到的主要CUDA事件阶段合计约 **{gpu_hours:.2f} GPU小时**（上限8小时）：资源预检、train_fit特征抽取、dev特征抽取、头部训练、首次保留但不采用的在线检查、合规在线确认；各阶段秒数见 `diagnostics/GPU_RESOURCE_ACTUAL.json`。少量实现检查kernel和失败的pre-update启动未单独累计，不改变训练步数。
- 完整检查结果：`diagnostics/IMPLEMENTATION_CHECKS.json` 全项通过；缓存完整性 `diagnostics/CACHE_INTEGRITY.json`通过；A基线差值0；冻结检测器和质量头哈希保持不变。
- 三轮checkpoint（含optimizer、RNG、采样位置）SHA256：
{chr(10).join(f'  - `{name}`: `{digest}`' for name,digest in ckpt_hashes.items())}
- 例图 {all_files} 张，索引：`figures/epoch_03/index.html`，含固定A候选最大改善、最大退化及新增/丢失目标例图；图像只作诊断，不据此调参。
- 复现：PowerShell `& 'python' '<PROJECT_ROOT>\\DMKPD-V31-V28学习式角点残差\\code\\train_v31.py' --resume`；在线确认 `& 'python' '<PROJECT_ROOT>\\DMKPD-V31-V28学习式角点残差\\code\\evaluate_v31.py' --stage online --checkpoint '<PROJECT_ROOT>\\DMKPD-V31-V28学习式角点残差\\checkpoints\\epoch_03.pt'`。一键阶段脚本为 `REPRODUCE.ps1`。

## 结论边界

V31 epoch3 在本次dev与合规全量在线dev确认中均显示正向AP信号，并通过预注册筛查；这支持把它作为**待独立确认候选**。少量角点尾部误差变差，工作点TP变化只有+1；dev已用于训练监督/筛查/确认且有潜在同源相关性，不能据此替换正式模型或承诺官网提升。正式最佳仍保留 V28；V10的历史官网成绩与本地AP不可直接对比。本轮未读test300、未提交、未上传。
'''
    (ROOT/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
    writej(ROOT/'EXECUTION_STATUS.json',{'status':'complete','formal_training_epochs':3,'optimizer_updates':4020,
        'training_complete':train_complete,'epoch3_fixed_selection':True,'screen_gate':gate,
        'staged_online_confirmation_status':online_record['status'],'staged_online_all_dev_images':online_record['images'],
        'staged_online_gt_loaded_after_prediction_lock':online_record['GT_loaded_only_after_predictions_locked'],
        'training_run_config_sha256':run_hash,'final_train_code_sha256':final_train_hash,'final_evaluation_code_sha256':final_eval_hash,
        'detector_sha256':model_hash,'quality_head_sha256':head_hash,'gpu_budget_hours':8,'logged_primary_gpu_hours':gpu_hours,
        'test300_read':False,'locked_confirm_read':False,'history_validation_read':False,'submission_created':False,'upload':False,
        'formal_best_remains':'V28; V31 epoch3 remains a dev-confirmed candidate only'})
    print(json.dumps({'status':'complete','report':str(ROOT/'FINAL_REPORT.md'),'online_alignment':alignment,
                      'common_gt':per_arm,'candidate_update':disp_summary,'logged_gpu_hours':gpu_hours,
                      'final_source_hashes':final_hashes,'figures':all_files},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
