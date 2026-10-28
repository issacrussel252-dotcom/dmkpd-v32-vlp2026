from __future__ import annotations
import csv,gzip,json,math,os,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(r"<PROJECT_ROOT>\DMKPD-V31-V28学习式角点残差")
os.environ["DMKPD_V32_RUN_ROOT"]=str(ROOT/"runs"/"seed_20261031_A")
os.environ["DMKPD_V32_SOURCE_ROOT"]=str(SOURCE)
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"code"/"source_snapshot"),str(ROOT/"code"/"source_refs")]
import v9_evaluate as ev
import v10_rank as rank
import metric_helpers as mh
from v31_core import MANIFEST,sha256,dump_json,jsonable

SEEDS=(20261031,20261101)
BOOT_SEED=20261102
REPS=1000

def load_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def stats(vals):
    a=np.asarray(vals,dtype=np.float64)
    return {"count":int(len(a)),"mean":float(a.mean()) if len(a) else None,
            "p95":float(np.percentile(a,95)) if len(a) else None,"p99":float(np.percentile(a,99)) if len(a) else None}
def read_gz(path):
    with gzip.open(path,"rt",encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def load_trace(path):
    result={}
    for row in read_gz(path):
        result[(row["stem"],int(row["candidate_id"]))]=row
    return result
def parse_json_cell(s): return json.loads(s) if s and s not in ("", "None", "null") else None
def dump_csv_gz(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,"wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["empty"]);w.writeheader();w.writerows(rows)
def normalize_corner_error(pred,gt):
    scale=max(math.sqrt(ev.polygon_area(gt)),1.0)
    return [math.dist(pred[i],gt[i])/scale for i in range(4)]
def get_area_metrics(metrics, predictions, samples, floor):
    return mh.custom_floor_metrics(samples,predictions,floor)[0]
def file_predictions(runroot):
    return load_json(runroot/"results"/"epochs"/"epoch_03_B_predictions.json")
def load_eval_samples(): return ev.load_dev(MANIFEST,"dev")

def render_examples(samples, pair_rows, gt_status_rows, image_map, outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    selected=[]
    fixed=sorted(pair_rows,key=lambda r:r["delta_B_minus_A_norm_mean"])
    for label,seq in (("largest_fixed_candidate_improvements",fixed[:5]),("largest_fixed_candidate_regressions",fixed[-5:][::-1])):
        for r in seq:selected.append((label,r["stem"],int(r["gt_index"]),r))
    for label,status in (("newly_detected_by_B","B_only"),("lost_from_A_to_B","A_only")):
        subset=[r for r in gt_status_rows if r["status"]==status]
        for r in subset[:5]:selected.append((label,r["stem"],int(r["gt_index"]),r))
    out=[]
    for i,(label,stem,gi,row) in enumerate(selected):
        path=image_map[stem]
        with Image.open(path) as im: image=im.convert("RGB")
        draw=ImageDraw.Draw(image)
        gt=np.asarray(samples[stem]["targets"][gi],dtype=float)
        # Colors: A blue, B green, GT red.
        quads=[]
        if "A_quad" in row: qa=parse_json_cell(row["A_quad"]); qb=parse_json_cell(row["B_quad"])
        else:
            qa=parse_json_cell(row.get("A_quad_json"));qb=parse_json_cell(row.get("B_quad_json"))
        for name,q,color in (("A",qa,(50,150,255)),("B",qb,(30,220,100)),("GT",gt,(255,60,60))):
            if q is None:continue
            pts=[(float(x),float(y)) for x,y in q]
            draw.line(pts+[pts[0]],fill=color,width=3)
            for j,(x,y) in enumerate(pts):
                draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
                draw.text((x+3,y+2),f"{name}{j}",fill=color)
        fn=f"{i:03d}_{label}_{stem}_gt{gi}.png"
        image.save(outdir/fn)
        out.append({"label":label,"stem":stem,"gt_index":gi,"file":fn,
                    "delta_B_minus_A_norm_mean":row.get("delta_B_minus_A_norm_mean")})
    html=["<!doctype html><meta charset='utf-8'><title>V32 paired dev examples</title>",
          "<style>body{font:14px sans-serif;background:#111;color:#eee}figure{display:inline-block;vertical-align:top;width:48%}img{max-width:100%}figcaption{padding:6px}</style>",
          "<p>Blue=A, green=B, red=GT. Examples follow the fixed selection rule in analyze_v32.py.</p>"]
    for r in out: html.append(f"<figure><img src='{r['file']}'><figcaption>{r['label']} | {r['stem']} GT {r['gt_index']} | Δ fixed-candidate normalized mean {r['delta_B_minus_A_norm_mean']}</figcaption></figure>")
    (outdir/"index.html").write_text("\n".join(html),encoding="utf-8")
    dump_json(outdir/"index.json",out)
    return out

def main():
    samples=load_eval_samples()
    if len(samples)!=859 or sum(len(s["targets"]) for s in samples.values())!=1981:raise RuntimeError("V32 dev manifest count changed")
    rows=rank.read_rows(MANIFEST,"dev")
    image_map={r["stem"]:Path(r["image"]) for r in rows}
    result={"dev_images":len(samples),"dev_gt":sum(len(s["targets"]) for s in samples.values()),"seeds":{},"bootstrap_seed":BOOT_SEED,"bootstrap_replicates":REPS}
    per_seed_indices=[]
    all_gate_seed={}
    all_metric_rows=[]
    for seed in SEEDS:
        rootA=ROOT/"runs"/f"seed_{seed}_A";rootB=ROOT/"runs"/f"seed_{seed}_B"
        predA=file_predictions(rootA);predB=file_predictions(rootB)
        for name,pred in (("A",predA),("B",predB)):
            if set(pred)!=set(samples):raise RuntimeError(f"prediction image keys mismatch seed={seed} arm={name}")
        dirout=ROOT/"results"/f"seed_{seed}";dirout.mkdir(parents=True,exist_ok=True)
        metA,_=ev.evaluate_predictions(samples,predA,dirout/"pr",f"seed{seed}_paired_A",write_curves=True)
        metB,_=ev.evaluate_predictions(samples,predB,dirout/"pr",f"seed{seed}_paired_B",write_curves=True)
        floorA={str(f):get_area_metrics(metA,predA,samples,f) for f in (.30,.45)}
        floorB={str(f):get_area_metrics(metB,predB,samples,f) for f in (.30,.45)}
        traceA=load_trace(rootA/"results"/"epochs"/"epoch_03_candidate_trace.csv.gz")
        traceB=load_trace(rootB/"results"/"epochs"/"epoch_03_candidate_trace.csv.gz")
        mapA,total=ev.matches_at_floor(samples,predA,.30);mapB,_=ev.matches_at_floor(samples,predB,.30)
        common_keys=set(mapA)&set(mapB); gained=set(mapB)-set(mapA); lost=set(mapA)-set(mapB)
        common_a=[];common_b=[];gtrows=[]
        all_gt_keys={(stem,i) for stem,sample in samples.items() for i in range(len(sample["targets"]))}
        for stem,gt_idx in sorted(all_gt_keys):
            a=mapA.get((stem,gt_idx));b=mapB.get((stem,gt_idx));status="common" if a and b else "B_only" if b else "A_only" if a else "neither"
            gt=np.asarray(samples[stem]["targets"][gt_idx],dtype=float)
            ea=normalize_corner_error(a["quad"],gt) if a else None
            eb=normalize_corner_error(b["quad"],gt) if b else None
            if a and b: common_a.extend(ea);common_b.extend(eb)
            gtrows.append({"stem":stem,"gt_index":gt_idx,"status":status,
                "A_candidate_id":int(a["candidate_id"]) if a else "","B_candidate_id":int(b["candidate_id"]) if b else "",
                "A_score":float(a["score"]) if a else "","B_score":float(b["score"]) if b else "",
                "A_quad_json":json.dumps(a["quad"]) if a else "","B_quad_json":json.dumps(b["quad"]) if b else "",
                "gt_quad_json":json.dumps(gt.tolist())})
        fixed=[]
        for (stem,gt_idx),a in sorted(mapA.items()):
            cid=int(a["candidate_id"]);ra=traceA.get((stem,cid));rb=traceB.get((stem,cid))
            if ra is None or rb is None:raise RuntimeError(f"fixed candidate missing from trace {stem}/{cid}")
            qa=parse_json_cell(ra["final_canonical_json"]);qb=parse_json_cell(rb["final_canonical_json"])
            gt=np.asarray(samples[stem]["targets"][gt_idx],dtype=float)
            if qa is None or qb is None:continue
            eA=normalize_corner_error(qa,gt);eB=normalize_corner_error(qb,gt)
            fixed.append({"stem":stem,"gt_index":gt_idx,"candidate_id":cid,"A_candidate_NMS_kept":ra["NMS_B_kept"],
                "B_candidate_NMS_kept":rb["NMS_B_kept"],"A_error_norm_mean":float(np.mean(eA)),"B_error_norm_mean":float(np.mean(eB)),
                "delta_B_minus_A_norm_mean":float(np.mean(eB)-np.mean(eA)),"A_error_norm_corners":json.dumps(eA),"B_error_norm_corners":json.dumps(eB),
                "A_quad_json":json.dumps(qa),"B_quad_json":json.dumps(qb),"gt_quad_json":json.dumps(gt.tolist())})
        dump_csv_gz(dirout/"GT_status_floor030.csv.gz",gtrows)
        dump_csv_gz(dirout/"fixed_A_GT_candidate_geometry.csv.gz",fixed)
        nms_rows=[]
        for stem in sorted(samples):
            ids_a={int(r["candidate_id"]) for r in predA[stem]}
            ids_b={int(r["candidate_id"]) for r in predB[stem]}
            nms_rows.append({"stem":stem,"A_prediction_count":len(ids_a),"B_prediction_count":len(ids_b),
                "candidate_ids_A_json":json.dumps(sorted(ids_a)),"candidate_ids_B_json":json.dumps(sorted(ids_b)),
                "removed_from_A_set_json":json.dumps(sorted(ids_a-ids_b)),"added_to_B_set_json":json.dumps(sorted(ids_b-ids_a)),
                "candidate_set_changed":bool(ids_a!=ids_b)})
        dump_csv_gz(dirout/"NMS_candidate_set_changes.csv.gz",nms_rows)
        boot_draws,boot_summary=mh.paired_bootstrap(samples,{"A":predA,"B":predB},seed=BOOT_SEED,replicates=REPS)
        per_seed_indices.append(np.asarray([r["sampled_indices"] for r in boot_draws],dtype=np.int32))
        dump_csv_gz(dirout/"paired_bootstrap_draws.csv.gz",boot_draws)
        dump_json(dirout/"paired_bootstrap_summary.json",boot_summary)
        guardA=load_json(rootA/"results"/"epochs"/"epoch_03_guard_summary.json")
        guardB=load_json(rootB/"results"/"epochs"/"epoch_03_guard_summary.json")
        delta={"local_oks_ap50_95":metB["local_oks_ap50_95_post_polygon_nms_no_floor"]-metA["local_oks_ap50_95_post_polygon_nms_no_floor"],
               "polygon_ap50_95":metB["polygon_ap50_95_post_polygon_nms_no_floor"]-metA["polygon_ap50_95_post_polygon_nms_no_floor"],
               "local_oks_AP75":metB["local_oks_ap_levels"][5]-metA["local_oks_ap_levels"][5],
               "polygon_AP75":metB["polygon_ap_levels"][5]-metA["polygon_ap_levels"][5]}
        op={}
        for budget in (100,200):
            a=metA["operating_points"][f"fp_budget_{budget}"];b=metB["operating_points"][f"fp_budget_{budget}"]
            op[str(budget)]={"A":a,"B":b,"delta_TP":b["tp"]-a["tp"],"delta_recall":b["recall"]-a["recall"]}
        fixed_rec={"A":floorA["0.3"],"B":floorB["0.3"]}
        common={"GT_count":len(common_keys),"A_only":len(lost),"B_only":len(gained),
                "A_normalized_corner_error":stats(common_a),"B_normalized_corner_error":stats(common_b)}
        counts={"total_gt":total,"common":len(common_keys),"A_only":len(lost),"B_only":len(gained),
                "both_final_invalid": [guardA["final_invalid_before_geometry_filter"],guardB["final_invalid_before_geometry_filter"]],
                "raw_invalid": [guardA["raw_invalid"],guardB["raw_invalid"]],
                "nms_changed_images_vs_V28": [guardA["images_nms_changed"],guardB["images_nms_changed"]],
                "nms_candidate_set_changed_images_A_vs_B":sum(bool(r["candidate_set_changed"]) for r in nms_rows),
                "nms_candidate_ids_removed_from_A":sum(len(json.loads(r["removed_from_A_set_json"])) for r in nms_rows),
                "nms_candidate_ids_added_to_B":sum(len(json.loads(r["added_to_B_set_json"])) for r in nms_rows)}
        seed_record={"seed":seed,"A_metrics":metA,"B_metrics":metB,"A_floor_metrics":floorA,"B_floor_metrics":floorB,
            "delta":delta,"AP50":{"local_oks":{"A":metA["local_oks_ap_levels"][0],"B":metB["local_oks_ap_levels"][0]},
                "polygon":{"A":metA["polygon_ap_levels"][0],"B":metB["polygon_ap_levels"][0]}},
            "AP75":{"local_oks":{"A":metA["local_oks_ap_levels"][5],"B":metB["local_oks_ap_levels"][5]},
                "polygon":{"A":metA["polygon_ap_levels"][5],"B":metB["polygon_ap_levels"][5]}},
            "AP95":{"local_oks":{"A":metA["local_oks_ap_levels"][9],"B":metB["local_oks_ap_levels"][9]},
                "polygon":{"A":metA["polygon_ap_levels"][9],"B":metB["polygon_ap_levels"][9]}},
            "operating_point_fp_budgets":op,"common_GT_normalized_error":common,"detection_status_counts":counts,
            "fixed_A_association_geometry":{"count":len(fixed),"A_mean":float(np.mean([r["A_error_norm_mean"] for r in fixed])),
                "B_mean":float(np.mean([r["B_error_norm_mean"] for r in fixed])),"delta_mean":float(np.mean([r["delta_B_minus_A_norm_mean"] for r in fixed])),
                "improved":sum(r["delta_B_minus_A_norm_mean"]<0 for r in fixed),"tied":sum(r["delta_B_minus_A_norm_mean"]==0 for r in fixed),
                "worsened":sum(r["delta_B_minus_A_norm_mean"]>0 for r in fixed)},
            "guard_summaries":{"A":guardA,"B":guardB},"bootstrap_summary":boot_summary}
        dump_json(dirout/"paired_metrics.json",jsonable(seed_record))
        example_index=render_examples(samples,fixed,gtrows,image_map,ROOT/"figures"/f"seed_{seed}")
        seed_record["example_index"]=[str((ROOT/"figures"/f"seed_{seed}"/"index.html").resolve())]
        result["seeds"][str(seed)]=seed_record
        all_gate_seed[str(seed)]={"delta":delta,"floorA":floorA["0.3"],"floorB":floorB["0.3"],
           "ops":op,"common":common,"bootstrap":boot_summary,
           "illegal_A":guardA["raw_invalid"],"illegal_B":guardB["final_invalid_before_geometry_filter"]}
        all_metric_rows.append({"seed":seed,"arm":"A","local_AP50_95":metA["local_oks_ap50_95_post_polygon_nms_no_floor"],
           "polygon_AP50_95":metA["polygon_ap50_95_post_polygon_nms_no_floor"],"local_AP50":metA["local_oks_ap_levels"][0],"polygon_AP50":metA["polygon_ap_levels"][0],
           "local_AP75":metA["local_oks_ap_levels"][5],"polygon_AP75":metA["polygon_ap_levels"][5],"floor030_TP":floorA["0.3"]["tp"],"floor030_FP":floorA["0.3"]["fp"],"floor030_FN":floorA["0.3"]["fn"]})
        all_metric_rows.append({"seed":seed,"arm":"B","local_AP50_95":metB["local_oks_ap50_95_post_polygon_nms_no_floor"],
           "polygon_AP50_95":metB["polygon_ap50_95_post_polygon_nms_no_floor"],"local_AP50":metB["local_oks_ap_levels"][0],"polygon_AP50":metB["polygon_ap_levels"][0],
           "local_AP75":metB["local_oks_ap_levels"][5],"polygon_AP75":metB["polygon_ap_levels"][5],"floor030_TP":floorB["0.3"]["tp"],"floor030_FP":floorB["0.3"]["fp"],"floor030_FN":floorB["0.3"]["fn"]})
    if not np.array_equal(per_seed_indices[0],per_seed_indices[1]):raise RuntimeError("paired bootstrap image draws differ across seeds")
    draws_path=ROOT/"results"/"bootstrap_draw_indices_seed20261102.npy";np.save(draws_path,per_seed_indices[0])
    result["bootstrap_draw_index_path"]=str(draws_path);result["bootstrap_draw_index_sha256"]=sha256(draws_path)
    result["bootstrap_same_draws_across_seeds"]=True
    # Conservative preregistered B-vs-A gate.
    local_both=all(all_gate_seed[str(s)]["delta"]["local_oks_ap50_95"]>=0 for s in SEEDS)
    poly_both=all(all_gate_seed[str(s)]["delta"]["polygon_ap50_95"]>=0 for s in SEEDS)
    gain_metric=None
    for metric,key in (("local_oks","local_oks_ap50_95"),("polygon","polygon_ap50_95")):
        if all(all_gate_seed[str(s)]["delta"][key]>=.002 and next(x for x in all_gate_seed[str(s)]["bootstrap"] if x["metric"]==metric)["ci95_low"]>0 for s in SEEDS):gain_metric=metric
    ap75=all(all_gate_seed[str(s)]["delta"]["local_oks_AP75"]>=0 and all_gate_seed[str(s)]["delta"]["polygon_AP75"]>=0 for s in SEEDS)
    fp=all(all_gate_seed[str(s)]["ops"][str(b)]["delta_TP"]>=0 for s in SEEDS for b in (100,200))
    groups=all(all_gate_seed[str(s)]["floorB"][f"{g}_detected"]>=all_gate_seed[str(s)]["floorA"][f"{g}_detected"] for s in SEEDS for g in ("small","tiny","boundary"))
    common_error=all(all_gate_seed[str(s)]["common"]["B_normalized_corner_error"][k]<=all_gate_seed[str(s)]["common"]["A_normalized_corner_error"][k] for s in SEEDS for k in ("mean","p95","p99"))
    illegal=all(all_gate_seed[str(s)]["illegal_B"]<=all_gate_seed[str(s)]["illegal_A"] for s in SEEDS)
    gates={"both_seed_local_AP_non_decrease":local_both,"both_seed_polygon_AP_non_decrease":poly_both,
      "same_AP_gain_ge_0_002_each_seed_and_CI_lower_positive":gain_metric is not None,"qualifying_shared_metric":gain_metric,
      "both_seed_local_and_polygon_AP75_non_decrease":ap75,"both_seed_TP_at_FP100_and_FP200_non_decrease":fp,
      "both_seed_floor030_small_tiny_boundary_non_decrease":groups,"both_seed_common_GT_error_mean_P95_P99_non_increase":common_error,
      "both_seed_final_illegal_geometry_non_increase":illegal}
    result["promotion_gate"]={"tests":gates,"all_pass":all(v for k,v in gates.items() if k!="qualifying_shared_metric"),
        "proceed_to_online_dev_confirmation":all(v for k,v in gates.items() if k!="qualifying_shared_metric"),
        "note":"same qualifying metric must satisfy per-seed gain+CI condition; metric name stored separately"}
    # Correct the gate's gain condition after the non-metric fields are excluded above.
    result["promotion_gate"]["all_pass"]=all(v for k,v in gates.items() if k!="qualifying_shared_metric")
    result["promotion_gate"]["proceed_to_online_dev_confirmation"]=result["promotion_gate"]["all_pass"]
    dump_json(ROOT/"results"/"RESULTS.json",jsonable(result))
    with (ROOT/"results"/"GROUPED_METRICS.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(all_metric_rows[0]));w.writeheader();w.writerows(all_metric_rows)
    # Per-epoch fixed metrics from all four runs, preserving arm identity.
    epoch_rows=[]
    for seed in SEEDS:
        for arm in ("A","B"):
            run=ROOT/"runs"/f"seed_{seed}_{arm}"
            for epoch in (1,2,3):
                m=load_json(run/"results"/"epochs"/f"epoch_{epoch:02d}_metrics.json")
                row={"seed":seed,"arm":arm,"epoch":epoch,
                     "local_AP50_95":m["local_oks_ap50_95_post_polygon_nms_no_floor"],
                     "polygon_AP50_95":m["polygon_ap50_95_post_polygon_nms_no_floor"],
                     "local_AP50":m["local_oks_ap_levels"][0],"local_AP75":m["local_oks_ap_levels"][5],"local_AP95":m["local_oks_ap_levels"][9],
                     "polygon_AP50":m["polygon_ap_levels"][0],"polygon_AP75":m["polygon_ap_levels"][5],"polygon_AP95":m["polygon_ap_levels"][9],
                     "floor030_TP":m["score_0.30"]["tp"],"floor030_FP":m["score_0.30"]["fp"],"floor030_FN":m["score_0.30"]["fn"]}
                epoch_rows.append(row)
    with (ROOT/"results"/"EPOCH_METRICS.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(epoch_rows[0]));w.writeheader();w.writerows(epoch_rows)
    print(json.dumps({"promotion_gate":result["promotion_gate"],"bootstrap_sha":result["bootstrap_draw_index_sha256"]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
