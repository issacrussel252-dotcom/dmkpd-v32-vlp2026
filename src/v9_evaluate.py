from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import os
import numpy as np
from PIL import Image

from v9_geometry import canonical_quad, hull, local_oks, ordered_quad_is_valid, parse_shapes, polygon_area, quad_iou
from v9_train import read_rows, sha256, write_json


def score_for(row: dict[str, Any], formula: str) -> float:
    quality = row.get("quality_score")
    p, q = float(row["class_score"]), 1.0 if quality is None or quality == "" else float(quality)
    if formula == "p_cls": return p
    if formula == "q": return q
    if formula == "p_cls*q": return p*q
    if formula == "p_cls*sqrt(q)": return p*math.sqrt(max(q,0.0))
    raise ValueError(formula)


def load_dev(manifest: Path, role: str) -> dict[str, dict[str, Any]]:
    samples = {}
    for row in read_rows(manifest, role):
        with Image.open(row["image"]) as image:
            width, height = image.size
        targets = parse_shapes(Path(row["annotation"]), row["stem"])
        areas = [polygon_area(t) for t in targets]
        small = [area / max(width*height,1) < .01 for area in areas]
        tiny = [min(math.dist(t[i],t[(i+1)%4]) for i in range(4)) < 16.0 for t in targets]
        boundary = [any(x <= 1.0 or y <= 1.0 or x >= width-2.0 or y >= height-2.0 for x,y in t) for t in targets]
        samples[row["stem"]] = {"targets":targets,"size":(width,height),"small":small,"tiny":tiny,"boundary":boundary,"dense_image":len(targets)>=3}
    return samples


def load_trace(path: Path, samples: dict[str,dict[str,Any]]) -> tuple[dict[str,list[dict[str,Any]]],dict[str,Any]]:
    pool: dict[str,list[dict[str,Any]]] = {stem:[] for stem in samples}
    stage: dict[str,list[dict[str,Any]]] = {stem:[] for stem in samples}
    rows=0; decoded=0; invalid=0; trunc=0
    with gzip.open(path,"rt",encoding="utf-8",newline="") as handle:
        for row in csv.DictReader(handle):
            rows+=1
            stem=row["stem"]
            if stem not in samples: continue
            row["candidate_id"]=int(row["candidate_id"])
            row["prekeypoint_cap_pass"]=int(row["prekeypoint_cap_pass"])
            row["geometry_valid"]=int(row["geometry_valid"])
            row["polygon_nms_kept"]=int(row["polygon_nms_kept"])
            row["final_output_kept"]=int(row["final_output_kept"])
            row["score"]=float(row["final_score"])
            row["quad"]=None
            if row["canonical_quad_json"]:
                try: row["quad"]=canonical_quad(json.loads(row["canonical_quad_json"]))
                except (ValueError,TypeError,IndexError,json.JSONDecodeError): invalid+=1
            if row["keypoint_decoded"]=="1": decoded+=1
            if row["removal_reason"]=="prekeypoint_candidate_cap": trunc+=1
            stage[stem].append(row)
            if row["prekeypoint_cap_pass"] and row["geometry_valid"] and row["quad"] is not None:
                pool[stem].append({"score":row["score"],"quad":row["quad"],"candidate_id":row["candidate_id"],"class_score":float(row["class_score"]),"quality_score":float(row["quality_score"] or 1.0),"row":row})
    info={"trace":str(path.resolve()),"trace_sha256":sha256(path),"rows":rows,"keypoints_decoded":decoded,"invalid_rows":invalid,"prekeypoint_truncated":trunc}
    return pool, {"stage":stage,"info":info}


def rerank_and_nms(pool: dict[str,list[dict[str,Any]]], formula: str, nms_iou: float, final_cap: int) -> dict[str,list[dict[str,Any]]]:
    output={}
    for stem,candidates in pool.items():
        ranked=sorted(candidates,key=lambda x:(-score_for(x,formula),x["candidate_id"]))
        kept=[]
        for row in ranked:
            if all(quad_iou(row["quad"],other["quad"])<nms_iou for other in kept):
                out={**row,"score":score_for(row,formula)}
                kept.append(out)
        output[stem]=kept[:final_cap]
    return output


def _ranked(samples:dict[str,dict[str,Any]],predictions:dict[str,list[dict[str,Any]]]):
    ranked=[]
    for stem,rows in predictions.items():
        for row in rows: ranked.append((float(row["score"]),stem,int(row["candidate_id"]),row["quad"]))
    ranked.sort(key=lambda x:(-x[0],x[1],x[2]))
    return ranked


def ranked_matches(samples,predictions,similarity:Callable,score_floor=None,match_threshold=.5):
    ranked=_ranked(samples,predictions)
    used:dict[str,set[int]]=defaultdict(set)
    records=[]
    total_gt=sum(len(s["targets"]) for s in samples.values())
    tp=fp=0
    for rank,(score,stem,cid,quad) in enumerate(ranked,1):
        if score_floor is not None and score<score_floor: continue
        choices=[(similarity(quad,target),i) for i,target in enumerate(samples[stem]["targets"]) if i not in used[stem]]
        sim,gt=max(choices,key=lambda choice:(choice[0],-choice[1]),default=(0.0,-1))
        is_tp=sim>=match_threshold
        if is_tp: used[stem].add(gt); tp+=1
        else: fp+=1
        records.append({"rank":rank,"score":score,"stem":stem,"candidate_id":cid,"tp":int(is_tp),"fp":int(not is_tp),"gt_index":gt if is_tp else -1,"similarity":sim if is_tp else 0.0,"cum_tp":tp,"cum_fp":fp,"precision":tp/max(tp+fp,1),"recall":tp/max(total_gt,1),"quad":quad})
    return records,total_gt


def ap_101(records,total_gt):
    if total_gt<=0 or not records:return 0.0
    recalls=np.asarray([r["cum_tp"]/total_gt for r in records])
    precisions=np.asarray([r["precision"] for r in records])
    vals=[]
    for recall in np.linspace(0,1,101):
        vals.append(float(precisions[recalls>=recall].max()) if np.any(recalls>=recall) else 0.0)
    return float(np.mean(vals))


def ap_metric(samples,predictions,similarity):
    levels=np.arange(.5,1.0,.05)
    scores=[]
    for threshold in levels:
        # Match the most similar still-unmatched GT before applying the AP
        # threshold. Binarizing similarity first loses the IoU/OKS ordering
        # when a candidate overlaps more than one target.
        records,total=ranked_matches(samples,predictions,similarity,match_threshold=threshold)
        scores.append(ap_101(records,total))
    return float(np.mean(scores)),[float(x) for x in scores]


def write_pr(path:Path,records:list[dict[str,Any]],total_gt:int):
    fields=["rank","score_threshold","stem","candidate_id","tp","fp","cum_tp","cum_fp","precision","recall","matched_gt_index","similarity"]
    with gzip.open(path,"wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:
            w.writerow({"rank":r["rank"],"score_threshold":r["score"],"stem":r["stem"],"candidate_id":r["candidate_id"],"tp":r["tp"],"fp":r["fp"],"cum_tp":r["cum_tp"],"cum_fp":r["cum_fp"],"precision":r["precision"],"recall":r["recall"],"matched_gt_index":r["gt_index"],"similarity":r["similarity"]})


def op_points(records,total_gt):
    result={}
    for target_recall in (.85,.88,.89):
        needed=math.ceil(target_recall*total_gt)
        match=next((r for r in records if r["cum_tp"]>=needed),None)
        result[f"recall_{target_recall:.2f}"]=None if match is None else {"fp":match["cum_fp"],"tp":match["cum_tp"],"recall":match["recall"],"score_threshold":match["score"],"rank":match["rank"]}
    for budget in (100,200):
        possible=[r for r in records if r["cum_fp"]<=budget]
        best=max(possible,key=lambda r:(r["cum_tp"],-r["cum_fp"],-r["rank"])) if possible else None
        result[f"fp_budget_{budget}"]=None if best is None else {"fp":best["cum_fp"],"tp":best["cum_tp"],"recall":best["cum_tp"]/max(total_gt,1),"score_threshold":best["score"],"rank":best["rank"]}
    return result


def matches_at_floor(samples,predictions,floor):
    records,total=ranked_matches(samples,predictions,lambda a,b:quad_iou(a,b),floor)
    mapping={}
    for r in records:
        if r["tp"]:
            mapping[(r["stem"],r["gt_index"])]=r
    return mapping,total


def corner_summary(samples,mapping):
    errors=[]; norm=[]; small_tp=small_n=tiny_tp=tiny_n=boundary_tp=boundary_n=dense_tp=dense_n=0
    matched_by_stem=defaultdict(set)
    for key,r in mapping.items():
        stem,gt=key; matched_by_stem[stem].add(gt)
        target=samples[stem]["targets"][gt]; quad=r["quad"]
        ce=[math.dist(quad[i],target[i]) for i in range(4)]
        errors.extend(ce); norm.extend([e/max(math.sqrt(polygon_area(target)),1.0) for e in ce])
    for stem,s in samples.items():
        for i in range(len(s["targets"])):
            hit=i in matched_by_stem[stem]
            if s["small"][i]: small_n+=1;small_tp+=int(hit)
            if s["tiny"][i]: tiny_n+=1;tiny_tp+=int(hit)
            if s["boundary"][i]: boundary_n+=1;boundary_tp+=int(hit)
            if s["dense_image"]: dense_n+=1;dense_tp+=int(hit)
    return {
        "matched_gt":len(mapping),"target_count":sum(len(s["targets"]) for s in samples.values()),"matching_coverage":len(mapping)/max(sum(len(s["targets"]) for s in samples.values()),1),
        "corner_error_mean_px":float(np.mean(errors)) if errors else None,"corner_error_median_px":float(np.median(errors)) if errors else None,
        "corner_error_mean_normalized_by_sqrt_area":float(np.mean(norm)) if norm else None,
        "small_detected":small_tp,"small_targets":small_n,"small_recall":small_tp/max(small_n,1),
        "tiny_detected":tiny_tp,"tiny_targets":tiny_n,"tiny_recall":tiny_tp/max(tiny_n,1),
        "boundary_detected":boundary_tp,"boundary_targets":boundary_n,"boundary_recall":boundary_tp/max(boundary_n,1),
        "dense_detected":dense_tp,"dense_targets":dense_n,"dense_recall":dense_tp/max(dense_n,1),
    }


def evaluate_predictions(samples,predictions,out_dir:Path,tag:str,write_curves=True):
    out_dir.mkdir(parents=True,exist_ok=True)
    poly_ap,poly_levels=ap_metric(samples,predictions,quad_iou)
    oks_ap,oks_levels=ap_metric(samples,predictions,local_oks)
    poly_records,total=ranked_matches(samples,predictions,lambda a,b:quad_iou(a,b))
    oks_records,_=ranked_matches(samples,predictions,lambda a,b:local_oks(a,b))
    if write_curves:
        write_pr(out_dir/f"{tag}_polygon_pr_iou50.csv.gz",poly_records,total)
        write_pr(out_dir/f"{tag}_local_oks_pr_050.csv.gz",oks_records,total)
    operating=op_points(poly_records,total)
    fixed_records,_=ranked_matches(samples,predictions,lambda a,b:quad_iou(a,b),.45)
    at45={}
    for r in fixed_records:
        if r["tp"]: at45[(r["stem"],r["gt_index"])]=r
    at_recall={};at_recall_maps={}
    for name in ("recall_0.85","recall_0.88","recall_0.89"):
        point=operating[name]
        mapping={}
        if point:
            for r in poly_records:
                if r["rank"]>point["rank"]:break
                if r["tp"]:mapping[(r["stem"],r["gt_index"])]=r
        at_recall[name]=corner_summary(samples,mapping)
        at_recall_maps[name]=mapping
    p45=corner_summary(samples,at45)
    # Keep per-GT mappings and polygon coordinates for paired localization
    # comparisons (e.g. V7/V9 common GT at the same recall operating point).
    detail_path=out_dir/f"{tag}_matched_gt_metrics.csv.gz"
    with gzip.open(detail_path,"wt",encoding="utf-8",newline="") as f:
        fields=["operation","stem","gt_index","candidate_id","score","polygon_iou","predicted_quad_json","gt_quad_json","mean_corner_error_px","mean_corner_error_normalized"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        detail_maps={"score_0.45":at45,**at_recall_maps}
        for operation,mapping in detail_maps.items():
            for (stem,gt_index),record in mapping.items():
                target=samples[stem]["targets"][gt_index]
                corner_errors=[math.dist(record["quad"][i],target[i]) for i in range(4)]
                area_scale=max(math.sqrt(polygon_area(target)),1.0)
                writer.writerow({
                    "operation":operation,"stem":stem,"gt_index":gt_index,"candidate_id":record["candidate_id"],
                    "score":record["score"],"polygon_iou":quad_iou(record["quad"],target),
                    "predicted_quad_json":json.dumps(record["quad"],separators=(",",":")),
                    "gt_quad_json":json.dumps(target,separators=(",",":")),
                    "mean_corner_error_px":float(np.mean(corner_errors)),
                    "mean_corner_error_normalized":float(np.mean(corner_errors))/area_scale,
                })
    result={
        "local_oks_ap50_95_post_polygon_nms_no_floor":oks_ap,"local_oks_ap_levels":oks_levels,
        "polygon_ap50_95_post_polygon_nms_no_floor":poly_ap,"polygon_ap_levels":poly_levels,
        "target_count":total,"predictions_after_polygon_nms_and_final_cap":len(poly_records),
        "operating_points":operating,
        "score_0.45_reference":{"tp":fixed_records[-1]["cum_tp"] if fixed_records else 0,"fp":fixed_records[-1]["cum_fp"] if fixed_records else 0,"fn":total-(fixed_records[-1]["cum_tp"] if fixed_records else 0),"precision":(fixed_records[-1]["cum_tp"]/max(fixed_records[-1]["cum_tp"]+fixed_records[-1]["cum_fp"],1)) if fixed_records else 0.0,"recall":(fixed_records[-1]["cum_tp"]/max(total,1)) if fixed_records else 0.0,**p45},
        "matched_corner_by_fixed_recall":at_recall,
        "full_pr_curve_polygon_iou50":str((out_dir/f"{tag}_polygon_pr_iou50.csv.gz").resolve()) if write_curves else None,
        "full_pr_curve_local_oks_050":str((out_dir/f"{tag}_local_oks_pr_050.csv.gz").resolve()) if write_curves else None,
        "matched_gt_detail":str(detail_path.resolve()),
        "local_oks_is_approximation_not_official_metric":True,
    }
    return result,poly_records


def stage_coverage(samples,stage):
    groups=["all","small","tiny","dense","boundary"]
    result={}
    thresholds=(.5,.75)
    stage_names=("rpn_box_xyxy","box_xyxy","cap_boxes","decoded_polygons","geometry_valid_polygons","polygon_nms","final_output")
    for name in stage_names:
        vals={g:{str(t):[0,0] for t in thresholds} for g in groups}
        for stem,s in samples.items():
            rows=stage.get(stem,[])
            if name=="rpn_box_xyxy": eligible=[r for r in rows if r["rpn_box_xyxy"]]
            elif name=="box_xyxy": eligible=[r for r in rows if r["box_xyxy"] and r["score_pass"]=="1"]
            elif name=="cap_boxes": eligible=[r for r in rows if r["prekeypoint_cap_pass"]==1]
            elif name=="decoded_polygons": eligible=[r for r in rows if r["keypoint_decoded"]=="1" and r["canonical_quad_json"]]
            elif name=="geometry_valid_polygons": eligible=[r for r in rows if r["geometry_valid"]==1 and r["quad"] is not None]
            elif name=="polygon_nms": eligible=[r for r in rows if r["polygon_nms_kept"]==1 and r["quad"] is not None]
            else: eligible=[r for r in rows if r["final_output_kept"]==1 and r["quad"] is not None]
            boxes=[]; quads=[]
            for r in eligible:
                if name in ("rpn_box_xyxy","box_xyxy","cap_boxes"):
                    try:
                        b=json.loads(r["rpn_box_xyxy"] if name=="rpn_box_xyxy" else r["box_xyxy"])
                        boxes.append(b)
                    except (ValueError,TypeError): pass
                elif r["quad"] is not None: quads.append(r["quad"])
            target_boxes=[]
            for gt in s["targets"]:
                xs=[p[0] for p in gt];ys=[p[1] for p in gt];target_boxes.append([min(xs),min(ys),max(xs),max(ys)])
            for i,gt in enumerate(s["targets"]):
                cat={"all":True,"small":s["small"][i],"tiny":s["tiny"][i],"dense":s["dense_image"],"boundary":s["boundary"][i]}
                if name in ("rpn_box_xyxy","box_xyxy","cap_boxes"):
                    vals_i=[box_iou_xyxy(box,target_boxes[i]) for box in boxes]
                else: vals_i=[quad_iou(q,gt) for q in quads]
                best=max(vals_i,default=0.0)
                for group,active in cat.items():
                    if active:
                        for t in thresholds:
                            vals[group][str(t)][1]+=1;vals[group][str(t)][0]+=int(best>=t)
        result[name]={g:{t:{"covered":v[0],"targets":v[1],"coverage":v[0]/max(v[1],1)} for t,v in gg.items()} for g,gg in vals.items()}
    # Label targets which were box-covered before the cap but no longer after it.
    trunc={"all":0,"small":0,"tiny":0,"dense":0,"boundary":0,"targets":0}
    for stem,s in samples.items():
        rows=stage.get(stem,[])
        full=[r for r in rows if r["score_pass"]=="1" and r["size_pass"]=="1" and r["box_nms_pass"]=="1"]
        capped=[r for r in full if r["prekeypoint_cap_pass"]==1]
        full_boxes=[];cap_boxes=[]
        for r in full:
            try:
                b=json.loads(r["box_xyxy"]);full_boxes.append([b[0],b[1],b[2],b[3]])
            except: pass
        for r in capped:
            try:
                b=json.loads(r["box_xyxy"]);cap_boxes.append([b[0],b[1],b[2],b[3]])
            except: pass
        for i,gt in enumerate(s["targets"]):
            xs=[p[0] for p in gt];ys=[p[1] for p in gt];gtb=[min(xs),min(ys),max(xs),max(ys)]
            before=max((box_iou_xyxy(b,gtb) for b in full_boxes),default=0.0)>=.5
            after=max((box_iou_xyxy(b,gtb) for b in cap_boxes),default=0.0)>=.5
            if before and not after:
                trunc["targets"]+=1
                for g,a in (("all",True),("small",s["small"][i]),("tiny",s["tiny"][i]),("dense",s["dense_image"]),("boundary",s["boundary"][i])):
                    if a:trunc[g]+=1
    return {"coverage":result,"axis_box_iou_ge_0.50_target_lost_to_prekeypoint_cap":trunc}


def box_iou_xyxy(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    aa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]);bb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])
    return inter/max(aa+bb-inter,1e-12)


def raw_geometry_diagnostics(samples,stage,formula):
    """Describe actual decoded point sets before canonicalization, without GT use.

    Degenerate means fewer than four distinct corners (1e-6 tolerance) or a
    convex-hull area <= 1e-9 px^2. A four-point set with one interior point is
    separately counted as nonconvex. Raw ordering errors can be repaired by
    canonicalization and therefore differ from geometry-filter rejections.
    """
    counts={key:0 for key in ("decoded_candidates","malformed_or_nonfinite","raw_ordered_illegal","duplicate_or_collapsed_corners","zero_area_or_collinear","degenerate_four_corner_candidates","nonconvex_four_distinct_point_set","raw_out_of_bounds","noncyclic_order_repaired","geometry_filter_rejected")}
    permutations=defaultdict(int); reasons=defaultdict(int)
    above={key:0 for key in ("decoded_candidates","raw_ordered_illegal","degenerate_four_corner_candidates","geometry_filter_rejected")}
    for stem,rows in stage.items():
        width,height=samples[stem]["size"]
        for row in rows:
            if row["keypoint_decoded"]!="1": continue
            counts["decoded_candidates"]+=1
            high=score_for(row,formula)>=.45
            above["decoded_candidates"]+=int(high)
            rejected=row["geometry_valid"]==0
            counts["geometry_filter_rejected"]+=int(rejected)
            above["geometry_filter_rejected"]+=int(high and rejected)
            if rejected: reasons[row.get("geometry_reason") or row["removal_reason"]]+=1
            try:
                raw=np.asarray(json.loads(row["raw_keypoints_json"]),dtype=np.float64)
            except (ValueError,TypeError,json.JSONDecodeError): raw=np.empty((0,2))
            if raw.shape!=(4,2) or not np.isfinite(raw).all():
                counts["malformed_or_nonfinite"]+=1
                counts["raw_ordered_illegal"]+=1
                above["raw_ordered_illegal"]+=int(high)
                continue
            points=[(float(x),float(y)) for x,y in raw]
            illegal=not ordered_quad_is_valid(points)
            counts["raw_ordered_illegal"]+=int(illegal)
            above["raw_ordered_illegal"]+=int(high and illegal)
            unique=[]
            for point in points:
                if not any(math.dist(point,other)<=1e-6 for other in unique): unique.append(point)
            outline=hull(unique)
            collapsed=len(unique)<4
            zero_area=polygon_area(outline)<=1e-9
            degenerate=collapsed or zero_area
            counts["duplicate_or_collapsed_corners"]+=int(collapsed)
            counts["zero_area_or_collinear"]+=int(zero_area)
            counts["degenerate_four_corner_candidates"]+=int(degenerate)
            above["degenerate_four_corner_candidates"]+=int(high and degenerate)
            counts["nonconvex_four_distinct_point_set"]+=int(len(unique)==4 and len(outline)<4 and not zero_area)
            counts["raw_out_of_bounds"]+=int(any(x<0 or y<0 or x>=width or y>=height for x,y in points))
            counts["noncyclic_order_repaired"]+=int(row.get("canonical_relation")=="hull_reordered")
            if row.get("canonical_quad_json"):
                canonical=json.loads(row["canonical_quad_json"])
                permutation=tuple(min(range(4),key=lambda j:math.dist(tuple(point),points[j])) for point in canonical)
                permutations[",".join(str(x) for x in permutation)]+=1
    return {"counts":counts,"above_final_score_0.45":above,"raw_to_canonical_permutation_counts":dict(sorted(permutations.items())),"geometry_filter_reason_counts":dict(sorted(reasons.items())),"definitions":{"degenerate":"<4 distinct corners (distance<=1e-6) or hull area<=1e-9 px^2","raw_ordered_illegal":"ordered_quad_is_valid on the four raw points fails, including reorderable self-crossing order","nonconvex_four_distinct_point_set":"four distinct points with a nonzero-area convex hull containing fewer than four vertices","reason_counts_overlap":"degenerate, ordering, and bounds flags can overlap; geometry_filter_rejected is the actual canonicalized legality filter count"}}


def evaluate(args):
    manifest=Path(args.manifest).resolve();trace=Path(args.trace).resolve();out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    samples=load_dev(manifest,args.role)
    pool,loaded=load_trace(trace,samples);stage=loaded["stage"]
    formula=args.formula
    predictions=rerank_and_nms(pool,formula,args.polygon_nms,args.final_output_cap)
    metrics,_=evaluate_predictions(samples,predictions,out,args.tag or trace.parent.name,True)
    trace_info=loaded["info"]
    all_stage_rows=[r for image_rows in stage.values() for r in image_rows]
    metrics["raw_invalid_or_out_of_bounds_decoded_candidates"]=sum(r["keypoint_decoded"]=="1" and r["geometry_valid"]==0 for r in all_stage_rows)
    metrics["raw_geometry_diagnostics"]=raw_geometry_diagnostics(samples,stage,formula)
    metrics["final_illegal_geometry_count"]=sum(not ordered_quad_is_valid(row["quad"]) or any(x<0 or y<0 or x>=samples[stem]["size"][0] or y>=samples[stem]["size"][1] for x,y in row["quad"]) for stem,rows in predictions.items() for row in rows)
    if args.stage_diagnostics:
        metrics["stage_diagnostics"]=stage_coverage(samples,stage)
    result={"status":"complete","role":args.role,"formula":formula,"prekeypoint_pool_size":args.candidate_budget,"final_output_cap":args.final_output_cap,"polygon_nms_iou":args.polygon_nms,"matching":"score-ranked greedy; choose greatest continuous similarity among unmatched GT before thresholding (lowest GT index on exact ties); fixed-point polygon IoU>=0.50 and AP thresholds 0.50:0.05:0.95","evaluator_source_sha256":sha256(Path(__file__)),"group_definitions":{"small":"GT polygon area/(original width*height)<0.01","tiny":"minimum GT polygon edge length<16 original-image pixels","dense":"image contains >=3 annotated GT","boundary":"a GT corner satisfies x<=1 or y<=1 or x>=width-2 or y>=height-2","box_stage_coverage":"axis-aligned proposal/regression box IoU with GT axis-aligned enclosing box; independent of final polygon recall"},"trace_info":trace_info,"manifest_sha256":sha256(manifest),"checkpoint":args.checkpoint,"metrics":metrics,"oks":"local four-corner approximation, not official evaluator"}
    write_json(out/f"{args.tag or trace.parent.name}_evaluation.json",result)
    print(json.dumps({"status":"complete","output":str(out),"metrics":{k:v for k,v in metrics.items() if k!="stage_diagnostics"}},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--trace",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--role",default="dev");p.add_argument("--formula",default="p_cls",choices=["p_cls","q","p_cls*q","p_cls*sqrt(q)"]);p.add_argument("--tag");p.add_argument("--checkpoint");p.add_argument("--candidate-budget",type=int,default=300);p.add_argument("--final-output-cap",type=int,default=100);p.add_argument("--polygon-nms",type=float,default=.30);p.add_argument("--stage-diagnostics",action="store_true");args=p.parse_args();evaluate(args)

if __name__=="__main__":main()
