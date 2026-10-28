"""Frozen V9 candidate quality ranking. GT is read only for train_fit targets.

Concepts: Mask Scoring R-CNN predicted-output features; GFL continuous quality
labels. This compact MLP is an adaptation, not a reproduction of either paper.
"""
import argparse,csv,gzip,json,math,time,hashlib,subprocess,sys,random
from pathlib import Path
import os
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision.transforms.functional import to_tensor
from v9_model import build_model,norm_inventory,parse_prediction
from v9_geometry import quad_iou,parse_shapes,polygon_area,local_oks
from v9_train import read_rows,sha256,write_json,atomic_torch_save

ROOT=Path(os.environ.get('DMKPD_QUALITY_RUN', 'runs/quality')).resolve()
MANIFEST=Path(os.environ.get('DMKPD_MANIFEST', 'data/manifest.csv'))
BASE=Path(os.environ.get('DMKPD_DETECTOR', 'weights/detector.pth'))
BASE_SHA='89cfc3b044db37c4232cbc6f9e839b0b40bbe3fe2beb51a2e250cc3e47633a88'

def features(pred,logits,box_features):
    """Identical GT-free FP32 features at collection and deployment."""
    ids=pred['candidate_ids'].long();p=pred['class_scores'].float().clamp(1e-6,1-1e-6)
    feat=box_features[ids].float(); n,k,h,w=logits.shape
    prob=logits.float().flatten(2).softmax(-1)
    xx=((torch.arange(w,device=logits.device)+.5)/w).repeat(h)
    yy=((torch.arange(h,device=logits.device)+.5)/h).repeat_interleave(w)
    mx=(prob*xx).sum(-1);my=(prob*yy).sum(-1)
    sx=((prob*(xx-mx[...,None])**2).sum(-1)+1e-12).sqrt()
    sy=((prob*(yy-my[...,None])**2).sum(-1)+1e-12).sqrt()
    top=prob.topk(2,dim=-1).values
    entropy=-(prob*prob.clamp_min(1e-12).log()).sum(-1)/math.log(h*w)
    peak=prob.argmax(-1);px=(peak%w+.5)/w;py=(peak//w+.5)/h
    heat=torch.stack([entropy,top[:,:,0],top[:,:,0]-top[:,:,1],mx,my,sx,sy,mx-px,my-py],-1).flatten(1)
    b=pred['boxes'].float();wh=(b[:,2:]-b[:,:2]).clamp_min(1e-6)
    xy=(pred['keypoints'][:,:,:2].float()-b[:,None,:2])/wh[:,None,:]
    edge=xy.roll(-1,1)-xy;length=edge.norm(dim=-1)
    cross=edge[:,:,0]*edge.roll(-1,1)[:,:,1]-edge[:,:,1]*edge.roll(-1,1)[:,:,0]
    area=.5*(xy[:,:,0]*xy.roll(-1,1)[:,:,1]-xy[:,:,1]*xy.roll(-1,1)[:,:,0]).sum(1).abs()
    diag=torch.stack([(xy[:,0]-xy[:,2]).norm(dim=-1),(xy[:,1]-xy[:,3]).norm(dim=-1)],1)
    geometry=torch.cat([xy.flatten(1),length,cross,area[:,None],diag,wh.log(),(wh[:,0]/wh[:,1]).log()[:,None]],1)
    x=torch.cat([torch.logit(p)[:,None],feat,heat,geometry],1)
    if not torch.isfinite(x).all():raise FloatingPointError('nonfinite deployment features')
    return x

def frozen_model():
    if sha256(BASE)!=BASE_SHA:raise ValueError('base checkpoint hash mismatch')
    torch.manual_seed(20261010)
    m=build_model(False,100,False,'p_cls','p_cls').cuda()
    m.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=False)['model'],strict=True)
    if norm_inventory(m)['normalization_modules']!={'FrozenBatchNorm2d':53}:raise ValueError('FrozenBN mismatch')
    m.requires_grad_(False);m.eval();return m

def collect(role,limit=None):
    if role not in ['train_fit','dev']:raise ValueError('only train_fit/dev allowed')
    out=ROOT/'data'/role;out.mkdir(parents=True,exist_ok=True)
    rows=read_rows(MANIFEST,role);rows=rows[:limit] if limit else rows
    model=frozen_model();captured={}
    model.roi_heads.box_head.register_forward_hook(lambda m,i,o:captured.__setitem__('roi',o.detach()))
    model.roi_heads.keypoint_predictor.register_forward_hook(lambda m,i,o:captured.__setitem__('heatmap',o.detach()))
    summaries=[]
    with torch.inference_mode():
        for i,r in enumerate(rows,1):
            path=out/(r['stem']+'.npz')
            if path.exists():
                with np.load(path,allow_pickle=False) as z:
                    if str(z['checkpoint_sha'])!=BASE_SHA or str(z['manifest_sha'])!=sha256(MANIFEST):raise ValueError('stale shard')
                    count=len(z['p'])
                summaries.append(dict(stem=r['stem'],candidates=count,cached=True));continue
            with Image.open(r['image']) as src:im=src.convert('RGB');width,height=im.size
            torch.cuda.synchronize();start=time.perf_counter()
            pred=model([to_tensor(im).cuda()])[0];x=features(pred,captured['heatmap'],captured['roi'])
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            raw=pred['keypoints'][:,:,:2].cpu().numpy();quads=[];valid=[]
            for points in raw:
                q,reason=parse_prediction(points.tolist(),width,height)
                quads.append(q if q is not None and not reason else [[float('nan')]*2]*4)
                valid.append(q is not None and not reason)
            target=np.full(len(raw),np.nan,dtype=np.float32)
            if role=='train_fit':
                gt=parse_shapes(Path(r['annotation']),r['stem'])
                for j,(q,v) in enumerate(zip(quads,valid)):
                    target[j]=max((quad_iou(q,g) for g in gt),default=0.) if v else 0.
            payload=dict(features=x.cpu().numpy().astype(np.float16),p=pred['class_scores'].cpu().numpy(),ids=pred['candidate_ids'].cpu().numpy(),boxes=pred['boxes'].cpu().numpy(),raw=raw,quads=np.asarray(quads,dtype=np.float32),valid=np.asarray(valid,dtype=bool),target=target,width=width,height=height,checkpoint_sha=BASE_SHA,manifest_sha=sha256(MANIFEST),feature_schema='logit_p,1024_RoI,36_heatmap,22_geometry',inference_seconds=seconds)
            tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**payload);tmp.replace(path)
            summaries.append(dict(stem=r['stem'],candidates=len(raw),cached=False))
            if i==1 or i%100==0 or i==len(rows):print(json.dumps(dict(role=role,images=f'{i}/{len(rows)}',seconds=seconds,feature_dim=x.shape[1]),ensure_ascii=False),flush=True)
    write_json(out/'collection_complete.json',dict(status='complete',role=role,images=len(rows),candidates=sum(s['candidates'] for s in summaries),checkpoint_sha256=BASE_SHA,manifest_sha256=sha256(MANIFEST),annotations_read=(role=='train_fit'),dev_targets_saved=False,model_frozen=True,all_feature_finite=True,shards=summaries))

class RankHead(nn.Module):
    def __init__(self,dim):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,p):
        delta=8*torch.tanh(self.net(x).squeeze(1))
        prior_logit=torch.logit(p.clamp(1e-6,1-1e-6))
        q=torch.sigmoid(prior_logit+delta)
        # Exact identity at zero residual while retaining d(score)/d(delta).
        # torch.where(delta==0,p,q) would mask every initial head gradient!
        return (p+(q-torch.sigmoid(prior_logit))).clamp(0,1)

def build_quality_targets():
    """Before fitting: local-OKS proxy on same GT matched by maximum polygon IoU."""
    rows=read_rows(MANIFEST,'train_fit');out=ROOT/'data/local_oks_targets';out.mkdir(parents=True,exist_ok=True)
    assert len(list((ROOT/'data/train_fit').glob('*.npz')))==len(rows)==7731
    stats={'background':0,'localization_negative':0,'foreground':0};sums=[0.,0.,0.];nums=[0,0,0]
    for n,r in enumerate(rows,1):
        path=out/(r['stem']+'.npy')
        with np.load(ROOT/'data/train_fit'/(r['stem']+'.npz'),allow_pickle=False) as z:
            if path.exists():values=np.load(path)
            else:
                gt=parse_shapes(Path(r['annotation']),r['stem']);values=[]
                for q,valid in zip(z['quads'],z['valid']):
                    choices=[(quad_iou(q.tolist(),g),i) for i,g in enumerate(gt)] if valid else []
                    overlap,index=max(choices,key=lambda c:(c[0],-c[1]),default=(0.,-1))
                    values.append(local_oks(q.tolist(),gt[index]) if overlap>=.1 else 0.)
                values=np.asarray(values,dtype=np.float32);tmp=path.with_suffix('.tmp.npy');np.save(tmp,values);tmp.replace(path)
            assert len(values)==len(z['p']) and np.isfinite(values).all()
            group=np.where(z['target']>=.5,2,np.where(z['target']>=.1,1,0))
            for g in range(3):sums[g]+=float(values[group==g].sum());nums[g]+=int((group==g).sum())
        if n==1 or n%500==0 or n==len(rows):print(json.dumps(dict(local_oks_targets=f'{n}/{len(rows)}')),flush=True)
    write_json(out/'target_summary.json',dict(status='complete',source='train_fit only',formula='V9 local four-corner OKS sigma .1, not official evaluator',GT_assignment='max polygon IoU with stable GT index tie',background_definition='polygon IoU<.1 or invalid ->target0',groups=nums,target_means=[s/max(n,1) for s,n in zip(sums,nums)],images=7731))

def fit(arm,seed=20261010):
    if arm not in ['roi_only','geometry']:raise ValueError(arm)
    if not (ROOT/'data/local_oks_targets/target_summary.json').exists():build_quality_targets()
    out=ROOT/'train'/f'{arm}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():print('already fitted',out);return
    files=sorted((ROOT/'data/train_fit').glob('*.npz'));assert len(files)==7731
    arrays=[];targets=[];priors=[];overlaps=[]
    for f in files:
        with np.load(f,allow_pickle=False) as z:
            arrays.append(z['features']);overlaps.append(z['target']);targets.append(np.load(ROOT/'data/local_oks_targets'/(f.stem+'.npy')));priors.append(z['p'])
    x=np.concatenate(arrays);t=np.concatenate(targets);p=np.concatenate(priors);iou=np.concatenate(overlaps);del arrays,targets,priors,overlaps
    if not np.isfinite(x).all() or not np.isfinite(t).all():raise ValueError('nonfinite training data')
    # Standardization derived exclusively from train_fit.
    stats_path=ROOT/'data/train_feature_standardization.npz'
    if stats_path.exists():
        with np.load(stats_path) as z:mean=z['mean'];std=z['std']
    else:
        sums=np.zeros(x.shape[1],dtype=np.float64);squares=sums.copy()
        for start in range(0,len(x),10000):
            chunk=x[start:start+10000].astype(np.float64);sums+=chunk.sum(0);squares+=(chunk*chunk).sum(0)
        mean=sums/len(x);std=np.sqrt(np.maximum(squares/len(x)-mean*mean,0.)).clip(.001)
        mean=mean.astype(np.float32);std=std.astype(np.float32);np.savez(stats_path,mean=mean,std=std,source='train_fit only')
    if arm=='roi_only':x=x[:,:1025].copy();mean=mean[:1025].copy();std=std[:1025].copy()
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    common=RankHead(1083)
    if x.shape[1]==1083:model=common.cuda()
    else:
        model=RankHead(x.shape[1]);state=common.state_dict();state['net.0.weight']=state['net.0.weight'][:,:x.shape[1]].clone();model.load_state_dict(state);model=model.cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,10,eta_min=.00005)
    xx=torch.from_numpy(x).cuda();tt=torch.from_numpy(t).cuda();pp=torch.from_numpy(p).cuda()
    mm=torch.from_numpy(mean).cuda();ss=torch.from_numpy(std).cuda()
    ii=torch.from_numpy(iou).cuda();groups=torch.where(ii>=.5,2,torch.where(ii>=.1,1,0)); counts=[int((groups==g).sum()) for g in range(3)]
    with torch.no_grad():assert torch.equal(model((xx[:100].float()-mm)/ss,pp[:100]),pp[:100])
    log=[];start_epoch=0
    if (out/'latest.pt').exists():
        saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        assert saved['checkpoint_sha']==BASE_SHA and saved['arm']==arm and saved['seed']==seed
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);log=saved['log'];start_epoch=saved['epoch']
        assert np.array_equal(mean,saved['mean']) and np.array_equal(std,saved['std'])
        assert saved['manifest_sha']==sha256(MANIFEST) and scheduler.last_epoch==start_epoch
        if 'numpy_rng' in saved:np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    for epoch in range(start_epoch+1,11):
        gen=torch.Generator(device='cuda').manual_seed(seed+epoch)
        order=torch.randperm(len(tt),device='cuda',generator=gen);sums=np.zeros(3);nums=np.zeros(3);grad_sum=0.;steps=0
        start=time.perf_counter();model.train()
        for batch in order.split(1024):
            opt.zero_grad(set_to_none=True);q=model((xx[batch].float()-mm)/ss,pp[batch]);target=tt[batch];g=groups[batch]
            per=torch.nn.functional.binary_cross_entropy(q.clamp(1e-6,1-1e-6),target,reduction='none')
            parts=[]
            for index in range(3):
                mask=g==index
                if mask.any():
                    value=per[mask].mean();parts.append(value);sums[index]+=float(per[mask].sum().detach());nums[index]+=int(mask.sum())
            loss=torch.stack(parts).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite rank loss')
            loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10))
            if not math.isfinite(grad):raise FloatingPointError('nonfinite rank gradient')
            opt.step();grad_sum+=grad;steps+=1
        scheduler.step()
        record=dict(epoch=epoch,background_loss=sums[0]/max(nums[0],1),localization_negative_loss=sums[1]/max(nums[1],1),foreground_loss=sums[2]/max(nums[2],1),gradient_mean=grad_sum/steps,steps=steps,lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-start)
        log.append(record)
        bundle=dict(model=model.state_dict(),mean=mean,std=std,dim=x.shape[1],arm=arm,seed=seed,epoch=epoch,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),checkpoint_sha=BASE_SHA,feature_schema='logit_p,1024_RoI,36_heatmap,22_geometry',groups=counts,score_formula='sigmoid(logit(p_clipped)+8*tanh(g(z))) plus numerical identity correction p-sigmoid(logit(p_clipped))',quality_meaning='balanced local OKS ranking proxy with background0, not calibrated foreground probability or official OKS',log=log,manifest_sha=sha256(MANIFEST),sampler_next_epoch_seed=seed+epoch+1,training_base_frozen=True)
        bundle.update(numpy_rng=np.random.get_state(),python_rng=random.getstate(),head_source_sha256=sha256(Path(__file__)),protocol_sha256=sha256(ROOT/'configs/OFFICIAL_METRIC_ADDENDUM.json'))
        atomic_torch_save(bundle,out/'latest.pt')
        atomic_torch_save(bundle,out/f'epoch_{epoch:02}.pt')
        with (out/'training_log.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(record));w.writeheader();w.writerows(log)
        print(json.dumps(dict(arm=arm,seed=seed,**record)),flush=True)
    with (out/'training_log.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(log[0]));w.writeheader();w.writerows(log)
    write_json(out/'complete.json',dict(status='complete',epochs=10,train_fit_images=len(files),candidates=len(t),groups=counts,zero_update_scores_equal=True,base_detector_updated=False,feature_dim=x.shape[1],seed=seed))

def emit(arm,seed=20261010):
    out=ROOT/'evaluation'/f'{arm}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    files=sorted((ROOT/'data/dev').glob('*.npz'));assert len(files)==859
    if arm!='baseline':
        ck=torch.load(ROOT/'train'/f'{arm}_seed{seed}'/'epoch_10.pt',map_location='cpu',weights_only=False)
        model=RankHead(ck['dim']).cuda();model.load_state_dict(ck['model'],strict=True);model.eval()
        mm=torch.from_numpy(ck['mean']).cuda();ss=torch.from_numpy(ck['std']).cuda()
    fields=['stem','candidate_id','roi_index','rpn_proposal_count','rpn_box_xyxy','class_score','quality_score','final_score','score_formula','pool_formula','box_xyxy','raw_keypoints_json','canonical_quad_json','canonical_relation','score_pass','size_pass','box_nms_pass','prekeypoint_cap_pass','keypoint_decoded','geometry_valid','geometry_reason','polygon_nms_kept','final_output_kept','removal_reason']
    trace=out/'dev_candidate_trace.csv.gz';start=time.perf_counter()
    with torch.inference_mode(),gzip.open(trace,'wt',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for path in files:
            with np.load(path,allow_pickle=False) as z:
                p=z['p'];x=z['features'];valid=z['valid'];quads=z['quads'];ids=z['ids'];raw=z['raw'];boxes=z['boxes']
                assert np.isnan(z['target']).all(),'dev target leakage'
                if arm=='baseline':scores=p
                else:scores=model((torch.from_numpy(x[:,:ck['dim']]).cuda().float()-mm)/ss,torch.from_numpy(p).cuda()).cpu().numpy()
                order=sorted([i for i,v in enumerate(valid) if v],key=lambda i:(-float(scores[i]),int(ids[i])))
                kept=[]
                for i in order:
                    if all(quad_iou(quads[i].tolist(),quads[j].tolist())<.30 for j in kept):kept.append(i)
                final=set(kept[:100]);kept=set(kept)
                for i in range(len(p)):
                    canonical,reason=parse_prediction(raw[i].tolist(),int(z['width']),int(z['height']))
                    relation='invalid'
                    if canonical is not None and not reason:
                        assert np.array_equal(np.asarray(canonical,dtype=np.float32),quads[i])
                        perm=tuple(int(np.argmin(np.linalg.norm(raw[i]-np.asarray(point),axis=1))) for point in canonical)
                        cyclic=any(perm==tuple((start+direction*k)%4 for k in range(4)) for start in range(4) for direction in [1,-1])
                        relation='cyclic_or_reversed' if cyclic else 'hull_reordered'
                    writer.writerow(dict(stem=path.stem,candidate_id=int(ids[i]),roi_index=int(ids[i]),rpn_proposal_count='',rpn_box_xyxy='[]',class_score=float(p[i]),quality_score=float(scores[i]),final_score=float(scores[i]),score_formula='q',pool_formula='p_cls',box_xyxy=json.dumps(boxes[i].tolist()),raw_keypoints_json=json.dumps(raw[i].tolist()),canonical_quad_json=json.dumps(canonical) if canonical is not None else '',canonical_relation=relation,score_pass=1,size_pass=1,box_nms_pass=1,prekeypoint_cap_pass=1,keypoint_decoded=1,geometry_valid=int(valid[i]),geometry_reason=reason,polygon_nms_kept=int(i in kept),final_output_kept=int(i in final),removal_reason='kept' if i in final else ('polygon_nms' if valid[i] else reason)))
    cmd=[sys.executable,str(ROOT/'code/v9_evaluate.py'),'--manifest',str(MANIFEST),'--trace',str(trace),'--output-dir',str(out),'--formula','q','--tag',arm,'--checkpoint',str(BASE),'--candidate-budget','100','--final-output-cap','100','--polygon-nms','.30']
    subprocess.run(cmd,check=True)
    write_json(out/'frozen_pool_audit.json',dict(status='complete',dev_images=859,decoded_candidates=85900,same_npz_shards_for_all_arms=True,boxes_and_raw_keypoints_unchanged=True,quality_GT_used_in_inference=False,stage_trace_scope='decoded top100 only; full RPN/pre-cap stages remain in original V9 trace; empty RPN boxes mean unavailable, not zero coverage',elapsed_seconds=time.perf_counter()-start))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['collect','targets','fit','emit']);p.add_argument('--role',choices=['train_fit','dev']);p.add_argument('--arm',choices=['baseline','roi_only','geometry']);p.add_argument('--seed',type=int,default=20261010);p.add_argument('--limit',type=int);a=p.parse_args()
    if a.action=='collect':collect(a.role,a.limit)
    elif a.action=='targets':build_quality_targets()
    elif a.action=='fit':fit(a.arm,a.seed)
    else:emit(a.arm,a.seed)
if __name__=='__main__':main()
