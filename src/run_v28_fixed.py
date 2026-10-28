"""Frozen V28 subpixel decoder, extracted without algorithm changes."""
import numpy as np
import torch
import torch.nn.functional as F

def refined_decode(logits, rois, original_size, resized_size):
    """Refine the existing bicubic integer maximum; never refit raw 56px grid.

    H negative definite, det>1e-12, diagonal<-1e-6; each offset<=0.5 resized
    pixel. Otherwise identity. Logits differ from log-softmax by a constant,
    so their derivatives coincide. This is not the Gaussian DARK pipeline.
    """
    raw_a=[];raw_b=[];patches=[];states=[];offsets=[]
    scale=np.array([original_size[0]/resized_size[1],original_size[1]/resized_size[0]])
    for i in range(len(rois)):
        roi=rois[i];wh=(roi[2:]-roi[:2]).clamp_min(1)
        w,h=int(wh[0].ceil()),int(wh[1].ceil())
        hm=F.interpolate(logits[i][:,None],size=(h,w),mode='bicubic',align_corners=False)[:,0]
        pos=hm.flatten(1).argmax(1);x=pos%w;y=pos//w
        a=torch.stack([(x.float()+.5)*(wh[0]/w)+roi[0],(y.float()+.5)*(wh[1]/h)+roi[1]],1)
        patch=torch.stack([hm[:,(y+dy).clamp(0,h-1),(x+dx).clamp(0,w-1)].diagonal()
                           for dy in (-1,0,1) for dx in (-1,0,1)],1).reshape(-1,3,3).double().cpu().numpy()
        inside=((x>0)&(x<w-1)&(y>0)&(y<h-1)).cpu().numpy()
        d=np.zeros((len(x),2),dtype=np.float64);state=np.zeros(len(x),dtype=np.int8)
        for k,p in enumerate(patch):
            if not inside[k]: state[k]=1;continue
            gx=(p[1,2]-p[1,0])/2;gy=(p[2,1]-p[0,1])/2
            xx=p[1,2]-2*p[1,1]+p[1,0];yy=p[2,1]-2*p[1,1]+p[0,1]
            xy=(p[2,2]-p[2,0]-p[0,2]+p[0,0])/4;det=xx*yy-xy*xy
            if not (xx < -1e-6 and yy < -1e-6 and det > 1e-12): state[k]=2;continue
            step=np.array([(-yy*gx+xy*gy)/det,(xy*gx-xx*gy)/det])
            if not np.isfinite(step).all() or (np.abs(step)>.5).any(): state[k]=3;continue
            d[k]=step
        # Preserve the identical native arithmetic path for A; add only delta for B.
        a=(a*torch.as_tensor(scale,device=a.device,dtype=a.dtype)).cpu().numpy()
        corr=wh.cpu().numpy()/[w,h]
        b=a.astype(np.float64)+d*corr*scale
        raw_a.append(a);raw_b.append(b)
        patches.append(patch.astype(np.float32));states.append(state);offsets.append(d)
    return (np.asarray(raw_a),np.asarray(raw_b),np.asarray(patches),np.asarray(states),np.asarray(offsets))
