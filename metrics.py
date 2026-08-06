"""Image-generation metrics for Round 4: FID, two IS variants and MSE."""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Standard plug-in FID between two full feature populations."""
    a=np.asarray(features_a,dtype=np.float64);b=np.asarray(features_b,dtype=np.float64)
    ma,mb=a.mean(0),b.mean(0);ca=np.cov(a,rowvar=False);cb=np.cov(b,rowvar=False)
    root=sqrtm(ca@cb)
    if np.iscomplexobj(root):root=root.real
    return float((ma-mb)@(ma-mb)+np.trace(ca+cb-2*root))


def frechet_distance_lowrank(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Exact FID, but with the trace term evaluated in the low-rank sample subspace.

    Identical in value to `frechet_distance`; it only avoids the 2048x2048 `sqrtm`. The nonzero
    eigenvalues of Ca@Cb are those of (A Bt)(B At), which is n_a x n_a. Use this when n << 2048,
    where the big sqrtm costs a minute per call and makes bootstrapping infeasible.
    """
    a=np.asarray(features_a,dtype=np.float64);b=np.asarray(features_b,dtype=np.float64)
    na,nb=len(a),len(b);d=a.shape[1];da,db=a-a.mean(0),b-b.mean(0)
    # Ca@Cb and (A Bt)(B At) share their nonzero eigenvalues, so use whichever matrix is smaller:
    # the n x n form when n << d (the small-sample case this was written for), the d x d form when
    # n > d. At n=10000, d=2048 the n x n route is 116x more work for the same answer.
    if min(na,nb)>d:
        # Ca@Cb is NOT symmetric, so eigvalsh on it is wrong. Ca^{1/2} Cb Ca^{1/2} is symmetric
        # PSD and shares its eigenvalues, so build that instead.
        ca=da.T@da/(na-1);cb=db.T@db/(nb-1)
        w,v=np.linalg.eigh((ca+ca.T)/2);root=v*np.sqrt(np.clip(w,0,None))@v.T
        eigenvalues=np.clip(np.linalg.eigvalsh(root@cb@root),0,None)
    else:
        # (A Bt)(B At) = (A Bt)(A Bt)^T is symmetric PSD by construction.
        cross=(da@db.T)@(db@da.T)/((na-1)*(nb-1))
        eigenvalues=np.clip(np.linalg.eigvalsh((cross+cross.T)/2),0,None)
    diff=a.mean(0)-b.mean(0)
    return float(diff@diff+(da*da).sum()/(na-1)+(db*db).sum()/(nb-1)-2*np.sqrt(eigenvalues).sum())


def fid_null_floor(real_a: np.ndarray, real_b: np.ndarray, n_generated: int,
                   repeats: int = 20, seed: int = 0) -> dict:
    """FID between two same-distribution real sets, at the n a generated set would use.

    Any reported FID must be read against this floor: it is what a *perfect* generator scores
    given the finite real reference. Plug-in FID is strongly n-dependent, so `n_generated` must
    match the arm being judged or the comparison is meaningless.
    """
    rng=np.random.default_rng(seed)
    values=[frechet_distance_lowrank(real_a,real_b[rng.integers(0,len(real_b),n_generated)])
            for _ in range(repeats)]
    return {"fid_floor_mean":float(np.mean(values)),"fid_floor_sd":float(np.std(values,ddof=1)),
            "fid_floor_n_generated":int(n_generated),"fid_floor_repeats":int(repeats)}


def inception_score(probabilities: np.ndarray, splits: int = 10) -> tuple[float,float]:
    """Mean and SD of exp(E_x KL[p(y|x)||p(y)]) over fixed contiguous splits."""
    p=np.asarray(probabilities,dtype=np.float64);p=np.clip(p,1e-12,1);scores=[]
    for chunk in np.array_split(p,splits):
        marginal=chunk.mean(0,keepdims=True)
        scores.append(float(np.exp(np.mean(np.sum(chunk*(np.log(chunk)-np.log(marginal)),axis=1)))))
    return float(np.mean(scores)),float(np.std(scores,ddof=1))


def reconstruction_error(images: np.ndarray, reconstructions: np.ndarray) -> float:
    return float(np.mean((np.asarray(images)-np.asarray(reconstructions))**2))


def inception_features_and_probs(images: np.ndarray, batch_size: int = 64, device: str = "cuda"):
    """Official ImageNet InceptionV3 pool3 features and class probabilities."""
    import torch
    import torch.nn.functional as F
    from torchvision.models import Inception_V3_Weights,inception_v3
    weights=Inception_V3_Weights.DEFAULT;model=inception_v3(weights=weights,aux_logits=True).to(device).eval()
    captured=[]
    hook=model.avgpool.register_forward_hook(lambda _m,_i,o:captured.append(torch.flatten(o,1)))
    feats=[];probs=[]
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            x=torch.as_tensor(images[start:start+batch_size],dtype=torch.float32,device=device)
            if x.ndim==3:x=x[:,None]
            x=(x+1)/2;x=x.repeat(1,3,1,1);x=F.interpolate(x,size=(299,299),mode='bilinear',align_corners=False);x=(x-mean)/std
            captured.clear();logits=model(x);feats.append(captured[0].cpu().numpy());probs.append(torch.softmax(logits,1).cpu().numpy())
    hook.remove();return np.concatenate(feats),np.concatenate(probs)


def fmnist_probs(images: np.ndarray, model_path: str, batch_size: int = 256, device: str = "cuda"):
    import torch
    import torch.nn.functional as F
    from experiments.phase_f import FashionCNN
    model=FashionCNN().to(device);model.load_state_dict(torch.load(model_path,map_location=device,weights_only=True));model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            x=torch.as_tensor(images[start:start+batch_size,None],dtype=torch.float32,device=device);out.append(torch.softmax(model(x),1).cpu().numpy())
    return np.concatenate(out)
