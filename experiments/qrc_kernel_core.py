"""Leakage-safe kernel machinery for vector epsilon denoising.

All selectors consume train/validation arrays only.  Test arrays are accepted only by ``predict``.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import tracemalloc

import numpy as np
from scipy import linalg, stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
GAMMA_MULTIPLIERS = (.25, .5, 1., 2.)
GAMMA_T_MULTIPLIERS = (.5, 1., 2.)
ALPHA_RES = (0., .25, .5, .75, 1.)


class Standardizer:
    def fit(self, x):
        self.mean_ = np.asarray(x, np.float64).mean(0)
        self.scale_ = np.asarray(x, np.float64).std(0)
        self.scale_[self.scale_ < 1e-10] = 1.
        return self
    def transform(self, x):
        return (np.asarray(x, np.float64) - self.mean_) / self.scale_


def sqdist(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.maximum((a*a).sum(1)[:, None] + (b*b).sum(1)[None] - 2*a@b.T, 0.)


def median_gamma(x, seed=0, max_rows=1000):
    rng = np.random.default_rng(seed)
    z = x[rng.choice(len(x), min(len(x), max_rows), replace=False)]
    d = sqdist(z, z)[np.triu_indices(len(z), 1)]
    med = float(np.median(d[d > 0])) if np.any(d > 0) else 1.
    return 1. / max(med, 1e-12), med


def kernel_matrix(a, b, kind, gamma=None, ta=None, tb=None, gamma_t=None):
    if kind == "linear":
        k = np.asarray(a, np.float64) @ np.asarray(b, np.float64).T
    elif kind in ("rbf", "composed_time"):
        k = np.exp(-float(gamma) * sqdist(a, b))
    else:
        raise ValueError(kind)
    if kind == "composed_time":
        if ta is None or tb is None or gamma_t is None:
            raise ValueError("composed_time needs ta, tb and gamma_t")
        k *= np.exp(-float(gamma_t) * (np.asarray(ta)[:, None]-np.asarray(tb)[None])**2)
    return k


def kernel_diagnostics(k, y=None):
    k = np.asarray(k, np.float64)
    sym = float(np.max(np.abs(k-k.T)))
    ks = (k+k.T)/2
    eig = np.linalg.eigvalsh(ks)
    pos = eig[eig > max(1e-12, eig.max()*1e-10)] if len(eig) else eig
    p = np.clip(eig, 0, None); p = p/p.sum() if p.sum() else p
    eff = float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))) if p.sum() else 0.
    out = dict(symmetry_max_abs=sym, diagonal_mean=float(np.diag(k).mean()),
               element_mean=float(k.mean()), element_sd=float(k.std()),
               numerical_rank=int(len(pos)), effective_rank=eff,
               eigen_min=float(eig.min()), eigen_max=float(eig.max()),
               negative_eigenvalues=int((eig < -max(1e-10, abs(eig.min())*1e-8)).sum()),
               condition_number=float(pos.max()/pos.min()) if len(pos) else np.inf,
               spectrum=eig.tolist())
    if y is not None:
        yy = np.asarray(y, np.float64) @ np.asarray(y, np.float64).T
        h = np.eye(len(k))-np.ones_like(k)/len(k)
        kc, yc = h@ks@h, h@yy@h
        den = np.linalg.norm(kc)*np.linalg.norm(yc)
        out["centered_kernel_target_alignment"] = float(np.sum(kc*yc)/den) if den else np.nan
        den0 = np.linalg.norm(ks)*np.linalg.norm(yy)
        out["kernel_target_alignment"] = float(np.sum(ks*yy)/den0) if den0 else np.nan
    return out


def centered_kernel_alignment(a, b):
    h = np.eye(len(a))-np.ones_like(a)/len(a)
    ac, bc = h@a@h, h@b@h
    den = np.linalg.norm(ac)*np.linalg.norm(bc)
    return float(np.sum(ac*bc)/den) if den else np.nan


@dataclass
class KRRModel:
    train_features: np.ndarray
    alpha: np.ndarray
    kind: str
    gamma: float | None
    train_t: np.ndarray | None = None
    gamma_t: float | None = None
    standardizer: Standardizer | None = None
    def predict(self, x, t=None):
        z = self.standardizer.transform(x) if self.standardizer else np.asarray(x)
        return kernel_matrix(z, self.train_features, self.kind, self.gamma,
                             t, self.train_t, self.gamma_t) @ self.alpha


def _stable_solve(k, y, lam):
    a = (k+k.T)/2 + float(lam)*np.eye(len(k))
    try:
        c, low = linalg.cho_factor(a, lower=True, check_finite=True)
        return linalg.cho_solve((c, low), y, check_finite=True), "cholesky"
    except linalg.LinAlgError:
        return np.linalg.solve(a, y), "solve"


def fit_krr(x, y, xv, yv, kind, lambdas=LAMBDAS, gamma_multipliers=GAMMA_MULTIPLIERS,
            t=None, tv=None, gamma_t_multipliers=GAMMA_T_MULTIPLIERS, seed=0):
    st = Standardizer().fit(x); z, zv = st.transform(x), st.transform(xv)
    base_gamma, med = median_gamma(z, seed)
    gammas = [None] if kind == "linear" else [base_gamma*m for m in gamma_multipliers]
    if kind == "composed_time":
        td = (np.asarray(t)[:, None]-np.asarray(t)[None])**2
        vals = td[np.triu_indices(len(t), 1)]; mt = float(np.median(vals[vals>0])) if np.any(vals>0) else 1.
        gt_values = [(1/max(mt,1e-12))*m for m in gamma_t_multipliers]
    else:
        mt, gt_values = np.nan, [None]
    best = None
    for gamma in gammas:
        for gamma_t in gt_values:
            k = kernel_matrix(z, z, kind, gamma, t, t, gamma_t)
            kv = kernel_matrix(zv, z, kind, gamma, tv, t, gamma_t)
            # One symmetric eigendecomposition serves every lambda exactly and avoids repeating
            # five O(n^3) factorizations for the same kernel during screening.
            ew, ev = np.linalg.eigh((k+k.T)/2)
            evty = ev.T @ y
            for lam in lambdas:
                alpha = ev @ (evty / (ew[:, None] + float(lam)))
                solver = "symmetric_eigendecomposition"
                mse = float(np.mean((kv@alpha-yv)**2))
                candidate = (mse, lam, gamma, gamma_t, alpha, solver, k)
                if best is None or candidate[0] < best[0]: best = candidate
    mse, lam, gamma, gamma_t, alpha, solver, k = best
    model = KRRModel(z, alpha, kind, gamma, np.asarray(t) if t is not None else None, gamma_t, st)
    return model, dict(val_mse=mse, lambda_=lam, gamma=gamma, gamma_t=gamma_t,
                       median_sqdist=med, median_t_sqdist=mt, solver=solver,
                       diagnostics=kernel_diagnostics(k, y))


def random_features(x, xv, dim, seed):
    """Matched tanh random projection; the map is fitted/drawn without validation targets."""
    st = Standardizer().fit(x); z, zv = st.transform(x), st.transform(xv)
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(z.shape[1], dim))/np.sqrt(z.shape[1]); b = rng.uniform(-np.pi, np.pi, dim)
    return np.tanh(z@w+b), np.tanh(zv@w+b), dict(seed=seed, input_mean=st.mean_.tolist(), input_scale=st.scale_.tolist(), w=w, b=b)


def grouped_oof_ridge(x, y, groups, alpha, n_splits=5):
    unique = np.unique(groups); folds = min(n_splits, len(unique))
    if folds < 2: raise ValueError("cross-fitting needs at least two groups")
    pred = np.full_like(y, np.nan, dtype=np.float64); seen = np.zeros(len(y), bool)
    for tr, ho in GroupKFold(folds).split(x, y, groups):
        if np.intersect1d(np.unique(groups[tr]), np.unique(groups[ho])).size:
            raise AssertionError("group leakage")
        pred[ho] = Ridge(alpha=alpha).fit(x[tr], y[tr]).predict(x[ho]); seen[ho] = True
    if not seen.all() or not np.isfinite(pred).all(): raise AssertionError("incomplete OOF")
    return pred


def select_ridge(x, y, xv, yv, alphas=(1e-6,1e-4,1e-2,1.,100.)):
    best = None
    for a in alphas:
        m = Ridge(alpha=a).fit(x, y); score=float(np.mean((m.predict(xv)-yv)**2))
        if best is None or score < best[0]: best=(score,a,m)
    return best


def select_residual_blend(base_val, residual_val, yv, tval):
    rows=[]
    for a in ALPHA_RES:
        p=base_val+a*residual_val
        rows.append((float(np.mean((p-yv)**2)),a))
    mse, alpha=min(rows)
    edges=np.quantile(tval,[0,1/3,2/3,1]); bins=np.clip(np.digitize(tval,edges[1:-1]),0,2)
    gate=[]
    for j in range(3):
        s=bins==j
        gate.append(bool(np.mean((base_val[s]+alpha*residual_val[s]-yv[s])**2) < np.mean((base_val[s]-yv[s])**2)))
    return alpha, np.asarray(gate), edges, mse


def timestep_metrics(y, pred, t, baseline=None, edges=None):
    if edges is None: edges=np.quantile(t,[0,1/3,2/3,1])
    bins=np.clip(np.digitize(t,edges[1:-1]),0,2); rows=[]
    for name,j in zip(("early_noise","middle_noise","late_denoising"),range(3)):
        s=bins==j; err=pred[s]-y[s]
        row=dict(timestep_bin=name,n=int(s.sum()),mse=float(np.mean(err**2)),
                 mae=float(np.mean(np.abs(err))),residual_norm=float(np.linalg.norm(err,axis=1).mean()))
        if baseline is not None: row["mse_improvement_vs_baseline"]=float(np.mean((baseline[s]-y[s])**2)-row["mse"])
        rows.append(row)
    return rows


def stratified_landmarks(t, n, seed):
    n=min(int(n),len(t)); rng=np.random.default_rng(seed); edges=np.quantile(t,[0,1/3,2/3,1]); b=np.clip(np.digitize(t,edges[1:-1]),0,2)
    out=[]
    for j in range(3):
        idx=np.flatnonzero(b==j); take=min(len(idx),n//3+(j < n%3)); out.extend(rng.choice(idx,take,False))
    if len(out)<n:
        rest=np.setdiff1d(np.arange(len(t)),out); out.extend(rng.choice(rest,n-len(out),False))
    return np.asarray(out,dtype=int)


@dataclass
class NystromKRR:
    standardizer: Standardizer; landmarks: np.ndarray; beta: np.ndarray
    kind: str; gamma: float; landmark_t: np.ndarray | None; gamma_t: float | None
    def predict(self,x,t=None):
        z=self.standardizer.transform(x)
        return kernel_matrix(z,self.landmarks,self.kind,self.gamma,t,self.landmark_t,self.gamma_t)@self.beta


def fit_nystrom(x,y,xv,yv,t,tv,kind,n_landmarks,seed=0,lambdas=LAMBDAS):
    st=Standardizer().fit(x); z,zv=st.transform(x),st.transform(xv); idx=stratified_landmarks(t,n_landmarks,seed)
    gamma,_=median_gamma(z,seed); gamma_t=None
    if kind=="composed_time":
        vals=(np.asarray(t)[:,None]-np.asarray(t)[None])**2; vals=vals[np.triu_indices(len(t),1)]
        gamma_t=1/max(float(np.median(vals[vals>0])),1e-12)
    c=kernel_matrix(z,z[idx],kind,gamma,t,np.asarray(t)[idx],gamma_t)
    cv=kernel_matrix(zv,z[idx],kind,gamma,tv,np.asarray(t)[idx],gamma_t)
    w=kernel_matrix(z[idx],z[idx],kind,gamma,np.asarray(t)[idx],np.asarray(t)[idx],gamma_t)
    ew,ev=np.linalg.eigh((w+w.T)/2); keep=ew>max(1e-10,ew.max()*1e-10)
    phi=c@((ev[:,keep]/np.sqrt(ew[keep]))); phiv=cv@((ev[:,keep]/np.sqrt(ew[keep])))
    best=None
    for lam in lambdas:
        coef=np.linalg.solve(phi.T@phi+lam*np.eye(phi.shape[1]),phi.T@y); mse=float(np.mean((phiv@coef-yv)**2))
        if best is None or mse<best[0]: best=(mse,lam,coef)
    beta=(ev[:,keep]/np.sqrt(ew[keep]))@best[2]
    return NystromKRR(st,z[idx],beta,kind,gamma,np.asarray(t)[idx],gamma_t), dict(val_mse=best[0],lambda_=best[1],n_landmarks=len(idx),rank=len(ew[keep]),landmark_indices=idx.tolist())


def paired_statistics(a,b):
    d=np.asarray(a)-np.asarray(b); n=len(d); mean=float(d.mean()); sd=float(d.std(ddof=1)) if n>1 else np.nan
    se=sd/np.sqrt(n) if n>1 else np.nan; ci=stats.t.interval(.95,n-1,loc=mean,scale=se) if n>1 else (np.nan,np.nan)
    p=float(stats.ttest_rel(a,b).pvalue) if n>1 else np.nan
    effect=mean/sd if sd and np.isfinite(sd) else np.nan
    mde=float((stats.t.ppf(.975,n-1)+stats.t.ppf(.8,n-1))*sd/np.sqrt(n)) if n>1 else np.nan
    return dict(n=n,mean_difference=mean,sd_difference=sd,ci95_low=float(ci[0]),ci95_high=float(ci[1]),p_value=p,effect_size_dz=effect,mde=mde)


def measured_call(fn,*args,**kwargs):
    tracemalloc.start(); start=perf_counter(); value=fn(*args,**kwargs); _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    return value, dict(wall_clock_s=perf_counter()-start,peak_python_memory_mb=peak/2**20)
