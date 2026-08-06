"""Phase 4: five-reset quadrature QRC with temporal multiplexing."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import numpy as np,pandas as pd,torch
from sklearn.linear_model import Ridge
from autoencoder import decode_numpy,encode_numpy,load_autoencoder
from data import load_fashion_mnist
from ddpm import cosine_schedule
from denoiser_qrc import fixed_slices,initial_state,inject_time,step,time_rotation
from denoiser_classical import sinusoidal_embedding
from latent_scaling import LatentScale
from metrics import frechet_distance,fmnist_probs,inception_features_and_probs,inception_score

def eps_data(z,abar,seed):
 rng=np.random.default_rng(seed);t=rng.integers(1,len(abar)+1,len(z));e=rng.normal(size=z.shape);a=abar[t-1,None];return np.sqrt(a)*z+np.sqrt(1-a)*e,e,t
@torch.inference_mode()
def features(xt_all,t_all,abar,V,seed,device='cuda',batch=1024,encoding='quadrature',correlations=False,input_scale=1.0,n_qubits=6,alpha=np.pi/4,depth=4,family='alpha_dial',slice_seed=None,time_mode=None,t_scale=1.0,time_angle=np.pi/2):
 """Reservoir design matrix [x_t, h, te].

 `time_mode` lets the reservoir see the timestep, which the original design does not:
   None      -- h = h(x_t), the inherited behaviour, bit-for-bit unchanged.
   'input'   -- Mechanism A: t enters the reset register as an extra qubit (needs n_qubits=7).
   'unitary' -- Mechanism B: U(t) = U_0 . R(t) conditions the dynamics instead.
 """
 out=[];slices=fixed_slices(V,seed if slice_seed is None else slice_seed,device,n_qubits,alpha,depth,family)
 for i in range(0,len(xt_all),batch):
  xt=torch.as_tensor(xt_all[i:i+batch],dtype=torch.float32,device=device);n=len(xt);t=torch.as_tensor(t_all[i:i+batch],device=device)
  xin=inject_time(xt,t,len(abar),t_scale) if time_mode=='input' else xt
  ph=time_rotation(t,2**n_qubits,len(abar),time_angle) if time_mode=='unitary' else None
  _,h=step(initial_state(n,device,n_qubits),xin,slices,encoding,correlations,input_scale,time_phase=ph);te=sinusoidal_embedding(t,10,len(abar));out.append(torch.cat([xt,h,te],1).cpu().numpy())
 return np.concatenate(out)
@torch.inference_mode()
def sample_qrc(noise,model,abar,V,seed,device='cuda',batch=256,steps=50,encoding='quadrature',correlations=False,x0_clip=.5,input_scale=1.0,n_qubits=6,alpha=np.pi/4,depth=4,family='alpha_dial',slice_seed=None,eta=0.,noise_seed=0,time_mode=None,t_scale=1.0,time_angle=np.pi/2):
 """QRC DDIM sampler. See experiments/phase2.py:sample for `x0_clip` and sample_base for `eta`."""
 result=[];gen=np.random.default_rng(noise_seed)
 for start in range(0,len(noise),batch):
  x=torch.as_tensor(noise[start:start+batch],dtype=torch.float32,device=device);state=initial_state(len(x),device,n_qubits);slices=fixed_slices(V,seed if slice_seed is None else slice_seed,device,n_qubits,alpha,depth,family);ab=torch.as_tensor(abar,dtype=torch.float32,device=device);grid=np.rint(np.linspace(1,len(abar),steps)).astype(int)[::-1]
  for i,tv in enumerate(grid):
   t=torch.full((len(x),),int(tv),device=device,dtype=torch.long)
   xin=inject_time(torch.clamp(x,-1,1),t,len(abar),t_scale) if time_mode=='input' else torch.clamp(x,-1,1)
   ph=time_rotation(t,2**n_qubits,len(abar),time_angle) if time_mode=='unitary' else None
   state,h=step(state,xin,slices,encoding,correlations,input_scale,time_phase=ph);design=torch.cat([torch.clamp(x,-1,1),h,sinusoidal_embedding(t,10,len(abar))],1).cpu().numpy();eps=model.predict(design);a=float(abar[tv-1]);prev=int(grid[i+1]) if i+1<len(grid) else 0;ap=1. if prev==0 else float(abar[prev-1]);x0=np.clip((x.cpu().numpy()-np.sqrt(1-a)*eps)/np.sqrt(a),-x0_clip,x0_clip);sig=eta*np.sqrt((1-ap)/(1-a))*np.sqrt(max(1-a/ap,0.)) if ap<1 else 0.;x=torch.as_tensor(np.sqrt(ap)*x0+np.sqrt(max(1-ap-sig*sig,0.))*eps+(sig*gen.normal(size=x0.shape) if sig>0 else 0.),dtype=torch.float32,device=device)
  result.append(x.cpu().numpy())
 return np.concatenate(result)
@torch.inference_mode()
def sample_base(noise,model,abar,device='cuda',batch=512,steps=50,x0_clip=.5,eta=0.,noise_seed=0):
 """No-reservoir ridge DDIM sampler. See experiments/phase2.py:sample for `x0_clip`.

 `eta` interpolates DDIM->DDPM: 0 is the deterministic map used everywhere before 2026-08-03,
 1 is ancestral sampling. Deterministic DDIM injects no noise after the initial draw, so sample
 diversity is bounded by the diversity of that draw -- which is what recall measures.
 """
 result=[];gen=np.random.default_rng(noise_seed);ab=torch.as_tensor(abar,dtype=torch.float32,device=device);grid=np.rint(np.linspace(1,len(abar),steps)).astype(int)[::-1]
 for start in range(0,len(noise),batch):
  x=torch.as_tensor(noise[start:start+batch],dtype=torch.float32,device=device)
  for i,tv in enumerate(grid):
   t=torch.full((len(x),),int(tv),device=device,dtype=torch.long);design=torch.cat([torch.clamp(x,-1,1),sinusoidal_embedding(t,10,len(abar))],1).cpu().numpy();eps=model.predict(design);a=float(abar[tv-1]);prev=int(grid[i+1]) if i+1<len(grid) else 0;ap=1. if prev==0 else float(abar[prev-1]);x0=np.clip((x.cpu().numpy()-np.sqrt(1-a)*eps)/np.sqrt(a),-x0_clip,x0_clip);sig=eta*np.sqrt((1-ap)/(1-a))*np.sqrt(max(1-a/ap,0.)) if ap<1 else 0.;x=torch.as_tensor(np.sqrt(ap)*x0+np.sqrt(max(1-ap-sig*sig,0.))*eps+(sig*gen.normal(size=x0.shape) if sig>0 else 0.),dtype=torch.float32,device=device)
  result.append(x.cpu().numpy())
 return np.concatenate(result)
def select_ridge(x,y,xv,yv):
 best=None
 for alpha in (1e-6,1e-4,1e-2,1.,100.):
  m=Ridge(alpha=alpha).fit(x,y);score=np.mean((m.predict(xv)-yv)**2)
  if best is None or score<best[0]:best=(score,m)
 return best[1]
def main():
 device='cuda';seed0=20260730;train_x,_,test_x,_=load_fashion_mnist('data/fashion-mnist/raw');ae=load_autoencoder('checkpoints/autoencoder_d10.pt',device);ztr=encode_numpy(ae,train_x);zte=encode_numpy(ae,test_x);_,abar=cosine_schedule(200);scale=LatentScale(ztr);ztr=scale.forward(ztr);print('latent scaling',scale.describe(),flush=True);real_f,_=inception_features_and_probs(test_x,batch_size=256,device=device);noise=np.random.default_rng(seed0+999).normal(size=(10000,10));rows=[];Path('checkpoints/qrc').mkdir(parents=True,exist_ok=True)
 for seed in range(5):
  x,e,t=eps_data(ztr,abar,seed0+seed);xv,ev,tv=eps_data(ztr[-10000:],abar,seed0+100+seed);no=np.column_stack([xv, sinusoidal_embedding(torch.as_tensor(tv),10,200).numpy()]);base=select_ridge(np.column_stack([x[:50000],sinusoidal_embedding(torch.as_tensor(t[:50000]),10,200).numpy()]),e[:50000],no,ev); 
  for V in (4,8,16):
   started=perf_counter();h=features(x,t,abar,V,seed0+seed);hv=features(xv,tv,abar,V,seed0+100+seed);m=select_ridge(h[:50000],e[:50000],hv,ev);torch.save({'V':V,'seed':seed},f'checkpoints/qrc/V{V}_s{seed}.pt');gen=decode_numpy(ae,scale.inverse(sample_qrc(noise,m,abar,V,seed0+seed,device,x0_clip=scale.x0_clip)));gf,_=inception_features_and_probs(gen,batch_size=256,device=device);p=fmnist_probs(gen,'results/fmnist_classifier.pt',batch_size=1024,device=device);isv,isd=inception_score(p);row={'arch':'qrc','arm':'reset_quadrature','latent_dim':10,'V':V,'seed':seed,'n_params_trainable':m.coef_.size,'n_params_quantum_trainable':0,'T_train':200,'ddim_steps':50,'fid':frechet_distance(real_f,gf),'is_fmnist':isv,'is_fmnist_sd':isd,'n_samples':10000,'wall_clock_s':perf_counter()-started};rows.append(row);print('QRC_RESULT '+json.dumps(row),flush=True)
  gen=decode_numpy(ae,scale.inverse(sample_base(noise,base,abar,device,x0_clip=scale.x0_clip)));gf,_=inception_features_and_probs(gen,batch_size=256,device=device);p=fmnist_probs(gen,'results/fmnist_classifier.pt',batch_size=1024,device=device);isv,isd=inception_score(p);rows.append({'arch':'qrc','arm':'no_reservoir','latent_dim':10,'V':0,'seed':seed,'n_params_trainable':base.coef_.size,'n_params_quantum_trainable':0,'T_train':200,'ddim_steps':50,'fid':frechet_distance(real_f,gf),'is_fmnist':isv,'is_fmnist_sd':isd,'n_samples':10000,'wall_clock_s':0.})
 pd.DataFrame(rows).to_parquet('results/phase4_qrc.parquet',index=False);Path('results/phase4_qrc_scaling.json').write_text(json.dumps(scale.describe(),indent=2)+'\n')
if __name__=='__main__':main()
