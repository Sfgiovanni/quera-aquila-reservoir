"""Quantum reservoir denoiser: n_data reset qubits plus n_memory retained qubits.

Qubit budget. The encoding puts two quadratures on each reset qubit, so
`n_data = latent_dim // 2` and is fixed at 5 while `autoencoder.py` uses latent_dim=10. Extra
qubits therefore have to be MEMORY qubits: they are not re-prepared by `reset_data`, so they carry
state across DDIM steps and, because the reservoir unitary entangles them with the data qubits,
they also add observables that depend on the input even when they start in |0>.

Cost is the binding constraint: the state is (batch, 2^n, 2^n) and `step` does 2 matmuls of 2^n
cubed per slice, so both memory and time scale as 8^n. n=6 is the original; n=9 is roughly the
practical ceiling at batch sizes that keep the GPU busy.
"""
from __future__ import annotations
import numpy as np,torch
from magicqrc.circuits import (alpha_dial_unitary,chaotic_ising_unitary,doped_clifford_circuit,haar_random_unitary,random_clifford_circuit)
from denoiser_classical import sinusoidal_embedding

DEFAULT_QUBITS=6

def _slice_unitary(family,n_qubits,alpha,depth,rng):
 if family=='alpha_dial':return alpha_dial_unitary(n_qubits,alpha,depth,rng)
 if family=='clifford':return random_clifford_circuit(n_qubits,depth,rng)
 if family=='doped':return doped_clifford_circuit(n_qubits,depth,max(1,int(alpha)),rng)
 if family=='haar':return haar_random_unitary(n_qubits,rng)
 if family=='ising':return chaotic_ising_unitary(n_qubits,t_evolve=alpha if alpha>0 else 1.0)
 raise ValueError(f"unknown reservoir family {family!r}")
def fixed_slices(V,seed,device="cuda",n_qubits=DEFAULT_QUBITS,alpha=np.pi/4,depth=4,family='alpha_dial'):
 """V reservoir unitaries drawn from `seed`.

 Defaults reproduce the original circuit exactly: alpha_dial at alpha=pi/4, depth 4. `alpha` is
 the magic dial (0 gives Clifford dynamics, hence zero dynamics-generated magic); for the 'doped'
 family it is reinterpreted as the T-gate count, and for 'ising' as the evolution time.
 """
 rng=np.random.default_rng(seed);out=[]
 for _ in range(V):out.append(torch.as_tensor(_slice_unitary(family,n_qubits,alpha,depth,rng),dtype=torch.complex64,device=device))
 return out
def initial_state(batch,device="cuda",n_qubits=DEFAULT_QUBITS):
 d=2**n_qubits;r=torch.zeros((batch,d,d),dtype=torch.complex64,device=device);r[:,0,0]=1;return r
def data_qubits(x, encoding):
 """How many reset qubits an encoding consumes for a given latent width."""
 return x.shape[1] if encoding == 'single' else x.shape[1] // 2
def _input_density(x, encoding='quadrature', input_scale=1.0):
 # Two quadratures per reset qubit; tanh/sqrt(2) guarantees x^2+y^2<=1.
 #
 # `input_scale` divides the input before tanh (quadrature) or multiplies the Bloch components
 # (multibase). It matters because everything downstream is linear in the density matrix -- U rho
 # U^dagger is linear and Z/ZZ expectations are linear functionals of rho -- so the ONLY
 # nonlinearity is the tensor product rho_1 (x) ... (x) rho_n, whose cross terms scale as the
 # per-qubit Bloch radius squared. Default 1.0 reproduces the pre-2026-08-02 behaviour exactly.
 n_data=data_qubits(x,encoding)
 if encoding == 'multibase':
  n=torch.linalg.vector_norm(x,dim=1,keepdim=True).clamp_min(1e-8);q=(x/n).reshape(len(x),n_data,2)/np.sqrt(2)*input_scale
  r=torch.linalg.vector_norm(q,dim=2,keepdim=True).clamp_min(1.0);q=q/r
 elif encoding == 'single':
  # One latent coordinate per qubit: b=0 and z closes the Bloch sphere, so every qubit sits on
  # the surface at full radius. Spreads the same input over twice as many tensor factors, which
  # is where the reservoir's only nonlinearity comes from.
  q=torch.stack([torch.tanh(x/input_scale),torch.zeros_like(x)],dim=2)
 else:
  q=torch.tanh(x/input_scale).reshape(len(x),n_data,2)/np.sqrt(2)
 dens=[]
 for i in range(n_data):
  a,b=q[:,i,0],q[:,i,1];z=torch.zeros_like(a) if encoding == 'multibase' else torch.sqrt(torch.clamp(1-a*a-b*b,min=0));rho=torch.empty((len(x),2,2),dtype=torch.complex64,device=x.device)
  rho[:,0,0]=(1+z)/2;rho[:,1,1]=(1-z)/2;rho[:,0,1]=(a-1j*b)/2;rho[:,1,0]=(a+1j*b)/2;dens.append(rho)
 out=dens[0]
 for rho in dens[1:]:out=torch.einsum("bij,bkl->bikjl",out,rho).reshape(len(x),out.shape[1]*2,out.shape[2]*2)
 return out
def inject_time(x,t,T=200,t_scale=1.0):
 """Mechanism A: append a (sin, cos) embedding of the timestep as one extra reset qubit.

 The reservoir never sees `t` in the original design -- `h = h(x_t)` is time-independent and the
 timestep enters the readout only as a separate additive block, so a linear readout cannot form
 the `x_t x t` interaction the denoiser needs. Feeding `t` into the reset register lets the
 reservoir's own dynamics mix it with `x_t`.

 Geometry: 5 reset qubits x 2 quadratures exactly consume the 10 latent dimensions, so there is no
 free slot; this takes m from 5 to 6 (N from 6 to 7), giving 12 slots -- 10 for `x_t`, 2 for `t`.

 `t_scale` is the amplitude of the t channel relative to the x_t channels. It is a real
 hyperparameter and is swept, not guessed.
 """
 phase=2*np.pi*t.to(x.dtype)/T
 return torch.cat([x,t_scale*torch.sin(phase)[:,None],t_scale*torch.cos(phase)[:,None]],dim=1)
def time_rotation(t,d,T=200,angle=np.pi/2):
 """Mechanism B: the diagonal of a global Z rotation R(t), for U(t) = U_0 . R(t).

 Conditions the reservoir dynamics on the timestep instead of the input. R(t) is fixed and
 non-trainable given t. Because Z rotations are diagonal, `R rho R^dagger` is an elementwise
 product by `exp(-i theta/2 (s_j - s_k))`, so this costs one outer product rather than a matmul.

 NOTE: this departs from the fixed-unitary premise of reservoir computing -- the reservoir is no
 longer a single fixed dynamical system. Reported as such.
 """
 n=d.bit_length()-1;idx=torch.arange(d,device=t.device)
 s=torch.stack([(1-2*((idx>>(n-1-q))&1)) for q in range(n)]).sum(0).to(torch.float32)
 theta=angle*t.to(torch.float32)/T
 return torch.exp(-0.5j*theta[:,None]*s[None])          # (batch, d)
def reset_data(state,x,encoding='quadrature',input_scale=1.0):
 """Trace out the data qubits, keep the memory register, re-prepare the data from x."""
 d=state.shape[1];dd=2**data_qubits(x,encoding);dm=d//dd
 mem=torch.einsum("baman->bmn",state.reshape(len(state),dd,dm,dd,dm))
 rin=_input_density(x,encoding,input_scale);return torch.einsum("bij,bkl->bikjl",rin,mem).reshape(len(x),d,d)
reset_five=reset_data  # retained name used by earlier phases
def step(state,x,slices,encoding='quadrature',correlations=False,input_scale=1.0,time_phase=None):
 """One reservoir step. `time_phase` is Mechanism B's R(t) diagonal; None reproduces the original."""
 state=reset_data(state,x,encoding,input_scale);features=[];d=state.shape[1];n=d.bit_length()-1
 idx=torch.arange(d,device=x.device);signs=[(1-2*((idx>>(n-1-q))&1)) for q in range(n)]
 for U in slices:
  if time_phase is not None:
   state=time_phase[:,:,None]*state*time_phase.conj()[:,None,:]   # R(t) rho R(t)^dagger
  state=U@state@U.mH;diag=state.diagonal(dim1=-2,dim2=-1).real
  for q in range(n):features.append((diag*signs[q][None]).sum(1))
  if correlations:
   for q in range(n):
    for r in range(q+1,n):features.append((diag*(signs[q]*signs[r])[None]).sum(1))
 return state,torch.stack(features,1)
def n_observables(V,n_qubits,correlations=True):
 per=n_qubits+(n_qubits*(n_qubits-1)//2 if correlations else 0);return V*per
def time_features(t,T=200):return sinusoidal_embedding(t,10,T)
