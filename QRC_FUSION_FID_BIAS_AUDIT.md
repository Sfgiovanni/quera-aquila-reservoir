# Auditoria de viés — comparação FID Ridge clássico vs Ridge clássico+QRC-B (Fashion-MNIST)

Data: 2026-08-05. Revisão do experimento reportado em `QRC_CLASSICAL_FEATURE_FUSION_REPORT.md`
§9 (commit 90eefc1) em resposta à pergunta: a comparação de FID é justa?

**Resposta curta: nenhuma das duas comparações é justa, e as duas invertem de sinal quando
corrigidas.** Dois defeitos mecânicos, ambos isolados por ablação:

| Contraste | Publicado | Corrigido | Defeito responsável |
|---|---:|---:|---|
| Fashion ΔFID (híbrido − clássico) | **+221,53** | **−5,60** (p=2,4e-5) | mismatch de estado treino/geração |
| Breast ΔFID | **+108,47** | **−16,19** (p=1,0e-7) | idem |
| Fashion Δtest MSE, n=500 | **+0,0172** | **+0,0010** (p=0,63) | piso de variância de 1e-10 |

O ΔFID publicado media divergência numérica do rollout: 11 das 15 células híbridas emitiram a
**mesma imagem** 10.000 vezes. O ΔMSE publicado vinha de dividir 12–30 observáveis
constantes-em-float32 por ~1e-8. Com as correções, o QRC **vence** o Ridge clássico em FID nos dois
datasets, e **empata** com um bloco de interação puramente clássico em Fashion (p=0,55).

A recomendação final do relatório — encerrar a hipótese de complementaridade QRC — continua
defensável, mas por um argumento diferente e mais fraco: não "o QRC piora", e sim "o QRC não supera
um bloco `x_t ⊗ te(t)` clássico de custo desprezível". Ver §5d.

## 0. Correção de premissa

Os braços não são "QRC+ridge (clássicas+quânticas) vs ridge (só quânticas)". São:

| Braço | Features | Nº |
|---|---|---:|
| `ridge_classical` | `[z_t, te(t)]` — **somente clássicas**, zero observáveis quânticos | 20 |
| `ridge_classical_qrc_B` | `[z_t, h_QRC, te(t)]` — clássicas **+** 84 observáveis Z/ZZ | 104 |

O braço de controle não tem nenhuma feature quântica. Como o modelo clássico está **aninhado** no
híbrido (`Ridge(φ_C) ⊂ Ridge([φ_C, φ_Q])`), com regularização bem ajustada e amostra infinita o
híbrido não pode ser pior. Ele só pode perder por (i) variância de amostra finita com n=500,
(ii) busca de λ truncada, (iii) incompatibilidade treino/inferência das features. As três estavam
presentes. Era correto achar o resultado estranho.

## 1. Achado principal: 11 das 15 células híbridas colapsaram para um único ponto

Em `results/qrc_classical_feature_fusion/fashion_fid_cells/`:

| data_seed | draw | FID | latent_energy | **diversity** |
|---:|---:|---:|---:|---:|
| 0 | 0 | 355,72 | 2,379 | **0,000000** |
| 0 | 1 | 306,27 | 2,476 | **0,000000** |
| 0 | 2 | 102,68 | 0,330 | 0,266 |
| 0 | 3 | 415,15 | 2,481 | **0,000000** |
| 0 | 4 | 346,06 | 1,795 | 0,135 |
| 1 | 0 | 381,13 | 2,370 | **0,000063** |
| 1 | 1 | 336,34 | 2,449 | **0,000000** |
| 1 | 2 | 80,19 | 0,124 | 0,277 |
| 1 | 3 | 327,22 | 2,492 | **0,000000** |
| 1 | 4 | 382,50 | 1,092 | 0,360 |
| 2 | 0 | 287,13 | 2,410 | **0,000000** |
| 2 | 1 | 336,68 | 2,493 | **0,000000** |
| 2 | 2 | 81,89 | 0,131 | 0,285 |
| 2 | 3 | 336,26 | 2,565 | **0,000000** |
| 2 | 4 | 332,44 | 1,637 | 0,197 |
| — | clássico | 70,53–75,66 | 0,027–0,046 | 0,167–0,181 |

`diversity = mean(std(enc, axis=0))`. No último passo DDIM `prev=0 → ap=1`, logo `x = x0` com
`x0 = clip(·, ±x0_clip)`. Um zero exato em 10 dimensões latentes × 10.000 amostras significa que
**todas as 10.000 amostras são idênticas e saturadas no mesmo limite de clip** — verificado:
`frac_at_clip = 1,000`. O híbrido emitiu a mesma imagem 10.000 vezes.

FID ≈ 330 é a distância de uma massa pontual até o Fashion-MNIST. Não é uma medida de denoising.
Todas as estatísticas de §9 (p=1,99e-6, dz=2,00, MDE 86,17) foram calculadas sobre células
degeneradas.

**O draw 2 elimina a explicação alternativa.** É o único draw que não colapsa nas três seeds, e é
justamente o de **pior** val_MSE nas seeds 1 e 2 (0,5927 e 0,5925). Ele dá FID 80–103, contra
70–76 do clássico. Qualidade preditiva não prevê colapso; o gap de FID é modo de falha numérica.

## 1b. BreastMNIST (§8) tem exatamente a mesma patologia

`results/qrc_classical_feature_fusion/generation_breastmnist.parquet`: **11 das 15** células
híbridas têm `diversity = 0,000000` **e** `recall = 0,000000`, com `latent_energy` 2,37–2,51.

O `recall = 0` é uma confirmação independente e mais forte que a diversity: vem de outro caminho
métrico (projeção Inception + kNN de 5 vizinhos) e uma massa pontual tem recall zero por construção.
Duas métricas distintas concordam que a saída é um único ponto.

O draw 2 é de novo o único que não colapsa nas três seeds, e ali o FID híbrido fica **abaixo** do
clássico (108,80 / 112,43 / 109,43 contra 131,35 / 120,90 / 131,59). O ΔFID publicado de **+108,47**
vem, portanto, inteiramente das células degeneradas.

**Mas as células não colapsadas não servem de estimativa do FID híbrido correto, em nenhuma
direção.** O traço do rollout mostra que o draw 2 sofre a mesma explosão — `mean|z_q|` = 219.732 e
`max|x|` = 457.713 nos passos 1–2 — e apenas por acaso é contrativo, decaindo para 0,78 no passo 10.
É o mesmo pipeline quebrado, num regime que não divergiu. Além disso, para o draw 2 a versão com
reset tem diversity *menor* (0,639) que a versão com estado carregado (1,252), ou seja, a célula
publicada não é uma variante degradada da configuração correta.

Conclusão de §1 e §1b: §8 e §9 não medem qualidade generativa. Elas medem se o rollout divergiu.

## 2. Causa raiz: colunas mortas amplificadas por incompatibilidade treino/geração

Dois defeitos que só se manifestam juntos.

**(a) O reservatório é stateless no treino e stateful na geração.** `experiments/phase4.py:features`
cria `initial_state` a cada chamada e dá **um** `step` por linha: o readout é ajustado sempre com o
qubit de memória em `|0⟩`. `sample_qrc` cria `initial_state` **uma vez por batch** e carrega o
estado pelos 50 passos DDIM. Com 6 qubits e encoding quadrature há 5 qubits de dado resetados e
1 de memória preservado, e a memória entra em todos os 84 observáveis por emaranhamento.
Já documentado em `QRC_STATEFUL_AUDIT.md` como "assimetria treino–geração", mas o efeito
quantitativo não havia sido medido.

**(b) 12 a 30 dos 84 observáveis são exatamente constantes no treino, e o `Standardizer` os
amplifica em vez de descartá-los.** Com a memória presa em `|0⟩`, a estrutura do circuito
`alpha_dial` torna vários Z/ZZ determinísticos — std de treino ~1e-8, que é ruído de float32
(ex.: `V0.Z1Z5` é constante igual a +1,0000; `V2.Z5`, `V2.Z0Z5`…`V2.Z4Z5` são constantes iguais a 0).
`experiments/qrc_kernel_core.py:27` usa piso de variância `scale_ < 1e-10 → 1`. Um piso de 1e-10
**não** captura std de 1e-8: essas colunas passam e são divididas por ~1e-8.

O §3 do relatório afirma "Não houve features constantes" — o teste usou exatamente esse limiar de
1e-10, então é um falso negativo.

Composição dos dois, medida diretamente (seed 0):

| | Δ bruto por 1 passo de estado carregado | Δ em unidades padronizadas |
|---|---:|---:|
| colunas mortas (draw 0, 12 colunas) | 0,0864 | **1,1e6 σ** |
| colunas bem condicionadas | 0,0564 | 0,32 σ |
| colunas mortas (draw 2, 30 colunas) | 0,0108 | **3,6e5 σ** |
| colunas bem condicionadas | 0,0076 | 0,10 σ |

Traço do rollout real (seed 0, draw 0), `mean|z_q|` = magnitude média das features QRC padronizadas
que entram no Ridge:

```
passo  0 (t=200, estado novo):  mean|z_q| = 0,67      max|x| = 3,44
passo  1 (t=196, 1 passo):      mean|z_q| = 160.777   max|z_q| = 1,3e7
passo  2 (t=192):               mean|z_q| = 144.948   max|x|   = 911.627   <- latente explodiu
passo 49 (t=1):                 mean|z_q| = 142.671
```

A cadeia é: memória sai de `|0⟩` no passo 1 → colunas mortas divididas por 1e-8 → excursão de 1e6 σ
→ predição de epsilon explode → `x0` satura o clip → todas as amostras iguais → FID ≈ 330. O draw 2
sofre a mesma explosão no passo 1 mas decai de volta (passo 10: `mean|z_q|` = 0,78), e é por isso
que ele é o único que produz imagens — sorte numérica, não mérito do modelo.

Aumentar λ **não** resgata: com λ=1e4 a norma de `W_q` cai para 0,091 e o colapso permanece
(0,091 × 1e6 continua enorme). O defeito não é de regularização.

## 3. Contrafactual: resetar o estado a cada passo DDIM

Alinhar geração com treino (reset completo antes de cada passo, que é o que `features()` faz) elimina
o colapso em todas as configurações testadas, seed 0, n=256:

| draw | λ | estado | val_MSE | ‖W_q‖ | diversity | frac_at_clip |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 10 | carregado | 0,57196 | 1,429 | 0,000005 **COLAPSO** | 1,000 |
| 0 | 10 | reset | 0,57196 | 1,429 | 0,733 | 0,000 |
| 0 | 100 | carregado | 0,56694 | 0,612 | 0,000005 **COLAPSO** | 1,000 |
| 0 | 100 | reset | 0,56694 | 0,612 | 0,742 | 0,000 |
| 0 | 1e4 | carregado | 0,90029 | 0,091 | 0,000005 **COLAPSO** | 1,000 |
| 0 | 1e4 | reset | 0,90029 | 0,091 | 1,718 | 0,007 |
| 2 | 10 | carregado | 0,57232 | 2,003 | 1,252 | 0,000 |
| 2 | 10 | reset | 0,57232 | 2,003 | 0,639 | 0,001 |

Referência clássica no mesmo rollout: diversity 0,820, frac_at_clip 0,000 (× sigma=0,2179 reproduz
exatamente a diversity 0,18 publicada, confirmando que o rollout de diagnóstico é fiel ao pipeline).

## 4. Contrafactual: teto de λ

`LAMBDAS=(1e-4 … 10.)` e **30/30 células publicadas selecionaram 10,0**, o topo da grade, nos dois
braços. Busca truncada na fronteira. Reexecutando o braço supervisionado com a grade estendida até
1e4 para **ambos** os braços, mesmas 3 seeds × 5 draws:

- O clássico continua escolhendo λ=10 mesmo com 100 disponível — o teto era o ótimo genuíno para ele.
- O híbrido migra para λ=100 em 6/15 células.
- ΔMSE validação cai de **+0,01722** para **+0,01462** (SD 0,00833, IC95% [+0,01001; +0,01923],
  p=8,59e-6, 1/15 vitórias híbridas).
- ΔMSE teste: **+0,01412** (SD 0,00970, IC95% [+0,00874; +0,01949], p=6,14e-5, 2/15).

**O negativo supervisionado sobrevive.** O teto de λ inflou o efeito em ~15%, não o criou. Isso é
consistente com a ausência de incompatibilidade treino/teste no braço supervisionado: ali `features()`
é usada nas duas pontas, então o defeito (a) não atua.

## 5. Vieses menores, não determinantes

- **Clamp de entrada.** `features()` recebe `x_t` sem clamp; `sample_qrc` e `sample_base` usam
  `torch.clamp(x,-1,1)` sobre latentes de std 1, truncando ~32% das coordenadas em t alto. É uma
  incompatibilidade treino/amostragem real, mas atinge os dois braços no bloco linear.
- **n=500 com 104 features.** O híbrido tem 5,2× mais features e o mesmo orçamento de amostras.
  Mede-se valor incremental, não eficiência por dimensão — o relatório já declara isso em §10.
- **`val_mse` reportado é o mínimo sobre a grade de λ**, selecionado na mesma validação. Otimista
  para os dois braços igualmente.

## 5b. Alcance: quais experimentos passados são afetados

23 arquivos em `experiments/` chamam `sample_qrc`, e **todos** ajustam o readout com
`phase4.features` (stateless) e amostram com `sample_qrc` (carrega estado). Logo o defeito (a) é
generalizado no projeto — inclui `phase4.py` (os FIDs originais de Phase 4), `time_injection.py`
(commit bd5762c), `qrc_observable_ablation.py`, `multibase_reservoir_ablation.py`, os
`breastmnist_*` e `qrc_factorial.py`.

**Mas o colapso não é generalizado, e a razão é instrutiva.** Varredura de todos os
`results/**/*.parquet` com coluna de diversidade e FID: as únicas células com diversidade ≈ 0 estão
em `results/qrc_classical_feature_fusion/`. Nenhum outro experimento apresenta a assinatura.

A causa é que o defeito (b) exige **padronização**, e só três arquivos importam `Standardizer`:
`qrc_kernel_core.py` (que a define), `qrc_classical_feature_fusion.py` e
`qrc_timestep_selective_core.py`. Os experimentos legados alimentam `h` **bruto** direto no Ridge —
`time_injection.Readout.predict` não padroniza nada, `phase4.main` chama
`select_ridge(h, e, hv, ev)` sobre features cruas. Sem divisão por 1e-8 as colunas mortas são apenas
features quase constantes com coeficientes pequenos, e a deriva de estado de ~0,09 em unidades brutas
é uma perturbação modesta em vez de uma excursão de 1e6 σ.

E `qrc_timestep_selective_core.selective_ddim` (linhas 205–227), que padroniza no controle de random
map, **recalcula as features a cada passo via `qrc_design` → `features()`**, que cria estado novo.
Stateless nas duas pontas: sem incompatibilidade, sem amplificação.

Portanto:

| | (a) mismatch de estado | (b) colunas mortas amplificadas | colapso |
|---|---|---|---|
| `qrc_classical_feature_fusion` | sim | **sim** | **sim** |
| `phase4`, `time_injection`, ablations | sim | não (sem padronização) | não |
| `qrc_timestep_selective` | não (recalcula por passo) | n/a | não |

Os resultados negativos anteriores de geração **não são invalidados por esta auditoria**. Eles
carregam (a) — um shift treino/inferência real e ainda não quantificado, que deve ser declarado como
limitação — mas não a falha catastrófica. O experimento de fusão é o único que combinou padronização
com colunas mortas, e foi essa combinação que produziu o ΔFID de +221.

## 5c. Reexecução corrigida (`experiments/qrc_fusion_fair_*`)

Nova implementação com todas as correções: reset por passo DDIM, piso de variância 1e-6, grade de λ
até 1e5, readout em GPU verificado contra o caminho numpy, guard rails que falham em vez de
reportar, e **dois controles novos** — `ridge_random` (84 features tanh aleatórias, casado em
dimensão) e `ridge_interaction` (100 produtos `x_t ⊗ te(t)`, o controle clássico estruturado usado
como baseline no time-injection e no timestep-selective).

Protocolo: 3 data seeds × 5 unitary draws, λ escolhido só na validação, **MSE de teste** como
primário (2000 pares de teste fixos, os mesmos para todo n_train).

### Atribuição do negativo supervisionado publicado

Fashion-MNIST, n_train=500, tudo idêntico exceto o piso de variância:

| Configuração | Δ test MSE (QRC − clássico) | p | vitórias |
|---|---:|---:|---:|
| publicado (§4 do relatório, ΔMSE validação) | +0,01722 | 4e-6 | 1/15 |
| piso 1e-10 (comportamento publicado), λ até 1e5 | **+0,01712** | 1,05e-7 | 0/15 |
| piso 1e-6 | **+0,00101** | **0,63** | 6/15 |

O piso de 1e-10 reproduz o número publicado quase exatamente, e trocá-lo por 1e-6 elimina o efeito
inteiro. Os braços `random` e `interaction` são bit-idênticos entre as duas linhas (+0,00560 e
−0,02514), porque não têm colunas mortas — o que isola a ablação ao bloco QRC. BreastMNIST replica:
+0,02283 (p=9e-7) com piso 1e-10, +0,00617 (p=0,016) com 1e-6.

**O negativo supervisionado publicado é, na maior parte, artefato do piso de variância**, não uma
propriedade dos observáveis. Com o piso correto e n=500 o QRC é um **nulo**, não um negativo.

### Sweep de n_train e o controle clássico estruturado

| dataset | n_train | Δ QRC | p | vit. | Δ random | Δ **interaction** | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| fashion | 500 | +0,00101 | 0,63 | 6/15 | +0,00560 | **−0,02514** | 1,1e-15 |
| fashion | 2000 | −0,02228 | 1,2e-9 | 15/15 | −0,00189 | **−0,05810** | 1,9e-18 |
| fashion | 5000 | −0,02864 | 3,9e-10 | 15/15 | −0,00935 | **−0,07080** | 3,8e-19 |
| breast | 500 | +0,00617 | 0,016 | 4/15 | +0,00823 | **−0,02936** | 1,4e-19 |
| breast | 2000 | −0,01551 | 6,8e-8 | 15/15 | +0,00421 | **−0,06369** | 1,4e-18 |
| breast | 5000 | −0,02174 | 4,6e-10 | 15/15 | −0,00116 (n.s.) | **−0,07706** | 1,4e-20 |

Duas leituras, ambas necessárias:

1. **O QRC não prejudica o denoising**, ao contrário do que o relatório afirma. Ele é nulo em n=500
   e melhora consistentemente em n≥2000 (15/15 nos dois datasets). O sinal do efeito publicado está
   invertido.
2. **Em MSE, o ganho parece ser capacidade e não quantumness.** `ridge_interaction` — 100 produtos
   de features que o modelo clássico já possui, custo computacional desprezível, nenhum simulador
   quântico — bate o QRC por 2,5× a 3,5× em todos os seis pontos, incluindo n=500 onde o QRC é nulo.
   E supera o
   controle `random` de dimensão comparável, então não é só contagem de features: parte é estrutura
   útil, e essa estrutura é clássica.

**Atenção: esta segunda leitura não sobrevive em FID.** O fator de 2,5–3,5× é específico do MSE de
epsilon; em geração o QRC empata com o interaction no Fashion. Ver §5d, subseção "QRC contra o
controle clássico estruturado". A conclusão de capacidade vale para MSE, não para FID.

## 5d. Geração corrigida: FID e Inception Score

144 células, 3 data seeds × 5 unitary draws × 4 braços × 2 valores de n_train × 2 datasets.
Fashion gera 10.000 amostras (comparável ao publicado); BreastMNIST gera **1.000**, não as 100 do
§8 publicado, então os FIDs absolutos de Breast **não** são comparáveis ao 127,94 — só as diferenças
pareadas internas o são.

Sanidade do pipeline: **0 células degeneradas** (publicado: 11/15), `max|z_q|` padronizado de
**14,4 σ** ao longo dos 50 passos (publicado: 1,3e7), e discordância máxima entre o readout em GPU e
o caminho numpy de 2,3e-5. O braço clássico reproduz o publicado: FID 72,35 contra 72,31.

### O ΔFID publicado desaparece e troca de sinal

| dataset | ΔFID publicado (híbrido − clássico) | ΔFID corrigido | p | vitórias |
|---|---:|---:|---:|---:|
| Fashion, n_train=500 | **+221,53** | **−5,598** | 2,4e-5 | 14/15 |
| Breast, n_train=500 | **+108,47** | **−16,188** | 1,0e-7 | 15/15 |

Uma oscilação de 227 pontos de FID no Fashion. O QRC **vence** o Ridge clássico em geração, nos dois
datasets, no protocolo publicado (n_train=500). Em Fashion com n_train=5000 o contraste vira nulo
(+1,493, p=0,114, 7/15).

### Tabela completa (FID menor melhor; IS maior melhor)

| dataset | n_train | braço | FID | SD | IS | recall | diversity |
|---|---:|---|---:|---:|---:|---:|---:|
| fashion | 500 | interaction | **66,11** | 1,36 | **5,023** | 0,554 | 1,012 |
| fashion | 500 | qrc | 66,75 | 4,57 | 4,704 | 0,478 | 0,888 |
| fashion | 500 | classical | 72,35 | 2,91 | 4,682 | 0,435 | 0,789 |
| fashion | 500 | random | 90,10 | 5,21 | 3,724 | 0,237 | 0,598 |
| fashion | 5000 | classical | **59,16** | 0,35 | 5,319 | 0,548 | 0,945 |
| fashion | 5000 | qrc | 60,66 | 3,45 | 5,252 | 0,567 | 1,038 |
| fashion | 5000 | interaction | 61,24 | 0,61 | **5,428** | 0,573 | 1,140 |
| fashion | 5000 | random | 77,27 | 5,58 | 3,939 | 0,347 | 0,687 |
| breast (n_gen=1000 ⚠) | 500 | interaction | **87,63** | 0,20 | — | 0,184 | 0,945 |
| breast (n_gen=1000 ⚠) | 500 | qrc | 99,31 | 4,01 | — | 0,115 | 0,762 |
| breast (n_gen=1000 ⚠) | 500 | classical | 115,50 | 6,32 | — | 0,090 | 0,633 |
| breast (n_gen=1000 ⚠) | 500 | random | 137,87 | 6,42 | — | 0,034 | 0,492 |
| breast (n_gen=1000 ⚠) | 5000 | interaction | **87,21** | 1,01 | — | 0,160 | 1,039 |
| breast (n_gen=1000 ⚠) | 5000 | qrc | 93,09 | 2,48 | — | 0,133 | 0,858 |
| breast (n_gen=1000 ⚠) | 5000 | classical | 107,61 | 2,72 | — | 0,079 | 0,713 |
| breast (n_gen=1000 ⚠) | 5000 | random | 126,76 | 5,70 | — | 0,053 | 0,577 |

⚠ **Os FIDs de Breast não são comparáveis ao 127,94 publicado**, que usou 100 amostras geradas
contra as mesmas 156 imagens reais. Só as diferenças pareadas internas a esta tabela são válidas.
Fashion usa 10.000 nos dois casos e é diretamente comparável.

### Por que o clássico ganha em Fashion com n_train=5000

O braço clássico melhora de FID 72,35 para 59,16 entre n_train=500 e 5000 — um salto maior que
qualquer diferença entre braços. Não é artefato; é visível no rollout:

| n_train | braço | λ | FID | `frac_at_clip` | latent_energy |
|---:|---|---:|---:|---:|---:|
| 500 | classical | 10 | 72,35 | 0,0000 | 0,0337 |
| 5000 | classical | 10 | **59,16** | 0,0001 | **0,0105** |
| 5000 | qrc | 10–100 | 60,66 | 0,0321 | 0,0124 |
| 5000 | interaction | 1 | 61,24 | **0,0851** | 0,0141 |

Com mais dados o Ridge de 20 features fica muito melhor calibrado — `latent_energy` cai de 0,034
para 0,010, a distância mais próxima da distribuição real de latentes em toda a tabela — e seu
rollout praticamente nunca toca o clip. Os dois braços de bloco largo, ao contrário, começam a
**saturar o clip** em n=5000: 3,2% para o QRC e 8,5% para o interaction, contra 0,01% do clássico.
Longe do limiar degenerado de 50%, mas sistemático e presente só no grupo em que eles perdem.

Este é o quarto caso de dissociação MSE↔FID neste projeto, e o único em que o mecanismo é visível:
em n=5000 o interaction tem o **melhor** MSE de epsilon (−0,071) e o **pior** FID dos três braços
não-aleatórios (+2,08 contra o clássico, p=0,027). Ganhar em predição de epsilon e perder em
geração por saturação de clip é um modo de falha concreto, não ruído.

### QRC contra o controle clássico estruturado

Reportado com **df corrigido**: média por data seed, 3 pares independentes. Os p-valores de 15 pares
(1,8e-8 e 1,2e-6 em Breast) são inflados, porque as 15 células QRC compartilham 3 valores de
baseline — ver a ressalva de potência abaixo.

| dataset | n_train | ΔFID (qrc − interaction) | SD | p (n=3) | por seed |
|---|---:|---:|---:|---:|---|
| fashion | 500 | +0,640 | 2,57 | **0,708** | −1,91 / +3,22 / +0,61 |
| fashion | 5000 | −0,586 | 0,60 | **0,232** | −1,05 / +0,09 / −0,80 |
| breast | 500 | +11,686 | 1,39 | **0,0047** | +10,42 / +11,46 / +13,18 |
| breast | 5000 | +5,872 | 1,82 | **0,0305** | +4,49 / +7,93 / +5,19 |

O efeito em Breast sobrevive à análise correta — mesmo sinal nas três seeds, magnitude 4–13 FID
contra MDE ~3 — mas com p de 0,005 e 0,03, não 1e-8. Em Fashion o nulo é robusto: df inflado
dificulta nulos, então p=0,71 e p=0,23 com 3 pares confirmam o p=0,55 e p=0,52 com 15.

**Correção de uma afirmação anterior desta auditoria.** Em §5c eu escrevi que `ridge_interaction`
bate o QRC "por 2,5× a 3,5×". Esse fator é de **MSE supervisionado de epsilon** e **não se
transfere para geração**. Em FID, no Fashion-MNIST os dois são estatisticamente
indistinguíveis nos dois n_train (p=0,55 e p=0,52, com MDE de ~3 FID). O interaction só vence
claramente no BreastMNIST, e por 6–12 FID, não por um fator multiplicativo.

Esta é a terceira vez que este projeto encontra a dissociação MSE↔FID: no §1 o draw 2 tinha o pior
val-MSE e o único FID não degenerado; aqui o interaction domina o MSE por 2,5× e empata em FID.
Ordenamento supervisionado não é evidência de ordenamento generativo neste pipeline.

### O que sustenta um resultado QRC positivo, e o que não

Sustenta: o controle **casado em dimensão** perde de forma esmagadora e consistente — `random` fica
+16 a +50 FID atrás do interaction e +17 a +22 atrás do clássico, 0/15 vitórias em todas as
células. Os 84 observáveis não são "84 features quaisquer": são muito melhores que 84 features
tanh aleatórias. Isso é um efeito real do reservatório, não de contagem de parâmetros.

Não sustenta: nenhuma vantagem sobre um bloco de interação puramente clássico, de custo
computacional desprezível e sem simulador quântico. O melhor caso para o QRC é **empate** em
Fashion; em Breast ele perde. O `random` também é um aprendiz deliberadamente fraco, então
"QRC ≫ random" é um piso, não uma demonstração de vantagem.

### Ressalva de potência

Os braços `classical` e `interaction` não dependem do unitary draw, logo têm 3 células por grupo.
Nos contrastes `qrc vs interaction` as 15 células QRC são pareadas contra o baseline da mesma seed,
o que dá 15 pares mas só 3 valores independentes de baseline — os graus de liberdade estão
inflados e os p-valores desses contrastes são otimistas. Os contrastes `interaction vs classical`
com 3 pares têm MDE de 5,0 (Fashion) e 19,7 (Breast, n_train=500): subpotentes por construção.

## 6. O que precisa mudar no relatório

1. **§8 e §9 (geração) devem ser retirados.** Não são "ruidosos" nem "conservadores": 11/15 células
   em cada dataset são massas pontuais, e o sinal do efeito inverte quando corrigido. Substituir
   pelos números de §5d.
2. **§4 (`SUPERVISED_NEGATIVE`) deve ser reclassificado para `SUPERVISED_NULL`** no protocolo
   publicado (n_train=500): +0,00101, p=0,63, 6/15. O negativo era o piso de variância.
3. A afirmação "Não houve features constantes" (§3) está incorreta; o limiar era 1e-10 e as colunas
   mortas têm std ~1e-8. Foram 12 a 30 delas, dependendo do draw.
4. §7 atribui condicionamento ~4,8e16 a "bloco muito mal condicionado" genericamente. A causa
   específica é rank-deficiência por observáveis determinísticos com a memória em `|0⟩` — as mesmas
   colunas mortas.
5. §10 lista a assimetria stateful como limitação secundária ("pode ampliar instabilidade"). Ela é a
   causa raiz do resultado de geração, não uma ressalva.
6. **A recomendação "Encerrar a hipótese de complementaridade QRC" pode ficar, com outro
   fundamento.** O fundamento válido é: o QRC não supera `x_t ⊗ te(t)`, que é clássico, gratuito e
   já disponível — empate em Fashion (p=0,55), derrota em Breast. O fundamento publicado ("os
   observáveis pioram MSE e desestabilizam a geração") é falso.
7. **Deve ser acrescentado o resultado positivo que o experimento produziu sem perceber:** os 84
   observáveis batem 84 features tanh aleatórias por +17 a +50 FID, 0/15 vitórias do controle em
   todas as células. O reservatório extrai estrutura; ele só não extrai mais que um produto
   cartesiano clássico.

## 7. Correções recomendadas no código

- `experiments/qrc_kernel_core.py:27` — piso de variância de 1e-10 é insuficiente para colunas
  constantes em float32 (std ~1e-8). Ver `FloorStandardizer` em
  `experiments/qrc_fusion_fair_core.py`, que usa 1e-6 e registra `n_floored`.
- `experiments/phase4.py:sample_qrc` — decidir explicitamente entre stateless (reset por passo,
  compatível com `features()`) e stateful (treino em trajetórias, como em `qrc_stateful_core.py`),
  e afirmar a escolha. O estado atual é a combinação incoerente das duas, em 23 arquivos.
- Toda célula de geração deve **falhar** quando as features padronizadas do rollout saírem da
  distribuição de treino, em vez de reportar o FID resultante. Ver o guard rail `--probe-limit` em
  `qrc_fusion_fair_generation.py`, e `degenerate()` para `std(latente)==0` / `frac_at_clip>0,5`.
- `metrics.inception_features_and_probs` recarrega os pesos da InceptionV3 a cada chamada. Com 144
  células isso domina o custo; cachear o modelo é a otimização de maior retorno.

## 8. Reprodução

Diagnóstico (scratchpad, exploratório): `diag_collapse.py` (contrafactual estado × λ),
`diag_constcols.py` (colunas mortas), `fair_supervised.py`, `fair_fid.py`.

Experimento corrigido (versionado, GPU):

```
python -m experiments.qrc_fusion_fair_supervised --device cuda
python -m experiments.qrc_fusion_fair_supervised --n-train 500 --variance-floor 1e-10 --device cuda
PAR=3 ./run_qrc_fusion_fair.sh
python -m experiments.qrc_fusion_fair_analyze
```

`PAR=3` é o ótimo medido numa RTX 3080 Ti de 12 GiB com `--inception-batch 96`: `PAR=5` estoura a
memória e é mais lento (a GPU já satura em 100%). O rollout de 10.000 amostras × 50 passos leva
~2 s em GPU, contra ~700 s por célula na versão CPU publicada.
