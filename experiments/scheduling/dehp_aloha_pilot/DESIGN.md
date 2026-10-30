# DEHP ALOHA pilot — setup, paper-vs-implementation choices

A fresh DEHP-style **execution-horizon** learning pilot on ALOHA Transfer Cube. Separate from the earlier R/N SAC
scheduler (SCH-01, `fixed_delay10_v2`) and from the denoising-steering experiments: no actor, critic, replay or prior
from those is reused. Only the frozen π0 sampler and the verified RTC port (`../src/sch01/sampler.py`,
`rtc_vendored.py`) are imported unchanged.

## Verified setup (2026-10-30)

| Item | Value | How verified |
|---|---|---|
| Action generator | frozen `pi0_aloha_sim`, `/N/scratch/jindjia/arli/checkpoints/pi0_aloha_sim` (fetched from gs://openpi-assets, MD5-checked; `params/_METADATA` sha256 `ce1369a7…`) | `FrozenPi0`, never passed to an optimizer |
| Prediction horizon / denoising | H = 50, 10 Euler steps (τ: 1 → 0) | `sch01.sampler.RTCSampler(n=10)` |
| Action representation | 14-d absolute joint targets (6 joints + normalised gripper per arm) | openpi ALOHA output transforms; executed actions in `gym_aloha` |
| Simulator | `gym-aloha 0.1.1`, `dm-control 1.0.14`, `mujoco 2.3.7`, `transfer_cube`, 50 Hz | `setup/pip_freeze_arli.txt` |
| Success | step reward 4 (left gripper holds the cube off the table); `gym_aloha` terminates on it | `gym_aloha/env.py` |
| Episode horizon | 400 control steps (terminal deadline in this experiment) | executor `MAX_T` |
| Fixed baseline | h = 25 actions per chunk in both runs (same executor) | `tests/test_dehp_cpu.py` [2] |

## What the paper specifies (DEHP, arXiv 2606.11408v2) vs our choices

Paper (Sec. 3–4, Table 3): factorisation π(a,h|s) = π_act(a|s)·π_len(h|s,a) with π_act frozen; categorical π_len over
{1..H} initialised uniform; actor input = state ++ flattened chunk; critic V(s) state-only, distributional with 201
atoms on [0,1], two-hot projection + cross-entropy; δ_k = R̄_k + γ^{h_k}V(s_{k+1}) − V(s_k);
Â_k = Σ_l (γλ)^{t_{k+l}−t_k} δ_{k+l}; target V_old + Â; clipped PPO with clip 0.05; DEHP LR 1e-4; critic warm-up for
several epochs before actor updates; sparse binary reward. **Not specified:** actor/critic sizes (Table 2's
[1024]×3/×7 MLPs belong to the diffusion policy), entropy bonus, minibatching, gradient clipping, advantage
normalisation, terminal handling, evaluation argmax vs sampling, any latency/RTC variant.

Pilot choices (ours): separate 2×256 ReLU MLPs with zero-initialised output layers (actor exactly uniform over valid
h; critic uniform over atoms, V = 0.5); critic LR 3e-4 (paper 1e-3); γ = 0.999 and λ = 0.99 per control step;
~256 decisions per batch, 4 epochs, one full-batch minibatch; advantage normalisation; no entropy bonus; global-norm
gradient clip 1.0; 20 critic-only warm-up epochs on the first batch; success and the 400-step deadline are terminal
(zero bootstrap); a non-terminal batch cut bootstraps with V(s'); episodes continue across batches.

## State features (identical in both runs, 42-d, `smdp.physical_state`)

agent_pos 14 · robot joint velocities 14 (gym_aloha `get_qvel`, ÷5) · cube position 3 · cube quaternion 4
(`qpos[16:23]`, free joint `red_box_joint`) · cube linear+angular velocity 6 (`qvel[16:22]`, ÷5) · remaining time
(400 − t)/400. No future states/rewards, no task-phase labels. π0 keeps its image + agent_pos inputs.

Actor input (both runs, 795-d): state 42 ++ executable plan 50×14 (time-aligned, zero-padded) ++ plan validity mask 50 ++
timing metadata 3 (executable offset e0/50, startup flag, plan observation age/50).
Critic input: run A = state 42 (paper's state-only critic); run B = the actor's 795-d input — an asynchronous adaptation
(the installed plan and its timing are part of the scheduler state), not the paper's state-only interface.

## Executors (`src/dehp/smdp.py`)

**A — no delay, RTC off, h ∈ {1..50}.** At decision time T the clock is frozen: observe o_T, generate new[0:50] from
o_T, pick h, execute new[0:h], discard the rest, repeat at T + h.

**B — delay 10, RTC on, h ∈ {10..40}.** At handoff T the next chunk is installed; executable positions new[e0:50] with
e0 = T − origin (10 in steady state). The actor reads o_T and that time-aligned plan and picks h. Automatically: launch
at L = T + h − 10 on o_L (actions keep executing), ready at L + 10 = T + h = next handoff, install new[10:50]. RTC
reference = current plan's normalised prediction shifted by L − origin (15 at the first launch of the h = 25 baseline,
then 25; derived from the actual origin, not assumed); committed full-weight prefix 10; exp soft mask until H − shift.
One request in flight, no underrun (both asserted). Startup (learned and fixed alike): synchronous chunk on o_0 without
RTC, origin 0 so e0 = 0; same action space h ∈ {10..40}; first launch at h − 10.

Termination: success or deadline inside an interval ends the decision early; its record holds the actual duration D
(< h), the in-interval discounted reward and terminal status. A selected h outside the valid range raises.

## Evaluation convention

10 held-out initial states (seeds 10,000,000–10,000,009, disjoint from training seeds < 10^7), π0 noise from a fixed
per-episode stream (`default_rng([777, i])`) shared by baseline and learned evaluations; the learned policy uses
deterministic argmax h. Note: the initial actor is exactly uniform, so its argmax is the first valid index — h = 1 in A,
h = 10 in B — i.e. the "initial" evaluation is that fixed horizon. Ten episodes are a pilot estimate; counts, no
significance.
