<!-- Tamil-medium simplified version of en.md -->

> **For Tamil-medium students:** This version uses simple English and Tamil hints to explain AI concepts.

# Dynamic Programming — Policy iteration (மறுயுதல் — மீண்டும் செய்தல்) & Value Iteration

> Dynamic programming is RL with cheating. You already know the transition and reward function (சார்பு — உள்ளீட்டுக்கு வெளியீடு)s. you just iterate the Bellman equation until `V` or `π` stops moving. It is the benchmark every sampling-based method tries to approach.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 minutes

## The Problem

You have an MDP with a known model: you can query `P(s' | s, a)` and `R(s, a, s')` for any state-action pair. An inventory manager knows the demand distribution. A board game has deterministic transitions. A gridworld is four lines of Python. You have a *model*.

Model-free RL (Q-learning, PPO, REINFORCE) was invented for the case where you don't have a model — you can only sample from the environment. But when you do have one, there are faster, better methods: dynamic programming. Bellman designed them in 1957. They still define correctness: when people say "optimal policy for this MDP," they mean the policy DP would return.

You need them in 2026 for three reasons. First, every tabular environment in RL research (GridWorld, FrozenLake, CliffWalking) is solved with DP to produce the gold-standard policy. Second, exact values let you *debug* sampling methods: if Q-learning's estimate for `V*(s_0)` disagrees with the DP answer by 30%, your Q-learning has a bug. Third, modern offline RL and planning methods (MCTS, AlphaZero's search, model-based RL in Phase 9 · 10) all iterate a Bellman backup over a learned or given model.

## The Concept

![Policy iteration (மறுயுதல் — மீண்டும் செய்தல்) and value iteration, side by side](../assets/dp.svg)

**Two algorithm (அல்கோரிதம் — வழிமுறை)s, both fixed-point iteration (மறுயுதல் — மீண்டும் செய்தல்) on Bellman.**

**Policy iteration (மறுயுதல் — மீண்டும் செய்தல்).** Alternates two steps until the policy stops changing.

1. *Evaluation:* given policy `π`, compute `V^π` by repeatedly applying `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` until it converges.
2. *Improvement:* given `V^π`, make `π` greedy w.r.t. `V^π`: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

convergence (ஒருங்கிணைப்பு — ஒரு பதிலில் சேர்ந்துவருதல்) is guaranteed because (a) each improvement step either keeps `π` the same or strictly increases `V^π` for some state, (b) the space of deterministic policies is finite. Usually converges in ~5–20 outer iteration (மறுயுதல் — மீண்டும் செய்தல்)s even for large state spaces.

**Value iteration (மறுயுதல் — மீண்டும் செய்தல்).** Collapses evaluation and improvement into one sweep. Apply the Bellman *optimality* equation:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

Repeat until `max_s |V_{new}(s) - V(s)| < ε`. Extract the policy at the end by taking the greedy action. Strictly faster per iteration (மறுயுதல் — மீண்டும் செய்தல்) — no inner evaluation loop — but usually needs more iterations to converge.

**Generalized policy iteration (மறுயுதல் — மீண்டும் செய்தல்) (GPI).** The unifying framing. Value function (சார்பு — உள்ளீட்டுக்கு வெளியீடு) and policy are locked in a two-way improvement loop. any method that drives both toward mutual consistency (async value iteration, modified policy iteration, Q-learning, actor-critic, PPO) is an instance of GPI.

**Why `γ < 1` matters.** The Bellman operator is a `γ`-contraction in the sup-norm: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. Contraction implies unique fixed point and geometric convergence (ஒருங்கிணைப்பு — ஒரு பதிலில் சேர்ந்துவருதல்). Drop `γ < 1` and you lose the guarantee — you need a finite horizon or an absorbing terminal state.

```figure
value-iteration (மறுயுதல் — மீண்டும் செய்தல்)-gamma
```

## Build It

### Step 1: build the GridWorld MDP model

Use the same 4×4 GridWorld from Lesson 01. We add a stochastic variant: with probability `0.1` the agent slips to a random perpendicular direction.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` returns a list of `(s', r, p)`. This is the entire model.

### Step 2: policy evaluation

Given a policy `π(s) = {action: prob}`, iterate the Bellman equation until `V` stops moving:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Step 3: policy improvement

Replace `π` with the greedy policy w.r.t. `V`. If `π` did not change, return — we are at the optimum.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Step 4: stitch them together

```python
def policy_iteration (மறுயுதல் — மீண்டும் செய்தல்)(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

Typical convergence (ஒருங்கிணைப்பு — ஒரு பதிலில் சேர்ந்துவருதல்) on 4×4: 4–6 outer iteration (மறுயுதல் — மீண்டும் செய்தல்)s. Outputs `V*(0,0) ≈ -6` and a policy that strictly decreases the step count.

### Step 5: value iteration (மறுயுதல் — மீண்டும் செய்தல்) (the one-loop version)

```python
def value_iteration (மறுயுதல் — மீண்டும் செய்தல்)(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

Same fixed point, fewer lines of code.

## Pitfalls

- **Forgetting to handle terminals.** If you apply Bellman to an absorbing state, it still picks up a "best action" that changes nothing. Guard with `if s == terminal: V[s] = 0`.
- **Sup-norm vs L2 convergence (ஒருங்கிணைப்பு — ஒரு பதிலில் சேர்ந்துவருதல்).** Use `max |V_new - V|`, not average. The theoretical guarantee is on the sup-norm.
- **In-place vs synchronous updates.** Updating `V[s]` in-place (Gauss-Seidel) converges faster than a separate `V_new` dict (Jacobi). Production code uses in-place.
- **Policy ties.** If two actions have equal Q-value, `argmax` may break ties differently each iteration (மறுயுதல் — மீண்டும் செய்தல்), causing the "policy stable" check to oscillate. Use a stable tie-break (first action in fixed order).
- **State-space explosion.** DP is `O(|S| · |A|)` per sweep. Works up to ~10⁷ states. Beyond that, you need function (சார்பு — உள்ளீட்டுக்கு வெளியீடு) approximation (Phase 9 · 05 onwards).

## Use It

In 2026, DP is the correctness baseline and the inner loop of planners:

| Use case | Method |
|----------|--------|
| Solve a small tabular MDP exactly | Value iteration (மறுயுதல் — மீண்டும் செய்தல்) (simpler) or policy iteration (fewer outer steps) |
| Verify a Q-learning / PPO how to build | Compare to DP-optimal V* on a toy environment |
| Model-based RL (Phase 9 · 10) | Bellman backup on a learned transition model |
| Planning in AlphaZero / MuZero | Monte Carlo Tree Search = async Bellman backup |
| Offline RL (CQL, IQL) | Conservative Q-iteration — DP with a penalty on OOD actions |

Every time someone says "the optimal value function (சார்பு — உள்ளீட்டுக்கு வெளியீடு)," they mean "the DP fixed point." When you see `V*` or `Q*` in a paper, picture this loop.

## Ship It

Save as `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration (மறுயுதல் — மீண்டும் செய்தல்) or value iteration. Report convergence (ஒருங்கிணைப்பு — ஒரு பதிலில் சேர்ந்துவருதல்) behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration (மறுயுதல் — மீண்டும் செய்தல்) vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. convergence (ஒருங்கிணைப்பு — ஒரு பதிலில் சேர்ந்துவருதல்) sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence (ஒருங்கிணைப்பு — ஒரு பதிலில் சேர்ந்துவருதல்) without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## Exercises

1. **Easy.** Run value iteration (மறுயுதல் — மீண்டும் செய்தல்) on the 4×4 GridWorld with `γ ∈ {0.9, 0.99}`. How many sweeps until `max |ΔV| < 1e-6`? Print `V*` as a 4×4 grid.
2. **Medium.** Compare policy iteration vs value iteration on the *stochastic* GridWorld (slip probability `0.1`). Count: sweeps, wall-clock time, final `V*(0,0)`. Which converges faster in iterations? In wall-clock?
3. **Hard.** Build modified policy iteration: in the evaluation step, run only `k` sweeps instead of to convergence (ஒருங்கிணைப்பு — ஒரு பதிலில் சேர்ந்துவருதல்). Plot `V*(0,0)` error vs `k` for `k ∈ {1, 2, 5, 10, 50}`. What does the curve tell you about the evaluation/improvement choice?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy iteration (மறுயுதல் — மீண்டும் செய்தல்) | "DP algorithm (அல்கோரிதம் — வழிமுறை)" | Alternating evaluation (`V^π`) and improvement (greedy `π` w.r.t. `V^π`) until the policy stops changing. |
| Value iteration | "Faster DP" | Bellman optimality backup applied in one sweep. converges to `V*` geometrically. |
| Bellman operator | "The recursion" | `(T V)(s) = max_a Σ P (r + γ V(s'))`. a `γ`-contraction in sup-norm. |
| Contraction | "Why DP converges" | Any operator `T` with `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` has a unique fixed point. |
| GPI | "Everything is DP" | Generalized Policy Iteration: any method driving `V` and `π` to mutual consistency. |
| Synchronous update | "Jacobi-style" | Use old `V` throughout a sweep. cleanly analyzable but slower. |
| In-place update | "Gauss-Seidel-style" | Use `V` as it's being updated. converges faster in practice. |

## Further Reading

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — the canonical presentation of policy iteration (மறுயுதல் — மீண்டும் செய்தல்) and value iteration.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — rigorous treatment of contraction-mapping arguments.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — modified policy iteration and its convergence (ஒருங்கிணைப்பு — ஒரு பதிலில் சேர்ந்துவருதல்) analysis.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — the original policy iteration paper.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — the bridge from DP to approximate-DP / deep RL used by every subsequent lesson.
