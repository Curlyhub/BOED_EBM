# Mixture Changed-Measure EBM

`ours_ebm_moe_*` variants add a router over EBM-induced changed measures on the finite hypothesis bank.

For bank logits `log_w_i` and expert energy `E_k(s_t, theta_i)`, each expert forms:

```text
log q_{k,i} = log_w_i - E_k(s_t, theta_i)
              - logsumexp_j(log_w_j - E_k(s_t, theta_j))
```

The router emits one global expert distribution per batch item:

```text
log_alpha = log_softmax(router(s_t))
log q_i = logsumexp_k(log_alpha_k + log q_{k,i})
```

This default `measure_mixture` mode is a true mixture of changed measures. The optional `energy_blend` mode is only an ablation.

Supported initial experts are `identity`, `standard`, and `cross`, reusing the existing EBM heads. The router weights are diagnostics; the policy still receives the existing EBM belief feature types (`legacy`, `moments`, `modal`).

This does not compare likelihood models and does not change `K(y | theta, d)`. The exact posterior/filter update and homeostatic admissibility filtering remain unchanged.
