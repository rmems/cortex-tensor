# Deterministic reference semantics (RM-1826)

The fixtures in `../reference_goldens.rs` freeze the existing CPU, dense, f32
reference implementation. Inputs, explicit parameters, expected arrays, shapes,
and derivation comments live beside each assertion. The sparse MoE input records
all nonzero indices and specifies zero for every omitted element. There are no
random constructors, seeds, downloaded checkpoints, services, or new dependencies.
These are reference semantics for overlapping operations in future adapters, not
a production engine specification.

## Comparison and reuse

`../support/golden.rs` accepts plain shape, dtype and flattened data, without a
cortex Tensor dependency. Future adapters can reuse that comparator and the
literal fixture data without calling reference kernels to manufacture an oracle.
Every fixture is row-major f32; token IDs and expert IDs are discrete integers.
A dtype, shape, or element-count mismatch fails before numerical comparison.
Unsupported adapter operations must be reported explicitly in their coverage
inventory, never silently skipped or coerced to another dtype.

- Integer-valued arithmetic, masks, simple exact norms and discrete decisions use
  exact equality. Signed zeros compare numerically; bit preservation is covered
  by `reference_serde.rs` and its signed-zero fixture.
- Transcendentals and composed calculations use
  `abs(actual - expected) <= 2e-6 + 2e-6 * abs(expected)`, evaluated in f64.
  This allows f32 rounding at intermediate operations relative to the independent
  real-arithmetic derivations below. It is deliberately tighter than a generic
  model-output tolerance and is specific to these small inputs.
- A finite expectation rejects NaN and both infinities. NaN matches only an
  explicit NaN expectation (payload not specified); infinity must match its sign.
  The explicit nonfinite fixtures exercise the documented finite-value policy.
- Every `repeat` fixture runs twice and compares f32 bits within the process;
  DenseSim also checks repeated selections, weights and hidden values. Debug and
  release runs both compare to the same constants. Cross-platform results must
  meet the numerical tolerances; identical transcendental bits across libm/CPU
  implementations are not assumed. CI currently exercises Linux x86_64.

## Coverage inventory

| Scoped behavior | Golden fixture / existing evidence / explicit boundary |
| --- | --- |
| Checked construction, scalar/zero axes, full/ones/zeros, shape/numel/strides | `checked_layout_and_arithmetic`; constructor overflow and storage mismatch in `checked_errors_are_structured_not_panic_expectations` |
| Copying row, materialized transpose, reshape and reshape in place | `checked_layout_and_arithmetic`; invalid rank/index/shape negatives |
| Same-shape add/subtract/multiply, scalar scale/add, finite reductions | `checked_layout_and_arithmetic`; equal-numel different-shape operands reject broadcasting |
| Matrix and batched products | `checked_products_embedding_and_mask`: rectangular, two distinct batches, right rank-2 broadcasting, rank-2 fallback, zero contraction; existing `tensor::ops` tests cover empty batches and allocation overflow |
| Embedding indexing and row order | `checked_products_embedding_and_mask`: reordered/duplicate IDs, empty IDs, OOV negative |
| Causal additive mask | `checked_products_embedding_and_mask`: diagonal/prefix zero, future negative infinity, empty sequence |
| ReLU, tanh-approximate GELU, SiLU, fast sigmoid surrogate | `activation_and_norm_conventions`; fast sigmoid means x/(1+abs(x)), not logistic |
| Softmax scalar, vector, matrix and empty last axis | `activation_and_norm_conventions`: row axis, finite/tied logits, NaN, positive infinities, fully masked rows; rank >2 is rejected |
| LayerNorm, RMSNorm | `activation_and_norm_conventions`: last axis of rank-2 inputs, population moments, epsilon inside square root, affine weights, constant/zero row, row-local nonfinite behavior; invalid epsilon/width/weight length negatives |
| Causal attention | `causal_attention_scaling_heads_and_sequence_boundaries`: fixed Q/K/V/output matrices and all biases, contiguous heads of width 2, 1/sqrt(head_dim) scaling, prefix-only softmax, empty/singleton/multiple tokens, future independence |
| Dense MLP | `dense_mlp_and_composed_hidden_boundaries`: xW1+b1, tanh GELU, W2+b2, row-tiled biases, signed inputs; no gated MLP implementation |
| Pre-norm block and residual order | Same fixture: norm1, attention, first residual, norm2, MLP, block egress all have independent numeric expectations; epsilon 1e-5 |
| Existing hidden ingress/egress and composed ANN | Same fixture: token+position ordering → block → final LayerNorm → LM head, `[seq,dim]` hidden and `[seq,vocab]` logits, empty sequence; frozen hidden values also injected directly into readout |
| Dense MoE selection and output | `dense_moe_selection_weights_and_hidden_combination`: explicit 2048-wide sparse input, synthetic chunk-sum gates, four experts, tie IDs, global softmax weights, selected mass scaling, constructor k clamping, repeatability, NaN/length errors |
| Checkpoint-backed gate weights | Existing `moe::tests::test_dense_sim_uses_real_gate_weights` and routing tests use synthetic local GGUF bytes. Checkpoint parsing/resampling/dequant are outside this golden baseline; no downloads needed |
| Raw-score ranking under softmax underflow; infinities and signed-zero ties | Existing `moe::routing::tests` covers these independently of DenseSim input combination; keep these tests in CI |
| Structured errors | `checked_errors_are_structured_not_panic_expectations`, plus existing `tensor::ops::tests::try_ops_error_table`, tensor/model unit tests |
| Stage-executor contracts | Already landed in RM-1822; `stage_execution.rs` owns topology, injected hidden states, dtype/input errors and `UnsupportedStage` for MoE/gated MLP. This baseline deliberately uses the older public tensor/model boundaries |
| Serde representation | `reference_serde.rs` owns version, shape, signed zero and bounded input contracts; numerical goldens do not replace that wire format |

## Independent arithmetic

Integer tensor results can be checked by hand: for example `[1,2,3]` dotted with
`[2,0,3]` is 11, and with `[1,-1,2]` is 5. Transpose changes storage order, while
reshape preserves `[1,2,3,4,5,6]`.

LayerNorm uses population variance: `[1,3]` has mean 2 and variance 1;
epsilon 3 gives denominator 2. With gamma `[2,4]` and beta `[0.5,-1]`, the result
is `[-0.5,1]`. `[5,5]` gives beta. RMSNorm on `[3,4]` has mean square 12.5;
epsilon 3.5 gives denominator 4 and gamma `[2,-1]` gives `[1.5,-1]`.

The activation constants use `G(x)=x/2*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`
and `SiLU(x)=x/(1+exp(-x))`, independently evaluated in Python's double-precision
`math`, then rounded to f32 literals. They do not use cortex outputs.

For the identity attention fixture's second token, probabilities are
`softmax([0,1/sqrt(2)])` in head 0 and `softmax([0,4/sqrt(2)])` in head 1.
The first token has only itself available. For the biased projection fixture,
the second query is `[0.5,1.25]`, keys are `[-0.5,1.5]` and `[0.5,0.5]`, and
scores are `[1.625,0.875]/sqrt(2)`. Values are `[3,2]` and `[1,1]`;
output is `[2*context[1]+0.25, context[0]-0.25]`.

For the composed fixture, tokens `[2,0,1]` plus positions yield rows
`[2.5,-1.5]`, `[0.5,3.25]`, `[-2,1.5]`. For any two-element row `[a,b]`,
unit-affine LayerNorm is `[d,-d]/sqrt(d*d+1e-5)` with `d=(a-b)/2`.
Q=K=0 makes attention the arithmetic mean of normalized prefix rows, followed
by `diag(0.5,-0.25)` and bias `[0.1,-0.2]`. Add the input, normalize again,
then apply the explicit MLP matrices and biases in `ffn()`, and add that result.
Apply the final affine norm `[1.25,0.75]`, bias `[0.1,-0.2]`, then the explicit
2-by-3 readout. All intermediate constants were evaluated independently with
Python double arithmetic using these equations, not captured from Rust execution.
For example, the standalone MLP's first preactivation is `[1.1,-1.2]`, which
produces `[0.640716215,-0.406793877]` after the second affine projection.

DenseSim's four chunk sums are `[0,1,1,-1]`. With
`Z=1+2*exp(1)+exp(-1)`, weights are `[1,exp(1),exp(1),exp(-1)]/Z`.
Experts 1 and 2 win, so hidden output is input times `2*exp(1)/Z`, approximately
0.7989726. Weights cover **all** experts and are not renormalized after selection.
There are no expert matrices or learned expert output combinations in DenseSim.

## Ambiguities and deferred cases

- Learned MoE experts, gated MLP, noncausal attention, padding/custom attention
  masks, batched transformer inputs and non-f32 execution are unsupported here.
- Fully negative-infinity softmax is intentionally uniform. In pathological
  attention rows that can give future positions mass; README documents this
  attention/mask-layer gap. Do not bless nonfinite attention output as a new
  causal contract. Finite attention is the baseline.
- Compatibility wrappers intentionally panic on errors. They are not desirable
  error semantics; negatives call checked APIs and assert variants/fields, not
  panic text. Allocation exhaustion is not a recoverable guarantee of all
  `try_*` methods. No OOM fixture is attempted.
- Public transformer fields permit inconsistent modules. Golden parameters obey
  constructor/wire invariants; malformed shapes are covered by checked error tests.
  Arbitrary combinations of mutated fields have not been exhaustively audited,
  so this baseline does not imply universal panic-freedom for such modules.
- Norm affine parameters currently validate element count rather than rank;
  MLP biases also flatten by element count. Only conventional vector norm weights
  and `[1,width]` MLP biases are frozen; permissive malformed layouts are deferred.
- Random initialization, diagnostic parameter counts, extreme-size allocations,
  nonfinite activation/reduction edge cases, and empty/tied argmax behavior are
  not this minimum forward-path baseline. Existing unit tests remain applicable;
  do not infer new adapter requirements from unlisted behavior.

## Running and changing goldens

Run `cargo test --test reference_goldens` and
`cargo test --release --test reference_goldens`. The existing default-feature and
all-feature CI test commands discover the integration suite automatically in debug mode. Contributors
run the release command locally as well. No external adapter or GPU is involved.

Golden changes require a PR explaining the semantic reason, affected boundary,
independent arithmetic, and adapter compatibility implications. Review changes
to inputs, parameters, tolerances and expectations together. Never update expected
values by copying failed actual output, widening tolerances to pass, or providing
an overwrite-on-failure command. Suspected bugs should be documented and fixed
separately before making them a contract. Re-run debug/release and the existing
full suites, formatting and Clippy gates after intentional changes.
