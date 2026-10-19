// SPDX-License-Identifier: MIT OR Apache-2.0

//! Routing math and embedding resampling for checkpoint-backed gates.
//!
//! Top-k expert selection is a pure helper with a documented total order:
//! finite scores descending, ascending expert ID on ties, typed NaN
//! rejection, and explicit ±Inf ranking. See [`top_k_indices`].

use crate::error::{HybridError, Result};
use crate::tensor::finite::{l2_normalize, softmax_vec};
use crate::types::ExtractTokenOptions;
use std::cmp::Ordering;

pub(super) fn checkpoint_gate_scores(
    dims: &[usize],
    weights: &[f32],
    model_path: &str,
    routing_tensor_name: &str,
    num_experts: usize,
    embedding: &[f32],
) -> Result<Vec<f32>> {
    if dims.len() != 2 {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor_name}' must be rank-2 F32 in '{model_path}', got dims={dims:?}"
        )));
    }
    let hidden_size = embedding.len();
    // Orientation is a property of the tensor, not of one element: resolve it
    // once instead of O(E·H) times inside the dot-product loop.
    let layout = resolve_gate_layout(dims, num_experts, hidden_size)?;
    // `element_count` already rejected an overflowing shape at bind time; this
    // is the local guard that lets the loop index without per-element checks.
    let required = dims[0]
        .checked_mul(dims[1])
        .ok_or_else(|| HybridError::ModelLoad {
            path: model_path.to_owned(),
            reason: format!("routing tensor '{routing_tensor_name}' dims {dims:?} overflow usize"),
        })?;
    if required > weights.len() {
        return Err(HybridError::ModelLoad {
            path: model_path.to_owned(),
            reason: format!(
                "routing tensor '{routing_tensor_name}' with dims {dims:?} needs {required} weights, payload holds {}",
                weights.len()
            ),
        });
    }
    let weights = &weights[..required];

    let mut gate_scores = Vec::with_capacity(num_experts);
    for expert_id in 0..num_experts {
        let mut score = 0.0f32;
        for (dim, &value) in embedding.iter().enumerate() {
            // Every index is in range: `expert_id < num_experts <= expert axis`
            // and `dim < hidden_size == hidden axis` were checked by `layout`.
            score += weights[layout.index(expert_id, dim)] * value;
        }
        gate_scores.push(score);
    }
    Ok(gate_scores)
}

pub(super) fn synthetic_gate_scores(num_experts: usize, embedding: &[f32]) -> Vec<f32> {
    let width = embedding.len().max(1);
    let chunk = (width / num_experts.max(1)).max(1);
    let mut gate_scores = Vec::with_capacity(num_experts);
    for expert_id in 0..num_experts {
        let start = (expert_id * chunk) % width;
        let end = (start + chunk).min(width);
        gate_scores.push(embedding[start..end].iter().sum());
    }
    gate_scores
}

pub(super) fn softmax(scores: &[f32]) -> Vec<f32> {
    softmax_vec(scores)
}

/// Deterministic top-k expert selection.
///
/// # Routing policy
///
/// * **NaN:** typed rejection ([`HybridError::NanRoutingScore`]) for the lowest
///   expert ID that is NaN. NaN is never ordered through `partial_cmp`.
/// * **±Inf:** ranked, not sanitized. `+Inf` is strictly above every finite
///   score; `-Inf` is strictly below every finite score. Equal infinities
///   break ties by ascending expert ID.
/// * **Finite scores:** descending numeric order. `+0.0` and `-0.0` are
///   equal; the lowest expert ID wins the tie.
/// * **`top_k == 0` or no experts:** empty selection.
/// * **`top_k > num_experts`:** every expert is returned, unique and in range,
///   already sorted by the total order above.
///
/// This helper is pure: the same `(scores, top_k)` pair always yields the same
/// `Result`. Softmax and normalization kernels are unchanged here.
///
/// Callers must rank these raw scores *before* softmax. Softmax currently maps
/// a non-finite exponential sum to uniform weights, which would otherwise
/// hide NaN / `+Inf` from this helper.
pub(super) fn top_k_indices(scores: &[f32], top_k: usize) -> Result<Vec<usize>> {
    reject_nan_routing_scores(scores)?;

    let take = top_k.min(scores.len());
    if take == 0 {
        return Ok(Vec::new());
    }

    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(cmp_routing_entries);
    Ok(indexed
        .into_iter()
        .take(take)
        .map(|(expert_id, _)| expert_id)
        .collect())
}

/// Reject NaN routing scores without ranking. Used to fail closed before
/// mutating any SNN backend state.
pub(crate) fn reject_nan_routing_scores(scores: &[f32]) -> Result<()> {
    if let Some(expert_id) = scores.iter().position(|score| score.is_nan()) {
        return Err(HybridError::NanRoutingScore { expert_id });
    }
    Ok(())
}

/// Rank raw routing scores, then softmax for the returned expert weights.
pub(crate) fn route_top_k(scores: &[f32], top_k: usize) -> Result<(Vec<f32>, Vec<usize>)> {
    let selected_experts = top_k_indices(scores, top_k)?;
    Ok((softmax(scores), selected_experts))
}

/// Total order for routing: higher score first, then lower expert ID.
///
/// Precondition: neither score is NaN. After that check, IEEE `<`/`>` is a
/// total order over finite values and infinities, and treats signed zeros as
/// equal so the expert-ID tie break is the only remaining discriminator.
fn cmp_routing_entries(left: &(usize, f32), right: &(usize, f32)) -> Ordering {
    debug_assert!(!left.1.is_nan() && !right.1.is_nan());
    if left.1 > right.1 {
        Ordering::Less
    } else if left.1 < right.1 {
        Ordering::Greater
    } else {
        left.0.cmp(&right.0)
    }
}

pub(super) fn resample_embedding(input: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 {
        return Vec::new();
    }
    if input.len() == target_len {
        return input.to_vec();
    }
    if input.is_empty() {
        return vec![0.0; target_len];
    }
    if target_len == 1 {
        return vec![input.iter().sum::<f32>() / input.len() as f32];
    }

    let scale = (input.len() - 1) as f32 / (target_len - 1) as f32;
    let mut out = Vec::with_capacity(target_len);
    for idx in 0..target_len {
        let source = idx as f32 * scale;
        let lo = source.floor() as usize;
        let hi = source.ceil().min((input.len() - 1) as f32) as usize;
        if lo == hi {
            out.push(input[lo]);
            continue;
        }
        let t = source - lo as f32;
        out.push(input[lo] * (1.0 - t) + input[hi] * t);
    }
    out
}

pub(super) fn normalize_l2(values: &mut [f32]) {
    l2_normalize(values);
}

pub(super) fn apply_extract_token_options(input: &[f32], options: ExtractTokenOptions) -> Vec<f32> {
    let mut out = if let Some(target_dim) = options.target_dim {
        resample_embedding(input, target_dim)
    } else {
        input.to_vec()
    };
    if options.l2_normalize {
        normalize_l2(&mut out);
    }
    out
}

/// Which axis of a rank-2 gate tensor is the expert axis.
///
/// `dims` is engram-parser's outermost-first shape, so `dims[1]` is the
/// contiguous on-disk axis (GGUF `ne[0]`), and `weights[r * dims[1] + c]` is
/// row `r`, column `c`. llama.cpp creates `ffn_gate_inp` as
/// `ne = [n_embd, n_expert]`, which arrives here as `[experts, hidden]`: expert
/// `e` is the contiguous row `weights[e * hidden + dim]`. The transposed
/// storage `[hidden, experts]` is also accepted; there expert `e` is the
/// strided column `weights[dim * experts + e]`. A square matrix has no shape
/// cue for which axis is which; expert rows win because that is llama.cpp's
/// layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateLayout {
    /// `[experts, hidden]`: expert rows, `hidden` contiguous.
    ExpertRows { hidden: usize },
    /// `[hidden, experts]`: expert columns, `experts` contiguous.
    ExpertColumns { experts: usize },
}

impl GateLayout {
    #[inline]
    fn index(self, expert_id: usize, dim: usize) -> usize {
        match self {
            GateLayout::ExpertRows { hidden } => expert_id * hidden + dim,
            GateLayout::ExpertColumns { experts } => dim * experts + expert_id,
        }
    }
}

/// Resolve the gate orientation from the outermost-first shape once per call.
fn resolve_gate_layout(
    dims: &[usize],
    num_experts: usize,
    hidden_size: usize,
) -> Result<GateLayout> {
    let d0 = dims[0];
    let d1 = dims[1];

    if d1 == hidden_size && d0 >= num_experts {
        return Ok(GateLayout::ExpertRows { hidden: d1 });
    }
    if d0 == hidden_size && d1 >= num_experts {
        return Ok(GateLayout::ExpertColumns { experts: d1 });
    }

    Err(HybridError::UnsupportedFormat(format!(
        "unsupported routing tensor orientation {dims:?} for hidden_size={hidden_size}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::HybridError;
    use std::collections::HashSet;

    #[test]
    fn resample_embedding_preserves_target_length() {
        let input = vec![0.0, 1.0, 0.0, -1.0];
        let out = resample_embedding(&input, 8);
        assert_eq!(out.len(), 8);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[7] + 1.0).abs() < 1e-6);
    }

    /// Gate value `100 * expert + dim` so every cell is distinguishable, laid
    /// out as `weights[expert * hidden + dim]` — the row-major view of a GGUF
    /// `ffn_gate_inp` with `ne = [hidden, experts]` (outermost-first
    /// `[experts, hidden]`), which is what engram-parser hands this crate.
    fn marker_gate(experts: usize, hidden: usize) -> Vec<f32> {
        let mut weights = vec![0.0f32; experts * hidden];
        for expert in 0..experts {
            for dim in 0..hidden {
                weights[expert * hidden + dim] = (100 * expert + dim) as f32;
            }
        }
        weights
    }

    fn one_hot(len: usize, hot: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; len];
        out[hot] = 1.0;
        out
    }

    #[test]
    fn gate_index_follows_llama_cpp_expert_rows() {
        // GGUF ne = [n_embd=32, n_expert=8]  →  outermost-first dims [8, 32].
        let (experts, hidden) = (8usize, 32usize);
        let dims = [experts, hidden];
        let weights = marker_gate(experts, hidden);
        let layout = resolve_gate_layout(&dims, experts, hidden).unwrap();
        assert_eq!(layout, GateLayout::ExpertRows { hidden });
        for expert in 0..experts {
            for dim in 0..hidden {
                let index = layout.index(expert, dim);
                assert_eq!(
                    weights[index],
                    (100 * expert + dim) as f32,
                    "expert {expert} dim {dim} resolved to index {index}"
                );
            }
        }
        // The issue's repro: input e_7 must see expert 5's cell, not a
        // transposed read of row 7's cell.
        assert_eq!(weights[layout.index(5, 7)], 507.0);
    }

    #[test]
    fn gate_index_accepts_transposed_expert_columns() {
        // Transposed storage: ne = [n_expert=8, n_embd=32] → dims [32, 8], so
        // expert `e` is a strided column `weights[dim * experts + e]`.
        let (experts, hidden) = (8usize, 32usize);
        let dims = [hidden, experts];
        let mut weights = vec![0.0f32; experts * hidden];
        for dim in 0..hidden {
            for expert in 0..experts {
                weights[dim * experts + expert] = (100 * expert + dim) as f32;
            }
        }
        let layout = resolve_gate_layout(&dims, experts, hidden).unwrap();
        assert_eq!(layout, GateLayout::ExpertColumns { experts });
        for expert in 0..experts {
            for dim in 0..hidden {
                assert_eq!(
                    weights[layout.index(expert, dim)],
                    (100 * expert + dim) as f32
                );
            }
        }
    }

    #[test]
    fn gate_scores_pick_marked_cell_for_one_hot_input() {
        let (experts, hidden) = (8usize, 32usize);
        let dims = [experts, hidden];
        let weights = marker_gate(experts, hidden);
        for dim in [0usize, 7, 31] {
            let scores = checkpoint_gate_scores(
                &dims,
                &weights,
                "marker",
                "gate",
                experts,
                &one_hot(hidden, dim),
            )
            .unwrap();
            let expected: Vec<f32> = (0..experts).map(|e| (100 * e + dim) as f32).collect();
            assert_eq!(scores, expected, "input e_{dim}");
        }
    }

    #[test]
    fn gate_scores_select_the_expert_whose_row_matches() {
        // Issue #89 repro: H=32, E=8, expert-5 row all ones, input e_0.
        // llama.cpp semantics: logit_e = Σ_d W[e*H + d]·x[d] → expert 5 wins.
        // A transposed read yields all-zero scores and selects expert 0.
        let (experts, hidden) = (8usize, 32usize);
        let dims = [experts, hidden];
        let mut weights = vec![0.0f32; experts * hidden];
        weights[5 * hidden..6 * hidden].fill(1.0);
        let scores = checkpoint_gate_scores(
            &dims,
            &weights,
            "repro",
            "gate",
            experts,
            &one_hot(hidden, 0),
        )
        .unwrap();
        let (_, selected) = route_top_k(&scores, 1).unwrap();
        assert_eq!(selected, vec![5]);
        assert_eq!(scores[5], 1.0);
        assert!(scores.iter().enumerate().all(|(e, s)| e == 5 || *s == 0.0));
    }

    #[test]
    fn gate_scores_with_more_rows_than_experts_still_use_expert_rows() {
        // Shape cue only requires `d0 >= num_experts`: a 64-row gate routed
        // over 8 experts reads rows 0..8.
        let (rows, experts, hidden) = (64usize, 8usize, 32usize);
        let dims = [rows, hidden];
        let weights = marker_gate(rows, hidden);
        let scores = checkpoint_gate_scores(
            &dims,
            &weights,
            "wide",
            "gate",
            experts,
            &one_hot(hidden, 3),
        )
        .unwrap();
        assert_eq!(scores.len(), experts);
        assert_eq!(scores[7], 703.0);
    }

    #[test]
    fn gate_scores_reject_short_payload_and_unknown_orientation() {
        let dims = [8usize, 32usize];
        let short = vec![0.0f32; 8 * 32 - 1];
        let err =
            checkpoint_gate_scores(&dims, &short, "short", "gate", 8, &one_hot(32, 0)).unwrap_err();
        assert!(matches!(err, HybridError::ModelLoad { .. }), "{err}");

        let err = resolve_gate_layout(&[8, 16], 8, 32).unwrap_err();
        assert!(matches!(err, HybridError::UnsupportedFormat(_)), "{err}");
        let err = resolve_gate_layout(&[4, 32], 8, 32).unwrap_err();
        assert!(matches!(err, HybridError::UnsupportedFormat(_)), "{err}");
    }

    #[test]
    fn softmax_empty_scores_are_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn softmax_all_masked_is_uniform() {
        let y = softmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
        assert!((y[0] - 0.5).abs() < 1e-6);
        assert!((y[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn softmax_two_pos_inf_split() {
        let y = softmax(&[f32::INFINITY, 0.0, f32::INFINITY]);
        assert_eq!(y, vec![0.5, 0.0, 0.5]);
    }

    #[test]
    fn l2_normalize_unit_vector() {
        let mut v = vec![3.0f32, 4.0];
        normalize_l2(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn projector_forward_options_resample_to_2048() {
        let input = vec![0.5; 3072];
        let out = apply_extract_token_options(&input, ExtractTokenOptions::for_projector_forward());
        assert_eq!(out.len(), crate::types::EMBEDDING_DIM);
    }

    /// Existing finite ranking fixture: strictly decreasing scores keep
    /// their previous selection `[0, 1]` for `top_k = 2`.
    #[test]
    fn route_top_k_ranks_raw_scores_before_softmax() {
        let scores = [0.0_f32, 1.0, f32::INFINITY];
        let (weights, selected) = route_top_k(&scores, 1).unwrap();
        assert_eq!(selected, vec![2]);
        // Finite-value policy: one +Inf → one-hot at that expert.
        assert_eq!(weights, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn route_top_k_preserves_raw_order_when_softmax_underflows() {
        // After max-subtraction, exp(-150) and exp(-140) both underflow to 0, so
        // ranking softmax weights would tie experts 1 and 2 and pick ID 1.
        let scores = [100.0_f32, -50.0, -40.0];
        let weights = softmax(&scores);
        assert_eq!(weights[1], 0.0);
        assert_eq!(weights[2], 0.0);
        let (_weights, selected) = route_top_k(&scores, 2).unwrap();
        assert_eq!(selected, vec![0, 2]);
    }

    #[test]
    fn top_k_zero_still_rejects_nan() {
        assert!(matches!(
            top_k_indices(&[f32::NAN], 0).unwrap_err(),
            HybridError::NanRoutingScore { expert_id: 0 }
        ));
    }

    #[test]
    fn finite_distinct_scores_keep_expected_selection() {
        let scores = [0.9_f32, 0.5, 0.3, 0.1];
        assert_eq!(top_k_indices(&scores, 2).unwrap(), vec![0, 1]);
        assert_eq!(top_k_indices(&scores, 1).unwrap(), vec![0]);
        assert_eq!(top_k_indices(&scores, 4).unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn all_equal_finite_scores_select_lowest_ids() {
        let scores = [0.25_f32, 0.25, 0.25, 0.25];
        assert_eq!(top_k_indices(&scores, 2).unwrap(), vec![0, 1]);
        assert_eq!(top_k_indices(&scores, 4).unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn partial_ties_prefer_lowest_expert_id() {
        let scores = [0.4_f32, 0.9, 0.4, 0.1];
        // 0.9 at expert 1, then the two 0.4s at 0 then 2.
        assert_eq!(top_k_indices(&scores, 3).unwrap(), vec![1, 0, 2]);
    }

    #[test]
    fn tied_high_scores_are_stable_across_id_permutations() {
        // Same three-way tie on the high value, assigned to different IDs.
        let permutations: [&[f32]; 3] = [
            &[1.0, 1.0, 0.0, 1.0],
            &[0.0, 1.0, 1.0, 1.0],
            &[1.0, 0.0, 1.0, 1.0],
        ];
        let expected: [&[usize]; 3] = [&[0, 1, 3], &[1, 2, 3], &[0, 2, 3]];
        for (scores, want) in permutations.iter().zip(expected) {
            assert_eq!(top_k_indices(scores, 3).unwrap(), want);
        }
    }

    #[test]
    fn repeated_runs_match_for_ties_and_infinities() {
        let scores = [1.0_f32, f32::INFINITY, 1.0, f32::NEG_INFINITY, 1.0];
        let first = top_k_indices(&scores, 4).unwrap();
        for _ in 0..16 {
            assert_eq!(top_k_indices(&scores, 4).unwrap(), first);
        }
        assert_eq!(first, vec![1, 0, 2, 4]);
    }

    #[test]
    fn nan_is_rejected_without_partial_cmp_fallback() {
        let scores = [0.2_f32, f32::NAN, 0.9];
        let err = top_k_indices(&scores, 2).unwrap_err();
        assert!(matches!(err, HybridError::NanRoutingScore { expert_id: 1 }));
        assert_eq!(err.to_string(), "NaN routing score at expert 1");
    }

    #[test]
    fn first_nan_is_reported_when_several_are_present() {
        let scores = [f32::NAN, 1.0, f32::NAN];
        assert!(matches!(
            top_k_indices(&scores, 1).unwrap_err(),
            HybridError::NanRoutingScore { expert_id: 0 }
        ));
    }

    #[test]
    fn pos_inf_ranks_above_finite_and_neg_inf_below() {
        let scores = [1.0_f32, f32::NEG_INFINITY, f32::INFINITY, 2.0];
        assert_eq!(top_k_indices(&scores, 4).unwrap(), vec![2, 3, 0, 1]);
    }

    #[test]
    fn equal_infinities_break_ties_by_expert_id() {
        let pos = [f32::INFINITY, 0.0, f32::INFINITY];
        assert_eq!(top_k_indices(&pos, 2).unwrap(), vec![0, 2]);
        let neg = [f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
        assert_eq!(top_k_indices(&neg, 3).unwrap(), vec![1, 0, 2]);
    }

    #[test]
    fn signed_zeros_are_equal_and_break_ties_by_id() {
        let scores = [-0.0_f32, 0.0, -0.0, 1.0];
        assert_eq!(top_k_indices(&scores, 3).unwrap(), vec![3, 0, 1]);
        assert_eq!(top_k_indices(&scores, 4).unwrap(), vec![3, 0, 1, 2]);
    }

    #[test]
    fn empty_scores_yield_empty_selection() {
        assert_eq!(top_k_indices(&[], 0).unwrap(), Vec::<usize>::new());
        assert_eq!(top_k_indices(&[], 3).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn top_k_zero_one_n_and_n_plus_one() {
        let scores = [0.1_f32, 0.4, 0.3];
        let n = scores.len();
        assert_eq!(top_k_indices(&scores, 0).unwrap(), Vec::<usize>::new());
        assert_eq!(top_k_indices(&scores, 1).unwrap(), vec![1]);
        assert_eq!(top_k_indices(&scores, n).unwrap(), vec![1, 2, 0]);
        assert_eq!(top_k_indices(&scores, n + 1).unwrap(), vec![1, 2, 0]);
    }

    fn assert_unique_in_range_and_monotonic(scores: &[f32], selected: &[usize]) {
        let mut seen = HashSet::with_capacity(selected.len());
        for &expert_id in selected {
            assert!(
                expert_id < scores.len(),
                "expert {expert_id} out of range for {} scores",
                scores.len()
            );
            assert!(seen.insert(expert_id), "duplicate expert {expert_id}");
        }
        for window in selected.windows(2) {
            let (left, right) = (window[0], window[1]);
            let (left_score, right_score) = (scores[left], scores[right]);
            assert!(
                left_score >= right_score,
                "selected scores must be non-increasing: {left_score} then {right_score}"
            );
            if left_score == right_score {
                assert!(
                    left < right,
                    "equal scores must break ties by ascending expert id, got {left} then {right}"
                );
            }
        }
    }

    #[test]
    fn property_unique_in_range_and_monotonic_selected_scores() {
        let fixtures: &[&[f32]] = &[
            &[0.1, 0.2, 0.3, 0.4],
            &[1.0, 1.0, 1.0],
            &[5.0, 1.0, 5.0, 1.0, 0.0],
            &[f32::INFINITY, 0.0, f32::NEG_INFINITY, f32::INFINITY],
            &[-0.0, 0.0, 0.0, -0.0],
            &[],
        ];
        for scores in fixtures {
            for top_k in [0, 1, scores.len(), scores.len().saturating_add(1)] {
                let selected = top_k_indices(scores, top_k).unwrap();
                assert_eq!(selected.len(), top_k.min(scores.len()));
                assert_unique_in_range_and_monotonic(scores, &selected);
            }
        }

        let mut rng = SplitMix64(0xC0FFEE_u64);
        for trial in 0..256 {
            let n = 1 + (rng.next_u64() as usize % 16);
            let mut scores = Vec::with_capacity(n);
            for _ in 0..n {
                scores.push(rng.next_routing_score());
            }
            if scores.iter().any(|score| score.is_nan()) {
                assert!(matches!(
                    top_k_indices(&scores, n.min(3)).unwrap_err(),
                    HybridError::NanRoutingScore { .. }
                ));
                continue;
            }
            let top_k = rng.next_u64() as usize % (n + 2);
            let selected = top_k_indices(&scores, top_k).unwrap();
            assert_eq!(selected.len(), top_k.min(n));
            assert_unique_in_range_and_monotonic(&scores, &selected);
            let again = top_k_indices(&scores, top_k).unwrap();
            assert_eq!(selected, again, "trial {trial} was not deterministic");
        }
    }

    /// SplitMix64 so property trials are reproducible without extra crates.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn next_routing_score(&mut self) -> f32 {
            match self.next_u64() % 32 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => 0.0,
                4 => -0.0,
                _ => {
                    let unit = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
                    unit * 20.0 - 10.0
                }
            }
        }
    }
}
