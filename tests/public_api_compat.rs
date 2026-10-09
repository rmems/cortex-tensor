// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compile-time / public API regression test for panic-style wrappers retained
//! for pre-1.0 source compatibility. If a signature here fails to compile, the
//! wrapper was removed or changed without a SemVer decision (see RM-1353).

use cortex_tensor::stage::{AnnStage, AnnTopology, ReferenceExecutor, StageId};
use cortex_tensor::tensor::ops::{
    batched_matmul, embedding, layer_norm, matmul, rms_norm, try_batched_matmul, try_causal_mask,
    try_embedding, try_layer_norm, try_matmul, try_rms_norm,
};
use cortex_tensor::transformer::TransformerLM;
use cortex_tensor::{Result, Tensor};

#[test]
fn retained_panic_wrappers_have_stable_signatures() {
    let _: fn(Vec<f32>, &[usize]) -> Tensor = Tensor::from_vec;
    let _: fn(&[usize]) -> Tensor = Tensor::zeros;
    let _: fn(&[usize]) -> Tensor = Tensor::ones;
    let _: fn(&[usize], f32) -> Tensor = Tensor::full;
    let _: fn(&[usize], f32, f32) -> Tensor = Tensor::randn;
    let _: fn(&Tensor, &Tensor) -> Tensor = matmul;
    let _: fn(&Tensor, &Tensor) -> Tensor = batched_matmul;
    let _: fn(&Tensor, &Tensor, &Tensor, f32) -> Tensor = layer_norm;
    let _: fn(&Tensor, &Tensor, f32) -> Tensor = rms_norm;
    let _: fn(&Tensor, &[u32]) -> Tensor = embedding;
}

#[test]
fn fallible_apis_are_public() {
    let _: fn(Vec<f32>, &[usize]) -> Result<Tensor> = Tensor::try_from_vec;
    let _: fn(&Tensor, &Tensor) -> Result<Tensor> = try_matmul;
    let _: fn(&Tensor, &Tensor) -> Result<Tensor> = try_batched_matmul;
    let _: fn(&Tensor, &[u32]) -> Result<Tensor> = try_embedding;
    let _: fn(&Tensor, &Tensor, &Tensor, f32) -> Result<Tensor> = try_layer_norm;
    let _: fn(&Tensor, &Tensor, f32) -> Result<Tensor> = try_rms_norm;
    let _: fn(usize) -> Result<Tensor> = try_causal_mask;
    let _: fn(&Tensor, &[usize]) -> Result<Tensor> = Tensor::try_reshape;
    let _: fn(&Tensor) -> Result<Tensor> = Tensor::try_transpose;
    let _: fn(&Tensor, &Tensor) -> Result<Tensor> = Tensor::try_add;
    let _: fn(&Tensor, &Tensor) -> Result<Tensor> = Tensor::try_sub;
    let _: fn(&Tensor, &Tensor) -> Result<Tensor> = Tensor::try_mul;
    let _: fn(&Tensor, usize) -> Result<Tensor> = Tensor::try_row;
    let _: fn(&Tensor) -> Result<Tensor> = Tensor::try_softmax_last;
}

#[test]
fn stage_execution_apis_have_stable_signatures() {
    // Fallible `StageId` constructor. `new` takes `impl Into<String>`; pin the
    // `String` instantiation, which is the form `from_transformer` relies on.
    let _: fn(String) -> Result<StageId> = StageId::new;
    // Validated topology constructor over an ordered stage list.
    let _: fn(Vec<AnnStage>) -> Result<AnnTopology> = AnnTopology::new;
    // Lifetime-generic model-to-topology builder: the returned executor borrows
    // from the model, tying its lifetime to the input. The fn item does not
    // coerce to a higher-ranked `for<'a> fn(...)` pointer, so pin the exact
    // signature structurally: this local function compiles only while
    // `ReferenceExecutor::from_transformer` keeps the `&'a TransformerLM ->
    // Result<(ReferenceExecutor<'a>, AnnTopology)>` shape.
    fn from_transformer_pin<'a>(
        model: &'a TransformerLM,
    ) -> Result<(ReferenceExecutor<'a>, AnnTopology)> {
        ReferenceExecutor::from_transformer(model)
    }
    // Reference the pin so it is not dead code; invoking it on a throwaway model
    // forces the compiler to resolve the exact borrowed-return signature.
    let model = TransformerLM::try_new(cortex_tensor::transformer::TransformerConfig {
        vocab_size: 4,
        dim: 8,
        num_heads: 2,
        num_layers: 1,
        ff_dim: 16,
        max_seq_len: 4,
    })
    .expect("valid pin model");
    let (_exec, _topo) = from_transformer_pin(&model).expect("pin resolves");
}

#[test]
fn external_adapter_contract_is_public() {
    use cortex_tensor::stage::{
        AdapterCapabilities, AnnCapabilities, BackendLimitations, DeviceClass, ExternalAdapter,
        ExternalAdapterMarker, HiddenStateBuffer, NegotiationDocument,
    };

    fn marker<T: ExternalAdapterMarker<Tensor = cortex_tensor::Tensor>>() {}
    marker::<ReferenceExecutor<'static>>();

    let caps = AnnCapabilities {
        backend_name: "pin",
        supported_tags: Default::default(),
        supported_dtypes: Default::default(),
        stateful: false,
    };
    let report = AdapterCapabilities {
        ann: caps,
        devices: Default::default(),
        batch: false,
        sequence_cache: false,
        hidden_state_io: true,
        limitations: BackendLimitations {
            unsupported_tags: Default::default(),
            unsupported_dtypes: Default::default(),
            unsupported_devices: Default::default(),
            notes: Vec::new(),
        },
    };
    let doc: NegotiationDocument = report.negotiation_document();
    let _: fn(&Tensor) -> cortex_tensor::Result<HiddenStateBuffer> =
        ReferenceExecutor::export_hidden;
    let _: fn(&HiddenStateBuffer) -> cortex_tensor::Result<Tensor> =
        ReferenceExecutor::import_hidden;
    let _: ExternalAdapter = ExternalAdapter;
    assert_eq!(DeviceClass::Cpu.as_str(), "cpu");
    assert_eq!(doc.backend_name, "pin");
}
