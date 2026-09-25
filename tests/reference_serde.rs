use cortex_tensor::Tensor;
use cortex_tensor::transformer::{
    MultiHeadAttention, TransformerBlock, TransformerConfig, TransformerLM,
};
use serde_json::json;

fn config() -> TransformerConfig {
    TransformerConfig {
        vocab_size: 5,
        dim: 4,
        num_heads: 2,
        num_layers: 1,
        ff_dim: 6,
        max_seq_len: 3,
    }
}

#[test]
fn finite_tensor_golden_json_round_trips() {
    let tensor = Tensor::from_vec(vec![0.0, -0.0, 2.5], &[3]);
    let wire = serde_json::to_string(&tensor).unwrap();
    assert_eq!(
        wire,
        include_str!("fixtures/reference_tensor_signed_zero.json").trim()
    );
    let restored: Tensor = serde_json::from_str(&wire).unwrap();
    assert_eq!(restored.shape(), &[3]);
    assert_eq!(restored.data()[1].to_bits(), (-0.0f32).to_bits());

    for (data, shape, golden) in [
        (
            vec![7.0],
            vec![],
            include_str!("fixtures/reference_tensor_scalar.json").trim(),
        ),
        (
            vec![],
            vec![2, 0],
            include_str!("fixtures/reference_tensor_zero_axis.json").trim(),
        ),
    ] {
        let tensor = Tensor::from_vec(data, &shape);
        assert_eq!(serde_json::to_string(&tensor).unwrap(), golden);
        let restored: Tensor = serde_json::from_str(golden).unwrap();
        assert_eq!(restored.shape(), shape);
    }
}

#[test]
fn tensor_rejects_hostile_json() {
    let cases = [
        json!({"schema_version":1,"data":[1.0,2.0],"shape":[3]}),
        json!({"schema_version":1,"data":[],"shape":[0,18446744073709551615u64,2]}),
        json!({"schema_version":1,"data":[1.0],"shape":[1],"strides":[1]}),
        json!({"schema_version":2,"data":[1.0],"shape":[1]}),
        json!({"data":[1.0],"shape":[1]}),
        json!({"schema_version":1,"data":[null],"shape":[1]}),
    ];
    for value in cases {
        let wire = value.to_string();
        let result: cortex_tensor::Result<Tensor> = serde_json::from_str(&wire).map_err(Into::into);
        assert!(result.is_err(), "accepted {wire}");
    }
}

#[test]
fn nonfinite_tensors_refuse_json_serialization() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let tensor = Tensor::from_vec(vec![value], &[1]);
        assert!(serde_json::to_string(&tensor).is_err());
    }
}

#[test]
fn transformer_payloads_are_versioned_and_shape_checked() {
    let cfg = config();
    let config_wire = serde_json::to_value(&cfg).unwrap();
    assert_eq!(config_wire["schema_version"], 1);
    let mut bad_config = config_wire.clone();
    bad_config["schema_version"] = json!(9);
    assert!(serde_json::from_value::<TransformerConfig>(bad_config).is_err());

    let model = TransformerLM::try_new(cfg).unwrap();
    let value = serde_json::to_value(&model).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["blocks"][0]["schema_version"], 1);
    assert_eq!(value["blocks"][0]["attn"]["schema_version"], 1);
    let _: TransformerLM = serde_json::from_value(value.clone()).unwrap();

    let mut bad = value.clone();
    bad["config"]["dim"] = json!(99);
    assert!(serde_json::from_value::<TransformerLM>(bad).is_err());
    let mut bad = value.clone();
    bad["blocks"][0]["attn"]["wq"]["shape"] = json!([2, 8]);
    assert!(serde_json::from_value::<TransformerLM>(bad).is_err());
    let mut bad = value.clone();
    bad["schema_version"] = json!(2);
    assert!(serde_json::from_value::<TransformerLM>(bad).is_err());
    let mut bad = value;
    bad["extra"] = json!(true);
    assert!(serde_json::from_value::<TransformerLM>(bad).is_err());
}

#[test]
fn standalone_transformer_parts_reject_inconsistent_weights() {
    let mut attn = serde_json::to_value(MultiHeadAttention::new(4, 2)).unwrap();
    attn["wb_q"]["shape"] = json!([4]);
    assert!(serde_json::from_value::<MultiHeadAttention>(attn).is_err());

    let mut block = serde_json::to_value(TransformerBlock::new(4, 2, 6)).unwrap();
    block["ln1_w"]["shape"] = json!([1, 4]);
    assert!(serde_json::from_value::<TransformerBlock>(block).is_err());
}
