// SPDX-License-Identifier: MIT OR Apache-2.0

/// The JSON format is scoped to this crate's deterministic reference backend.
/// Each public transformer value carries its own version and rejects drift.
macro_rules! reference_serde {
    ($name:ident, $wire:ident { $($field:ident : $ty:ty),+ $(,)? }) => {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct $wire {
            schema_version: u32,
            $($field: $ty,)+
        }

        impl serde::Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
            where S: serde::Serializer {
                use serde::ser::{Error, SerializeStruct};
                self.validate_wire().map_err(S::Error::custom)?;
                let mut wire = serializer.serialize_struct(stringify!($name), 1 + [$(stringify!($field)),+].len())?;
                wire.serialize_field("schema_version", &1u32)?;
                $(wire.serialize_field(stringify!($field), &self.$field)?;)+
                wire.end()
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
            where D: serde::Deserializer<'de> {
                let wire = <$wire as serde::Deserialize>::deserialize(deserializer)?;
                if wire.schema_version != 1 {
                    return Err(serde::de::Error::custom(format!("unsupported {} schema_version {}", stringify!($name), wire.schema_version)));
                }
                let value = Self { $($field: wire.$field,)+ };
                value.validate_wire().map_err(serde::de::Error::custom)?;
                Ok(value)
            }
        }
    };
}

fn expect_shape(tensor: &crate::tensor::Tensor, expected: &[usize]) -> crate::error::Result<()> {
    if tensor.shape() != expected {
        return Err(crate::error::CortexError::ShapeMismatch {
            expected: expected.to_vec(),
            got: tensor.shape().to_vec(),
        });
    }
    Ok(())
}

pub mod attention;
pub mod block;
pub mod model;

pub use attention::MultiHeadAttention;
pub use block::TransformerBlock;
pub use model::{TransformerConfig, TransformerLM};
