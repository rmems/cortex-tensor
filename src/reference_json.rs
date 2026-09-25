// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded loading for untrusted reference-backend JSON fixtures.
//!
//! Direct Serde deserialization remains available for trusted inputs. Callers
//! processing untrusted JSON should choose a byte limit for the complete input
//! before parsing; the limit is applied before any tensor data is allocated.

use crate::error::{CortexError, Result};
use serde::de::{DeserializeOwned, MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Deserializer};
use std::marker::PhantomData;

/// Deserialize a complete reference JSON payload only when it fits `max_bytes`.
/// The limit covers the entire JSON document, including nested model weights.
pub fn from_slice_with_limit<T: DeserializeOwned>(input: &[u8], max_bytes: usize) -> Result<T> {
    if input.len() > max_bytes {
        return Err(CortexError::SerdeInputTooLarge {
            max_bytes,
            actual_bytes: input.len(),
        });
    }
    Ok(serde_json::from_slice(input)?)
}

/// Require an object at each versioned reference-wire boundary. Derived struct
/// deserializers accept positional sequences too, which JSON fixtures do not.
pub(crate) fn deserialize_object<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct ObjectVisitor<T>(PhantomData<T>);

    impl<'de, T: Deserialize<'de>> Visitor<'de> for ObjectVisitor<T> {
        type Value = T;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a versioned reference JSON object")
        }

        fn visit_map<A: MapAccess<'de>>(self, map: A) -> std::result::Result<T, A::Error> {
            T::deserialize(MapAccessDeserializer::new(map))
        }
    }

    deserializer.deserialize_map(ObjectVisitor(PhantomData))
}
