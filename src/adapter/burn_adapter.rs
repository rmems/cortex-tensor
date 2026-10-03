// SPDX-License-Identifier: MIT OR Apache-2.0

//! Burn adapter placeholder.
//!
//! v0.3 ships the adapter slot without a `burn` crate dependency. Published
//! `burn` 0.21.0 declares `rust-version = "1.92"`, which is a minimum and is
//! compatible with this crate's 1.98.1 pin. The feature stays a policy stub so
//! core remains framework-free.

super::define_policy_stub_adapter!(BurnAdapter, BurnTensorView, "burn", "burn");
