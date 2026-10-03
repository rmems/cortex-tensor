// SPDX-License-Identifier: MIT OR Apache-2.0

//! Candle adapter placeholder.
//!
//! v0.3 ships the adapter slot without a `candle-core` dependency so the core
//! crate remains framework-free. A sibling integration crate can implement the
//! cortex-owned adapter contract without changing core traits.

super::define_policy_stub_adapter!(CandleAdapter, CandleTensorView, "candle", "candle");
