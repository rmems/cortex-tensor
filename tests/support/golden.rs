// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-independent expected tensor data and comparison rules for RM-1826.
//! An adapter supplies its own flattened values, shape and dtype to `compare`.

#[derive(Clone, Copy)]
pub struct Golden<'a> {
    pub shape: &'a [usize],
    pub data: &'a [f32],
    pub atol: f64,
    pub rtol: f64,
}

impl<'a> Golden<'a> {
    pub const fn exact(shape: &'a [usize], data: &'a [f32]) -> Self {
        Self {
            shape,
            data,
            atol: 0.0,
            rtol: 0.0,
        }
    }

    pub const fn approximate(shape: &'a [usize], data: &'a [f32]) -> Self {
        Self {
            shape,
            data,
            atol: 2e-6,
            rtol: 2e-6,
        }
    }

    /// Shape and dtype must match before values are compared. NaN matches only
    /// explicit NaN; infinities must have the same sign. Signed zeros compare
    /// numerically (their serialization contract has separate bitwise tests).
    pub fn compare(&self, shape: &[usize], dtype: &str, data: &[f32]) -> Result<(), String> {
        if dtype != "f32" || shape != self.shape || data.len() != self.data.len() {
            return Err(format!(
                "layout mismatch: {dtype} {shape:?}, {} values; expected f32 {:?}, {} values",
                data.len(),
                self.shape,
                self.data.len()
            ));
        }
        for (i, (&got, &want)) in data.iter().zip(self.data).enumerate() {
            let matches = if want.is_nan() {
                got.is_nan()
            } else if want.is_infinite() {
                got == want
            } else {
                got.is_finite()
                    && (f64::from(got) - f64::from(want)).abs()
                        <= self.atol + self.rtol * f64::from(want).abs()
            };
            if !matches {
                return Err(format!(
                    "element {i}: got {got:?}, expected {want:?} (atol={}, rtol={})",
                    self.atol, self.rtol
                ));
            }
        }
        Ok(())
    }
}
