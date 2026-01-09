//! # muscat
//! muscat is a library implementing state-of-the-art side channel attacks (SCAs) algorithms.
//!
//! Traces are processed by processors that exposes a streaming interface. A
//! processor is a structure implementing an `update` method to add a trace to
//! the computation, allowing to incrementally process traces, and a `finalize`
//! method to finalize the computation and return the result.
//!
//! # Supported algorithms
//! - CPA
//! - DPA
//! - SNR
//! - NICV
//! - Welch's T-Test
//! - Elastic alignment
//!
//! # Getting started
//! Here is an example of how to use the CPA processor to recover the first byte of the AES key of the given traces:
//! ```rust
//! use ndarray::array;
//! use std::iter::zip;
//!
//! use muscat::distinguishers::cpa::CpaProcessor;
//! use muscat::leakage_model::aes::sbox;
//!
//! let traces = array![
//!     [77u8, 137, 51, 91],
//!     [72, 61, 91, 83],
//!     [39, 49, 52, 23],
//!     [26, 114, 63, 45],
//!     [30, 8, 97, 91],
//!     [13, 68, 7, 45],
//!     [17, 181, 60, 34],
//!     [43, 88, 76, 78],
//!     [0, 36, 35, 0],
//!     [93, 191, 49, 26],
//! ];
//! let plaintexts = array![
//!     [1usize, 2],
//!     [2, 1],
//!     [1, 2],
//!     [1, 2],
//!     [2, 1],
//!     [2, 1],
//!     [1, 2],
//!     [1, 2],
//!     [2, 1],
//!     [2, 1],
//! ];
//!
//! fn leakage_model(plaintext_byte: usize, guess: usize) -> usize {
//!     sbox((plaintext_byte ^ guess) as u8) as usize
//! }
//!
//! let mut processor = CpaProcessor::new(traces.shape()[1], 256);
//! for (trace, plaintext) in zip(traces.rows(), plaintexts.rows()) {
//!     processor.update(trace.view(), plaintext[0], leakage_model);
//! }
//! let cpa = processor.finalize(leakage_model);
//! let best_guess = cpa.best_guess();
//! println!("Best subkey guess: {best_guess:?}");
//! ```
//!
//! More examples are available in the [examples](https://github.com/Ledger-Donjon/muscat/tree/master/examples) directory.
//!
//! # Performance
//! To get the best performance out of muscat, it is recommended to compile in release mode.

// Re-export public dependencies
pub use ndarray;
pub use serde;

pub mod asymmetric;
pub mod distinguishers;
pub mod error;
pub mod leakage_detection;
pub mod leakage_model;
pub mod preprocessors;
pub mod processors;
#[cfg(feature = "quicklog")]
pub mod quicklog;
pub mod trace;
pub mod util;

use std::ops::{Add, AddAssign, Mul};

use num_traits::{AsPrimitive, Zero};

pub use crate::error::Error;

/// Sample type that can be processed by processors.
/// This trait is used to restrict the traces sample type processed by processors.
///
/// # Dyn compatibility
/// This trait is not [dyn compatible](https://doc.rust-lang.org/nightly/reference/items/traits.html#dyn-compatibility).
///
/// # Limitations
/// We are assuming that the sum of [`Container`] types will not overflow.
pub trait Sample: Sized {
    /// Bigger container type to perform computations (such as sums) of [`Self`] types that could
    /// otherwise overflow.
    type Container: Zero
        + Add
        + AddAssign
        + Mul<Output = Self::Container>
        + AsPrimitive<f32>
        + Clone
        + Copy
        + From<Self>;
}

macro_rules! impl_sample {
    ($($t:ty),* => $c:ty) => {
        $(
            impl Sample for $t {
                type Container = $c;
            }
        )*
    };
}

impl_sample! { u8, u16, u32, u64 => u64 }
impl_sample! { i8, i16, i32, i64 => i64 }
impl_sample! { f32 => f32 }
