//! Traces processing algorithms
use ndarray::{Array1, ArrayView1};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::Add;

use crate::Sample;

/// Processes traces to calculate mean and variance using a numerically stable online algorithm
/// (Welford's method).
#[derive(Serialize, Deserialize)]
pub struct MeanVarProcessor<T>
where
    T: Sample,
{
    /// Running mean per sample position
    mean: Array1<f64>,
    /// Sum of squares of differences from the current mean
    m2: Array1<f64>,
    /// Number of traces processed
    count: usize,
    _marker: PhantomData<T>,
}

impl<T> MeanVarProcessor<T>
where
    T: Sample + Copy,
{
    /// Creates a new mean and variance processor.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples per trace
    pub fn new(size: usize) -> Self {
        Self {
            mean: Array1::zeros(size),
            m2: Array1::zeros(size),
            count: 0,
            _marker: PhantomData,
        }
    }

    /// Processes an input trace to update internal accumulators using Welford's algorithm.
    ///
    /// # Panics
    /// Panics in debug if the length of the trace is different form the size of [`MeanVarProcessor`].
    pub fn process(&mut self, trace: ArrayView1<T>) {
        debug_assert!(trace.len() == self.size());

        self.count += 1;

        for i in 0..trace.len() {
            let sample = <T as Sample>::Container::from(trace[i]).as_() as f64;

            let delta = sample - self.mean[i];
            self.mean[i] += delta / self.count as f64;

            let delta2 = sample - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    /// Returns the sample mean.
    pub fn mean(&self) -> Array1<f32> {
        self.mean.mapv(|x| x as f32)
    }

    /// Returns the biased sample variance.
    pub fn var(&self) -> Array1<f32> {
        if self.count == 0 {
            return self.m2.mapv(|x| x as f32);
        }

        self.m2.mapv(|x| (x / self.count as f64) as f32)
    }

    /// Returns the trace size handled.
    pub fn size(&self) -> usize {
        self.mean.len()
    }

    /// Returns the number of traces processed.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Determine if two [`MeanVarProcessor`] are compatible for addition.
    ///
    /// If they were created with the same parameters, they are compatible.
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.size() == other.size()
    }
}

impl<T> Add for MeanVarProcessor<T>
where
    T: Sample + Copy,
{
    type Output = Self;

    /// Merge computations of two [`MeanVarProcessor`]. Processors need to be compatible to be merged
    /// together, otherwise it can panic or yield incoherent result (see
    /// [`MeanVarProcessor::is_compatible_with`]).
    ///
    /// # Panics
    /// Panics in debug if the processors are not compatible.
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(self.is_compatible_with(&rhs));

        if self.count == 0 {
            return rhs;
        }
        if rhs.count == 0 {
            return self;
        }

        let count_a = self.count as f64;
        let count_b = rhs.count as f64;
        let count_total = (self.count + rhs.count) as f64;

        let mut mean = Array1::zeros(self.size());
        let mut m2 = Array1::zeros(self.size());
        for i in 0..self.size() {
            let mean_a = self.mean[i];
            let mean_b = rhs.mean[i];
            let m2_a = self.m2[i];
            let m2_b = rhs.m2[i];

            let delta = mean_b - mean_a;

            mean[i] = (count_a * mean_a + count_b * mean_b) / count_total;
            m2[i] = m2_a + m2_b + delta * delta * (count_a * count_b / count_total);
        }

        Self {
            mean,
            m2,
            count: (count_total as usize),
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MeanVarProcessor;
    use ndarray::{Array1, array};

    fn assert_all_close(a: &Array1<f32>, b: &Array1<f32>, rtol: f32, atol: f32) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            let diff = (a[i] - b[i]).abs();
            let tol = atol + rtol * b[i].abs();
            assert!(
                diff <= tol,
                "index {i}: left={} right={} diff={} tol={}",
                a[i],
                b[i],
                diff,
                tol
            );
        }
    }

    #[test]
    fn test_mean_var() {
        let mut processor = MeanVarProcessor::new(4);
        processor.process(array![28038i16, 22066i16, -20614i16, -9763i16].view());
        assert_eq!(
            processor.mean(),
            array![28038f32, 22066f32, -20614f32, -9763f32]
        );
        assert_eq!(processor.var(), array![0f32, 0f32, 0f32, 0f32]);
        processor.process(array![31377, -6950, -15666, 26773].view());
        processor.process(array![24737, -18311, 24742, 17207].view());
        processor.process(array![12974, -29255, -28798, 18988].view());
        assert_all_close(
            &processor.mean(),
            &array![24281.5f32, -8112.5f32, -10084f32, 13301.25f32],
            1e-6,
            1e-7,
        );
        assert_all_close(
            &processor.var(),
            &array![48131136.0, 365777020.0, 426275900.0, 190260430.0],
            1e-6,
            1e-2,
        );
    }

    #[test]
    fn test_mean_var_numerical_stability_constant_f32() {
        // Many large identical values should yield near-zero variance (within tiny FP error)
        let mut processor = MeanVarProcessor::new(1);
        for _ in 0..10_000 {
            processor.process(array![1_000_000_000f32].view());
        }
        let var = processor.var();
        assert!(var[0] >= -1e-6);
        assert!(var[0].abs() <= 1e-3);
    }

    #[test]
    fn test_mean_var_numerical_stability_small_variance_f32() {
        // Values around a large mean with small deltas should keep a stable small variance
        let mut processor = MeanVarProcessor::new(1);
        let base = 1_000_000f32;
        let delta = 3.0f32; // expected variance = delta^2 = 9
        for _ in 0..5_000 {
            processor.process(array![base + delta].view());
            processor.process(array![base - delta].view());
        }
        let var = processor.var();
        assert!(var[0] >= -1e-5);
        assert!((var[0] - 9.0).abs() <= 2e-2);
    }

    #[test]
    fn test_mean_var_numerical_stability_large_integers() {
        // Large-magnitude integers with small spread; container arithmetic should avoid overflow
        let mut processor = MeanVarProcessor::new(1);
        // Two values equally frequent: variance should be (10/2)^2 = 25
        for _ in 0..100_000 {
            processor.process(array![1_000_000i32].view());
            processor.process(array![1_000_010i32].view());
        }
        let var = processor.var();
        assert!(var[0] >= -1e-5);
        assert!((var[0] - 25.0).abs() <= 2e-2);
    }
}
