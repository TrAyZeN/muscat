//! Leakage detection methods
use crate::{processors::MeanVar, Error};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::iter::{ParallelBridge, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{fs::File, iter::zip, ops::Add, path::Path};

/// Compute the SNR of the given traces using [`SnrProcessor`].
///
/// `get_class` is a function returning the class of the given trace by index.
///
/// # Examples
/// ```
/// use muscat::leakage_detection::snr;
/// use ndarray::array;
///
/// let traces = array![
///     [77, 137, 51, 91],
///     [72, 61, 91, 83],
///     [39, 49, 52, 23],
///     [26, 114, 63, 45],
///     [30, 8, 97, 91],
///     [13, 68, 7, 45],
///     [17, 181, 60, 34],
///     [43, 88, 76, 78],
///     [0, 36, 35, 0],
///     [93, 191, 49, 26],
/// ];
/// let plaintexts = array![
///     [1usize, 2],
///     [2, 1],
///     [1, 2],
///     [1, 2],
///     [2, 1],
///     [2, 1],
///     [1, 2],
///     [1, 2],
///     [2, 1],
///     [2, 1],
/// ];
/// let snr = snr(traces.view(), 256, |i| plaintexts.row(i)[0].into(), 2);
/// ```
///
/// # Panics
/// - Panic if `batch_size` is 0.
pub fn snr<T, F>(
    traces: ArrayView2<T>,
    classes: usize,
    get_class: F,
    batch_size: usize,
) -> Array1<f64>
where
    T: Into<i64> + Copy + Sync,
    F: Fn(usize) -> usize + Sync,
{
    assert!(batch_size > 0);

    // From benchmarks fold + reduce_with is faster than map + reduce/reduce_with and fold + reduce
    traces
        .axis_chunks_iter(Axis(0), batch_size)
        .enumerate()
        .par_bridge()
        .fold(
            || SnrProcessor::new(traces.shape()[1], classes),
            |mut snr, (batch_idx, trace_batch)| {
                for i in 0..trace_batch.shape()[0] {
                    snr.process(trace_batch.row(i), get_class(batch_idx * batch_size + i));
                }
                snr
            },
        )
        .reduce_with(|a, b| a + b)
        .unwrap()
        .snr()
}

/// A Processor that computes the Signal-to-Noise Ratio of the given traces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnrProcessor {
    mean_var: MeanVar,
    /// Sum of traces per class
    classes_sum: Array2<i64>,
    /// Counts the number of traces per class
    classes_count: Array1<usize>,
}

impl SnrProcessor {
    /// Create a new [`SnrProcessor`].
    ///
    /// # Arguments
    ///
    /// - `size` - Size of the input traces
    /// - `num_classes` - Number of classes
    pub fn new(size: usize, num_classes: usize) -> Self {
        Self {
            mean_var: MeanVar::new(size),
            classes_sum: Array2::zeros((num_classes, size)),
            classes_count: Array1::zeros(num_classes),
        }
    }

    /// Process an input trace to update internal accumulators.
    ///
    /// # Panics
    /// - Panics in debug if the length of the trace is different from the size of [`SnrProcessor`].
    pub fn process<T: Into<i64> + Copy>(&mut self, trace: ArrayView1<T>, class: usize) {
        debug_assert!(trace.len() == self.size());
        debug_assert!(class < self.num_classes());

        self.mean_var.process(trace);

        for i in 0..self.size() {
            self.classes_sum[[class, i]] += trace[i].into();
        }

        self.classes_count[class] += 1;
    }

    /// Finalize the processor computation and return the Signal-to-Noise Ratio.
    pub fn snr(&self) -> Array1<f64> {
        // SNR = V[E[L|X]] / E[V[L|X]]

        let size = self.size();

        let mut acc: Array1<f64> = Array1::zeros(size);
        for class in 0..self.num_classes() {
            if self.classes_count[class] == 0 {
                continue;
            }

            let class_sum = self.classes_sum.slice(s![class, ..]);
            for i in 0..size {
                acc[i] += (class_sum[i] as f64).powi(2) / (self.classes_count[class] as f64);
            }
        }

        let var = self.mean_var.var();
        let mean = self.mean_var.mean();
        // V[E[L|X]]
        let velx = (acc / self.mean_var.count() as f64) - mean.mapv(|x| x.powi(2));
        1f64 / (var / velx - 1f64)
    }

    /// Return the trace size handled
    pub fn size(&self) -> usize {
        self.classes_sum.shape()[1]
    }

    /// Return the number of classes handled.
    pub fn num_classes(&self) -> usize {
        self.classes_count.len()
    }

    /// Save the [`SnrProcessor`] to a file.
    ///
    /// # Warning
    /// The file format is not stable as muscat is active development. Thus, the format might
    /// change between versions.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let file = File::create(path)?;
        serde_json::to_writer(file, self)?;

        Ok(())
    }

    /// Load a [`SnrProcessor`] from a file.
    ///
    /// # Warning
    /// The file format is not stable as muscat is active development. Thus, the format might
    /// change between versions.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path)?;
        let p = serde_json::from_reader(file)?;

        Ok(p)
    }

    /// Determine if two [`SnrProcessor`] are compatible for addition.
    ///
    /// If they were created with the same parameters, they are compatible.
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.size() == other.size() && self.num_classes() == other.num_classes()
    }
}

impl Add for SnrProcessor {
    type Output = Self;

    /// Merge computations of two [`SnrProcessor`]. Processors need to be compatible to be merged
    /// together, otherwise it can panic or yield incoherent result (see
    /// [`SnrProcessor::is_compatible_with`]).
    ///
    /// # Panics
    /// - Panics in debug if the processors are not compatible.
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(self.is_compatible_with(&rhs));

        Self {
            mean_var: self.mean_var + rhs.mean_var,
            classes_sum: self.classes_sum + rhs.classes_sum,
            classes_count: self.classes_count + rhs.classes_count,
        }
    }
}

/// Compute the Welch's T-test of the given traces using [`TTestProcessor`].
///
/// # Examples
/// ```
/// use muscat::leakage_detection::ttest;
/// use ndarray::array;
///
/// let traces = array![
///     [77, 137, 51, 91],
///     [72, 61, 91, 83],
///     [39, 49, 52, 23],
///     [26, 114, 63, 45],
///     [30, 8, 97, 91],
///     [13, 68, 7, 45],
///     [17, 181, 60, 34],
///     [43, 88, 76, 78],
///     [0, 36, 35, 0],
///     [93, 191, 49, 26],
/// ];
/// let trace_classes =
///     array![true, false, false, true, false, false, true, false, false, true];
/// let ttest = ttest(traces.view(), trace_classes.view(), 2);
/// ```
///
/// # Panics
/// - Panic if `traces.shape()[0] != trace_classes.shape()[0]`
/// - Panic if `batch_size` is 0.
pub fn ttest<T>(
    traces: ArrayView2<T>,
    trace_classes: ArrayView1<bool>,
    batch_size: usize,
) -> Array1<f64>
where
    T: Into<i64> + Copy + Sync,
{
    assert_eq!(traces.shape()[0], trace_classes.shape()[0]);
    assert!(batch_size > 0);

    zip(
        traces.axis_chunks_iter(Axis(0), batch_size),
        trace_classes.axis_chunks_iter(Axis(0), batch_size),
    )
    .par_bridge()
    .fold(
        || TTestProcessor::new(traces.shape()[1]),
        |mut ttest, (trace_batch, trace_classes_batch)| {
            for i in 0..trace_batch.shape()[0] {
                ttest.process(trace_batch.row(i), trace_classes_batch[i]);
            }
            ttest
        },
    )
    .reduce_with(|a, b| a + b)
    .unwrap()
    .ttest()
}

/// A Processor that computes the Welch's T-Test of the given traces.
#[derive(Debug, Serialize, Deserialize)]
pub struct TTestProcessor {
    mean_var_1: MeanVar,
    mean_var_2: MeanVar,
}

impl TTestProcessor {
    /// Create a new [`TTestProcessor`].
    ///
    /// # Arguments
    /// * `size` - Number of samples per trace
    pub fn new(size: usize) -> Self {
        Self {
            mean_var_1: MeanVar::new(size),
            mean_var_2: MeanVar::new(size),
        }
    }

    /// Process an input trace to update internal accumulators.
    ///
    /// # Arguments
    /// * `trace` - Input trace.
    /// * `class` - Indicates to which of the two partitions the given trace belongs.
    ///
    /// # Panics
    /// Panics in debug if `trace.len() != self.size()`.
    pub fn process<T: Into<i64> + Copy>(&mut self, trace: ArrayView1<T>, class: bool) {
        debug_assert!(trace.len() == self.size());

        if class {
            self.mean_var_2.process(trace);
        } else {
            self.mean_var_1.process(trace);
        }
    }

    /// Calculate and return Welch's T-Test result.
    pub fn ttest(&self) -> Array1<f64> {
        // E(X1) - E(X2)
        let q = self.mean_var_1.mean() - self.mean_var_2.mean();

        // √(σ1²/N1 + σ2²/N2)
        let d = ((self.mean_var_1.var() / self.mean_var_1.count() as f64)
            + (self.mean_var_2.var() / self.mean_var_2.count() as f64))
            .mapv(f64::sqrt);
        q / d
    }

    /// Return the trace size handled.
    pub fn size(&self) -> usize {
        self.mean_var_1.size()
    }

    /// Save the [`TTestProcessor`] to a file.
    ///
    /// # Warning
    /// The file format is not stable as muscat is active development. Thus, the format might
    /// change between versions.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let file = File::create(path)?;
        serde_json::to_writer(file, self)?;

        Ok(())
    }

    /// Load a [`TTestProcessor`] from a file.
    ///
    /// # Warning
    /// The file format is not stable as muscat is active development. Thus, the format might
    /// change between versions.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path)?;
        let p = serde_json::from_reader(file)?;

        Ok(p)
    }

    /// Determine if two [`TTestProcessor`] are compatible for addition.
    ///
    /// If they were created with the same parameters, they are compatible.
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.size() == other.size()
    }
}

impl Add for TTestProcessor {
    type Output = Self;

    /// Merge computations of two [`TTestProcessor`]. Processors need to be compatible to be merged
    /// together, otherwise it can panic or yield incoherent result (see
    /// [`TTestProcessor::is_compatible_with`]).
    ///
    /// # Panics
    /// Panics in debug if the processors are not compatible.
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(self.is_compatible_with(&rhs));

        Self {
            mean_var_1: self.mean_var_1 + rhs.mean_var_1,
            mean_var_2: self.mean_var_2 + rhs.mean_var_2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{snr, ttest, SnrProcessor, TTestProcessor};
    use ndarray::array;

    #[test]
    fn test_snr_helper() {
        let traces = array![
            [77, 137, 51, 91],
            [72, 61, 91, 83],
            [39, 49, 52, 23],
            [26, 114, 63, 45],
            [30, 8, 97, 91],
            [13, 68, 7, 45],
            [17, 181, 60, 34],
            [43, 88, 76, 78],
            [0, 36, 35, 0],
            [93, 191, 49, 26],
        ];
        let classes = [1, 3, 1, 2, 3, 2, 2, 1, 3, 1];

        let mut processor = SnrProcessor::new(traces.shape()[1], 256);
        for (trace, class) in std::iter::zip(traces.rows(), classes.iter()) {
            processor.process(trace, *class);
        }
        assert_eq!(processor.snr(), snr(traces.view(), 256, |i| classes[i], 2));
    }

    #[test]
    fn test_ttest() {
        let traces = [
            array![77, 137, 51, 91],
            array![72, 61, 91, 83],
            array![39, 49, 52, 23],
            array![26, 114, 63, 45],
            array![30, 8, 97, 91],
            array![13, 68, 7, 45],
            array![17, 181, 60, 34],
            array![43, 88, 76, 78],
            array![0, 36, 35, 0],
            array![93, 191, 49, 26],
        ];

        let mut processor = TTestProcessor::new(4);
        for (i, trace) in traces.iter().enumerate() {
            processor.process(trace.view(), i % 3 == 0);
        }

        assert_eq!(
            processor.ttest(),
            array![
                -1.0910344547297484,
                -5.524921845887032,
                0.29385284736362266,
                0.23308466737856662
            ]
        );
    }

    #[test]
    fn test_ttest_helper() {
        let traces = array![
            [77, 137, 51, 91],
            [72, 61, 91, 83],
            [39, 49, 52, 23],
            [26, 114, 63, 45],
            [30, 8, 97, 91],
            [13, 68, 7, 45],
            [17, 181, 60, 34],
            [43, 88, 76, 78],
            [0, 36, 35, 0],
            [93, 191, 49, 26],
        ];
        let trace_classes =
            array![true, false, false, true, false, false, true, false, false, true];

        let mut processor = TTestProcessor::new(4);
        for (i, trace) in traces.rows().into_iter().enumerate() {
            processor.process(trace, trace_classes[i]);
        }

        assert_eq!(
            processor.ttest(),
            ttest(traces.view(), trace_classes.view(), 2)
        );
    }
}
