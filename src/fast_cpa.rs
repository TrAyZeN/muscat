use std::{iter::zip, ops::Add};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::{iter::ParallelBridge, prelude::ParallelIterator};

use crate::distinguishers::cpa::Cpa;

pub fn cpa<T, F>(
    leakages: ArrayView2<T>,
    plaintexts: ArrayView2<T>,
    guess_range: usize,
    plaintext_range: usize,
    target_byte: usize,
    leakage_func: F,
    chunk_size: usize,
) -> Cpa
where
    T: Into<usize> + Copy + Sync,
    F: Fn(usize, usize) -> usize + Send + Sync + Copy,
{
    assert_eq!(leakages.shape()[0], plaintexts.shape()[0]);
    assert!(chunk_size > 0);

    zip(
        leakages.axis_chunks_iter(Axis(0), chunk_size),
        plaintexts.axis_chunks_iter(Axis(0), chunk_size),
    )
    .par_bridge()
    .fold(
        || {
            FastCpaProcessor::new(
                leakages.shape()[1],
                guess_range,
                plaintext_range,
                target_byte,
                leakage_func,
            )
        },
        |mut cpa, (leakages_chunk, plaintexts_chunk)| {
            for i in 0..leakages_chunk.shape()[0] {
                cpa.update(leakages_chunk.row(i), plaintexts_chunk.row(i));
            }

            cpa
        },
    )
    .reduce_with(|a, b| a + b)
    .unwrap()
    .finalize()
}

/// It has less accuracy though
///
/// It implements the algorithm from [^1].
///
/// [^1]: <https://hal.science/hal-02172200/document>
pub struct FastCpaProcessor<F>
where
    F: Fn(usize, usize) -> usize + Sync,
{
    num_samples: usize,
    target_byte: usize,
    guess_range: usize,
    plaintext_range: usize,
    num_values: Array2<usize>,
    sum_values: Array2<usize>,
    //mean_power: Array2<f32>,
    leakage_func: F,
}

impl<F> FastCpaProcessor<F>
where
    F: Fn(usize, usize) -> usize + Sync,
{
    pub fn new(
        num_samples: usize,
        guess_range: usize,
        plaintext_range: usize,
        target_byte: usize,
        leakage_func: F,
    ) -> Self {
        Self {
            num_samples,
            target_byte,
            guess_range,
            num_values: Array2::zeros((plaintext_range, num_samples)),
            sum_values: Array2::zeros((plaintext_range, num_samples)),
            //mean_power: Array2::zeros((plaintext_range, num_samples)),
            leakage_func,
            plaintext_range,
        }
    }

    pub fn update<T, P>(&mut self, trace: ArrayView1<T>, plaintext: ArrayView1<P>)
    where
        T: Into<usize> + Copy,
        P: Into<usize> + Copy,
    {
        let plaintext_byte = plaintext[self.target_byte].into();
        for i in 0..self.num_samples {
            self.num_values[[plaintext_byte, i]] += 1;
            self.sum_values[[plaintext_byte, i]] += trace[i].into();
            //self.mean_power[[plaintext_byte, i]] += (trace[i].into() as f32
            // - self.mean_power[[plaintext_byte, i]])
            // / self.num_values[[plaintext_byte, i]] as f32;
        }
    }

    pub fn finalize(&self) -> Cpa {
        let mean_power = self.sum_values.mapv(|x| x as f32) / self.num_values.mapv(|x| x as f32);
        let mut mean_mean_power = Array1::zeros(mean_power.shape()[1]);
        let mut var_mean_power = Array1::zeros(mean_power.shape()[1]);
        for i in 0..self.num_samples {
            mean_mean_power[i] = mean_power.column(i).sum() / mean_power.shape()[1] as f32;
            let sum_squared = mean_power.column(i).mapv(|x| x.powi(2)).sum();
            var_mean_power[i] =
                sum_squared / mean_power.shape()[1] as f32 - mean_mean_power[i].powi(2);
        }

        let mut corr = Array2::zeros((self.guess_range, self.num_samples));
        for guess in 0..self.guess_range {
            let mut modeled_leakages = Array1::zeros(self.plaintext_range);
            for pt in 0..self.plaintext_range {
                modeled_leakages[pt] = (self.leakage_func)(pt, guess);
            }
            let mean_modeled_leakages = modeled_leakages.sum() as f32 / self.plaintext_range as f32;
            let sum_squared_modeled_leakages = modeled_leakages.mapv(|x| x.pow(2)).sum();
            let var_modeled_leakages = sum_squared_modeled_leakages as f32
                / self.plaintext_range as f32
                - mean_modeled_leakages.powi(2);

            for i in 0..self.num_samples {
                corr[[guess, i]] = f32::abs(self.comp_cc(
                    mean_power.column(i),
                    mean_mean_power[i],
                    var_mean_power[i],
                    modeled_leakages.view(),
                    mean_modeled_leakages,
                    var_modeled_leakages,
                ));
            }
        }

        Cpa { corr }
    }

    /// See algorithm 3
    fn comp_cc(
        &self,
        u: ArrayView1<f32>,
        mean_u: f32,
        var_u: f32,
        v: ArrayView1<usize>,
        mean_v: f32,
        var_v: f32,
    ) -> f32 {
        let mut mu_uv = 0f32;
        for guess in 0..self.guess_range {
            mu_uv += (u[guess] * v[guess] as f32 - mu_uv) / (guess + 1) as f32;
        }

        (mu_uv - mean_u * mean_v) / f32::sqrt(var_u * var_v)
    }

    /// Determine if two [`FastCpaProcessor`] are compatible for addition.
    ///
    /// If they were created with the same parameters, they are compatible.
    ///
    /// Note: [`FastCpaProcessor::leakage_func`] cannot be checked for equality, but they must have
    /// the same leakage functions in order to be compatible.
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.num_samples == other.num_samples
            && self.target_byte == other.target_byte
            && self.guess_range == other.guess_range
            && self.plaintext_range == other.plaintext_range
    }
}

impl<F> Add for FastCpaProcessor<F>
where
    F: Fn(usize, usize) -> usize + Sync,
{
    type Output = Self;

    /// Merge computations of two [`FastCpaProcessor`]. Processors need to be compatible to be
    /// merged together, otherwise it can panic or yield incoherent result (see
    /// [`FastCpaProcessor::is_compatible_with`]).
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(self.is_compatible_with(&rhs));

        // for pt in 0..self.plaintext_range {
        //     for i in 0..self.num_samples {
        //         self.mean_power[[pt, i]] = (self.mean_power[[pt, i]]
        //             * self.num_values[[pt, i]] as f32
        //             + rhs.mean_power[[pt, i]] * rhs.num_values[[pt, i]] as f32)
        //             / (self.num_values[[pt, i]] + rhs.num_values[[pt, i]]) as f32;
        //         self.num_values[[pt, i]] += rhs.num_values[[pt, i]];
        //     }
        // }

        Self {
            num_samples: self.num_samples,
            target_byte: self.target_byte,
            guess_range: self.guess_range,
            plaintext_range: self.plaintext_range,
            num_values: self.num_values + rhs.num_values,
            sum_values: self.sum_values + rhs.sum_values,
            //mean_power: self.mean_power,
            leakage_func: self.leakage_func,
        }
    }
}
