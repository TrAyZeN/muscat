use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use muscat::cpa::{Cpa, CpaProcessor};
use muscat::leakage::{hw, sbox};
use ndarray::{Array2, ArrayView2, Axis};
use ndarray_rand::rand::{rngs::StdRng, SeedableRng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::iter::zip;

pub fn leakage_model(value: usize, guess: usize) -> usize {
    hw(sbox((value ^ guess) as u8) as usize)
}

fn cpa_sequential(leakages: &Array2<f64>, plaintexts: &Array2<u8>) -> Cpa {
    let mut cpa = CpaProcessor::new(leakages.shape()[1], 256, 0, leakage_model);

    for i in 0..leakages.shape()[0] {
        cpa.update(
            leakages.row(i).map(|&x| x as usize).view(),
            plaintexts.row(i).map(|&y| y as usize).view(),
        );
    }

    cpa.finalize()
}

pub fn cpa_map_reduce<T>(
    leakages: ArrayView2<T>,
    plaintexts: ArrayView2<T>,
    guess_range: usize,
    target_byte: usize,
    leakage_func: fn(usize, usize) -> usize,
    chunk_size: usize,
) -> Cpa
where
    T: Into<usize> + Copy + Sync,
{
    assert_eq!(leakages.shape()[0], plaintexts.shape()[0]);
    assert!(chunk_size > 0);

    zip(
        leakages.axis_chunks_iter(Axis(0), chunk_size),
        plaintexts.axis_chunks_iter(Axis(0), chunk_size),
    )
    .par_bridge()
    .map(|(leakages_chunk, plaintexts_chunk)| {
        let mut cpa =
            CpaProcessor::new(leakages.shape()[1], guess_range, target_byte, leakage_func);

        for i in 0..leakages_chunk.shape()[0] {
            cpa.update(leakages_chunk.row(i), plaintexts_chunk.row(i));
        }

        cpa
    })
    .reduce(
        || CpaProcessor::new(leakages.shape()[1], guess_range, target_byte, leakage_func),
        |a, b| a + b,
    )
    .finalize()
}

pub fn cpa_map_reduce_with<T>(
    leakages: ArrayView2<T>,
    plaintexts: ArrayView2<T>,
    guess_range: usize,
    target_byte: usize,
    leakage_func: fn(usize, usize) -> usize,
    chunk_size: usize,
) -> Cpa
where
    T: Into<usize> + Copy + Sync,
{
    assert_eq!(leakages.shape()[0], plaintexts.shape()[0]);
    assert!(chunk_size > 0);

    zip(
        leakages.axis_chunks_iter(Axis(0), chunk_size),
        plaintexts.axis_chunks_iter(Axis(0), chunk_size),
    )
    .par_bridge()
    .map(|(leakages_chunk, plaintexts_chunk)| {
        let mut cpa =
            CpaProcessor::new(leakages.shape()[1], guess_range, target_byte, leakage_func);

        for i in 0..leakages_chunk.shape()[0] {
            cpa.update(leakages_chunk.row(i), plaintexts_chunk.row(i));
        }

        cpa
    })
    .reduce_with(|a, b| a + b)
    .unwrap()
    .finalize()
}

pub fn cpa_fold_reduce<T>(
    leakages: ArrayView2<T>,
    plaintexts: ArrayView2<T>,
    guess_range: usize,
    target_byte: usize,
    leakage_func: fn(usize, usize) -> usize,
    chunk_size: usize,
) -> Cpa
where
    T: Into<usize> + Copy + Sync,
{
    assert_eq!(leakages.shape()[0], plaintexts.shape()[0]);
    assert!(chunk_size > 0);

    zip(
        leakages.axis_chunks_iter(Axis(0), chunk_size),
        plaintexts.axis_chunks_iter(Axis(0), chunk_size),
    )
    .par_bridge()
    .fold(
        || CpaProcessor::new(leakages.shape()[1], guess_range, target_byte, leakage_func),
        |mut cpa, (leakages_chunk, plaintexts_chunk)| {
            for i in 0..leakages_chunk.shape()[0] {
                cpa.update(leakages_chunk.row(i), plaintexts_chunk.row(i));
            }

            cpa
        },
    )
    .reduce(
        || CpaProcessor::new(leakages.shape()[1], guess_range, target_byte, leakage_func),
        |a, b| a + b,
    )
    .finalize()
}

pub fn cpa_fold_reduce_with<T>(
    leakages: ArrayView2<T>,
    plaintexts: ArrayView2<T>,
    guess_range: usize,
    target_byte: usize,
    leakage_func: fn(usize, usize) -> usize,
    chunk_size: usize,
) -> Cpa
where
    T: Into<usize> + Copy + Sync,
{
    assert_eq!(leakages.shape()[0], plaintexts.shape()[0]);
    assert!(chunk_size > 0);

    zip(
        leakages.axis_chunks_iter(Axis(0), chunk_size),
        plaintexts.axis_chunks_iter(Axis(0), chunk_size),
    )
    .par_bridge()
    .fold(
        || CpaProcessor::new(leakages.shape()[1], guess_range, target_byte, leakage_func),
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

fn bench_cpa_helper(c: &mut Criterion) {
    // Seed rng to get the same output each run
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("cpa_helper");

    group.measurement_time(std::time::Duration::from_secs(60));

    for nb_traces in [5000, 10000, 25000].into_iter() {
        let leakages = Array2::random_using((nb_traces, 5000), Uniform::new(-2., 2.), &mut rng);
        let plaintexts = Array2::random_using(
            (nb_traces, 16),
            Uniform::new_inclusive(0u8, 255u8),
            &mut rng,
        );

        group.bench_with_input(
            BenchmarkId::new("cpa_sequential", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| b.iter(|| cpa_sequential(leakages, plaintexts)),
        );
        group.bench_with_input(
            BenchmarkId::new("cpa_map_reduce", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| {
                b.iter(|| {
                    cpa_map_reduce(
                        leakages.map(|&x| x as usize).view(),
                        plaintexts.map(|&x| x as usize).view(),
                        256,
                        0,
                        leakage_model,
                        500,
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cpa_map_reduce_with", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| {
                b.iter(|| {
                    cpa_map_reduce(
                        leakages.map(|&x| x as usize).view(),
                        plaintexts.map(|&x| x as usize).view(),
                        256,
                        0,
                        leakage_model,
                        500,
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cpa_fold_reduce", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| {
                b.iter(|| {
                    cpa_fold_reduce(
                        leakages.map(|&x| x as usize).view(),
                        plaintexts.map(|&x| x as usize).view(),
                        256,
                        0,
                        leakage_model,
                        500,
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cpa_fold_reduce_with", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| {
                b.iter(|| {
                    cpa_fold_reduce_with(
                        leakages.map(|&x| x as usize).view(),
                        plaintexts.map(|&x| x as usize).view(),
                        256,
                        0,
                        leakage_model,
                        500,
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_cpa_helper);
criterion_main!(benches);
