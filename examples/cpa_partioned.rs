use anyhow::Result;
use indicatif::ProgressIterator;
use muscat::distinguishers::cpa::CpaProcessor;
use muscat::leakage_model::{aes::sbox, hw};
use muscat::util::{progress_bar, read_array2_from_npy_file, save_array};
use rayon::prelude::{ParallelBridge, ParallelIterator};

// traces format
type FormatTraces = i16;
type FormatMetadata = i32;

// leakage model
pub fn leakage_model(value: usize, guess: usize) -> usize {
    hw(sbox((value ^ guess) as u8) as usize)
}

// multi-threading cpa
fn cpa() -> Result<()> {
    let size = 5000; // Number of samples
    let guess_range = 256; // 2**(key length)
    let target_byte = 1;
    let folder = String::from("../../data"); // Directory of traces and metadata
    let nfiles = 5; // Number of files in the directory. TBD: Automating this value

    /* Parallel operation using multi-threading on batches */
    let cpa = (0..nfiles)
        .progress_with(progress_bar(nfiles))
        .map(|n| {
            let dir_l = format!("{folder}/l{n}.npy");
            let dir_p = format!("{folder}/p{n}.npy");
            let traces = read_array2_from_npy_file::<FormatTraces>(&dir_l).unwrap();
            let plaintext = read_array2_from_npy_file::<FormatMetadata>(&dir_p).unwrap();
            (traces, plaintext)
        })
        .par_bridge()
        .map(|batch| {
            let mut c = CpaProcessor::new(size, guess_range, target_byte);
            for i in 0..batch.0.shape()[0] {
                c.update(
                    batch.0.row(i).map(|x| *x as usize).view(),
                    batch.1.row(i).map(|y| *y as usize).view(),
                    leakage_model,
                );
            }
            c
        })
        .reduce(
            || CpaProcessor::new(size, guess_range, target_byte),
            |a, b| a + b,
        );

    let cpa_result = cpa.finalize(leakage_model);
    println!("Guessed key = {}", cpa_result.best_guess());

    // save corr key curves in npy
    save_array("../results/corr.npy", &cpa_result.corr())?;

    Ok(())
}

fn main() -> Result<()> {
    cpa()?;

    Ok(())
}
