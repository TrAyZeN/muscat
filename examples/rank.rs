use anyhow::Result;
use indicatif::ProgressIterator;
use muscat::distinguishers::cpa::CpaProcessor;
use muscat::leakage_model::{aes::sbox, hw};
use muscat::util::progress_bar;
use ndarray::{Array2, s};
use ndarray_npy::{read_npy, write_npy};
use rayon::prelude::{ParallelBridge, ParallelIterator};

// traces format
type FormatTraces = i16;
type FormatMetadata = i32;

// leakage model
pub fn leakage_model(value: usize, guess: usize) -> usize {
    hw(sbox((value ^ guess) as u8) as usize)
}

fn rank() -> Result<()> {
    let size = 5000; // Number of samples
    let guess_range = 256; // 2**(key length)
    let target_byte = 1;
    let folder = String::from("../../data");
    let nfiles = 5;
    let batch_size = 3000;
    let mut rank = CpaProcessor::new(size, guess_range);
    for file in (0..nfiles).progress_with(progress_bar(nfiles)) {
        let dir_l = format!("{folder}/l{file}.npy");
        let dir_p = format!("{folder}/p{file}.npy");
        let traces: Array2<FormatTraces> = read_npy(&dir_l)?;
        let plaintext: Array2<FormatMetadata> = read_npy(&dir_p)?;
        for sample in (0..traces.shape()[0]).step_by(batch_size) {
            let l_sample: ndarray::ArrayBase<
                ndarray::ViewRepr<&FormatTraces>,
                ndarray::Dim<[usize; 2]>,
            > = traces.slice(s![sample..sample + batch_size, ..]);
            let p_sample = plaintext.slice(s![sample..sample + batch_size, ..]);
            let x = (0..batch_size)
                .par_bridge()
                .fold(
                    || CpaProcessor::new(size, guess_range),
                    |mut r, n| {
                        r.update(
                            l_sample.row(n).view(),
                            p_sample.row(n)[target_byte] as usize,
                            leakage_model,
                        );
                        r
                    },
                )
                .reduce(
                    || CpaProcessor::new(size, guess_range),
                    |lhs, rhs| lhs + rhs,
                );
            rank = rank + x;
        }
    }

    let rank = rank.finalize(leakage_model);

    // save rank key curves in npy
    write_npy("../results/rank.npy", &rank.rank().map(|&x| x as u64))?;

    Ok(())
}

fn main() -> Result<()> {
    rank()?;

    Ok(())
}
