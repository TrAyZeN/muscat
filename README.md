# Multithreaded Side Channel Attacks Tool

muscat is a Rust library implementing state-of-the-art side channel attacks (SCAs) algorithms.

Supported algorithms:
- CPA
- DPA
- SNR
- NICV
- Welch's T-Test
- Elastic alignment

Python bindings are also provided see [muscatpy](https://github.com/Ledger-Donjon/muscat/tree/master/muscatpy).

## Getting started
Here is an example of how to recover the first byte of the AES key of the given traces:
```rust
use ndarray::Array2;
use ndarray_npy::read_npy;
use std::{env, iter::zip, path::PathBuf};

use muscat::distinguishers::cpa::CpaProcessor;
use muscat::leakage_model::aes::sbox;

fn leakage_model(plaintext_byte: usize, guess: usize) -> usize {
    sbox((plaintext_byte ^ guess) as u8) as usize
}

fn main() {
    let traces_dir =
        PathBuf::from(env::var("TRACES_DIR").expect("Missing TRACES_DIR environment variable"));

    let traces: Array2<f64> =
          read_npy(traces_dir.join("traces.npy")).expect("Failed to read traces.npy");
    let plaintexts: Array2<u8> =
          read_npy(traces_dir.join("plaintexts.npy")).expect("Failed to read plaintexts.npy");
    assert_eq!(traces.shape()[0], plaintexts.shape()[0]);

    let mut processor = CpaProcessor::new(traces.shape()[1], 256);
    for (trace, plaintext) in zip(traces.rows(), plaintexts.rows()) {
        processor.update(trace.view(), plaintext[0], leakage_model);
    }
    let cpa = processor.finalize(leakage_model);
    let best_guess = cpa.best_guess();
    println!("Best subkey guess: {best_guess:?}");
}
```

More examples are available in the [examples](https://github.com/Ledger-Donjon/muscat/tree/master/examples) directory.

## Benchmark
To reduce benchmark variance it is advised to follow [these instructions](https://google.github.io/benchmark/reducing_variance.html).

Then to run benchmarks, run
```sh
cargo bench
```

Benchmark report can found under `target/criterion/report/index.html`.

## License
Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
