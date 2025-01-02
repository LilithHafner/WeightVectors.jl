use dynamic_weighted_index::DynamicWeightedIndex;
use rand::distributions::Distribution;
use rand::Rng;
use rand::SeedableRng;
use pcg_rand::Pcg64;
use std::time::Instant;

fn setup(rng: &mut Pcg64, size: usize) -> DynamicWeightedIndex<f64> {
    let mut dw_index = DynamicWeightedIndex::new(size);
    for i in 0..size {
        let weight = rng.gen_range(0.0..(10.0_f64).powi(200));
        dw_index.set_weight(i, weight);
    }
    dw_index
}

fn sample_fixed(
    rng: &mut Pcg64,
    dw_index: &DynamicWeightedIndex<f64>,
    n: usize,
) -> Vec<usize> {
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        samples.push(dw_index.sample(rng).unwrap());
    }
    samples
}

fn sample_variable(
    rng: &mut Pcg64,
    dw_index: &mut DynamicWeightedIndex<f64>,
    n: usize,
) -> Vec<usize> {
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let index_to_modify = rng.gen_range(0..n);
        dw_index.remove_weight(index_to_modify);
        samples.push(dw_index.sample(rng).unwrap());
        let new_weight = rng.gen_range(0.0..(10.0_f64).powi(200));
        dw_index.set_weight(index_to_modify, new_weight);
    }
    samples
}

fn main() {
    let mut rng = Pcg64::seed_from_u64(42);
    let sample_sizes: Vec<usize> = (3..=8).map(|i| 10usize.pow(i)).collect();

    for &size in &sample_sizes {
        let setup_start = Instant::now();
        let mut dw_index = setup(&mut rng, size);
        let setup_time = setup_start.elapsed();

        let fixed_start = Instant::now();
        sample_fixed(&mut rng, &dw_index, size);
        let fixed_time = fixed_start.elapsed();

        let variable_start = Instant::now();
        sample_variable(&mut rng, &mut dw_index, size);
        let variable_time = variable_start.elapsed();

        println!(
            "Size: {}, Setup time: {:.2} ns/element, Fixed sampling: {:.2} ns/sample, Variable sampling: {:.2} ns/sample",
            size,
            (setup_time.as_nanos() as f64 / size as f64),
            (fixed_time.as_nanos() as f64 / size as f64),
            (variable_time.as_nanos() as f64 / size as f64),
        );
    }
}
