mod benchmarks {
    use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, BatchSize};
    use miden_gpu::HashFn;
    use winter_prover::math::StarkField;
    use winter_prover::{CompositionPolyTrace, Prover, StarkDomain};
    use air::Felt;
    use processor::crypto::{Rpo256, RpoRandomCoin};
    use processor::math::fft;
    use crate::gpu::cuda::CudaExecutionProver;
    use crate::gpu::cuda::tests::{create_test_prover, get_random_values};



    fn cuda_bench(_c: &mut Criterion) {
        const BENCHMARK_INPUT_SIZES: [usize; 5] = [8, 12, 16, 20, 24];
        const NUM_BASE: usize = 8;

        pub fn cuda_bench_rpo(c: &mut Criterion, name: &str) {
            let mut group = c.benchmark_group(name);
            group.sample_size(10);

            let gpu_prover = CudaExecutionProver::new(create_test_prover::<RpoRandomCoin, Rpo256>(false), HashFn::Rpo256);

            for n in BENCHMARK_INPUT_SIZES {
                let input_size = n;
                let num_rows = 1 << input_size;
                let ce_blowup_factor = 2;

                let values = get_random_values::<Felt>(num_rows * ce_blowup_factor);
                let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), ce_blowup_factor, Felt::GENERATOR);

                group.bench_with_input(BenchmarkId::new("RpoConstraint", input_size), &input_size, |b, _| {
                    b.iter_batched(
                        || CompositionPolyTrace::new(values.clone()),
                        |poly_trace| {
                            gpu_prover.build_constraint_commitment(poly_trace, 8, &domain);
                        },
                        BatchSize::SmallInput,
                    )
                });
            }

            group.finish();
        }

        pub fn cuda_bench_rpx(c: &mut Criterion, name: &str) {
            let mut group = c.benchmark_group(name);
            group.sample_size(10);

            let gpu_prover = CudaExecutionProver::new(create_test_prover::<RpoRandomCoin, Rpo256>(true), HashFn::Rpx256);

            for n in BENCHMARK_INPUT_SIZES {
                let input_size = n;
                let num_rows = 1 << input_size;
                let ce_blowup_factor = 2;

                let values = get_random_values::<Felt>(num_rows * ce_blowup_factor);
                let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), ce_blowup_factor, Felt::GENERATOR);

                group.bench_with_input(BenchmarkId::new("RpxConstraint", input_size), &input_size, |b, _| {
                    b.iter_batched(
                        || CompositionPolyTrace::new(values.clone()),
                        |poly_trace| {
                            gpu_prover.build_constraint_commitment(poly_trace, 8, &domain);
                        },
                        BatchSize::SmallInput,
                    )
                });
            }

            group.finish();
        }
    }

    criterion_group!(benches, cuda_bench);
    criterion_main!(benches);
}