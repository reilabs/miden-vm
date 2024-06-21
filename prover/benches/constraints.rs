
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use miden_gpu::HashFn;
use winter_prover::{CompositionPolyTrace, Prover, StarkDomain};
use winter_prover::math::fields::CubeExtension;
use winter_prover::math::StarkField;

use air::{Felt, FieldElement, ProvingOptions};
use processor::crypto::{ElementHasher, RandomCoin, Rpo256, RpoRandomCoin, Rpx256, RpxRandomCoin};
use processor::math::fft;

#[cfg(feature = "cuda")]
use miden_prover::gpu::cuda::CudaExecutionProver;
use miden_prover::{ExecutionProver, StackInputs, StackOutputs};
#[cfg(all(feature = "metal", target_arch = "aarch64", target_os = "macos"))]
use crate::gpu::metal::MetalExecutionProver;

// This will be multiplied be CE_BLOWUP, so we get 1 << 20,21,22,23 values
const BENCHMARK_INPUT_SIZES: [usize; 4] = [17,18,19,20];
const CE_BLOWUP: usize = 8;

type CubeFelt = CubeExtension<Felt>;

pub fn create_test_prover<
    R: RandomCoin<BaseField=Felt, Hasher=H>,
    H: ElementHasher<BaseField=Felt>,
>(
    use_rpx: bool,
) -> ExecutionProver<H, R> {
    if use_rpx {
        ExecutionProver::new(
            ProvingOptions::with_128_bit_security_rpx(),
            StackInputs::default(),
            StackOutputs::default(),
        )
    } else {
        ExecutionProver::new(
            ProvingOptions::with_128_bit_security(true),
            StackInputs::default(),
            StackOutputs::default(),
        )
    }
}

pub fn get_random_values<E: FieldElement>(num_rows: usize) -> Vec<E> {
    (0..num_rows).map(|i| E::from(i as u32)).collect()
}



    #[cfg(feature = "cuda")]
    pub fn cuda_bench_constraint_rpo(c: &mut Criterion) {
        let mut group = c.benchmark_group("cuda rpo");
        group.sample_size(10);

        let prover = CudaExecutionProver::new(create_test_prover::<RpoRandomCoin, Rpo256>(false), HashFn::Rpo256);

        for n in BENCHMARK_INPUT_SIZES {
            let input_size = n;
            let num_rows = 1 << input_size;

            let values = get_random_values::<CubeFelt>(num_rows * CE_BLOWUP);
            let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), CE_BLOWUP, Felt::GENERATOR);

            group.bench_function(BenchmarkId::new("CudaRpoConstraint", input_size + 3), |b| {
                b.iter_batched(
                    || CompositionPolyTrace::new(values.clone()),
                    |poly_trace| {
                        prover.build_constraint_commitment(poly_trace, 72, &domain);
                    },
                    BatchSize::SmallInput,
                )
            });
        }

        group.finish();
    }

        #[cfg(feature = "cuda")]
        pub fn cuda_bench_constraint_rpx(c: &mut Criterion) {
            let mut group = c.benchmark_group("cuda rpx");
            group.sample_size(10);

            let prover = CudaExecutionProver::new(create_test_prover::<RpxRandomCoin, Rpx256>(true), HashFn::Rpx256);

            for n in BENCHMARK_INPUT_SIZES {
                let input_size = n;
                let num_rows = 1 << input_size;

                let values = get_random_values::<CubeFelt>(num_rows * CE_BLOWUP);
                let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), CE_BLOWUP, Felt::GENERATOR);

                group.bench_function(BenchmarkId::new("CudaRpxConstraint", input_size + 3), |b| {
                    b.iter_batched(
                        || CompositionPolyTrace::new(values.clone()),
                        |poly_trace| {
                            prover.build_constraint_commitment(poly_trace, 72, &domain);
                        },
                        BatchSize::SmallInput,
                    )
                });
            }

            group.finish();
        }

    pub fn cpu_bench_constraint_rpo(c: &mut Criterion) {
        let mut group = c.benchmark_group("cpu rpo");
        group.sample_size(10);

        let prover = create_test_prover::<RpoRandomCoin, Rpo256>(false);

        for n in BENCHMARK_INPUT_SIZES {
            let input_size = n;
            let num_rows = 1 << input_size;


            let values = get_random_values::<CubeFelt>(num_rows * CE_BLOWUP);
            let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), CE_BLOWUP, Felt::GENERATOR);

            group.bench_function(BenchmarkId::new("CpuRpoConstraint", input_size + 3), |b| {
                b.iter_batched(
                    || CompositionPolyTrace::new(values.clone()),
                    |poly_trace| {
                        prover.build_constraint_commitment(poly_trace, 72, &domain);
                    },
                    BatchSize::SmallInput,
                )
            });
        }

        group.finish();
    }

    pub fn cpu_bench_constraint_rpx(c: &mut Criterion) {
        let mut group = c.benchmark_group("cpu rpx");
        group.sample_size(10);

        let prover = create_test_prover::<RpxRandomCoin, Rpx256>(true);

        for n in BENCHMARK_INPUT_SIZES {
            let input_size = n;
            let num_rows = 1 << input_size;

            let values = get_random_values::<CubeFelt>(num_rows * CE_BLOWUP);
            let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), CE_BLOWUP, Felt::GENERATOR);

            group.bench_function(BenchmarkId::new("CpuRpxConstraint", input_size + 3), |b| {
                b.iter_batched(
                    || CompositionPolyTrace::new(values.clone()),
                    |poly_trace| {
                        prover.build_constraint_commitment(poly_trace, 72, &domain);
                    },
                    BatchSize::SmallInput,
                )
            });
        }

        group.finish();
    }

criterion_group!(cuda_bench,
    cuda_bench_constraint_rpo,
    cuda_bench_constraint_rpx,
);
criterion_main!(cuda_bench);