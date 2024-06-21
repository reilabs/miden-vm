
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use miden_gpu::HashFn;
use winter_air::TraceInfo;
use winter_prover::{CompositionPolyTrace, Prover, StarkDomain};
use winter_prover::math::fields::CubeExtension;
use winter_prover::math::StarkField;
use winter_prover::matrix::ColMatrix;

use air::{Felt, FieldElement, ProvingOptions};
use processor::crypto::{ElementHasher, RandomCoin, Rpo256, RpoRandomCoin, Rpx256, RpxRandomCoin};
use processor::math::fft;

#[cfg(feature = "cuda")]
use miden_prover::gpu::cuda::CudaExecutionProver;
use miden_prover::{ExecutionProver, StackInputs, StackOutputs};
#[cfg(all(feature = "metal", target_arch = "aarch64", target_os = "macos"))]
use crate::gpu::metal::MetalExecutionProver;

// Random trace is built from RATE columns with 1 << 17,18,19,20 rows,
// making the full trace length equal to 1 << 20,21,22,23
const BENCHMARK_INPUT_SIZES: [usize; 4] = [17,18,19,20];
const CE_BLOWUP: usize = 8;

const RATE: usize = Rpo256::RATE_RANGE.end - Rpo256::RATE_RANGE.start;

type CubeFelt = CubeExtension<Felt>;

fn create_test_prover<
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

fn get_random_values<E: FieldElement>(num_rows: usize) -> Vec<E> {
    (0..num_rows).map(|i| E::from(i as u32)).collect()
}

fn get_trace_info(num_cols: usize, num_rows: usize) -> TraceInfo {
    TraceInfo::new(num_cols, num_rows)
}

fn gen_random_trace(num_rows: usize, num_cols: usize) -> ColMatrix<Felt> {
    ColMatrix::new((0..num_cols as u64).map(|col| vec![Felt::new(col); num_rows]).collect())
}

#[cfg(feature = "cuda")]
pub fn cuda_bench_trace_rpo(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda rpo");
    group.sample_size(10);

    let prover = CudaExecutionProver::new(create_test_prover::<RpoRandomCoin, Rpo256>(false), HashFn::Rpo256);

    for n in BENCHMARK_INPUT_SIZES {
        let input_size = n;
        let num_rows = 1 << input_size;
        let ce_blowup_factor = CE_BLOWUP;

        let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), ce_blowup_factor, Felt::GENERATOR);

        let trace_info = get_trace_info(1, num_rows);
        let trace = gen_random_trace(num_rows, RATE);

        group.bench_function(BenchmarkId::new("CudaRpoTrace", input_size + 3), |b| {
            b.iter(|| {
                    prover.new_trace_lde::<CubeFelt>(&trace_info,&trace, &domain);
                },
            )
        });
    }

    group.finish();
}

#[cfg(feature = "cuda")]
pub fn cuda_bench_trace_rpx(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda rpx");
    group.sample_size(10);

    let prover = CudaExecutionProver::new(create_test_prover::<RpxRandomCoin, Rpx256>(true), HashFn::Rpx256);

    for n in BENCHMARK_INPUT_SIZES {
        let input_size = n;
        let num_rows = 1 << input_size;
        let ce_blowup_factor = CE_BLOWUP;

        let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), ce_blowup_factor, Felt::GENERATOR);

        let trace_info = get_trace_info(1, num_rows);
        let trace = gen_random_trace(num_rows, RATE);

        group.bench_function(BenchmarkId::new("CudaRpxTrace", input_size + 3), |b| {
            b.iter(|| {
                    prover.new_trace_lde::<CubeFelt>(&trace_info, &trace, &domain);
                }
            )
        });
    }

    group.finish();
}

pub fn cpu_bench_trace_rpo(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu rpo");
    group.sample_size(10);

    let prover = create_test_prover::<RpoRandomCoin, Rpo256>(false);

    for n in BENCHMARK_INPUT_SIZES {
        let input_size = n;
        let num_rows = 1 << input_size;

        let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), CE_BLOWUP, Felt::GENERATOR);


        let trace_info = get_trace_info(1, num_rows);
        let trace = gen_random_trace(num_rows, RATE);

        group.bench_function(BenchmarkId::new("CpuRpoTrace", input_size + 3), |b| {
            b.iter(|| {
                    prover.new_trace_lde::<CubeFelt>(&trace_info, &trace, &domain);
                }
            )
        });
    }

    group.finish();
}

pub fn cpu_bench_trace_rpx(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu rpx");
    group.sample_size(10);

    let prover = create_test_prover::<RpxRandomCoin, Rpx256>(true);

    for n in BENCHMARK_INPUT_SIZES {
        let input_size = n;
        let num_rows = 1 << input_size;

        let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), CE_BLOWUP, Felt::GENERATOR);

        let trace_info = get_trace_info(1, num_rows);
        let trace = gen_random_trace(num_rows, RATE);

        group.bench_function(BenchmarkId::new("CpuRpxTrace", input_size + 3), |b| {
            b.iter(|| {
                    prover.new_trace_lde::<CubeFelt>(&trace_info, &trace, &domain);
                }
            )
        });
    }

    group.finish();
}

criterion_group!(cuda_bench,
    cuda_bench_trace_rpo,
    cuda_bench_trace_rpx,
);
criterion_main!(cuda_bench);