use crate::*;
use air::{StarkField};
use alloc::vec::Vec;
use gpu::cuda::{CudaExecutionProver, DIGEST_SIZE};
use processor::{
    crypto::{Hasher, RpoDigest, RpoRandomCoin, Rpx256, RpxDigest, RpxRandomCoin},
    math::fft,
};
use std::{assert_eq, matches, assert_ne};
use winter_prover::{crypto::Digest, math::fields::CubeExtension, CompositionPolyTrace, TraceLde};
use crate::gpu::tests::{create_test_prover, gen_random_trace, get_random_values, get_trace_info};

const RATE: usize = Rpo256::RATE_RANGE.end - Rpo256::RATE_RANGE.start;

type CubeFelt = CubeExtension<Felt>;

fn build_trace_commitment_on_gpu_with_padding_matches_cpu<
    R: RandomCoin<BaseField = Felt, Hasher = H> + Send,
    H: ElementHasher<BaseField = Felt> + Hasher<Digest = D>,
    D: Digest + for<'a> From<&'a [Felt; DIGEST_SIZE]>,
>(
    hash_fn: HashFn,
) {
    let is_rpx = matches!(hash_fn, HashFn::Rpx256);

    let cpu_prover = create_test_prover::<R, H>(is_rpx);
    let gpu_prover = CudaExecutionProver::new(create_test_prover::<R, H>(is_rpx), hash_fn);
    let num_rows = 1 << 8;
    let trace_info = get_trace_info(8, num_rows);
    let trace = gen_random_trace(num_rows, 8);
    let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), 8, Felt::GENERATOR);

    let (cpu_trace_lde, cpu_polys) =
        cpu_prover.new_trace_lde::<CubeFelt>(&trace_info, &trace, &domain);
    let (gpu_trace_lde, gpu_polys) =
        gpu_prover.new_trace_lde::<CubeFelt>(&trace_info, &trace, &domain);

    assert_eq!(
        cpu_trace_lde.get_main_trace_commitment(),
        gpu_trace_lde.get_main_trace_commitment()
    );
    assert_eq!(
        cpu_polys.main_trace_polys().collect::<Vec<_>>(),
        gpu_polys.main_trace_polys().collect::<Vec<_>>()
    );
}

fn build_trace_commitment_on_gpu_without_padding_matches_cpu<
    R: RandomCoin<BaseField = Felt, Hasher = H> + Send,
    H: ElementHasher<BaseField = Felt> + Hasher<Digest = D>,
    D: Digest + for<'a> From<&'a [Felt; DIGEST_SIZE]>,
>(
    hash_fn: HashFn,
) {
    let is_rpx = matches!(hash_fn, HashFn::Rpx256);

    let cpu_prover = create_test_prover::<R, H>(is_rpx);
    let gpu_prover = CudaExecutionProver::new(create_test_prover::<R, H>(is_rpx), hash_fn);
    let num_rows = 1 << 8;
    let trace_info = get_trace_info(1, num_rows);
    let trace = gen_random_trace(num_rows, RATE);
    let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), 8, Felt::GENERATOR);

    let (cpu_trace_lde, cpu_polys) =
        cpu_prover.new_trace_lde::<CubeFelt>(&trace_info, &trace, &domain);
    let (gpu_trace_lde, gpu_polys) =
        gpu_prover.new_trace_lde::<CubeFelt>(&trace_info, &trace, &domain);

    assert_eq!(
        cpu_trace_lde.get_main_trace_commitment(),
        gpu_trace_lde.get_main_trace_commitment()
    );
    assert_eq!(
        cpu_polys.main_trace_polys().collect::<Vec<_>>(),
        gpu_polys.main_trace_polys().collect::<Vec<_>>()
    );
}

fn build_constraint_commitment_on_gpu_with_padding_matches_cpu<
    R: RandomCoin<BaseField = Felt, Hasher = H> + Send,
    H: ElementHasher<BaseField = Felt> + Hasher<Digest = D>,
    D: Digest + for<'a> From<&'a [Felt; DIGEST_SIZE]>,
>(
    hash_fn: HashFn,
) {
    let is_rpx = matches!(hash_fn, HashFn::Rpx256);

    let cpu_prover = create_test_prover::<R, H>(is_rpx);
    let gpu_prover = CudaExecutionProver::new(create_test_prover::<R, H>(is_rpx), hash_fn);
    let num_rows = 1 << 8;
    let ce_blowup_factor = 2;
    let values = get_random_values::<CubeFelt>(num_rows * ce_blowup_factor);
    let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), 8, Felt::GENERATOR);

    let (commitment_cpu, composition_poly_cpu) = cpu_prover.build_constraint_commitment(
        CompositionPolyTrace::new(values.clone()),
        2,
        &domain,
    );
    let (commitment_gpu, composition_poly_gpu) =
        gpu_prover.build_constraint_commitment(CompositionPolyTrace::new(values), 2, &domain);

    assert_eq!(commitment_cpu.root(), commitment_gpu.root());
    assert_ne!(0, composition_poly_cpu.data().num_base_cols() % RATE);
    assert_eq!(composition_poly_cpu.into_columns(), composition_poly_gpu.into_columns());
}

fn build_constraint_commitment_on_gpu_without_padding_matches_cpu<
    R: RandomCoin<BaseField = Felt, Hasher = H> + Send,
    H: ElementHasher<BaseField = Felt> + Hasher<Digest = D>,
    D: Digest + for<'a> From<&'a [Felt; DIGEST_SIZE]>,
>(
    hash_fn: HashFn,
) {
    let is_rpx = matches!(hash_fn, HashFn::Rpx256);

    let cpu_prover = create_test_prover::<R, H>(is_rpx);
    let gpu_prover = CudaExecutionProver::new(create_test_prover::<R, H>(is_rpx), hash_fn);
    let num_rows = 1 << 8;
    let ce_blowup_factor = 8;
    let values = get_random_values::<Felt>(num_rows * ce_blowup_factor);
    let domain = StarkDomain::from_twiddles(fft::get_twiddles(num_rows), 8, Felt::GENERATOR);

    let (commitment_cpu, composition_poly_cpu) = cpu_prover.build_constraint_commitment(
        CompositionPolyTrace::new(values.clone()),
        8,
        &domain,
    );
    let (commitment_gpu, composition_poly_gpu) =
        gpu_prover.build_constraint_commitment(CompositionPolyTrace::new(values), 8, &domain);

    assert_eq!(commitment_cpu.root(), commitment_gpu.root());
    assert_eq!(0, composition_poly_cpu.data().num_base_cols() % RATE);
    assert_eq!(composition_poly_cpu.into_columns(), composition_poly_gpu.into_columns());
}

// #[test]
// fn rpo_build_trace_commitment_on_gpu_with_padding_matches_cpu() {
//     build_trace_commitment_on_gpu_with_padding_matches_cpu::<RpoRandomCoin, Rpo256, RpoDigest>(
//         HashFn::Rpo256,
//     );
// }
//
// #[test]
// fn rpx_build_trace_commitment_on_gpu_with_padding_matches_cpu() {
//     build_trace_commitment_on_gpu_with_padding_matches_cpu::<RpxRandomCoin, Rpx256, RpxDigest>(
//         HashFn::Rpx256,
//     );
// }

#[test]
fn rpo_build_trace_commitment_on_gpu_without_padding_matches_cpu() {
    build_trace_commitment_on_gpu_without_padding_matches_cpu::<RpoRandomCoin, Rpo256, RpoDigest>(
        HashFn::Rpo256,
    );
}

// #[test]
// fn rpx_build_trace_commitment_on_gpu_without_padding_matches_cpu() {
//     build_trace_commitment_on_gpu_without_padding_matches_cpu::<RpxRandomCoin, Rpx256, RpxDigest>(
//         HashFn::Rpx256,
//     );
// }
//
// #[test]
// fn rpo_build_constraint_commitment_on_gpu_with_padding_matches_cpu() {
//     build_constraint_commitment_on_gpu_with_padding_matches_cpu::<RpoRandomCoin, Rpo256, RpoDigest>(
//         HashFn::Rpo256,
//     );
// }
//
// #[test]
// fn rpx_build_constraint_commitment_on_gpu_with_padding_matches_cpu() {
//     build_constraint_commitment_on_gpu_with_padding_matches_cpu::<RpxRandomCoin, Rpx256, RpxDigest>(
//         HashFn::Rpx256,
//     );
// }
//
// #[test]
// fn rpo_build_constraint_commitment_on_gpu_without_padding_matches_cpu() {
//     build_constraint_commitment_on_gpu_without_padding_matches_cpu::<
//         RpoRandomCoin,
//         Rpo256,
//         RpoDigest,
//     >(HashFn::Rpo256);
// }
//
// #[test]
// fn rpx_build_constraint_commitment_on_gpu_without_padding_matches_cpu() {
//     build_constraint_commitment_on_gpu_without_padding_matches_cpu::<
//         RpxRandomCoin,
//         Rpx256,
//         RpxDigest,
//     >(HashFn::Rpx256);
// }
