use alloc::vec::Vec;
use winter_air::proof::Queries;
use winter_prover::crypto::MerkleTree;
use winter_prover::matrix::RowMatrix;
use processor::crypto::{ElementHasher, Hasher};
use processor::Felt;
use processor::math::FieldElement;

#[cfg(all(feature = "metal", target_arch = "aarch64", target_os = "macos"))]
pub mod metal;

#[cfg(all(feature = "cuda", target_arch = "x86_64"))]
pub mod cuda;

// In the case where none of the gpu features are used, this won't be used as well
#[allow(unused)]
fn build_segment_queries<
    E: FieldElement<BaseField = Felt>,
    H: Hasher + ElementHasher<BaseField = E::BaseField>,
>(
    segment_lde: &RowMatrix<E>,
    segment_tree: &MerkleTree<H>,
    positions: &[usize],
) -> Queries {
    // for each position, get the corresponding row from the trace segment LDE and put all these
    // rows into a single vector
    let trace_states =
        positions.iter().map(|&pos| segment_lde.row(pos).to_vec()).collect::<Vec<_>>();

    // build Merkle authentication paths to the leaves specified by positions
    let trace_proof = segment_tree
        .prove_batch(positions)
        .expect("failed to generate a Merkle proof for trace queries");

    Queries::new(trace_proof, trace_states)
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};
    use winter_air::TraceInfo;
    use winter_prover::matrix::ColMatrix;
    use air::ProvingOptions;
    use processor::crypto::{ElementHasher, RandomCoin};
    use processor::Felt;
    use processor::math::FieldElement;
    use crate::{ExecutionProver, StackInputs, StackOutputs};

    #[cfg(test)]
    #[allow(unused)]
    pub(crate) fn gen_random_trace(num_rows: usize, num_cols: usize) -> ColMatrix<Felt> {
        ColMatrix::new((0..num_cols as u64).map(|col| vec![Felt::new(col); num_rows]).collect())
    }

    #[cfg(test)]
    #[allow(unused)]
    pub(crate) fn get_random_values<E: FieldElement>(num_rows: usize) -> Vec<E> {
        (0..num_rows).map(|i| E::from(i as u32)).collect()
    }

    #[cfg(test)]
    #[allow(unused)]
    pub(crate) fn get_trace_info(num_cols: usize, num_rows: usize) -> TraceInfo {
        TraceInfo::new(num_cols, num_rows)
    }

    #[cfg(test)]
    #[allow(unused)]
    pub(crate) fn create_test_prover<
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
}