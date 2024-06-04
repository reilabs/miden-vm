use alloc::vec::Vec;
use winter_air::proof::Queries;
use winter_prover::crypto::MerkleTree;
use winter_prover::matrix::RowMatrix;
use air::{Felt, FieldElement};
use processor::crypto::{ElementHasher, Hasher};

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