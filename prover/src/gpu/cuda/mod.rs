//! This module contains GPU acceleration logic for devices supporting NVIDIA CUDA.
//! For now, the logic is limited to GPU accelerating trace and constraint commitments,
//! using the RPO 256 or RPX 256 hash functions.

use crate::{
    crypto::{RandomCoin, Rpo256},
    ExecutionProver, ExecutionTrace, Felt, FieldElement, ProcessorAir, PublicInputs,
    WinterProofOptions,
};

use miden_gpu;
use processor::{
    crypto::{ElementHasher, Hasher},
};
use std::{marker::PhantomData, time::Instant, vec, vec::Vec};
use std::mem::size_of;
use miden_gpu::HashFn;
use miden_gpu::cuda::{CudaOptions, init_gpu, lde_into_rescue_batch, Trace, alloc_pinned, cleanup_gpu, set_transpose, Transpose};
use tracing::{event, Level};
use winter_air::LagrangeKernelEvaluationFrame;
use winter_prover::{crypto::{Digest, MerkleTree}, matrix::{ColMatrix, RowMatrix}, proof::Queries, CompositionPoly, CompositionPolyTrace, ConstraintCommitment, ConstraintCompositionCoefficients, DefaultConstraintEvaluator, EvaluationFrame, Prover, StarkDomain, TraceInfo, TraceLde, TracePolyTable};
use air::AuxRandElements;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod bench;

// CONSTANTS
// ================================================================================================
const DIGEST_SIZE: usize = Rpo256::DIGEST_RANGE.end - Rpo256::DIGEST_RANGE.start;
const RATE: usize = Rpo256::RATE_RANGE.end - Rpo256::RATE_RANGE.start;
const CAPACITY: usize = 4;
const MERKLE_TREE_HEADER_SIZE: usize = 2;

// METAL RPO/RPX PROVER
// ================================================================================================

/// Wraps an [ExecutionProver] and provides GPU acceleration for building trace commitments.
pub(crate) struct CudaExecutionProver<H, D, R>
    where
        H: Hasher<Digest = D> + ElementHasher<BaseField = R::BaseField>,
        D: Digest + for<'a> From<&'a [Felt; DIGEST_SIZE]>,
        R: RandomCoin<BaseField = Felt, Hasher = H>,
{
    pub execution_prover: ExecutionProver<H, R>,
    pub hash_fn: HashFn,
    phantom_data: PhantomData<D>,
}

impl<H, D, R> CudaExecutionProver<H, D, R>
    where
        H: Hasher<Digest = D> + ElementHasher<BaseField = R::BaseField>,
        D: Digest + for<'a> From<&'a [Felt; DIGEST_SIZE]>,
        R: RandomCoin<BaseField = Felt, Hasher = H>,
{
    pub fn new(execution_prover: ExecutionProver<H, R>, hash_fn: HashFn) -> Self {
        CudaExecutionProver {
            execution_prover,
            hash_fn,
            phantom_data: PhantomData,
        }
    }
}

impl<H, D, R> Prover for CudaExecutionProver<H, D, R>
    where
        H: Hasher<Digest = D> + ElementHasher<BaseField = R::BaseField>,
        D: Digest + for<'a> From<&'a [Felt; DIGEST_SIZE]>,
        R: RandomCoin<BaseField = Felt, Hasher = H> + Send,
{
    type BaseField = Felt;
    type Air = ProcessorAir;
    type Trace = ExecutionTrace;
    type HashFn = H;
    type RandomCoin = R;
    type TraceLde<E: FieldElement<BaseField = Felt>> = CudaTraceLde<E, H>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Felt>> =
    DefaultConstraintEvaluator<'a, ProcessorAir, E>;

    fn get_pub_inputs(&self, trace: &ExecutionTrace) -> PublicInputs {
        self.execution_prover.get_pub_inputs(trace)
    }

    fn options(&self) -> &WinterProofOptions {
        &self.execution_prover.options
    }

    fn new_trace_lde<E: FieldElement<BaseField = Felt>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Felt>,
        domain: &StarkDomain<Felt>,
    ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
        CudaTraceLde::new(trace_info, main_trace, domain, self.hash_fn)
    }

    fn new_evaluator<'a, E: FieldElement<BaseField = Felt>>(
        &self,
        air: &'a ProcessorAir,
        aux_rand_elements: Option<AuxRandElements<E>>,
        composition_coefficients: ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        self.execution_prover
            .new_evaluator(air, aux_rand_elements, composition_coefficients)
    }

    /// Evaluates constraint composition polynomial over the LDE domain and builds a commitment
    /// to these evaluations.
    ///
    /// The evaluation is done by evaluating each composition polynomial column over the LDE
    /// domain.
    ///
    /// The commitment is computed by hashing each row in the evaluation matrix, and then building
    /// a Merkle tree from the resulting hashes.
    ///
    /// The composition polynomial columns are evaluated on the CPU. Afterward the commitment
    /// is computed on the GPU.
    fn build_constraint_commitment<E: FieldElement<BaseField = Felt>>(
        &self,
        composition_poly_trace: CompositionPolyTrace<E>,
        num_trace_poly_columns: usize,
        domain: &StarkDomain<Felt>,
    ) -> (ConstraintCommitment<E, Self::HashFn>, CompositionPoly<E>) {
        // evaluate composition polynomial columns over the LDE domain

        let lde_domain_size = domain.lde_domain_size();
        let blowup = domain.trace_to_lde_blowup();


        init_gpu(lde_domain_size.ilog2() as usize, blowup.ilog2() as usize).expect("Could not initialize GPU.");

        let now = Instant::now();

        let composition_poly =
            CompositionPoly::new(composition_poly_trace, domain, num_trace_poly_columns);

        let num_cols = composition_poly.num_columns();
        let partition_count = (2 *  num_cols - 1) / num_cols;

        // allocate space to store all the results
        let mut lde_out = alloc_pinned(lde_domain_size * num_cols * size_of::<E>(), E::ZERO);
        let mut tmp = alloc_pinned(lde_domain_size * num_cols * size_of::<E>(), E::ZERO);
        let mut hash_states_out = alloc_pinned(partition_count * CAPACITY * lde_domain_size * size_of::<E>(), D::default());
        let mut merkle_tree_out = alloc_pinned((2 * lde_domain_size - 1) * CAPACITY * size_of::<E>(), D::default());

        set_transpose(Transpose::Full, 4);

        let mut data = composition_poly.data().columns().flat_map(|c| c.to_vec()).collect::<Vec<E>>();

        lde_into_rescue_batch(Trace {
            tmp: Some(&mut tmp),
            data: data.as_mut_slice(),
            num_cols: composition_poly.num_columns(),
            domain_size: domain.lde_domain_size(),
        },
            CudaOptions { blowup_factor: blowup, partition_size: composition_poly.num_columns() as u32 },
            &mut lde_out,
            &mut hash_states_out,
            &mut merkle_tree_out,
            self.hash_fn)
            .expect("failed to compute lde into rescue batch.");

        // NOTE: `merkle_tree_out` contains leaves. Currently, the bellow code is not correct.
        // NOTE: `hash_states_out` should be transposed to row-major if we want to match the merkle_tree order.
        // Leaves in merkle start at tree_offset:
        // let tree_offset = MERKLE_TREE_HEADER_SIZE + num_cols * lde_domain_size;

        let num_base_columns =
            composition_poly.num_columns() * <E as FieldElement>::EXTENSION_DEGREE;
        let (nodes, leaves) = merkle_tree_out.split_at(MERKLE_TREE_HEADER_SIZE + num_base_columns * lde_domain_size);

        let commitment = MerkleTree::<H>::from_raw_parts(nodes.to_vec(), leaves.to_vec()).expect("failed to build Merkle tree");

        let composed_evaluations = RowMatrix::<E>::new(lde_out.to_vec(), lde_out.len() / num_cols, lde_out.len() / num_cols);
        let constraint_commitment = ConstraintCommitment::new(composed_evaluations, commitment);

        event!(
            Level::INFO,
            "Computed constraint evaluation commitment on the GPU (Merkle tree of depth {}) in {} ms",
            constraint_commitment.tree_depth(),
            now.elapsed().as_millis()
        );
        cleanup_gpu();
        (constraint_commitment, composition_poly)
    }
}

// TRACE LOW DEGREE EXTENSION (CUDA)
// ================================================================================================

/// Contains all segments of the extended execution trace, the commitments to these segments, the
/// LDE blowup factor, and the [TraceInfo].
///
/// Segments are stored in two groups:
/// - Main segment: this is the first trace segment generated by the prover. Values in this segment
///   will always be elements in the base field (even when an extension field is used).
/// - Auxiliary segments: a list of 0 or more segments for traces generated after the prover
///   commits to the first trace segment. Currently, at most 1 auxiliary segment is possible.
pub struct CudaTraceLde<E: FieldElement<BaseField = Felt>, H: Hasher> {
    // low-degree extension of the main segment of the trace
    main_segment_lde: RowMatrix<Felt>,
    // commitment to the main segment of the trace
    main_segment_tree: MerkleTree<H>,
    // low-degree extensions of the auxiliary segments of the trace
    aux_segment_ldes: Vec<RowMatrix<E>>,
    // commitment to the auxiliary segments of the trace
    aux_segment_trees: Vec<MerkleTree<H>>,
    blowup: usize,
    trace_info: TraceInfo,
    hash_fn: HashFn,
}

impl<
    E: FieldElement<BaseField = Felt>,
    H: Hasher<Digest = D> + ElementHasher<BaseField = E::BaseField>,
    D: Digest + for<'a> From<&'a [Felt; DIGEST_SIZE]>,
> CudaTraceLde<E, H>
{
    /// Takes the main trace segment columns as input, interpolates them into polynomials in
    /// coefficient form, evaluates the polynomials over the LDE domain, commits to the
    /// polynomial evaluations, and creates a new [DefaultTraceLde] with the LDE of the main trace
    /// segment and the commitment.
    ///
    /// Returns a tuple containing a [TracePolyTable] with the trace polynomials for the main trace
    /// segment and the new [DefaultTraceLde].
    pub fn new(
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Felt>,
        domain: &StarkDomain<Felt>,
        hash_fn: HashFn,
    ) -> (Self, TracePolyTable<E>) {
        // extend the main execution trace and build a Merkle tree from the extended trace
        let (main_segment_lde, main_segment_tree, main_segment_polys) =
            build_trace_commitment(main_trace, domain, hash_fn);

        let trace_poly_table = TracePolyTable::new(main_segment_polys);
        let trace_lde = CudaTraceLde {
            main_segment_lde,
            main_segment_tree,
            aux_segment_ldes: Vec::new(),
            aux_segment_trees: Vec::new(),
            blowup: domain.trace_to_lde_blowup(),
            trace_info: trace_info.clone(),
            hash_fn,
        };

        (trace_lde, trace_poly_table)
    }

    // TEST HELPERS
    // --------------------------------------------------------------------------------------------

    /// Returns number of columns in the main segment of the execution trace.
    #[allow(unused)]
    #[cfg(test)]
    pub fn main_segment_width(&self) -> usize {
        self.main_segment_lde.num_cols()
    }

    /// Returns a reference to [Matrix] representing the main trace segment.
    #[allow(unused)]
    #[cfg(test)]
    pub fn get_main_segment(&self) -> &RowMatrix<Felt> {
        &self.main_segment_lde
    }

    /// Returns the entire trace for the column at the specified index.
    #[allow(unused)]
    #[cfg(test)]
    pub fn get_main_segment_column(&self, col_idx: usize) -> Vec<Felt> {
        (0..self.main_segment_lde.num_rows())
            .map(|row_idx| self.main_segment_lde.get(col_idx, row_idx))
            .collect()
    }
}

impl<
    E: FieldElement<BaseField = Felt>,
    H: Hasher<Digest = D> + ElementHasher<BaseField = E::BaseField>,
    D: Digest + for<'a> From<&'a [Felt; DIGEST_SIZE]>,
> TraceLde<E> for CudaTraceLde<E, H>
{
    type HashFn = H;

    /// Returns the commitment to the low-degree extension of the main trace segment.
    fn get_main_trace_commitment(&self) -> D {
        let root_hash = self.main_segment_tree.root();
        *root_hash
    }

    /// Takes auxiliary trace segment columns as input, interpolates them into polynomials in
    /// coefficient form, evaluates the polynomials over the LDE domain, and commits to the
    /// polynomial evaluations.
    ///
    /// Returns a tuple containing the column polynomials in coefficient from and the commitment
    /// to the polynomial evaluations over the LDE domain.
    ///
    /// # Panics
    ///
    /// This function will panic if any of the following are true:
    /// - the number of rows in the provided `aux_trace` does not match the main trace.
    /// - this segment would exceed the number of segments specified by the trace layout.
    fn set_aux_trace(
        &mut self,
        aux_trace: &ColMatrix<E>,
        domain: &StarkDomain<Felt>,
    ) -> (ColMatrix<E>, D) {
        // extend the auxiliary trace segment and build a Merkle tree from the extended trace
        let (aux_segment_lde, aux_segment_tree, aux_segment_polys) =
            build_trace_commitment::<E, H, D>(aux_trace, domain, self.hash_fn);

        // check errors
        assert!(
            self.aux_segment_ldes.len() < self.trace_info.num_aux_segments(),
            "the specified number of auxiliary segments has already been added"
        );
        assert_eq!(
            self.main_segment_lde.num_rows(),
            aux_segment_lde.num_rows(),
            "the number of rows in the auxiliary segment must be the same as in the main segment"
        );

        // save the lde and commitment
        self.aux_segment_ldes.push(aux_segment_lde);
        let root_hash = *aux_segment_tree.root();
        self.aux_segment_trees.push(aux_segment_tree);

        (aux_segment_polys, root_hash)
    }

    /// Reads current and next rows from the main trace segment into the specified frame.
    fn read_main_trace_frame_into(&self, lde_step: usize, frame: &mut EvaluationFrame<Felt>) {
        // at the end of the trace, next state wraps around and we read the first step again
        let next_lde_step = (lde_step + self.blowup()) % self.trace_len();

        // copy main trace segment values into the frame
        frame.current_mut().copy_from_slice(self.main_segment_lde.row(lde_step));
        frame.next_mut().copy_from_slice(self.main_segment_lde.row(next_lde_step));
    }

    /// Reads current and next rows from the auxiliary trace segment into the specified frame.
    ///
    /// # Panics
    /// This currently assumes that there is exactly one auxiliary trace segment, and will panic
    /// otherwise.
    fn read_aux_trace_frame_into(&self, lde_step: usize, frame: &mut EvaluationFrame<E>) {
        // at the end of the trace, next state wraps around and we read the first step again
        let next_lde_step = (lde_step + self.blowup()) % self.trace_len();

        // copy auxiliary trace segment values into the frame
        let segment = &self.aux_segment_ldes[0];
        frame.current_mut().copy_from_slice(segment.row(lde_step));
        frame.next_mut().copy_from_slice(segment.row(next_lde_step));
    }

    /// Returns trace table rows at the specified positions along with Merkle authentication paths
    /// from the commitment root to these rows.
    fn query(&self, positions: &[usize]) -> Vec<Queries> {
        // build queries for the main trace segment
        let mut result = vec![super::build_segment_queries(
            &self.main_segment_lde,
            &self.main_segment_tree,
            positions,
        )];

        // build queries for auxiliary trace segments
        for (i, segment_tree) in self.aux_segment_trees.iter().enumerate() {
            let segment_lde = &self.aux_segment_ldes[i];
            result.push(super::build_segment_queries(segment_lde, segment_tree, positions));
        }

        result
    }

    /// Returns the number of rows in the execution trace.
    fn trace_len(&self) -> usize {
        self.main_segment_lde.num_rows()
    }

    /// Returns blowup factor which was used to extend the original execution trace into trace LDE.
    fn blowup(&self) -> usize {
        self.blowup
    }

    /// Populates the provided Lagrange kernel frame starting at the current row (as defined by
    /// `lde_step`).
    ///
    /// Note that unlike [`EvaluationFrame`], the Lagrange kernel frame includes only the Lagrange
    /// kernel column (as opposed to all columns).
    fn read_lagrange_kernel_frame_into(&self, lde_step: usize, col_idx: usize, frame: &mut LagrangeKernelEvaluationFrame<E>) {
        let frame = frame.frame_mut();
        frame.truncate(0);

        let aux_segment = &self.aux_segment_ldes[0];

        frame.push(aux_segment.get(col_idx, lde_step));

        let frame_length = self.trace_info.length().ilog2() as usize + 1;
        for i in 0..frame_length - 1 {
            let shift = self.blowup() * (1 << i);
            let next_lde_step = (lde_step + shift) % self.trace_len();

            frame.push(aux_segment.get(col_idx, next_lde_step));
        }
    }

    /// Returns the trace info for the execution trace.
    fn trace_info(&self) -> &TraceInfo {
        &self.trace_info
    }
}

/// Computes a low-degree extension (LDE) of the provided execution trace over the specified
/// domain and builds a commitment to the extended trace.
///
/// The extension is performed by interpolating each column of the execution trace into a
/// polynomial of degree = trace_length - 1, and then evaluating the polynomial over the LDE
/// domain.
///
/// Trace commitment is computed by hashing each row of the extended execution trace, and then
/// building a Merkle tree from the resulting hashes.
///
/// Interpolations, evaluations and hashes are simultaneously
/// computed on the GPU
/// ```
fn build_trace_commitment<
    E: FieldElement<BaseField = Felt>,
    H: Hasher<Digest = D> + ElementHasher<BaseField = E::BaseField>,
    D: Digest + for<'a> From<&'a [Felt; DIGEST_SIZE]>,
>(
    trace: &ColMatrix<E>,
    domain: &StarkDomain<Felt>,
    hash_fn: HashFn,
) -> (RowMatrix<E>, MerkleTree<H>, ColMatrix<E>) {
    let now = Instant::now();

    let lde_domain_size = domain.lde_domain_size();
    let blowup = lde_domain_size.ilog2() as usize;
    let num_base_columns = trace.num_base_cols();
    let partition_count = (2 *  num_base_columns - 1) / num_base_columns;

    // initialize the GPU
    init_gpu(lde_domain_size.ilog2() as usize, blowup).expect("Could not initialize GPU.");

    // allocate space to store all the results
    let mut lde_out = alloc_pinned(lde_domain_size * num_base_columns * size_of::<E>(), E::ZERO);
    let mut lde_out_transposed = alloc_pinned(lde_domain_size * num_base_columns * size_of::<E>(), E::ZERO);
    let mut hash_states_out = alloc_pinned(partition_count * CAPACITY * lde_domain_size * size_of::<E>(), D::default());
    let mut merkle_tree_out = alloc_pinned((2 * lde_domain_size - 1) * CAPACITY * size_of::<E>(), D::default());

    // Number of threads needs to be tested
    set_transpose(Transpose::Full, 4);

    let mut data = trace.columns().flat_map(|c| c.to_vec()).collect::<Vec<E>>();

    // all the outputs are written in column-major order
    lde_into_rescue_batch(Trace {
        tmp: Some(&mut lde_out_transposed),
        data: data.as_mut_slice(),
        num_cols: trace.num_cols(),
        domain_size: trace.num_rows(),
    },
        CudaOptions { blowup_factor: blowup, partition_size: trace.num_cols() as u32 },
        &mut lde_out,
        &mut hash_states_out,
        &mut merkle_tree_out,
        hash_fn)
        .expect("failed to compute lde into rescue batch.");

    // NOTE: `merkle_tree_out` contains leaves.
    // NOTE: `hash_states_out` should be transposed to row-major if we want to match the merkle_tree order.
    // Leaves in merkle start at tree_offset:
    // let tree_offset = MERKLE_TREE_HEADER_SIZE + num_cols * lde_domain_size;

    let trace_lde = RowMatrix::<E>::new(lde_out.to_vec(), lde_out.len() / num_base_columns, lde_out.len() / num_base_columns);

    let tpolys: Vec<Vec<E>> = lde_out.chunks(lde_out.len() / num_base_columns).map(|e| e.to_vec()).collect();
    let trace_polys: ColMatrix<E> = ColMatrix::new(tpolys);

    let (nodes, leaves) = merkle_tree_out.split_at(MERKLE_TREE_HEADER_SIZE + num_base_columns * lde_domain_size);
    let trace_tree = MerkleTree::from_raw_parts(nodes.to_vec(), leaves.to_vec()).unwrap();

    event!(
            Level::INFO,
            "Extended and committed (on GPU) to an execution trace of {} columns from 2^{} to 2^{} steps in {} ms",
            trace.num_cols(),
            trace.num_rows().ilog2(),
            trace_lde.num_rows().ilog2(),
            now.elapsed().as_millis()
        );

    cleanup_gpu();

    (trace_lde, trace_tree, trace_polys)
}