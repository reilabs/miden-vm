use crate::{
    gkr::{CompositionPolynomial, LagrangeKernel, MultiLinear},
    utils::get_trace_len,
    CodeBlock, DefaultHost, ExecutionOptions, ExecutionTrace, Kernel, Operation, Process,
    StackInputs, Vec,
};
use alloc::sync::Arc;
use core::marker::PhantomData;
use miden_air::trace::{
    chiplets::MEMORY_D0_COL_IDX,
    decoder::{DECODER_OP_BITS_OFFSET, DECODER_USER_OP_HELPERS_OFFSET},
    range::{M_COL_IDX, V_COL_IDX},
    CHIPLETS_OFFSET, CHIPLETS_WIDTH,
};
use test_utils::{crypto::Rpo256, rand::rand_value};
use vm_core::{
    crypto::random::RpoRandomCoin, CodeBlockTable, Felt, FieldElement, StarkField, ONE, ZERO,
};
use winter_prover::{crypto::RandomCoin, math::log2};

#[test]
fn test() {
    // --- bunch of ops requiring range checks ----------------------------------------------------
    let e = 10;
    let stack: Vec<_> = (0..(1 << e)).into_iter().collect();
    let operations: Vec<_> = (0..(1 << e))
        .flat_map(|_| {
            vec![Operation::U32split, Operation::U32add, Operation::U32xor, Operation::MStoreW]
        })
        .collect();

    let (trace, trace_len) = build_full_trace(&stack, operations, Kernel::default());

    // TODO: this should be generated using the transcript up to when the prover sends the commitment
    // of the main trace.
    let alphas = vec![rand_value()];

    // the composition polynomials defining the numerators/denominators
    let composition_polys: Vec<Vec<Arc<dyn CompositionPolynomial<Felt>>>> = vec![
        // left num
        vec![
            Arc::new(RangeCheckMultiplicity::default()),
            Arc::new(ZeroNumerator::default()),
            Arc::new(MemoryFlagChiplet::default()),
            Arc::new(MemoryFlagChiplet::default()),
        ],
        // right num
        vec![
            Arc::new(U32RangeCheckFlag::default()),
            Arc::new(U32RangeCheckFlag::default()),
            Arc::new(U32RangeCheckFlag::default()),
            Arc::new(U32RangeCheckFlag::default()),
        ],
        // left den
        vec![
            Arc::new(TableValue::new(alphas.clone())),
            Arc::new(OneDenominator::default()),
            Arc::new(MemoryValue::new(0, alphas.clone())),
            Arc::new(MemoryValue::new(1, alphas.clone())),
        ],
        // right den
        vec![
            Arc::new(StackValue::new(0, alphas.clone())),
            Arc::new(StackValue::new(1, alphas.clone())),
            Arc::new(StackValue::new(2, alphas.clone())),
            Arc::new(StackValue::new(3, alphas.clone())),
        ],
    ];

    let mut mls = vec![];
    for col in trace {
        let ml = MultiLinear::from_values(&col);
        mls.push(ml)
    }

    // run the GKR prover to obtain:
    // 1. The fractional sum circuit output.
    // 2. GKR proofs up to the last circuit layer counting backwards.
    // 3. GKR proof (i.e., a sum-check proof) for the last circuit layer counting backwards.
    // 4. A claim on the evaluation of the oracles at a random query point. This is used by the
    // verifier to the final check of the final sum-check. These evaluation are then checked by
    // the STARK verifier using two auxiliary columns to simulate openings of a multi-linear
    // commitment.
    let seed = [Felt::ZERO; 4]; // TODO: should be initialized with the appropriate transcript
    let mut transcript = RpoRandomCoin::new(seed.into());
    let num_rounds_pre_switch = log2(composition_polys[0].len()) as usize;
    let (circuit_outputs, gkr_before_last_proof, final_layer_proof, gkr_final_evaluation_claim) =
        crate::gkr::circuit::prove_v_bus_(
            composition_polys.clone(),
            num_rounds_pre_switch,
            &mut mls,
            &mut transcript,
        );

    let seed = [Felt::ZERO; 4];
    let mut transcript = RpoRandomCoin::new(seed.into());

    // run the GKR verifier to obtain a final evaluation claim. This last one is composed of:
    //
    // 1. Randomness defining the point of evaluation of the Lagrange kernel.
    // 2. The claimed openings of the oracles at the previous point.
    //
    // This final claim is then proven using the STARK in the next step.
    let gkr_final_evaluation_claim = crate::gkr::circuit::verify_v_bus_(
        gkr_before_last_proof,
        composition_polys.clone(),
        final_layer_proof,
        gkr_final_evaluation_claim,
        &circuit_outputs,
        num_rounds_pre_switch,
        5,
        &mut transcript,
    );

    // the final verification step is querying the oracles for the openings at the evaluation point. This will be done by the
    // (outer STARK) prover using:
    //      a. The Lagrange kernel (auxiliary) column at the evaluation point.
    //      b. An extra (auxiliary) column to compute an inner product between two vectors, the first
    //      being the Lagrange kernel and the second being  (\sum_{j=0}^l mls[j][i] * \lambda_j)_{i\in\{0,..,n\}

    // The query (i.e., openning) point
    let evaluation_point = gkr_final_evaluation_claim.evaluation_point.clone();

    // These are the (claimed) queries to the oracles.
    let ml_evals = gkr_final_evaluation_claim.ml_evaluations;

    // The verifier absorbs the claimed openings and generates batching randomness lambda
    transcript.reseed(Rpo256::hash_elements(&ml_evals));
    let lambdas: Vec<Felt> = (0..ml_evals.len()).map(|_i| transcript.draw().unwrap()).collect();

    // Compute the opening of the batched oracles
    let batched_query = (0..ml_evals.len())
        .map(|i| ml_evals[i] * lambdas[i])
        .fold(Felt::ZERO, |acc, term| acc + term);

    // The prover generates the Lagrange kernel as an auxiliary column
    let evaluation_point = evaluation_point[num_rounds_pre_switch..].to_vec();
    let lagrange_kernel = LagrangeKernel::new(evaluation_point.to_vec()).evaluations();

    // The prover generates the additional auxiliary column for the inner product
    let tmp_col: Vec<Felt> = (0..mls[0].len())
        .map(|i| {
            (0..ml_evals.len())
                .map(|j| mls[j][i] * lambdas[j])
                .fold(Felt::ZERO, |acc, term| acc + term)
        })
        .collect();

    let mut running_sum_col = vec![Felt::ZERO; tmp_col.len()];
    running_sum_col[0] = tmp_col[0] * lagrange_kernel[0];
    for i in 1..tmp_col.len() {
        running_sum_col[i] = running_sum_col[i - 1] + tmp_col[i] * lagrange_kernel[i];
    }

    // Boundary constraint to check correctness of openings
    assert_eq!(batched_query, *running_sum_col.last().unwrap());

    // Lagrange kernel constraints enforced by the STARK
    let trace_len = trace_len + 1;
    let nu = log2(trace_len);
    let g = Felt::get_root_of_unity(trace_len.ilog2());

    // Boundary constraints
    assert_eq!(
        evaluation_point.iter().fold(Felt::ONE, |acc, &term| (ONE - term) * acc),
        lagrange_kernel[0]
    );

    // Transition constraints
    for k in 1..=nu {
        let mut count_sub_group_elem = 0_usize;
        let divisor_k_minus_1: Vec<Felt> = (0..trace_len)
            .into_iter()
            .map(|i| g.exp(i as u64 * 2_u64.pow(k - 1)) - ONE)
            .collect();
        for row in (0..(1 << nu)).step_by(1 << (nu - (k - 1))) {
            let constraint = (evaluation_point[(nu - k) as usize]) * lagrange_kernel[row]
                - (ONE - evaluation_point[(nu - k) as usize])
                    * lagrange_kernel[row + 2_usize.pow(nu - k)];
            if divisor_k_minus_1[row] == ZERO {
                count_sub_group_elem += 1;
                assert_eq!(constraint, ZERO)
            } else {
                // not necessary nor necessarly true but should hold with high probability
                assert!(constraint != ZERO)
            }
        }
        assert_eq!(count_sub_group_elem, 1 << (k - 1))
    }
}

fn build_full_trace(
    stack_inputs: &[u64],
    operations: Vec<Operation>,
    kernel: Kernel,
) -> (Vec<Vec<Felt>>, usize) {
    let stack_inputs = StackInputs::try_from_values(stack_inputs.iter().copied()).unwrap();
    let host = DefaultHost::default();
    let mut process = Process::new(kernel, stack_inputs, host, ExecutionOptions::default());
    let program = CodeBlock::new_span(operations);
    process.execute_code_block(&program, &CodeBlockTable::default()).unwrap();
    let (trace, _, _) = ExecutionTrace::test_finalize_trace(process);
    let trace_len = get_trace_len(&trace) - ExecutionTrace::NUM_RAND_ROWS;

    (trace.to_vec().try_into().expect("failed to convert vector to array"), trace_len)
}

#[derive(Default)]
pub struct U32RangeCheckFlag<E>
where
    E: FieldElement,
{
    phantom: PhantomData<E>,
}

impl<E> CompositionPolynomial<E> for U32RangeCheckFlag<E>
where
    E: FieldElement,
{
    fn num_variables(&self) -> usize {
        1
    }

    fn max_degree(&self) -> usize {
        3
    }

    fn evaluate(&self, query: &[E]) -> E {
        let op_bit_4 = query[DECODER_OP_BITS_OFFSET + 4];
        let op_bit_5 = query[DECODER_OP_BITS_OFFSET + 5];
        let op_bit_6 = query[DECODER_OP_BITS_OFFSET + 6];

        (E::ONE - E::from(op_bit_4)) * (E::ONE - E::from(op_bit_5)) * E::from(op_bit_6)
    }
}

#[derive(Default)]
pub struct MemoryFlagChiplet<E>
where
    E: FieldElement,
{
    phantom: PhantomData<E>,
}

impl<E> CompositionPolynomial<E> for MemoryFlagChiplet<E>
where
    E: FieldElement,
{
    fn num_variables(&self) -> usize {
        1
    }

    fn max_degree(&self) -> usize {
        3
    }

    fn evaluate(&self, query: &[E]) -> E {
        let mem_selec0 = query[CHIPLETS_OFFSET];
        let mem_selec1 = query[CHIPLETS_OFFSET + 1];
        let mem_selec2 = query[CHIPLETS_OFFSET + 2];

        E::from(mem_selec0) * E::from(mem_selec1) * (E::ONE - E::from(mem_selec2))
    }
}

#[derive(Default)]
pub struct RangeCheckMultiplicity<E>
where
    E: FieldElement,
{
    phantom: PhantomData<E>,
}

impl<E> CompositionPolynomial<E> for RangeCheckMultiplicity<E>
where
    E: FieldElement,
{
    fn num_variables(&self) -> usize {
        1
    }

    fn max_degree(&self) -> usize {
        1
    }

    fn evaluate(&self, query: &[E]) -> E {
        query[M_COL_IDX]
    }
}

pub struct StackValue<E>
where
    E: FieldElement,
{
    i: usize,
    alphas: Vec<E>,
}

impl<E> StackValue<E>
where
    E: FieldElement,
{
    pub fn new(i: usize, alphas: Vec<E>) -> Self {
        assert!(i < 4);
        Self { i, alphas }
    }
}

impl<E> CompositionPolynomial<E> for StackValue<E>
where
    E: FieldElement,
{
    fn num_variables(&self) -> usize {
        1
    }

    fn max_degree(&self) -> usize {
        1
    }

    fn evaluate(&self, query: &[E]) -> E {
        -(self.alphas[0] - query[DECODER_USER_OP_HELPERS_OFFSET + self.i])
    }
}

pub struct MemoryValue<E>
where
    E: FieldElement,
{
    i: usize,
    alphas: Vec<E>,
}

impl<E> MemoryValue<E>
where
    E: FieldElement,
{
    pub fn new(i: usize, alphas: Vec<E>) -> Self {
        assert!(i < 2);
        Self { i, alphas }
    }
}

impl<E> CompositionPolynomial<E> for MemoryValue<E>
where
    E: FieldElement,
{
    fn num_variables(&self) -> usize {
        1
    }

    fn max_degree(&self) -> usize {
        1
    }

    fn evaluate(&self, query: &[E]) -> E {
        -(self.alphas[0] - query[MEMORY_D0_COL_IDX + self.i])
    }
}

pub struct TableValue<E>
where
    E: FieldElement,
{
    alphas: Vec<E>,
}

impl<E> TableValue<E>
where
    E: FieldElement,
{
    pub fn new(alphas: Vec<E>) -> Self {
        Self { alphas }
    }
}

impl<E> CompositionPolynomial<E> for TableValue<E>
where
    E: FieldElement,
{
    fn num_variables(&self) -> usize {
        1
    }

    fn max_degree(&self) -> usize {
        1
    }

    fn evaluate(&self, query: &[E]) -> E {
        self.alphas[0] - query[V_COL_IDX]
    }
}

/// For padding to the next power of 2.
#[derive(Default)]
pub struct ZeroNumerator<E>
where
    E: FieldElement,
{
    phantom: PhantomData<E>,
}

impl<E> CompositionPolynomial<E> for ZeroNumerator<E>
where
    E: FieldElement,
{
    fn num_variables(&self) -> usize {
        1
    }

    fn max_degree(&self) -> usize {
        1
    }

    fn evaluate(&self, _query: &[E]) -> E {
        E::ZERO
    }
}

/// For padding to the next power of 2.
#[derive(Default)]
pub struct OneDenominator<E>
where
    E: FieldElement,
{
    phantom: PhantomData<E>,
}

impl<E> CompositionPolynomial<E> for OneDenominator<E>
where
    E: FieldElement,
{
    fn num_variables(&self) -> usize {
        1
    }

    fn max_degree(&self) -> usize {
        1
    }

    fn evaluate(&self, _query: &[E]) -> E {
        E::ONE
    }
}
