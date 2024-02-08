use alloc::sync::Arc;
use vm_core::{Felt, FieldElement};
use winter_prover::crypto::{ElementHasher, RandomCoin};

use crate::gkr::sumcheck::{
    sum_check_verify, FinalEvaluationClaim_, PartialProof as SumCheckFullProof,
};

use crate::gkr::multivariate::{
    gkr_merge_composition_from_composition_polys, CompositionPolynomial, MultiLinear,
};
use crate::gkr::sumcheck::PartialProof as SumcheckInstanceProof;

use super::{CircuitProof, GkrFinalEvaluationClaim};

/// Checks the validity of a `LayerProof`.
///
/// It first reduces the 2 claims to 1 claim using randomness and then checks that the sumcheck
/// protocol was correctly executed.
///
/// The method outputs:
///
/// 1. A vector containing the randomness sent by the verifier throughout the course of the
/// sum-check protocol.
/// 2. The (claimed) evaluation of the inner polynomial (i.e., the one being summed) at the this random vector.
/// 3. The random value used in the 2-to-1 reduction of the 2 sumchecks.  
pub fn verify_sum_check_proof<
    E: FieldElement<BaseField = Felt> + 'static,
    C: RandomCoin<Hasher = H, BaseField = Felt>,
    H: ElementHasher<BaseField = Felt>,
>(
    proof: &SumcheckInstanceProof<E>,
    num_rounds_pre_switch: usize,
    max_degree_post_switch: usize,
    claim: (E, E),
    transcript: &mut C,
) -> (FinalEvaluationClaim_<E>, E) {
    // Absorb the claims
    let data = vec![claim.0, claim.1];
    transcript.reseed(H::hash_elements(&data));

    // Squeeze challenge to reduce two sumchecks to one
    let r_sum_check: E = transcript.draw().unwrap();

    // Run the sumcheck protocol

    // Given r_sum_check and claim, we create a Claim with the GKR composer and then call the generic sum-check verifier
    let reduced_claim = claim.0 + claim.1 * r_sum_check;

    let mut eval_point = vec![];

    let reduced_gkr_claim = if num_rounds_pre_switch > 0 {
        let (proof_pre, proof_post) = proof.split_at(num_rounds_pre_switch);
        let FinalEvaluationClaim_ {
            evaluation_point,
            claimed_evaluation,
        } = sum_check_verify(reduced_claim, 3, proof_pre.clone(), transcript);
        eval_point.extend_from_slice(&evaluation_point);
        let FinalEvaluationClaim_ {
            evaluation_point,
            claimed_evaluation,
        } = sum_check_verify(claimed_evaluation, max_degree_post_switch, proof_post, transcript);
        eval_point.extend_from_slice(&evaluation_point);

        FinalEvaluationClaim_ {
            evaluation_point: eval_point,
            claimed_evaluation,
        }
    } else {
        sum_check_verify(reduced_claim, 3, proof.clone(), transcript)
    };

    (reduced_gkr_claim, r_sum_check)
}

pub fn verify_virtual_bus<
    E: FieldElement<BaseField = Felt> + 'static,
    C: RandomCoin<Hasher = H, BaseField = Felt>,
    H: ElementHasher<BaseField = Felt>,
>(
    proof: CircuitProof<E>,
    composition_polys: Vec<Vec<Arc<dyn CompositionPolynomial<E>>>>,
    final_layer_proof: SumCheckFullProof<E>,
    gkr_final_eval_claim: GkrFinalEvaluationClaim<E>,
    claims_sum_vec: &[E],
    num_rounds_pre_switch: usize,
    max_degree_post_switch: usize,
    transcript: &mut C,
) -> GkrFinalEvaluationClaim<E> {
    let num_layers = proof.proof.len() as usize;
    let mut rand: Vec<E> = Vec::new();

    // Check that a/b + d/e is equal to 0
    assert_ne!(claims_sum_vec[2], E::ZERO);
    assert_ne!(claims_sum_vec[3], E::ZERO);
    assert_eq!(
        claims_sum_vec[0] * claims_sum_vec[3] + claims_sum_vec[1] * claims_sum_vec[2],
        E::ZERO
    );

    let data = claims_sum_vec;
    transcript.reseed(H::hash_elements(&data));

    let r_cord = transcript.draw().unwrap();

    let p_poly_coef = vec![claims_sum_vec[0], claims_sum_vec[1]];
    let q_poly_coef = vec![claims_sum_vec[2], claims_sum_vec[3]];

    let p_poly = MultiLinear::new(p_poly_coef);
    let q_poly = MultiLinear::new(q_poly_coef);
    let p_eval = p_poly.evaluate(&[r_cord]);
    let q_eval = q_poly.evaluate(&[r_cord]);

    let mut reduced_claim = (p_eval, q_eval);

    // I) Verify all GKR layers but for the last one counting backwards.
    rand.push(r_cord);
    for (_num_rounds, i) in (0..num_layers).enumerate() {
        let (
            FinalEvaluationClaim_ {
                evaluation_point: rand_sumcheck,
                claimed_evaluation: claim_last,
            },
            r_two_sumchecks,
        ) = verify_sum_check_proof(&proof.proof[i].proof, 0, 0, reduced_claim, transcript);

        let claims_sum_p0 = &proof.proof[i].claims_sum_p0;
        let claims_sum_p1 = &proof.proof[i].claims_sum_p1;
        let claims_sum_q0 = &proof.proof[i].claims_sum_q0;
        let claims_sum_q1 = &proof.proof[i].claims_sum_q1;

        let data = vec![
            claims_sum_p0.clone(),
            claims_sum_p1.clone(),
            claims_sum_q0.clone(),
            claims_sum_q1.clone(),
        ];
        transcript.reseed(H::hash_elements(&data));

        assert_eq!(rand.len(), rand_sumcheck.len());

        let eq: E = (0..rand.len())
            .map(|i| rand[i] * rand_sumcheck[i] + (E::ONE - rand[i]) * (E::ONE - rand_sumcheck[i]))
            .fold(E::ONE, |acc, term| acc * term);

        let claim_expected: E = (*claims_sum_p0 * *claims_sum_q1
            + *claims_sum_p1 * *claims_sum_q0
            + r_two_sumchecks * *claims_sum_q0 * *claims_sum_q1)
            * eq;

        assert_eq!(claim_expected, claim_last);

        // Produce a random challenge to condense claims into a single claim
        let r_layer = transcript.draw().unwrap();

        reduced_claim = (
            *claims_sum_p0 + r_layer * (*claims_sum_p1 - *claims_sum_p0),
            *claims_sum_q0 + r_layer * (*claims_sum_q1 - *claims_sum_q0),
        );

        let mut ext = rand_sumcheck;
        ext.push(r_layer);
        rand = ext;
    }

    // II) Verify the final GKR layer counting backwards.
    let (
        FinalEvaluationClaim_ {
            evaluation_point: eval_point,
            claimed_evaluation,
        },
        r_sum_check,
    ) = verify_sum_check_proof(
        &final_layer_proof,
        num_rounds_pre_switch,
        max_degree_post_switch,
        reduced_claim,
        transcript,
    );

    // III) Execute the final evaluation check
    let GkrFinalEvaluationClaim {
        evaluation_point,
        ml_evaluations,
    } = gkr_final_eval_claim.clone();
    assert_eq!(eval_point, evaluation_point);

    let gkr_final_composed_ml = gkr_merge_composition_from_composition_polys(
        &composition_polys,
        r_sum_check,
        (&evaluation_point[..num_rounds_pre_switch]).to_vec(),
        1 << (num_layers + 1),
    );

    // TODO: use `rand` to compute the evaluation of the Lagrange kernel at `rand` at `evaluation_point`
    let final_eval = gkr_final_composed_ml.evaluate(&ml_evaluations);
    assert_eq!(final_eval, claimed_evaluation);

    // IV) Pass the claimed openings for verification by the STARK
    gkr_final_eval_claim
}
