use super::{FinalEvaluationClaim, PartialProof};
use crate::gkr::utils::{barycentric_weights, evaluate_barycentric};
use vm_core::{Felt, FieldElement};
use winter_prover::crypto::{ElementHasher, RandomCoin};

pub fn sum_check_verify<
    E: FieldElement<BaseField = Felt>,
    C: RandomCoin<Hasher = H, BaseField = Felt>,
    H: ElementHasher<BaseField = Felt>,
>(
    claim: E,
    degree: usize,
    round_proofs: PartialProof<E>,
    coin: &mut C,
) -> FinalEvaluationClaim<E> {
    let points: Vec<E> = (0..degree + 1).map(|x| E::from(x as u8)).collect();
    let mut claimed_evaluation = claim;
    let mut evaluation_point = vec![];
    for proof in round_proofs.round_proofs {
        let partial_evals = proof.poly_evals.clone();
        coin.reseed(H::hash_elements(&partial_evals));

        // get r
        let r: E = coin.draw().unwrap();
        let evals = proof.to_evals(claimed_evaluation);

        let point_evals: Vec<_> = points.iter().zip(evals.iter()).map(|(x, y)| (*x, *y)).collect();
        let weights = barycentric_weights(&point_evals);
        claimed_evaluation = evaluate_barycentric(&point_evals, r, &weights);
        evaluation_point.push(r);
    }
    FinalEvaluationClaim {
        evaluation_point,
        claimed_evaluation,
    }
}
