use super::{RoundProof, Witness};
use crate::gkr::sumcheck::{reduce_claim, PartialProof, RoundClaim, RoundOutput};
use vm_core::{Felt, FieldElement};
use winter_prover::crypto::{ElementHasher, RandomCoin};

pub fn sum_check_prove<
    E: FieldElement<BaseField = Felt>,
    C: RandomCoin<Hasher = H, BaseField = Felt>,
    H: ElementHasher<BaseField = Felt>,
>(
    claim: E,
    num_rounds: usize,
    witness: Witness<E>,
    coin: &mut C,
) -> (Vec<E>, Vec<E>, E, PartialProof<E>) {
    // Setup first round
    let mut prev_claim = RoundClaim {
        partial_eval_point: vec![],
        current_claim: claim.clone(),
    };
    let prev_proof = PartialProof {
        round_proofs: vec![],
    };
    let prev_output = RoundOutput {
        proof: prev_proof,
        witness,
    };

    let mut output = sumcheck_round(prev_output);
    let poly_evals = &output.proof.round_proofs[0].poly_evals;
    coin.reseed(H::hash_elements(&poly_evals));

    for i in 1..num_rounds {
        let round_challenge = coin.draw().unwrap();
        let new_claim = reduce_claim(
            output.proof.round_proofs.last().unwrap().clone(),
            prev_claim,
            round_challenge,
        );
        output.witness.polynomial = output.witness.polynomial.bind(round_challenge);

        output = sumcheck_round(output);
        prev_claim = new_claim;

        let poly_evals = &output.proof.round_proofs[i].poly_evals;
        coin.reseed(H::hash_elements(&poly_evals));
    }

    let round_challenge = coin.draw().unwrap();
    output.witness.polynomial = output.witness.polynomial.bind(round_challenge);
    let witness_evals: Vec<E> = output
        .witness
        .polynomial
        .multi_linears
        .iter()
        .flat_map(|ml| ml.evaluations.clone())
        .collect();

    let RoundClaim {
        partial_eval_point,
        current_claim,
    } = reduce_claim(
        output.proof.round_proofs.last().unwrap().clone(),
        prev_claim,
        round_challenge,
    );

    (witness_evals, partial_eval_point, current_claim, output.proof)
}

fn sumcheck_round<E: FieldElement>(prev_proof: RoundOutput<E>) -> RoundOutput<E> {
    let RoundOutput { mut proof, witness } = prev_proof;

    let polynomial = witness.polynomial;
    let num_ml = polynomial.num_ml();
    let num_vars = polynomial.num_variables_ml();
    let num_rounds = num_vars - 1;

    let mut evals_zero = vec![E::ZERO; num_ml];
    let mut evals_one = vec![E::ZERO; num_ml];
    let mut deltas = vec![E::ZERO; num_ml];
    let mut evals_x = vec![E::ZERO; num_ml];

    let total_evals = (0..1 << num_rounds).into_iter().map(|i| {
        for (j, ml) in polynomial.multi_linears.iter().enumerate() {
            evals_zero[j] = ml.evaluations[(i << 1) as usize];
            evals_one[j] = ml.evaluations[(i << 1) + 1];
        }
        let mut total_evals = vec![E::ZERO; polynomial.degree()];
        total_evals[0] = polynomial.composer.evaluate(&evals_one);
        evals_zero
            .iter()
            .zip(evals_one.iter().zip(deltas.iter_mut().zip(evals_x.iter_mut())))
            .for_each(|(a0, (a1, (delta, evx)))| {
                *delta = *a1 - *a0;
                *evx = *a1;
            });
        total_evals.iter_mut().skip(1).for_each(|e| {
            evals_x.iter_mut().zip(deltas.iter()).for_each(|(evx, delta)| {
                *evx += *delta;
            });
            *e = polynomial.composer.evaluate(&evals_x);
        });
        total_evals
    });
    let evaluations = total_evals.fold(vec![E::ZERO; polynomial.degree()], |mut acc, evals| {
        acc.iter_mut().zip(evals.iter()).for_each(|(a, ev)| *a += *ev);
        acc
    });
    let proof_update = RoundProof {
        poly_evals: evaluations,
    };
    proof.round_proofs.push(proof_update);
    RoundOutput {
        proof,
        witness: Witness { polynomial },
    }
}
