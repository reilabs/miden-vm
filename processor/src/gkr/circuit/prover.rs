use super::super::sumcheck::{
    PartialProof as SumcheckInstanceProof, RoundProof as SumCheckRoundProof, Witness,
};
use super::{CircuitProof, LayerProof};
use crate::gkr::circuit::{
    evaluate_composition_polys, FractionalSumCircuit, GkrClaim, GkrFinalEvaluationClaim,
};
use crate::gkr::multivariate::LagrangeKernel;
use crate::gkr::multivariate::{
    gkr_merge_composition_from_composition_polys, ComposedMultiLinears, CompositionPolynomial,
    MultiLinear,
};
use crate::gkr::sumcheck::{sum_check_prove, PartialProof as SumCheckFullProof};
use alloc::sync::Arc;
use vm_core::{Felt, FieldElement};
use winter_prover::crypto::{ElementHasher, RandomCoin};

pub fn prove_virtual_bus<
    E: FieldElement<BaseField = Felt> + 'static,
    C: RandomCoin<Hasher = H, BaseField = Felt>,
    H: ElementHasher<BaseField = Felt>,
>(
    composition_polys: Vec<Vec<Arc<dyn CompositionPolynomial<E>>>>,
    num_rounds_pre_switch: usize,
    mls: &mut Vec<MultiLinear<E>>,
    transcript: &mut C,
) -> (Vec<E>, CircuitProof<E>, SumCheckFullProof<E>, GkrFinalEvaluationClaim<E>) {
    // I) Evaluate the numerators and denominators over the boolean hyper-cube
    let input: Vec<MultiLinear<E>> = evaluate_composition_polys(&mls, &composition_polys);

    // II) Evaluate the GKR fractional sum circuit
    let mut circuit = FractionalSumCircuit::new(&input);

    // III) Run the GKR prover for all layers except the last one counting backwards
    let (
        gkr_proofs,
        GkrClaim {
            evaluation_point,
            claimed_evaluation,
        },
    ) = prove_before_final_circuit_layer(&mut circuit, transcript);

    // IV) Run the sum-check prover for the last GKR layer (counting backwards i.e., first layer
    // in the circuit)

    // 1) Run a sum-check instance for the merged virtual polynomials for `num_rounds_pre_switch` rounds.
    // In the `num_rounds_pre_switch` first variables, each of the 4 virtual polynomials is linear and
    // thus the total degree in each of these variables is at most 3.

    // a) Build the EQ polynomial (Lagrange kernel) at the randomness sampled during the previous
    // sum-check protocol run
    let mut poly_x = LagrangeKernel::new_ml(evaluation_point.clone());

    // b) Get the mls representing the 4 merged virtual polynomials
    let mut poly_a = circuit.p_0_vec[0].to_owned();
    let mut poly_b = circuit.p_1_vec[0].to_owned();
    let mut poly_c = circuit.q_0_vec[0].to_owned();
    let mut poly_d = circuit.q_1_vec[0].to_owned();
    let merged_mls = (&mut poly_a, &mut poly_b, &mut poly_c, &mut poly_d, &mut poly_x);

    // c) Run the pre-switch sum-check protocol for `num_rounds_pre_switch` rounds
    let comb_func = |a: &E, b: &E, c: &E, d: &E, x: &E, rho: &E| -> E {
        (*a * *d + *b * *c + *rho * *c * *d) * *x
    };
    let (mut proof_pre_switch, mut rand_merge, _, r_sum_check, reduced_claim) =
        sum_check_prover_plain(
            claimed_evaluation,
            num_rounds_pre_switch,
            merged_mls,
            comb_func,
            transcript,
        );

    assert_eq!(poly_x.evaluations.len(), poly_a.evaluations.len());
    mls.push(poly_x);

    // 2) Run the sum-check protocol on the merged virtual polynomials with the first `num_rounds_pre_switch`
    // fixed to the randomness generated during the course of the pre-switch sum-check.

    // a) Create the composed ML post-switch
    let gkr_composition = gkr_merge_composition_from_composition_polys(
        &composition_polys,
        r_sum_check,
        rand_merge.clone(),
        1 << mls[0].num_variables,
    );

    // b) Create the claim for the post-switch sum-check protocol.
    let claim = reduced_claim;

    // c) Create the witness for the post-switch sum-check claim.
    let composed_ml = ComposedMultiLinears::new(Arc::new(gkr_composition.clone()), mls.to_vec());
    let witness = Witness {
        polynomial: composed_ml,
    };

    // d) Run the post switch sum-check
    let (ml_evaluations, evaluation_point, _claimed_evaluation, proof_post_switch) =
        sum_check_prove(claim, witness.polynomial.num_variables_ml(), witness, transcript);

    // V) Create the prover output

    // a) Create the final GKR evaluation claim
    rand_merge.extend_from_slice(&evaluation_point);

    let gkr_final_eval = GkrFinalEvaluationClaim {
        evaluation_point: rand_merge,
        ml_evaluations,
    };

    // b) Create the final GKR proof
    proof_pre_switch.round_proofs.extend_from_slice(&proof_post_switch.round_proofs);
    let proof_final_layer = proof_pre_switch;

    // c) Create the claimed output of the circuit.
    let circuit_outputs = vec![
        circuit.p_0_vec.last().unwrap()[0],
        circuit.p_1_vec.last().unwrap()[0],
        circuit.q_0_vec.last().unwrap()[0],
        circuit.q_1_vec.last().unwrap()[0],
    ];

    // d) Return:
    //  1. The claimed circuit outputs.
    //  2. GKR proofs of all circuit layers except the initial layer.
    //  3. Output of the final sum-check protocol.
    (circuit_outputs, gkr_proofs, proof_final_layer, gkr_final_eval)
}

fn prove_before_final_circuit_layer<
    E: FieldElement<BaseField = Felt> + 'static,
    C: RandomCoin<Hasher = H, BaseField = Felt>,
    H: ElementHasher<BaseField = Felt>,
>(
    circuit: &mut FractionalSumCircuit<E>,
    transcript: &mut C,
) -> (CircuitProof<E>, GkrClaim<E>) {
    let mut proof_layers: Vec<LayerProof<E>> = Vec::new();
    let num_layers = circuit.p_1_vec.len();

    let data = vec![
        circuit.p_0_vec.last().unwrap()[0],
        circuit.p_1_vec.last().unwrap()[0],
        circuit.q_0_vec.last().unwrap()[0],
        circuit.q_1_vec.last().unwrap()[0],
    ];
    transcript.reseed(H::hash_elements(&data));

    // Challenge to reduce p1, p0, q1, q0 to pr, qr
    let r_cord = transcript.draw().unwrap();

    // Compute the (2-to-1 folded) claim
    let mut claims_to_verify = circuit.evaluate(r_cord);
    let mut all_rand = Vec::new();

    let mut rand = Vec::new();
    rand.push(r_cord);
    for layer_id in (1..num_layers - 1).rev() {
        let len = circuit.p_1_vec[layer_id].len();

        // Construct the Lagrange kernel evaluated at previous GKR round randomness.
        // TODO: Treat the direction of doing sum-check more robustly.
        let eq_evals = LagrangeKernel::new(rand.clone()).evaluations();
        let mut poly_x = MultiLinear::from_values(&eq_evals);
        assert_eq!(poly_x.len(), len);

        let num_rounds = poly_x.len().ilog2() as usize;

        // 1. A is a polynomial containing the evaluations `p_1`.
        // 2. B is a polynomial containing the evaluations `p_0`.
        // 3. C is a polynomial containing the evaluations `q_1`.
        // 4. D is a polynomial containing the evaluations `q_0`.
        let poly_a: &mut MultiLinear<E>;
        let poly_b: &mut MultiLinear<E>;
        let poly_c: &mut MultiLinear<E>;
        let poly_d: &mut MultiLinear<E>;
        poly_a = &mut circuit.p_0_vec[layer_id];
        poly_b = &mut circuit.p_1_vec[layer_id];
        poly_c = &mut circuit.q_0_vec[layer_id];
        poly_d = &mut circuit.q_1_vec[layer_id];

        let poly_vec = (poly_a, poly_b, poly_c, poly_d, &mut poly_x);

        let claim = claims_to_verify;

        // The (non-linear) polynomial combining the multilinear polynomials
        let comb_func = |a: &E, b: &E, c: &E, d: &E, x: &E, rho: &E| -> E {
            (*a * *d + *b * *c + *rho * *c * *d) * *x
        };

        // Run the sumcheck protocol
        let (proof, rand_sumcheck, claims_sum, _, _) =
            sum_check_prover_plain(claim, num_rounds, poly_vec, comb_func, transcript);

        let (claims_sum_p1, claims_sum_p0, claims_sum_q1, claims_sum_q0, _claims_eq) = claims_sum;

        let data = vec![claims_sum_p1, claims_sum_p0, claims_sum_q1, claims_sum_q0];
        transcript.reseed(H::hash_elements(&data));

        // Produce a random challenge to condense claims into a single claim
        let r_layer = transcript.draw().unwrap();

        claims_to_verify = (
            claims_sum_p1 + r_layer * (claims_sum_p0 - claims_sum_p1),
            claims_sum_q1 + r_layer * (claims_sum_q0 - claims_sum_q1),
        );

        // Collect the randomness used for the current layer in order to construct the random
        // point where the input multilinear polynomials were evaluated.
        let mut ext = rand_sumcheck;
        ext.push(r_layer);
        all_rand.push(rand);
        rand = ext;

        proof_layers.push(LayerProof {
            proof,
            claims_sum_p0: claims_sum_p1,
            claims_sum_p1: claims_sum_p0,
            claims_sum_q0: claims_sum_q1,
            claims_sum_q1: claims_sum_q0,
        });
    }
    let gkr_claim = GkrClaim {
        evaluation_point: rand.clone(),
        claimed_evaluation: claims_to_verify,
    };

    (
        CircuitProof {
            proof: proof_layers,
        },
        gkr_claim,
    )
}

fn sum_check_prover_plain<
    E: FieldElement<BaseField = Felt>,
    C: RandomCoin<Hasher = H, BaseField = Felt>,
    H: ElementHasher<BaseField = Felt>,
>(
    claim: (E, E),
    num_rounds: usize,
    ml_polys: (
        &mut MultiLinear<E>,
        &mut MultiLinear<E>,
        &mut MultiLinear<E>,
        &mut MultiLinear<E>,
        &mut MultiLinear<E>,
    ),
    comb_func: impl Fn(&E, &E, &E, &E, &E, &E) -> E,
    transcript: &mut C,
) -> (SumcheckInstanceProof<E>, Vec<E>, (E, E, E, E, E), E, E) {
    // Absorb the claims
    let data = vec![claim.0, claim.1];
    transcript.reseed(H::hash_elements(&data));

    // Squeeze challenge to reduce two sumchecks to one
    let r_sum_check = transcript.draw().unwrap();

    let (poly_a, poly_b, poly_c, poly_d, poly_x) = ml_polys;

    let mut e = claim.0 + claim.1 * r_sum_check;

    let mut r: Vec<E> = Vec::new();
    let mut round_proofs: Vec<SumCheckRoundProof<E>> = Vec::new();

    for _j in 0..num_rounds {
        let evals: (E, E, E) = {
            let mut eval_point_0 = E::ZERO;
            let mut eval_point_2 = E::ZERO;
            let mut eval_point_3 = E::ZERO;

            let len = poly_a.len() / 2;
            for i in 0..len {
                // The interpolation formula for a linear function is:
                // z * A(x) + (1 - z) * A (y)
                // z * A(1) + (1 - z) * A(0)

                // eval at z = 0: A(1)
                eval_point_0 += comb_func(
                    &poly_a[i << 1],
                    &poly_b[i << 1],
                    &poly_c[i << 1],
                    &poly_d[i << 1],
                    &poly_x[i << 1],
                    &r_sum_check,
                );

                let poly_a_u = poly_a[(i << 1) + 1];
                let poly_a_v = poly_a[i << 1];
                let poly_b_u = poly_b[(i << 1) + 1];
                let poly_b_v = poly_b[i << 1];
                let poly_c_u = poly_c[(i << 1) + 1];
                let poly_c_v = poly_c[i << 1];
                let poly_d_u = poly_d[(i << 1) + 1];
                let poly_d_v = poly_d[i << 1];
                let poly_x_u = poly_x[(i << 1) + 1];
                let poly_x_v = poly_x[i << 1];

                // eval at z = 2: 2 * A(1) - A(0)
                let poly_a_extrapolated_point = poly_a_u + poly_a_u - poly_a_v;
                let poly_b_extrapolated_point = poly_b_u + poly_b_u - poly_b_v;
                let poly_c_extrapolated_point = poly_c_u + poly_c_u - poly_c_v;
                let poly_d_extrapolated_point = poly_d_u + poly_d_u - poly_d_v;
                let poly_x_extrapolated_point = poly_x_u + poly_x_u - poly_x_v;
                eval_point_2 += comb_func(
                    &poly_a_extrapolated_point,
                    &poly_b_extrapolated_point,
                    &poly_c_extrapolated_point,
                    &poly_d_extrapolated_point,
                    &poly_x_extrapolated_point,
                    &r_sum_check,
                );

                // eval at z = 3: 3 * A(1) - 2 * A(0) = 2 * A(1) - A(0) + A(1) - A(0)
                // hence we can compute the evaluation at z + 1 from that of z for z > 1
                let poly_a_extrapolated_point = poly_a_extrapolated_point + poly_a_u - poly_a_v;
                let poly_b_extrapolated_point = poly_b_extrapolated_point + poly_b_u - poly_b_v;
                let poly_c_extrapolated_point = poly_c_extrapolated_point + poly_c_u - poly_c_v;
                let poly_d_extrapolated_point = poly_d_extrapolated_point + poly_d_u - poly_d_v;
                let poly_x_extrapolated_point = poly_x_extrapolated_point + poly_x_u - poly_x_v;

                eval_point_3 += comb_func(
                    &poly_a_extrapolated_point,
                    &poly_b_extrapolated_point,
                    &poly_c_extrapolated_point,
                    &poly_d_extrapolated_point,
                    &poly_x_extrapolated_point,
                    &r_sum_check,
                );
            }

            (eval_point_0, eval_point_2, eval_point_3)
        };

        let eval_0 = evals.0;
        let eval_2 = evals.1;
        let eval_3 = evals.2;

        let evals = vec![e - eval_0, eval_2, eval_3];
        let compressed_poly = SumCheckRoundProof { poly_evals: evals };

        // append the prover's message to the transcript
        transcript.reseed(H::hash_elements(&compressed_poly.poly_evals));

        // derive the verifier's challenge for the next round
        let r_j = transcript.draw().unwrap();
        r.push(r_j);

        poly_a.bind_assign(r_j);
        poly_b.bind_assign(r_j);
        poly_c.bind_assign(r_j);
        poly_d.bind_assign(r_j);

        poly_x.bind_assign(r_j);

        e = compressed_poly.evaluate(e, r_j);

        round_proofs.push(compressed_poly);
    }
    let claims_sum = (poly_a[0], poly_b[0], poly_c[0], poly_d[0], poly_x[0]);

    (SumcheckInstanceProof { round_proofs }, r, claims_sum, r_sum_check, e)
}
