use super::multivariate::{CompositionPolynomial, MultiLinear};
use super::sumcheck::PartialProof as SumcheckInstanceProof;

use alloc::sync::Arc;
use vm_core::{Felt, FieldElement};

mod prover;
mod verifier;

#[cfg(test)]
mod test;

/// Layered circuit for computing a sum of fractions.
///
/// The circuit computes a sum of fractions based on the formula a / c + b / d = (a * d + b * c) / (c * d)
/// which defines a "gate" ((a, b), (c, d)) --> (a * d + b * c, c * d) upon which the `FractionalSumCircuit`
/// is built.
#[derive(Debug)]
pub struct FractionalSumCircuit<E: FieldElement> {
    p_0_vec: Vec<MultiLinear<E>>,
    p_1_vec: Vec<MultiLinear<E>>,
    q_0_vec: Vec<MultiLinear<E>>,
    q_1_vec: Vec<MultiLinear<E>>,
}

impl<E: FieldElement> FractionalSumCircuit<E> {
    /// Computes The values of the gates outputs for each of the layers of the fractional sum circuit.
    pub fn new(num_den: &Vec<MultiLinear<E>>) -> Self {
        let mut p_0_vec: Vec<MultiLinear<E>> = Vec::new();
        let mut p_1_vec: Vec<MultiLinear<E>> = Vec::new();
        let mut q_0_vec: Vec<MultiLinear<E>> = Vec::new();
        let mut q_1_vec: Vec<MultiLinear<E>> = Vec::new();

        let num_layers = num_den[0].len().ilog2() as usize;

        p_0_vec.push(num_den[0].to_owned());
        p_1_vec.push(num_den[1].to_owned());
        q_0_vec.push(num_den[2].to_owned());
        q_1_vec.push(num_den[3].to_owned());

        for i in 0..num_layers {
            let (output_p_0, output_p_1, output_q_0, output_q_1) =
                FractionalSumCircuit::compute_layer(
                    &p_0_vec[i],
                    &p_1_vec[i],
                    &q_0_vec[i],
                    &q_1_vec[i],
                );
            p_0_vec.push(output_p_0);
            p_1_vec.push(output_p_1);
            q_0_vec.push(output_q_0);
            q_1_vec.push(output_q_1);
        }

        FractionalSumCircuit {
            p_0_vec,
            p_1_vec,
            q_0_vec,
            q_1_vec,
        }
    }

    /// Compute the output values of the layer given a set of input values
    fn compute_layer(
        inp_p_1: &MultiLinear<E>,
        inp_p_0: &MultiLinear<E>,
        inp_q_1: &MultiLinear<E>,
        inp_q_0: &MultiLinear<E>,
    ) -> (MultiLinear<E>, MultiLinear<E>, MultiLinear<E>, MultiLinear<E>) {
        let len = inp_q_1.len();
        let outp_p_1 = (0..len / 2)
            .map(|i| inp_p_1[i] * inp_q_0[i] + inp_p_0[i] * inp_q_1[i])
            .collect::<Vec<E>>();
        let outp_p_0 = (len / 2..len)
            .map(|i| inp_p_1[i] * inp_q_0[i] + inp_p_0[i] * inp_q_1[i])
            .collect::<Vec<E>>();
        let outp_q_1 = (0..len / 2).map(|i| inp_q_1[i] * inp_q_0[i]).collect::<Vec<E>>();
        let outp_q_0 = (len / 2..len).map(|i| inp_q_1[i] * inp_q_0[i]).collect::<Vec<E>>();

        (
            MultiLinear::new(outp_p_1),
            MultiLinear::new(outp_p_0),
            MultiLinear::new(outp_q_1),
            MultiLinear::new(outp_q_0),
        )
    }

    /// Given a value r, computes the evaluation of the last layer at r when interpreted as (two)
    /// multilinear polynomials.
    pub fn evaluate(&self, r: E) -> (E, E) {
        let len = self.p_0_vec.len();
        assert_eq!(self.p_0_vec[len - 1].num_variables(), 0);
        assert_eq!(self.p_1_vec[len - 1].num_variables(), 0);
        assert_eq!(self.q_0_vec[len - 1].num_variables(), 0);
        assert_eq!(self.q_1_vec[len - 1].num_variables(), 0);

        let mut p_ = self.p_0_vec[len - 1].clone();
        p_.extend(&self.p_1_vec[len - 1]);
        let mut q_ = self.q_0_vec[len - 1].clone();
        q_.extend(&self.q_1_vec[len - 1]);

        (p_.evaluate(&[r]), q_.evaluate(&[r]))
    }
}

/// A proof for reducing a claim on the correctness of the output of a layer to that of:
///
/// 1. Correctness of a sumcheck proof on the claimed output.
/// 2. Correctness of the evaluation of the input (to the said layer) at a random point when
/// interpreted as multilinear polynomial.
///
/// The verifier will then have to work backward and:
///
/// 1. Verify that the sumcheck proof is valid.
/// 2. Recurse on the (claimed evaluations) using the same approach as above.
///
/// Note that the following struct batches proofs for many circuits of the same type that
/// are independent i.e., parallel.
#[derive(Debug)]
pub struct LayerProof<E: FieldElement> {
    pub proof: SumcheckInstanceProof<E>,
    pub claims_sum_p0: E,
    pub claims_sum_p1: E,
    pub claims_sum_q0: E,
    pub claims_sum_q1: E,
}

#[derive(Debug)]
pub struct CircuitProof<E: FieldElement + 'static> {
    pub proof: Vec<LayerProof<E>>,
}

#[derive(Debug)]
pub struct GkrClaim<E: FieldElement + 'static> {
    evaluation_point: Vec<E>,
    claimed_evaluation: (E, E),
}

#[derive(Debug, Clone)]
pub struct GkrFinalEvaluationClaim<E: FieldElement + 'static> {
    pub evaluation_point: Vec<E>,
    pub ml_evaluations: Vec<E>,
}

fn evaluate_composition_polys<E: FieldElement<BaseField = Felt> + 'static>(
    mls: &Vec<MultiLinear<E>>,
    composition_polys: &Vec<Vec<Arc<dyn CompositionPolynomial<E>>>>,
) -> Vec<MultiLinear<E>> {
    let num_evaluations = 1 << mls[0].num_variables();

    let mut num_den: Vec<Vec<E>> = vec![vec![]; 4];
    for i in 0..num_evaluations {
        for j in 0..4 {
            let query: Vec<E> = mls.iter().map(|ml| ml[i]).collect();

            composition_polys[j].iter().for_each(|c| {
                let evaluation = c.as_ref().evaluate(&query);
                num_den[j].push(evaluation);
            });
        }
    }
    let input: Vec<MultiLinear<E>> =
        (0..4).map(|i| MultiLinear::from_values(&num_den[i])).collect();
    input
}
