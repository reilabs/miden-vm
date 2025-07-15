use core::iter;

use miden_core::{Felt, Word};
use miden_utils_testing::{ONE, StackInputs, ZERO, felt_slice_to_ints, rand::seeded_word};

#[test]
fn test_word_eq() {
    miden_utils_testing::init_logging();
    const SOURCE: &str = "
        use.std::word

        begin
            exec.word::eq
        end
    ";

    let mut seed = 0xf20;

    for _ in 0..256 {
        let lhs: Word = seeded_word(&mut seed);
        let rhs: Word = seeded_word(&mut seed);

        let operand_stack = iter::empty()
            .chain(rhs.iter().rev())
            .chain(lhs.iter().rev())
            .map(Felt::as_int)
            .collect::<Vec<_>>();

        let result_felt = if lhs == rhs { ONE } else { ZERO };

        let result = vec![result_felt.as_int()];

        build_debug_test!(SOURCE, &operand_stack).expect_stack(&result);
    }
}
