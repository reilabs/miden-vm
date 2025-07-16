use core::cmp::Ordering;

use miden_core::{Felt, LexicographicWord, Word};
use miden_utils_testing::rand;
use rstest::rstest;

/// Note that adding a word to the *beginning* of a Vec adds it to the *bottom* of the stack.
fn prepend_word(stack: &mut Vec<u64>, word: Word) {
    // Actual insertion happens when this iterator is dropped.
    let _iterator = stack.splice(0..0, word.iter().map(Felt::as_int));
}

#[rstest]
#[case("gt", &[Ordering::Greater])]
#[case("gte", &[Ordering::Greater, Ordering::Equal])]
#[case("eq", &[Ordering::Equal])]
#[case("lt", &[Ordering::Less])]
#[case("lte", &[Ordering::Less, Ordering::Equal])]
fn test_word_comparison(#[case] proc_name: &str, #[case] valid_ords: &[Ordering]) {
    let source = &format!(
        "
        use.std::word

        begin
            exec.word::{proc_name}
        end
    "
    );

    let mut seed = 0xfacade;

    for _ in 0..1000 {
        let lhs = rand::seeded_word(&mut seed);
        let rhs = rand::seeded_word(&mut seed);

        let expected_cmp = LexicographicWord::cmp(&lhs.into(), &rhs.into());

        let mut operand_stack: Vec<u64> = Default::default();
        prepend_word(&mut operand_stack, lhs);
        prepend_word(&mut operand_stack, rhs);
        // => [LHS, RHS]

        let expected = u64::from(valid_ords.contains(&expected_cmp));

        build_test!(source, &operand_stack).expect_stack(&[expected]);
    }
}
