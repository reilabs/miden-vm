use core::cmp::Ordering;

use miden_core::{Felt, LexicographicWord, Word};
use miden_utils_testing::rand;
use num::Integer;
use rstest::rstest;

/// Note that adding a word to the *beginning* of a Vec adds it to the *bottom* of the stack.
fn prepend_word(stack: &mut Vec<u64>, word: Word) {
    // Actual insertion happens when this iterator is dropped.
    let _iterator = stack.splice(0..0, word.iter().map(Felt::as_int));
}

#[rstest]
#[case::gt("gt", &[Ordering::Greater])]
#[case::gte("gte", &[Ordering::Greater, Ordering::Equal])]
#[case::eq("eq", &[Ordering::Equal])]
#[case::lt("lt", &[Ordering::Less])]
#[case::lte("lte", &[Ordering::Less, Ordering::Equal])]
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

    for i in 0..1000 {
        let lhs = rand::seeded_word(&mut seed);
        let rhs = if i.is_even() { rand::seeded_word(&mut seed) } else { lhs };

        let expected_cmp = LexicographicWord::cmp(&lhs.into(), &rhs.into());

        let mut operand_stack: Vec<u64> = Default::default();
        prepend_word(&mut operand_stack, rhs);
        prepend_word(&mut operand_stack, lhs);
        // => [RHS, LHS]

        let expected = u64::from(valid_ords.contains(&expected_cmp));

        build_test!(source, &operand_stack).expect_stack(&[expected]);
    }
}

#[test]
fn test_reverse() {
    const SOURCE: &str = "
        use.std::word

        begin
            exec.word::reverse
        end
    ";

    let mut seed = 0xfacade;
    for _ in 0..1000 {
        let mut operand_stack: Vec<u64> = Default::default();
        prepend_word(&mut operand_stack, rand::seeded_word(&mut seed));

        // This looks extremely weird, but `build_test!()` and `expect_stack()` take opposite
        // stack orders, so this is actually correct.
        build_test!(SOURCE, &operand_stack).expect_stack(&operand_stack);
    }
}

#[test]
fn test_eqz() {
    const SOURCE: &str = "
        use.std::word

        begin
            exec.word::eqz
        end
    ";

    build_test!(SOURCE, &[0, 0, 0, 0]).expect_stack(&[1]);
    build_test!(SOURCE, &[0, 1, 2, 3]).expect_stack(&[0]);
}

#[test]
fn test_preserving_eqz() {
    const SOURCE: &str = "
        use.std::word
        use.std::sys

        begin
            exec.word::testz
            exec.sys::truncate_stack
        end
    ";

    build_test!(SOURCE, &[0, 0, 0, 0]).expect_stack(&[1, 0, 0, 0, 0]);
    build_test!(SOURCE, &[0, 1, 2, 3]).expect_stack(&[0, 3, 2, 1, 0]);
}
