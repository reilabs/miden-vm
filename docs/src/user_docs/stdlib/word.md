# Word procedures

Module `std::word` contains utilities for manipulating *words* &mdash; sequences of four *field elements*.
See [Terms and notations](./main.md#Terms-and-notations) for more information.

| Procedure      | Description   |
| -------------- | ------------- |
| reverse        | Reverses order of the first four elements on the stack.<br /><br />Inputs: `[a, b, c, d, ...]`<br />Output: `[d, c, b, a, ...]`<br /><br />Cycles: 3|
| eqz            | Returns a boolean indicating whether the input word `[0, 0, 0, 0]`.<br /><br />Inputs: `[INPUT_WORD]`<br />Outputs: `[is_word_zero, INPUT_WORD]`<br /><br />Where:<br />- `INPUT_WORD` is the word to compare against `[0, 0, 0, 0]`.<br />- `is_word_zero` is a boolean indicating whether `INPUT_WORD` is all zeros.<br /><br />Cycles: 11|
| gt             | Returns true if `LHS` is strictly greater than `RHS`, false otherwise.<br /><br />It compares words using the same ordering as Merkle tree key comparison. This implementation avoids branching for performance reasons.<br /><br />Inputs: `[LHS, RHS]`<br />Output: `[is_lhs_greater]`<br /><br />Cycles: 121 |
| gte            | Returns true if `LHS` is greater than or equal to `RHS`.<br /><br />Inputs: `[LHS, RHS]`<br />Outputs: `[is_lhs_greater_or_equal]`<br /><br />Cycles: 118 |
| lt             | Returns true if `LHS` is strictly less than `RHS`, false otherwise.<br /><br />The implementation avoids branching for performance reasons.<br /><br />From an implementation standpoint this is exactly the same as `word::gt` except it uses `lt` rather than `gt`. See its docs for details.<br /><br />Inputs: `[LHS, RHS]`<br />Output: `[is_lhs_lesser]`<br /><br />Cycles: 117 |
| lte            | Returns true if `LHS` is exactly equal to `RHS`, false otherwise.<br /><br />The implementation does not branch, and always performs the same number of comparisons.<br /><br />This is currently equivalent the `eqw` instruction.<br /><br />Inputs: `[LHS, RHS]`.<br />Output: `[lhs_eq_rhs]`<br /><br />Cycles: 26 |
