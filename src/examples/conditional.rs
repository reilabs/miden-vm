use super::{utils::parse_args, Example};
use distaff::{assembly, ProgramInputs};
use winterfell::math::{fields::f128::BaseElement, StarkField};

pub fn get_example(args: &[String]) -> Example {
    // get flag value and proof options from the arguments
    let (flag, options) = parse_args(args);
    let flag = BaseElement::new(flag as u128);

    // determine the expected result
    let expected_result = match flag.as_int() {
        0 => BaseElement::new(15),
        1 => BaseElement::new(8),
        _ => panic!("flag must be a binary value"),
    };

    // construct the program which either adds or multiplies two numbers
    // based on the value provided via secret inputs
    let program = assembly::compile(
        "
    begin
        push.3
        push.5
        read
        if.true
            add
        else
            mul
        end
    end",
    )
    .unwrap();

    println!(
        "Generated a program to test conditional execution; expected result: {}",
        expected_result
    );

    // put the flag as the only secret input for tape A
    let inputs = ProgramInputs::new(&[], &[flag], &[]);

    // a single element from the top of the stack will be the output
    let num_outputs = 1;

    Example {
        program,
        inputs,
        options,
        expected_result: vec![expected_result],
        num_outputs,
    }
}
