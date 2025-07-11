#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

use alloc::{
    format,
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::iter;

use miden_assembly::{KernelLibrary, Library, Parse, diagnostics::reporting::PrintDiagnostic};
pub use miden_assembly::{
    LibraryPath,
    debuginfo::{DefaultSourceManager, SourceFile, SourceLanguage, SourceManager},
    diagnostics::Report,
};
pub use miden_core::{
    EMPTY_WORD, Felt, FieldElement, ONE, StackInputs, StackOutputs, StarkField, WORD_SIZE, Word,
    ZERO,
    chiplets::hasher::{STATE_WIDTH, hash_elements},
    stack::MIN_STACK_DEPTH,
    utils::{IntoBytes, ToElements, collections, group_slice_elements},
};
use miden_core::{ProgramInfo, chiplets::hasher::apply_permutation};
pub use miden_processor::{
    AdviceInputs, AdviceProvider, ContextId, ExecutionError, ExecutionOptions, ExecutionTrace,
    Process, ProcessState, VmStateIterator,
};
use miden_processor::{Program, fast::FastProcessor};
use miden_prover::utils::range;
pub use miden_prover::{MerkleTreeVC, ProvingOptions, prove};
pub use miden_verifier::{AcceptableOptions, VerifierError, verify};
pub use pretty_assertions::{assert_eq, assert_ne, assert_str_eq};
#[cfg(not(target_family = "wasm"))]
use proptest::prelude::{Arbitrary, Strategy};
pub use test_case::test_case;

pub mod math {
    pub use winter_prover::math::{
        ExtensionOf, FieldElement, StarkField, ToElements, fft, fields::QuadExtension, polynom,
    };
}

pub mod serde {
    pub use miden_core::utils::{
        ByteReader, ByteWriter, Deserializable, DeserializationError, Serializable, SliceReader,
    };
}

pub mod crypto;

pub mod host;
use host::TestHost;

#[cfg(not(target_family = "wasm"))]
pub mod rand;

mod test_builders;

#[cfg(not(target_family = "wasm"))]
pub use proptest;

// CONSTANTS
// ================================================================================================

/// A value just over what a [u32] integer can hold.
pub const U32_BOUND: u64 = u32::MAX as u64 + 1;

/// A source code of the `truncate_stack` procedure.
pub const TRUNCATE_STACK_PROC: &str = "
proc.truncate_stack.4
    loc_storew.0 dropw movupw.3
    sdepth neq.16
    while.true
        dropw movupw.3
        sdepth neq.16
    end
    loc_loadw.0
end
";

// TEST HANDLER
// ================================================================================================

/// Asserts that running the given assembler test will result in the expected error.
#[cfg(all(feature = "std", not(target_family = "wasm")))]
#[macro_export]
macro_rules! expect_assembly_error {
    ($test:expr, $(|)? $( $pattern:pat_param )|+ $( if $guard: expr )? $(,)?) => {
        let error = $test.compile().expect_err("expected assembly to fail");
        match error.downcast::<::miden_assembly::AssemblyError>() {
            Ok(error) => {
                ::miden_core::assert_matches!(error, $( $pattern )|+ $( if $guard )?);
            }
            Err(report) => {
                panic!(r#"
assertion failed (expected assembly error, but got a different type):
    left: `{:?}`,
    right: `{}`"#, report, stringify!($($pattern)|+ $(if $guard)?));
            }
        }
    };
}

/// Asserts that running the given execution test will result in the expected error.
#[cfg(all(feature = "std", not(target_family = "wasm")))]
#[macro_export]
macro_rules! expect_exec_error_matches {
    ($test:expr, $(|)? $( $pattern:pat_param )|+ $( if $guard: expr )? $(,)?) => {
        match $test.execute() {
            Ok(_) => panic!("expected execution to fail @ {}:{}", file!(), line!()),
            Err(error) => ::miden_core::assert_matches!(error, $( $pattern )|+ $( if $guard )?),
        }
    };
}

/// Like [miden_assembly::testing::assert_diagnostic], but matches each non-empty line of the
/// rendered output to a corresponding pattern.
///
/// So if the output has 3 lines, the second of which is empty, and you provide 2 patterns, the
/// assertion passes if the first line matches the first pattern, and the third line matches the
/// second pattern - the second line is ignored because it is empty.
#[cfg(not(target_family = "wasm"))]
#[macro_export]
macro_rules! assert_diagnostic_lines {
    ($diagnostic:expr, $($expected:expr),+) => {{
        use miden_assembly::testing::Pattern;
        let actual = format!("{}", miden_assembly::diagnostics::reporting::PrintDiagnostic::new_without_color($diagnostic));
        let lines = actual.lines().filter(|l| !l.trim().is_empty()).zip([$(Pattern::from($expected)),*].into_iter());
        for (actual_line, expected) in lines {
            expected.assert_match_with_context(actual_line, &actual);
        }
    }};
}

#[cfg(not(target_family = "wasm"))]
#[macro_export]
macro_rules! assert_assembler_diagnostic {
    ($test:ident, $($expected:literal),+) => {{
        let error = $test
            .compile()
            .expect_err("expected diagnostic to be raised, but compilation succeeded");
        assert_diagnostic_lines!(error, $($expected),*);
    }};

    ($test:ident, $($expected:expr),+) => {{
        let error = $test
            .compile()
            .expect_err("expected diagnostic to be raised, but compilation succeeded");
        assert_diagnostic_lines!(error, $($expected),*);
    }};
}

/// This is a container for the data required to run tests, which allows for running several
/// different types of tests.
///
/// Types of valid result tests:
/// - Execution test: check that running a program compiled from the given source has the specified
///   results for the given (optional) inputs.
/// - Proptest: run an execution test inside a proptest.
///
/// Types of failure tests:
/// - Assembly error test: check that attempting to compile the given source causes an AssemblyError
///   which contains the specified substring.
/// - Execution error test: check that running a program compiled from the given source causes an
///   ExecutionError which contains the specified substring.
#[derive(Debug)]
pub struct Test {
    pub source_manager: Arc<dyn SourceManager + Send + Sync>,
    pub source: Arc<SourceFile>,
    pub kernel_source: Option<Arc<SourceFile>>,
    pub stack_inputs: StackInputs,
    pub advice_inputs: AdviceInputs,
    pub in_debug_mode: bool,
    pub libraries: Vec<Library>,
    pub add_modules: Vec<(LibraryPath, String)>,
}

impl Test {
    // CONSTRUCTOR
    // --------------------------------------------------------------------------------------------

    /// Creates the simplest possible new test, with only a source string and no inputs.
    pub fn new(name: &str, source: &str, in_debug_mode: bool) -> Self {
        let source_manager = Arc::new(miden_assembly::DefaultSourceManager::default());
        let source = source_manager.load(SourceLanguage::Masm, name.into(), source.to_string());
        Test {
            source_manager,
            source,
            kernel_source: None,
            stack_inputs: StackInputs::default(),
            advice_inputs: AdviceInputs::default(),
            in_debug_mode,
            libraries: Vec::default(),
            add_modules: Vec::default(),
        }
    }

    /// Add an extra module to link in during assembly
    pub fn add_module(&mut self, path: miden_assembly::LibraryPath, source: impl ToString) {
        self.add_modules.push((path, source.to_string()));
    }

    // TEST METHODS
    // --------------------------------------------------------------------------------------------

    /// General purpose method to assert expectations of the final program state.
    ///
    /// [`Test::expect_stack()`] and [`Test::expect_stack_and_memory()`] are shortcuts for this
    /// method.
    #[track_caller]
    pub fn expect(&self, expectations: Expect) {
        let Expect { stack, mem, cycles } = expectations;

        // Compile the program.
        let (program, mut host) = self.get_program_and_host();

        let mut process = self.get_process(&program);

        // Execute the test.
        let stack_result = process.execute(&program, &mut host).unwrap();
        // Validate state.
        self.assert_outputs_with_fast_processor(&stack_result);

        // Validate expectations.

        if let Some((start_addr, mem)) = mem {
            for (addr, mem_value) in iter::zip(range(start_addr as usize, mem.len()), mem.iter()) {
                let mem_state = process
                    .chiplets
                    .memory
                    .get_value(ContextId::root(), addr as u32)
                    .unwrap_or(ZERO);

                assert_eq!(
                    *mem_value,
                    mem_state.as_int(),
                    "Expected memory [{addr}] => {mem_value:?}, found {mem_state:?}",
                );
            }
        }

        let trace = ExecutionTrace::new(process, stack_result);

        if let Some(stack) = stack {
            let found = trace.last_stack_state().as_int_vec();
            let expected = resize_to_min_stack_depth(stack);
            assert_eq!(expected, found, "Expected stack to be {expected:?}, found {found:?}");
        }

        if let Some(expected) = cycles {
            let found = trace.get_trace_len();
            assert_eq!(expected, found, "Expected execution cycles {expected:?}, found {found:?}");
        }
    }

    /// Builds a final stack from the provided stack-ordered array and asserts that executing the
    /// test will result in the expected final stack state.
    #[track_caller]
    pub fn expect_stack(&self, final_stack: &[u64]) {
        self.expect(Expect::with_stack(final_stack));
    }

    /// Executes the test and validates that the process memory has the elements of `expected_mem`
    /// at address `mem_start_addr` and that the end of the stack execution trace matches the
    /// `final_stack`.
    #[track_caller]
    pub fn expect_stack_and_memory(
        &self,
        final_stack: &[u64],
        mem_start_addr: u32,
        expected_mem: &[u64],
    ) {
        let expectations = Expect::default().mem(mem_start_addr, expected_mem).stack(final_stack);
        self.expect(expectations);
    }

    /// Asserts that executing the test inside a proptest results in the expected final stack state.
    /// The proptest will return a test failure instead of panicking if the assertion condition
    /// fails.
    #[cfg(not(target_family = "wasm"))]
    pub fn prop_expect_stack(
        &self,
        final_stack: &[u64],
    ) -> Result<(), proptest::prelude::TestCaseError> {
        let result = self.get_last_stack_state().as_int_vec();
        proptest::prop_assert_eq!(resize_to_min_stack_depth(final_stack), result);

        Ok(())
    }

    // UTILITY METHODS
    // --------------------------------------------------------------------------------------------

    /// Compiles a test's source and returns the resulting Program together with the associated
    /// kernel library (when specified).
    ///
    /// # Errors
    /// Returns an error if compilation of the program source or the kernel fails.
    pub fn compile(&self) -> Result<(Program, Option<KernelLibrary>), Report> {
        use miden_assembly::{Assembler, ParseOptions, ast::ModuleKind};

        let (assembler, kernel_lib) = if let Some(kernel) = self.kernel_source.clone() {
            let kernel_lib =
                Assembler::new(self.source_manager.clone()).assemble_kernel(kernel).unwrap();

            (
                Assembler::with_kernel(self.source_manager.clone(), kernel_lib.clone()),
                Some(kernel_lib),
            )
        } else {
            (Assembler::new(self.source_manager.clone()), None)
        };

        let mut assembler = self
            .add_modules
            .iter()
            .fold(assembler, |mut assembler, (path, source)| {
                let module = source
                    .parse_with_options(
                        &assembler.source_manager(),
                        ParseOptions::new(ModuleKind::Library, path.clone()).unwrap(),
                    )
                    .expect("invalid masm source code");
                assembler.compile_and_statically_link(module).expect("failed to link module");
                assembler
            })
            .with_debug_mode(self.in_debug_mode);
        for library in &self.libraries {
            assembler.link_dynamic_library(library).unwrap();
        }

        Ok((assembler.assemble_program(self.source.clone())?, kernel_lib))
    }

    /// Compiles the test's source to a Program and executes it with the tests inputs. Returns a
    /// resulting execution trace or error.
    ///
    /// Internally, this also checks that the slow and fast processors agree on the stack
    /// outputs.
    #[track_caller]
    pub fn execute(&self) -> Result<ExecutionTrace, ExecutionError> {
        let (program, mut host) = self.get_program_and_host();

        // slow processor
        let mut process = self.get_process(&program);
        let slow_stack_result = process.execute(&program, &mut host);

        // compare fast and slow processors' stack outputs
        self.assert_result_with_fast_processor(&slow_stack_result);

        match slow_stack_result {
            Ok(slow_stack_outputs) => {
                let trace = ExecutionTrace::new(process, slow_stack_outputs);
                assert_eq!(&program.hash(), trace.program_hash(), "inconsistent program hash");

                Ok(trace)
            },
            Err(err) => Err(err),
        }
    }

    /// Compiles the test's source to a Program and executes it with the tests inputs. Returns the
    /// process once execution is finished.
    pub fn execute_process(&self) -> Result<(Process, TestHost), ExecutionError> {
        let (program, mut host) = self.get_program_and_host();

        let mut process = self.get_process(&program);

        let stack_result = process.execute(&program, &mut host);
        self.assert_result_with_fast_processor(&stack_result);

        match stack_result {
            Ok(_) => Ok((process, host)),
            Err(err) => Err(err),
        }
    }

    /// Compiles the test's code into a program, then generates and verifies a proof of execution
    /// using the given public inputs and the specified number of stack outputs. When `test_fail`
    /// is true, this function will force a failure by modifying the first output.
    pub fn prove_and_verify(&self, pub_inputs: Vec<u64>, test_fail: bool) {
        let (program, mut host) = self.get_program_and_host();
        let stack_inputs = StackInputs::try_from_ints(pub_inputs).unwrap();
        let (mut stack_outputs, proof) = miden_prover::prove(
            &program,
            stack_inputs.clone(),
            self.advice_inputs.clone(),
            &mut host,
            ProvingOptions::default(),
            self.source_manager.clone(),
        )
        .unwrap();

        self.assert_outputs_with_fast_processor(&stack_outputs);

        let program_info = ProgramInfo::from(program);
        if test_fail {
            stack_outputs.stack_mut()[0] += ONE;
            assert!(
                miden_verifier::verify(program_info, stack_inputs, stack_outputs, proof).is_err()
            );
        } else {
            let result = miden_verifier::verify(program_info, stack_inputs, stack_outputs, proof);
            assert!(result.is_ok(), "error: {result:?}");
        }
    }

    /// Compiles the test's source to a Program and executes it with the tests inputs. Returns a
    /// VmStateIterator that allows us to iterate through each clock cycle and inspect the process
    /// state.
    pub fn execute_iter(&self) -> VmStateIterator {
        let (program, mut host) = self.get_program_and_host();

        let mut process = self.get_process(&program);
        let result = process.execute(&program, &mut host);

        self.assert_result_with_fast_processor(&result);

        if result.is_ok() {
            assert_eq!(
                program.hash(),
                process.decoder.program_hash().into(),
                "inconsistent program hash"
            );
        }
        VmStateIterator::new(process, result)
    }

    /// Returns the last state of the stack after executing a test.
    #[track_caller]
    pub fn get_last_stack_state(&self) -> StackOutputs {
        let trace = self.execute().expect("failed to execute");

        trace.last_stack_state()
    }

    // HELPERS
    // ------------------------------------------------------------------------------------------

    /// Returns the program and host for the test.
    ///
    /// The host is initialized with the advice inputs provided in the test, as well as the kernel
    /// and library MAST forests.
    fn get_program_and_host(&self) -> (Program, TestHost) {
        let (program, kernel) = self.compile().expect("Failed to compile test source.");
        let mut host = TestHost::default();
        if let Some(kernel) = kernel {
            host.load_mast_forest(kernel.mast_forest().clone()).unwrap();
        }
        for library in &self.libraries {
            host.load_mast_forest(library.mast_forest().clone()).unwrap();
        }

        (program, host)
    }

    /// Returns the [`Process`] for this test, using the passed [`Program`] and the test's
    /// configured source manager, debug mode, and stack & advice inputs.
    fn get_process(&self, program: &Program) -> Process {
        Process::new(
            program.kernel().clone(),
            self.stack_inputs.clone(),
            self.advice_inputs.clone(),
            ExecutionOptions::default().with_debugging(self.in_debug_mode),
        )
        .with_source_manager(self.source_manager.clone())
    }

    /// Runs the program on the fast processor, and asserts that the stack outputs match the slow
    /// processor's stack outputs.
    fn assert_outputs_with_fast_processor(&self, slow_stack_outputs: &StackOutputs) {
        let (program, mut host) = self.get_program_and_host();
        let stack_inputs: Vec<Felt> = self.stack_inputs.clone().into_iter().rev().collect();
        let advice_inputs = self.advice_inputs.clone();
        let fast_process = FastProcessor::new_with_advice_inputs(&stack_inputs, advice_inputs);
        let fast_stack_outputs = fast_process.execute_sync(&program, &mut host).unwrap();

        assert_eq!(
            slow_stack_outputs, &fast_stack_outputs,
            "stack outputs do not match between slow and fast processors"
        );
    }

    fn assert_result_with_fast_processor(
        &self,
        slow_result: &Result<StackOutputs, ExecutionError>,
    ) {
        let (program, mut host) = self.get_program_and_host();
        let stack_inputs: Vec<Felt> = self.stack_inputs.clone().into_iter().rev().collect();
        let advice_inputs: AdviceInputs = self.advice_inputs.clone();
        let fast_process = FastProcessor::new_with_advice_inputs(&stack_inputs, advice_inputs)
            .with_source_manager(self.source_manager.clone());
        let fast_result = fast_process.execute_sync(&program, &mut host);

        match slow_result {
            Ok(slow_stack_outputs) => {
                let fast_stack_outputs = fast_result.unwrap();
                assert_eq!(
                    slow_stack_outputs, &fast_stack_outputs,
                    "stack outputs do not match between slow and fast processors"
                );
            },
            Err(slow_err) => {
                assert!(fast_result.is_err(), "expected error, but got success");
                let fast_err = fast_result.unwrap_err();

                // assert that diagnostics match
                let slow_diagnostic = format!("{}", PrintDiagnostic::new_without_color(slow_err));
                let fast_diagnostic = format!("{}", PrintDiagnostic::new_without_color(fast_err));

                // Note: This assumes that the tests are run WITHOUT the `no_err_ctx` feature
                assert_eq!(
                    slow_diagnostic, fast_diagnostic,
                    "diagnostics do not match between slow and fast processors:\nSlow: {}\nFast: {}",
                    slow_diagnostic, fast_diagnostic
                );
            },
        }
    }
}

/// A 'pattern' type used with [`Test::expect()`] to match on desired test results.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Expect<'s> {
    /// If set, [`Test::expect()`] asserts that executing the test results in a final stack that
    /// matches matches this stack-ordered slice.
    pub stack: Option<&'s [u64]>,

    /// `(mem_start_addr, final_mem)`. If set, [`Test::expect()`] asserts that executing the test
    /// results in memory that, using the first tuple element as the starting address, matches the
    /// second tuple element's contents.
    pub mem: Option<(u32, &'s [u64])>,

    /// If set, [`Test::expect()`] asserts that executing the test takes exactly this number of
    /// cycles.
    pub cycles: Option<usize>,
}

impl<'s> Expect<'s> {
    // CONSTRUCTORS
    // ============================================================================================

    pub fn with_stack(stack: &'s [u64]) -> Self {
        Self { stack: Some(stack), ..Default::default() }
    }

    pub fn with_mem(start_addr: u32, mem: &'s [u64]) -> Self {
        Self {
            mem: Some((start_addr, mem)),
            ..Default::default()
        }
    }

    pub fn with_cycles(cycles: usize) -> Self {
        Self {
            cycles: Some(cycles),
            ..Default::default()
        }
    }

    // BUILDERS
    // ============================================================================================

    pub fn stack(self, stack: &'s [u64]) -> Self {
        Self { stack: Some(stack), ..self }
    }

    pub fn mem(self, start_addr: u32, mem: &'s [u64]) -> Self {
        Self { mem: Some((start_addr, mem)), ..self }
    }

    pub fn cycles(self, cycles: usize) -> Self {
        Self { cycles: Some(cycles), ..self }
    }
}

// HELPER FUNCTIONS
// ================================================================================================

/// Converts a slice of Felts into a vector of u64 values.
pub fn felt_slice_to_ints(values: &[Felt]) -> Vec<u64> {
    values.iter().map(|e| (*e).as_int()).collect()
}

pub fn resize_to_min_stack_depth(values: &[u64]) -> Vec<u64> {
    let mut result: Vec<u64> = values.to_vec();
    result.resize(MIN_STACK_DEPTH, 0);
    result
}

/// A proptest strategy for generating a random word with 4 values of type T.
#[cfg(not(target_family = "wasm"))]
pub fn prop_randw<T: Arbitrary>() -> impl Strategy<Value = Vec<T>> {
    use proptest::prelude::{any, prop};
    prop::collection::vec(any::<T>(), 4)
}

/// Given a hasher state, perform one permutation.
///
/// The values of `values` should be:
/// - 0..4 the capacity
/// - 4..12 the rate
///
/// Return the result of the permutation in stack order.
pub fn build_expected_perm(values: &[u64]) -> [Felt; STATE_WIDTH] {
    let mut expected = [ZERO; STATE_WIDTH];
    for (&value, result) in values.iter().zip(expected.iter_mut()) {
        *result = Felt::new(value);
    }
    apply_permutation(&mut expected);
    expected.reverse();

    expected
}

pub fn build_expected_hash(values: &[u64]) -> [Felt; 4] {
    let digest = hash_elements(&values.iter().map(|&v| Felt::new(v)).collect::<Vec<_>>());
    let mut expected: [Felt; 4] = digest.into();
    expected.reverse();

    expected
}

// Generates the MASM code which pushes the input values during the execution of the program.
#[cfg(all(feature = "std", not(target_family = "wasm")))]
pub fn push_inputs(inputs: &[u64]) -> String {
    let mut result = String::new();

    inputs.iter().for_each(|v| result.push_str(&format!("push.{v}\n")));
    result
}
