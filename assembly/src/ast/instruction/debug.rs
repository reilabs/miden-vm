use core::fmt;

use crate::{
    AssemblyError,
    assembler::ProcedureContext,
    ast::{ImmU8, ImmU16, ImmU32},
};

// DEBUG OPTIONS
// ================================================================================================

/// A proxy for [vm_core::DebugOptions], but with [super::Immediate] values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DebugOptions {
    StackAll,
    StackTop(ImmU8),
    MemAll,
    MemInterval(ImmU32, ImmU32),
    LocalInterval(ImmU16, ImmU16),
    LocalRangeFrom(ImmU16),
    LocalAll,
}

impl DebugOptions {
    /// Compiles the AST representation of a `debug` instruction into its VM representation.
    ///
    /// This function does not currently return any errors, but may in the future.
    ///
    /// See [crate::Assembler] for an overview of AST compilation.
    pub fn compile(
        &self,
        proc_ctx: &ProcedureContext,
    ) -> Result<vm_core::DebugOptions, AssemblyError> {
        type Ast = DebugOptions;
        type Vm = vm_core::DebugOptions;

        let compiled = match self {
            Ast::StackAll => Vm::StackAll,
            Ast::StackTop(n) => Vm::StackTop(n.expect_value()),
            Ast::MemAll => Vm::MemAll,
            Ast::MemInterval(start, end) => {
                Vm::MemInterval(start.expect_value(), end.expect_value())
            },
            Ast::LocalInterval(start, end) => {
                let (start, end) = (start.expect_value(), end.expect_value());
                Vm::LocalInterval(start, end, proc_ctx.num_locals())
            },
            Ast::LocalRangeFrom(index) => {
                let index = index.expect_value();
                Vm::LocalInterval(index, index, proc_ctx.num_locals())
            },
            Ast::LocalAll => Vm::LocalInterval(0, proc_ctx.num_locals(), proc_ctx.num_locals()),
        };

        Ok(compiled)
    }
}

impl crate::prettier::PrettyPrint for DebugOptions {
    fn render(&self) -> crate::prettier::Document {
        crate::prettier::display(self)
    }
}

impl fmt::Display for DebugOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StackAll => write!(f, "stack"),
            Self::StackTop(n) => write!(f, "stack.{n}"),
            Self::MemAll => write!(f, "mem"),
            Self::MemInterval(n, m) => write!(f, "mem.{n}.{m}"),
            Self::LocalAll => write!(f, "local"),
            Self::LocalRangeFrom(start) => write!(f, "local.{start}"),
            Self::LocalInterval(start, end) => {
                write!(f, "local.{start}.{end}")
            },
        }
    }
}
