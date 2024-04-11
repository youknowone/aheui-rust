#[cfg(feature = "clap")]
pub mod clap;

use malachite_bigint::BigInt;
use rustpython::vm::{
    builtins::{PyBytes, PyInt, PyStr},
    Interpreter, PyObjectRef, PyResult,
};

pub const FROZEN: &rustpython::vm::frozen::FrozenLib =
    rustpython_derive::py_freeze!(dir = "./rpaheui", crate_name = "rustpython::vm");

pub struct Aheui {
    python: Interpreter,
    aheui_module: PyObjectRef,
    prepare_compiler: PyObjectRef,
}

#[derive(Clone)]
pub struct Object {
    inner: PyObjectRef,
}

impl Default for Aheui {
    fn default() -> Self {
        Self::new()
    }
}

impl Aheui {
    pub fn new() -> Self {
        let python = rustpython::InterpreterConfig::new()
            .init_stdlib()
            .init_hook(Box::new(|vm| {
                vm.add_frozen(FROZEN);
            }))
            .interpreter();

        let (aheui_module, prepare_compiler) = python.enter_and_expect(
            |vm| {
                let top = vm.import("aheui.aheui", 0)?;
                let aheui_module = top.get_attr("aheui", vm)?;
                let prepare_compiler = aheui_module.get_attr("prepare_compiler", vm)?;
                Ok((aheui_module, prepare_compiler))
            },
            "importing aheui.aheui.prepare_compiler unexpectedly failed.",
        );

        Self {
            python,
            aheui_module,
            prepare_compiler,
        }
    }

    pub fn compile(&mut self, code: &str) -> PyResult<Object> {
        let object = self.python.enter_and_expect(
            |vm| {
                let code = vm.ctx.new_bytes(code.as_bytes().to_owned());
                self.prepare_compiler.call((code, 1, "text"), vm)
            },
            "Compilation failed. This is a bug.",
        );
        Ok(Object { inner: object })
    }

    pub fn compile_asm(&mut self, asm: &str) -> PyResult<Object> {
        let object = self.python.enter_and_expect(
            |vm| {
                let code = vm.ctx.new_bytes(asm.as_bytes().to_owned());
                self.prepare_compiler.call((code, 1, "asm"), vm)
            },
            "Compilation failed. This is a bug.",
        );
        Ok(Object { inner: object })
    }

    pub fn load_bytecode(&mut self, bytecode: &[u8]) -> PyResult<Object> {
        let object = self.python.enter_and_expect(
            |vm| {
                let code = vm.ctx.new_bytes(bytecode.to_owned());
                self.prepare_compiler.call((code, 1, "bytecode"), vm)
            },
            "Compilation failed. This is a bug.",
        );
        Ok(Object { inner: object })
    }

    pub fn run(&mut self, object: &Object) -> PyResult<BigInt> {
        Ok(self.python.enter_and_expect(
            |vm| {
                let run_with_compiler = self.aheui_module.get_attr("run_with_compiler", vm)?;
                let exit_code = run_with_compiler.call((object.inner.clone(),), vm)?;
                let exit_code = exit_code.downcast::<PyInt>().unwrap();
                Ok(exit_code.as_bigint().clone())
            },
            "run failed",
        ))
    }

    pub fn make_asm(&mut self, object: &Object, commented: bool) -> PyResult<String> {
        let asm = self.python.enter(|vm| {
            let write_asm = object.inner.get_attr("write_asm", vm)?;
            let asm = write_asm.call((commented,), vm)?;
            let asm = asm.downcast::<PyStr>().expect("rpaheui bug");
            Ok(asm)
        })?;
        Ok(asm.as_str().to_string())
    }

    pub fn make_bytecode(&mut self, object: &Object) -> PyResult<Vec<u8>> {
        let bytecode = self.python.enter(|vm| {
            let write_bytecode = object.inner.get_attr("write_bytecode", vm)?;
            let bytecode = write_bytecode.call((), vm)?;
            let bytecode = bytecode.downcast::<PyBytes>().expect("rpaheui bug");
            Ok(bytecode)
        })?;
        Ok(bytecode.as_bytes().to_vec())
    }
}

#[test]
fn test_asm() {
    let mut aheui = Aheui::new();
    let object = aheui.compile("반받망희").unwrap();
    let asm = aheui.make_asm(&object, false).unwrap();
    println!("{}", asm);
    assert!(asm.contains("PUSH 2"));
    assert!(asm.contains("PUSH 3"));
    assert!(asm.contains("POPNUM"));
    assert!(asm.contains("HALT"));

    let object2 = aheui.compile_asm(&asm).expect("recompilation failed");
    let asm2 = aheui.make_asm(&object2, false).unwrap();
    assert_eq!(asm, asm2);
    let exit_code = aheui.run(&object);
    assert_eq!(exit_code.unwrap(), BigInt::from(2));
}

#[test]
fn test_bytecode() {
    let mut aheui = Aheui::new();
    let object = aheui.compile("반망희");
    let bytecode = aheui.make_bytecode(&object.unwrap()).unwrap();

    let object2 = aheui.load_bytecode(&bytecode).unwrap();
    let bytecode2 = aheui.make_bytecode(&object2).unwrap();
    assert_eq!(bytecode, bytecode2);
    let exit_code = aheui.run(&object2);
    assert_eq!(exit_code.unwrap(), BigInt::from(0));
}
