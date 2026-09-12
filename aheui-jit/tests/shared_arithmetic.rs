//! Exercise the shared arithmetic in compiled band, stack, queue and port paths.

mod common;

use ahsembler::consts::*;
use common::{Asm, LABEL_END, LABEL_LOOP, NONE};

#[test]
fn shared_arithmetic() {
    let Ok(case) = std::env::var("AHEUI_ARITHMETIC_CASE") else {
        // Each interpreter owns process-global GC/JIT state: isolate each case.
        for pool in [0, 1, VAL_QUEUE, VAL_PORT] {
            for op in [OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD, OP_CMP] {
                for tagged in [false, true] {
                    let case = format!("{pool},{op},{tagged}");
                    let output = std::process::Command::new(std::env::current_exe().unwrap())
                        .args(["--exact", "shared_arithmetic", "--nocapture"])
                        .env("AHEUI_ARITHMETIC_CASE", &case)
                        .env("AHEUI_BANDS", "1")
                        .env("AHEUI_CAP", "2")
                        .env_remove("AHEUI_BAND_ARMS")
                        .stdin(std::process::Stdio::null())
                        .output()
                        .unwrap();
                    assert!(
                        output.status.success(),
                        "case {case}:\n{}\n{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
            }
        }
        return;
    };
    let fields: Vec<_> = case.split(',').collect();
    let pool: usize = fields[0].parse().unwrap();
    let op: u8 = fields[1].parse().unwrap();
    let tagged: bool = fields[2].parse().unwrap();
    let mut asm = Asm::new();
    if tagged {
        // Enter tagged mode before the subject loop, then discard the bigint.
        asm.emit(OP_SEL, 26)
            .emit(OP_PUSH, i32::MAX)
            .emit(OP_DUP, NONE)
            .emit(OP_MUL, NONE)
            .emit(OP_DUP, NONE)
            .emit(OP_MUL, NONE)
            .emit(OP_POP, NONE);
    }
    asm.emit(OP_SEL, pool as i32);
    let mut values = std::collections::VecDeque::new();
    for i in 0..common::ITERATIONS {
        let value = if i % 3 == 0 { -1 } else { 1 };
        values.push_back(value as i64);
        asm.emit(OP_PUSH, value);
    }
    while values.len() > 1 {
        let r1 = if pool == VAL_QUEUE {
            values.pop_front()
        } else {
            values.pop_back()
        }
        .unwrap();
        let r2 = if pool == VAL_QUEUE {
            values.pop_front()
        } else {
            values.pop_back()
        }
        .unwrap();
        let result = match op {
            OP_ADD => r2 + r1,
            OP_SUB => r2 - r1,
            OP_MUL => r2 * r1,
            OP_DIV | OP_MOD if r1 == 0 => 0,
            OP_DIV | OP_MOD => {
                let rem = r2 % r1;
                let correction = i64::from(rem != 0 && (rem < 0) != (r1 < 0));
                if op == OP_DIV {
                    r2 / r1 - correction
                } else {
                    rem + correction * r1
                }
            }
            OP_CMP => i64::from(r2 >= r1),
            _ => unreachable!(),
        };
        values.push_back(result);
    }
    asm.label(LABEL_LOOP)
        .emit(OP_BRPOP2, LABEL_END)
        .emit(op, NONE)
        .emit(OP_JMP, LABEL_LOOP)
        .label(LABEL_END)
        .emit(OP_HALT, NONE);
    assert_eq!(common::run(&asm.build()), values[0], "{case}");
    common::assert_compiled(&case);
    majit_metainterp::assert_no_degraded_dispatch_arms("AheuiState");
}
