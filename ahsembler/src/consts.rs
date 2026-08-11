//! Aheui language constants and Korean character decomposition.
//!
//! Ported from rpaheui/aheui/const.py and rpaheui/aheui/compile.py.

// Opcodes — initial consonant (초성) index
// Indices 0, 1, 13, 15 are unused (ㄱ, ㄲ, ㅉ, ㅋ)
pub const OP_DIV: u8 = 2; // ㄴ
pub const OP_ADD: u8 = 3; // ㄷ
pub const OP_MUL: u8 = 4; // ㄸ
pub const OP_MOD: u8 = 5; // ㄹ
pub const OP_POP: u8 = 6; // ㅁ
pub const OP_PUSH: u8 = 7; // ㅂ
pub const OP_DUP: u8 = 8; // ㅃ
pub const OP_SEL: u8 = 9; // ㅅ
pub const OP_MOV: u8 = 10; // ㅆ
pub const OP_NONE: u8 = 11; // ㅇ
pub const OP_CMP: u8 = 12; // ㅈ
pub const OP_BRZ: u8 = 14; // ㅊ
pub const OP_SUB: u8 = 16; // ㅌ
pub const OP_SWAP: u8 = 17; // ㅍ
pub const OP_HALT: u8 = 18; // ㅎ

// Extended opcodes (assigned after primitive range)
pub const OP_POPNUM: u8 = 19;
pub const OP_POPCHAR: u8 = 20;
pub const OP_PUSHNUM: u8 = 21;
pub const OP_PUSHCHAR: u8 = 22;
pub const OP_BRPOP2: u8 = 23; // Python: -3
pub const OP_BRPOP1: u8 = 24; // Python: -2
pub const OP_JMP: u8 = 25; // Python: -1

pub const NUM_OPS: usize = 26;

// Storage constants
pub const STORAGE_COUNT: usize = 28;
pub const VAL_QUEUE: usize = 21;
pub const VAL_PORT: usize = 27;

// Special value indices for OP_POP/OP_PUSH sub-operations
pub const VAL_NUMBER: i32 = 21;
pub const VAL_UNICODE: i32 = 27;

/// Required stack size for each opcode.
pub const OP_REQSIZE: [usize; NUM_OPS] = [
    0, 0, 2, 2, 2, 2, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 2, 2, 0, 1, 1, 0, 0, 2, 1, 0,
];

/// Stack elements consumed by each opcode.
pub const OP_STACKDEL: [i32; NUM_OPS] = [
    0, 0, 2, 2, 2, 2, 1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 2, 2, 0, 1, 1, 0, 0, 0, 0, 0,
];

/// Stack elements produced by each opcode.
pub const OP_STACKADD: [i32; NUM_OPS] = [
    0, 0, 1, 1, 1, 1, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 1, 1, 0, 0, 0,
];

/// Constants pushed by final consonant (종성) value.
pub const VAL_CONSTS: [i64; 28] = [
    //() ㄱ ㄲ ㄳ ㄴ ㄵ ㄶ ㄷ ㄹ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅁ ㅂ ㅄ ㅅ ㅆ ㅇ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ
    0, 2, 4, 4, 2, 5, 5, 3, 5, 7, 9, 9, 7, 9, 9, 8, 4, 4, 6, 2, 4, 1, 3, 4, 3, 4, 4, 3,
];

/// Whether the opcode has a valid operand.
pub const OP_HASOP: [bool; NUM_OPS] = [
    false, false, true, true, true, true, true, true, true, true, true, false, true, false, true,
    false, true, true, true, true, true, true, true, true, true, true,
];

/// Whether the opcode uses its operand value.
pub const OP_USEVAL: [bool; NUM_OPS] = [
    false, false, false, false, false, false, false, true, false, true, true, false, false, false,
    true, false, false, false, false, false, false, false, false, true, true, true,
];

/// Opcode names for debug/asm output.
pub const OP_NAMES: [Option<&str>; NUM_OPS] = [
    None,
    None,
    Some("DIV"),
    Some("ADD"),
    Some("MUL"),
    Some("MOD"),
    Some("POP"),
    Some("PUSH"),
    Some("DUP"),
    Some("SEL"),
    Some("MOV"),
    None,
    Some("CMP"),
    None,
    Some("BRZ"),
    None,
    Some("SUB"),
    Some("SWAP"),
    Some("HALT"),
    Some("POPNUM"),
    Some("POPCHAR"),
    Some("PUSHNUM"),
    Some("PUSHCHAR"),
    Some("BRPOP2"),
    Some("BRPOP1"),
    Some("JMP"),
];

// Movement direction constants (중성 index)
pub const MV_NONE: i32 = -1;
pub const MV_RIGHT: i32 = 0; // ㅏ
pub const MV_RIGHT2: i32 = 2; // ㅑ
pub const MV_LEFT: i32 = 4; // ㅓ
pub const MV_LEFT2: i32 = 6; // ㅕ
pub const MV_UP: i32 = 8; // ㅗ
pub const MV_UP2: i32 = 12; // ㅛ
pub const MV_DOWN: i32 = 13; // ㅜ
pub const MV_DOWN2: i32 = 17; // ㅠ
pub const MV_HWALL: i32 = 18; // ㅡ
pub const MV_WALL: i32 = 19; // ㅢ
pub const MV_VWALL: i32 = 20; // ㅣ

// Direction constants
pub const DIR_DOWN: i32 = 1;
pub const DIR_RIGHT: i32 = 2;
pub const DIR_UP: i32 = -1;
pub const DIR_LEFT: i32 = -2;

/// Movement codes that deterministically set direction.
pub const MV_DETERMINISTICS: [i32; 8] = [
    MV_DOWN, MV_DOWN2, MV_UP, MV_UP2, MV_LEFT, MV_LEFT2, MV_RIGHT, MV_RIGHT2,
];

/// Check if opcode is a jump/branch.
pub fn is_jump_op(op: u8) -> bool {
    matches!(op, OP_BRZ | OP_BRPOP1 | OP_BRPOP2 | OP_JMP)
}

/// Check if opcode is a binary arithmetic operation.
pub fn is_binary_op(op: u8) -> bool {
    matches!(op, OP_DIV | OP_ADD | OP_MUL | OP_MOD | OP_CMP | OP_SUB)
}

/// `OP_DIV` on a non-zero divisor: the quotient rounds toward negative
/// infinity, as `int.__floordiv__` and `rbigint.div` both do.
///
/// Rust's `/` and `wrapping_div` truncate toward zero, which answers the same
/// as flooring whenever the operands share a sign or the division is exact and
/// differs by one otherwise. A constant folder that truncates here would give
/// a literal `-7 / 2` a different answer from the same division performed at
/// run time.
///
/// `div_euclid` is a third convention — it forces a non-negative remainder —
/// and is not this one.
#[inline]
pub fn floor_div_i64(a: i64, b: i64) -> i64 {
    let q = a.wrapping_div(b);
    let r = a.wrapping_rem(b);
    if r != 0 && (r < 0) != (b < 0) {
        q.wrapping_sub(1)
    } else {
        q
    }
}

/// `OP_MOD` on a non-zero divisor: the remainder carries the divisor's sign,
/// the twin of [`floor_div_i64`].
///
/// The corrected remainder cannot overflow — it is only computed when `r` and
/// `b` have opposite signs, so `|r + b| < |b|`.
#[inline]
pub fn floor_mod_i64(a: i64, b: i64) -> i64 {
    let r = a.wrapping_rem(b);
    if r != 0 && (r < 0) != (b < 0) {
        r.wrapping_add(b)
    } else {
        r
    }
}

/// Decompose a Korean syllable into (초성, 중성, 종성).
/// Returns None for non-Korean characters.
pub fn decompose(ch: char) -> Option<(u8, i32, i32)> {
    let code = ch as u32;
    if !(0xAC00..=0xD7A3).contains(&code) {
        return None;
    }
    let base = code - 0xAC00;
    let op = (base / 588) as u8;
    let mv = ((base / 28) % 21) as i32;
    let val = (base % 28) as i32;
    Some((op, mv, val))
}

/// Determine direction and step from movement code.
pub fn dir_from_mv(mv_code: i32, direction: i32, step: i32) -> (i32, i32) {
    match mv_code {
        MV_RIGHT => (DIR_RIGHT, 1),
        MV_RIGHT2 => (DIR_RIGHT, 2),
        MV_LEFT => (DIR_LEFT, 1),
        MV_LEFT2 => (DIR_LEFT, 2),
        MV_UP => (DIR_UP, 1),
        MV_UP2 => (DIR_UP, 2),
        MV_DOWN => (DIR_DOWN, 1),
        MV_DOWN2 => (DIR_DOWN, 2),
        MV_WALL => (-direction, step),
        MV_HWALL => {
            if direction == DIR_UP || direction == DIR_DOWN {
                (-direction, step)
            } else {
                (direction, step)
            }
        }
        MV_VWALL => {
            if direction == DIR_RIGHT || direction == DIR_LEFT {
                (-direction, step)
            } else {
                (direction, step)
            }
        }
        _ => (direction, step),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_korean() {
        assert_eq!(decompose('가'), Some((0, 0, 0)));
        assert_eq!(decompose('힣'), Some((18, 20, 27)));
    }

    #[test]
    fn test_decompose_non_korean() {
        assert_eq!(decompose('A'), None);
        assert_eq!(decompose(' '), None);
    }

    #[test]
    fn test_decompose_specific() {
        let (op, mv, val) = decompose('반').unwrap();
        assert_eq!(op, OP_PUSH);
        assert_eq!(mv, MV_RIGHT);
        assert_eq!(val, 4);
    }

    #[test]
    fn test_dir_from_mv() {
        assert_eq!(dir_from_mv(MV_RIGHT, DIR_DOWN, 1), (DIR_RIGHT, 1));
        assert_eq!(dir_from_mv(MV_WALL, DIR_DOWN, 1), (DIR_UP, 1));
        assert_eq!(dir_from_mv(MV_HWALL, DIR_DOWN, 1), (DIR_UP, 1));
        assert_eq!(dir_from_mv(MV_HWALL, DIR_RIGHT, 1), (DIR_RIGHT, 1));
        assert_eq!(dir_from_mv(MV_VWALL, DIR_RIGHT, 1), (DIR_LEFT, 1));
        assert_eq!(dir_from_mv(MV_VWALL, DIR_DOWN, 1), (DIR_DOWN, 1));
    }
}
