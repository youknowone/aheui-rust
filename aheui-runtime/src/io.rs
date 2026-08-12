//! Buffered I/O for Aheui interpreter.
//!
//! Ported from rpaheui/aheui/option.py and target/aheui IO routines.

use std::io::Write;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::io::{self, Read};

use crate::value::*;

// `wasm32-unknown-unknown` 타겟에는 실제 stdin/stdout 이 없으므로
// thread-local 버퍼로 I/O 를 대신한다. 호스트 코드 (예: aheui-wasm)
// 에서 `set_input` 으로 입력을 주입하고 `take_output` 으로 출력을
// 회수한다. 다른 타겟에서는 평소대로 std::io 경로를 사용한다.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod wasm_buf {
    use std::cell::RefCell;

    thread_local! {
        static INPUT: RefCell<(Vec<u8>, usize)> = RefCell::new((Vec::new(), 0));
        static OUTPUT: RefCell<Vec<u8>> = RefCell::new(Vec::new());
    }

    pub fn set_input(bytes: &[u8]) {
        INPUT.with(|cell| {
            let mut inner = cell.borrow_mut();
            inner.0.clear();
            inner.0.extend_from_slice(bytes);
            inner.1 = 0;
        });
    }

    pub fn take_output() -> Vec<u8> {
        OUTPUT.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
    }

    pub(super) fn write_all(bytes: &[u8]) {
        OUTPUT.with(|cell| cell.borrow_mut().extend_from_slice(bytes));
    }

    pub(super) fn drain_input_into(dst: &mut Vec<u8>) {
        INPUT.with(|cell| {
            let mut inner = cell.borrow_mut();
            let end = inner.0.len();
            if inner.1 < end {
                dst.extend_from_slice(&inner.0[inner.1..end]);
                inner.1 = end;
            }
        });
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn output_write_all(bytes: &[u8]) {
    wasm_buf::write_all(bytes);
}

// Mirrors rpaheui/aheui/aheui.py:196-220 OutputBuffer. Each thread buffers
// only pending output bytes, not interpreter state, and drains them at the
// threshold or `output_flush` to avoid locking stdout for every small write.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
const OUTPUT_BUFFER_THRESHOLD: usize = 4096;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
thread_local! {
    static OUTPUT_BUFFER: std::cell::RefCell<Vec<u8>> =
        std::cell::RefCell::new(Vec::with_capacity(OUTPUT_BUFFER_THRESHOLD));
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn output_buffer_drain(buf: &mut Vec<u8>) {
    if buf.is_empty() {
        return;
    }
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    let _ = handle.write_all(buf);
    buf.clear();
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn output_write_all(bytes: &[u8]) {
    OUTPUT_BUFFER.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.extend_from_slice(bytes);
        if buf.len() >= OUTPUT_BUFFER_THRESHOLD {
            output_buffer_drain(&mut buf);
        }
    });
}

/// Fast decimal encoding for i64, avoids alloc.
fn encode_decimal_i64(value: i64, buf: &mut [u8; 20]) -> &[u8] {
    if value == 0 {
        buf[19] = b'0';
        return &buf[19..];
    }
    let negative = value < 0;
    // Take the magnitude unsigned because negating i64::MIN overflows.
    let mut magnitude = value.unsigned_abs();
    let mut pos = 20;
    while magnitude > 0 {
        pos -= 1;
        buf[pos] = b'0' + (magnitude % 10) as u8;
        magnitude /= 10;
    }
    if negative {
        pos -= 1;
        buf[pos] = b'-';
    }
    &buf[pos..]
}

fn output_write_number_i64(value: i64) {
    let mut buf = [0u8; 20];
    output_write_all(encode_decimal_i64(value, &mut buf));
}

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
pub fn output_write_number(value: &Val) {
    output_write_number_i64(*value);
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
pub fn output_write_number(value: &Val) {
    if let Some(v) = value.try_to_i64() {
        output_write_number_i64(v);
    } else {
        let text = value.to_string();
        output_write_all(text.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::encode_decimal_i64;

    #[test]
    fn encode_decimal_i64_handles_full_range_and_common_values() {
        let cases = [
            (i64::MIN, "-9223372036854775808"),
            (i64::MAX, "9223372036854775807"),
            (0, "0"),
            (-1, "-1"),
            (1, "1"),
            (-10, "-10"),
        ];

        for (value, expected) in cases {
            let mut buf = [0u8; 20];
            assert_eq!(encode_decimal_i64(value, &mut buf), expected.as_bytes());
        }
    }
}

pub fn output_write_utf8(value: &Val) {
    output_write_utf8_i64(val_to_i64(value));
}

fn output_write_utf8_i64(code: i64) {
    if let Some(ch) = char::from_u32(code as u32) {
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        output_write_all(s.as_bytes());
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub fn output_flush() {
    // wasm32-unknown-unknown: 출력은 thread-local 버퍼에 모이고
    // 호스트 코드에서 `wasm_buf::take_output` 으로 회수한다.
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub fn output_flush() {
    OUTPUT_BUFFER.with(|cell| {
        let mut buf = cell.borrow_mut();
        output_buffer_drain(&mut buf);
    });
    let stdout = io::stdout();
    let _ = stdout.lock().flush();
}

// Generated and JIT-compiled code writes through caller-provided buffers.

pub fn write_number(value: &Val, writer: &mut impl Write) {
    #[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
    {
        let mut buf = [0u8; 20];
        let _ = writer.write_all(encode_decimal_i64(*value, &mut buf));
    }
    #[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
    {
        if value.is_small() {
            let mut buf = [0u8; 20];
            let _ = writer.write_all(encode_decimal_i64(value.to_i64(), &mut buf));
        } else {
            let _ = write!(writer, "{}", value);
        }
    }
}

/// Write a Val as a Unicode codepoint (UTF-8) to stdout.
pub fn write_utf8(value: &Val, writer: &mut impl Write) {
    let code = val_to_i64(value);
    if let Some(ch) = char::from_u32(code as u32) {
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        let _ = writer.write_all(s.as_bytes());
    }
}

/// Buffered input for Aheui I/O operations.
pub struct InputBuffer {
    buffer: Vec<u8>,
    pos: usize,
}

impl InputBuffer {
    pub fn new() -> Self {
        InputBuffer {
            buffer: Vec::new(),
            pos: 0,
        }
    }

    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    fn fill_line(&mut self) {
        self.buffer.clear();
        self.pos = 0;
        wasm_buf::drain_input_into(&mut self.buffer);
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    fn fill_line(&mut self) {
        self.buffer.clear();
        self.pos = 0;
        let stdin = io::stdin();
        let mut handle = stdin.lock();
        let _ = handle.read_to_end(&mut self.buffer);
    }

    fn ensure_data(&mut self) {
        if self.pos >= self.buffer.len() {
            self.fill_line();
        }
    }

    pub fn read_number(&mut self) -> i64 {
        self.ensure_data();
        // Skip whitespace
        while self.pos < self.buffer.len() && self.buffer[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        // Read digits
        let start = self.pos;
        let negative = self.pos < self.buffer.len() && self.buffer[self.pos] == b'-';
        if negative {
            self.pos += 1;
        }
        while self.pos < self.buffer.len() && self.buffer[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if start == self.pos {
            return 0;
        }
        let s = std::str::from_utf8(&self.buffer[start..self.pos]).unwrap_or("0");
        s.parse::<i64>().unwrap_or(0)
    }

    pub fn read_utf8(&mut self) -> i64 {
        self.ensure_data();
        if self.pos >= self.buffer.len() {
            return -1;
        }
        // Decode one UTF-8 character
        let remaining = &self.buffer[self.pos..];
        let s = std::str::from_utf8(remaining).unwrap_or("");
        if let Some(ch) = s.chars().next() {
            self.pos += ch.len_utf8();
            ch as i64
        } else {
            self.pos += 1;
            -1
        }
    }
}
