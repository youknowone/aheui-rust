/// Buffered I/O for Aheui interpreter.
///
/// Ported from rpaheui/aheui/option.py and target/aheui IO routines.
use std::io::{self, Read, Write};

use crate::value::*;

// ── Output helpers ──────────────────────────────────────────────────

fn output_write_all(bytes: &[u8]) {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    let _ = handle.write_all(bytes);
}

/// Fast decimal encoding for i64, avoids alloc.
fn encode_decimal_i64(mut value: i64, buf: &mut [u8; 20]) -> &[u8] {
    if value == 0 {
        buf[19] = b'0';
        return &buf[19..];
    }
    let negative = value < 0;
    if negative {
        value = -value;
    }
    let mut pos = 20;
    while value > 0 {
        pos -= 1;
        buf[pos] = b'0' + (value % 10) as u8;
        value /= 10;
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

pub fn output_flush() {
    let stdout = io::stdout();
    let _ = stdout.lock().flush();
}

// ── JIT I/O shims (stack-allocated decimal encoding) ────────────────

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

// ── Buffered input ──────────────────────────────────────────────────

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
