#[cfg(not(feature = "cranelift"))]
fn main() {
    eprintln!("cranelift_snippet_runner requires --features cranelift");
    std::process::exit(2);
}

#[cfg(feature = "cranelift")]
mod enabled {
    use compaheuiler::jit::{self, SpecialStorage};
    use std::io::{Read, Write};
    use std::sync::{LazyLock, Mutex};
    use std::time::Instant;

    #[derive(Default)]
    struct IoState {
        input: Vec<u8>,
        pos: usize,
        output: Vec<u8>,
    }

    static IO: LazyLock<Mutex<IoState>> = LazyLock::new(|| Mutex::new(IoState::default()));

    fn reset_io(input: &[u8]) {
        let mut io = IO.lock().unwrap_or_else(|e| e.into_inner());
        io.input.clear();
        io.input.extend_from_slice(input);
        io.pos = 0;
        io.output.clear();
    }

    fn take_output() -> Vec<u8> {
        std::mem::take(&mut IO.lock().unwrap_or_else(|e| e.into_inner()).output)
    }

    extern "C" fn write_char(value: i64) {
        let Some(ch) = char::from_u32(value as u32) else {
            return;
        };
        let mut encoded = [0; 4];
        IO.lock()
            .unwrap_or_else(|e| e.into_inner())
            .output
            .extend_from_slice(ch.encode_utf8(&mut encoded).as_bytes());
    }

    extern "C" fn write_bytes(data: *const u8, len: usize) {
        let bytes = unsafe { std::slice::from_raw_parts(data, len) };
        IO.lock()
            .unwrap_or_else(|e| e.into_inner())
            .output
            .extend_from_slice(bytes);
    }

    extern "C" fn write_num(value: i64) {
        let mut io = IO.lock().unwrap_or_else(|e| e.into_inner());
        let _ = write!(io.output, "{value}");
    }

    extern "C" fn read_char() -> i64 {
        let mut io = IO.lock().unwrap_or_else(|e| e.into_inner());
        if io.pos >= io.input.len() {
            return -1;
        }
        let remaining = &io.input[io.pos..];
        let Ok(text) = std::str::from_utf8(remaining) else {
            io.pos += 1;
            return -1;
        };
        let Some(ch) = text.chars().next() else {
            return -1;
        };
        io.pos += ch.len_utf8();
        ch as i64
    }

    extern "C" fn read_num() -> i64 {
        let mut io = IO.lock().unwrap_or_else(|e| e.into_inner());
        while io.pos < io.input.len() && io.input[io.pos].is_ascii_whitespace() {
            io.pos += 1;
        }
        let start = io.pos;
        if io.pos < io.input.len() && io.input[io.pos] == b'-' {
            io.pos += 1;
        }
        while io.pos < io.input.len() && io.input[io.pos].is_ascii_digit() {
            io.pos += 1;
        }
        std::str::from_utf8(&io.input[start..io.pos])
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    }

    fn execute_once(function: &jit::JitFunction, input: &[u8]) -> (Vec<u8>, i64, u128) {
        let mut storage: Vec<Vec<i64>> = (0..28).map(|_| vec![0; 65_536]).collect();
        let mut bases = [std::ptr::null_mut(); 28];
        for (base, values) in bases.iter_mut().zip(&mut storage) {
            *base = values.as_mut_ptr();
        }
        let mut lengths = [0_i32; 28];
        let mut special = SpecialStorage::new();
        let special_ptr = &mut special as *mut SpecialStorage as *mut u8;
        reset_io(input);

        let started = Instant::now();
        let exit = unsafe {
            function.execute_buffered(
                &mut bases,
                &mut lengths,
                write_char,
                write_bytes,
                write_num,
                read_char,
                read_num,
                special_ptr,
                jit::sp_push,
                jit::sp_pop,
                jit::sp_depth,
                jit::sp_dup,
                jit::sp_swap,
            )
        };
        let elapsed = started.elapsed().as_nanos();
        (take_output(), exit, elapsed)
    }

    pub fn main() {
        let mut args = std::env::args_os().skip(1);
        let source_path = args.next().unwrap_or_else(|| {
            eprintln!("usage: cranelift_snippet_runner SOURCE [REPEATS]");
            std::process::exit(2);
        });
        let repeats = args
            .next()
            .and_then(|value| value.to_string_lossy().parse::<usize>().ok())
            .unwrap_or(3)
            .max(1);
        if args.next().is_some() {
            eprintln!("usage: cranelift_snippet_runner SOURCE [REPEATS]");
            std::process::exit(2);
        }

        let source = std::fs::read_to_string(&source_path).unwrap_or_else(|error| {
            eprintln!("cannot read {}: {error}", source_path.to_string_lossy());
            std::process::exit(2);
        });
        let mut input = Vec::new();
        std::io::stdin().read_to_end(&mut input).unwrap();

        let cfg = compaheuiler::pipeline::optimize(&source, ahsembler::OptimizationLevel::O3);
        let function = jit::compile_cfg(&cfg).unwrap_or_else(|error| {
            eprintln!("cranelift compilation failed: {error}");
            std::process::exit(2);
        });

        let mut times = Vec::with_capacity(repeats);
        let mut first_output = None;
        let mut first_exit = None;
        for _ in 0..repeats {
            let (output, exit, elapsed) = execute_once(&function, &input);
            if let Some(expected) = &first_output {
                assert_eq!(expected, &output, "repeated execution changed stdout");
            } else {
                first_output = Some(output);
            }
            if let Some(expected) = first_exit {
                assert_eq!(expected, exit, "repeated execution changed the exit code");
            } else {
                first_exit = Some(exit);
            }
            times.push(elapsed);
        }
        times.sort_unstable();
        let median_ns = times[times.len() / 2];
        let output = first_output.unwrap_or_default();
        let exit = first_exit.unwrap_or(0);

        std::io::stdout().write_all(&output).unwrap();
        eprintln!("[snippet-matrix] median_ns={median_ns} repeats={repeats} exit={exit}");
        std::process::exit(exit as i32);
    }
}

#[cfg(feature = "cranelift")]
fn main() {
    enabled::main();
}
