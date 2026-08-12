#[cfg(not(feature = "runtime-rbigint"))]
mod conventional {
    #[cfg(feature = "malachite-bigint")]
    use malachite_bigint::BigInt;
    #[cfg(all(feature = "num-bigint", not(feature = "malachite-bigint")))]
    use num_bigint::BigInt;
    use num_traits::{Signed, ToPrimitive, Zero};

    pub type Big = BigInt;

    pub fn big_from_i64(v: i64) -> Big {
        BigInt::from(v)
    }

    pub fn big_to_i64(b: &Big) -> Option<i64> {
        b.to_i64()
    }

    pub fn big_add(a: &Big, b: &Big) -> Big {
        a + b
    }

    pub fn big_sub(a: &Big, b: &Big) -> Big {
        a - b
    }

    pub fn big_mul(a: &Big, b: &Big) -> Big {
        a * b
    }

    /// Whether a truncating division of a bignum needs the floor correction, the
    /// [`super::super::floor_correction_needed`] twin.
    #[inline]
    fn floor_correction_needed_big(r: &Big, b: &Big) -> bool {
        !r.is_zero() && (r.is_negative() != b.is_negative())
    }

    /// `a // b` for a non-zero `b`, floored. `rbigint.div` is `floordiv`, which is
    /// `divmod`'s quotient; the backing crates' `/` truncates instead.
    pub fn big_floor_div(a: &Big, b: &Big) -> Big {
        let q = a / b;
        if floor_correction_needed_big(&(a % b), b) {
            q - BigInt::from(1)
        } else {
            q
        }
    }

    /// `a % b` for a non-zero `b`, floored. `rbigint.mod` is `divmod`'s remainder.
    pub fn big_floor_mod(a: &Big, b: &Big) -> Big {
        let r = a % b;
        if floor_correction_needed_big(&r, b) {
            r + b
        } else {
            r
        }
    }

    pub fn big_ge(a: &Big, b: &Big) -> bool {
        a >= b
    }

    pub fn big_from_dec_str(s: &str) -> Option<Big> {
        s.parse::<BigInt>().ok()
    }

    pub fn big_clone(b: &Big) -> Big {
        b.clone()
    }

    pub fn big_fmt(b: &Big, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", b)
    }
}

#[cfg(feature = "runtime-rbigint")]
mod runtime {
    //! RBigInt digits live in a `TypedItemsBlock` allocated by majit-rlib's
    //! `try_alloc_typed_items_block_nursery`. Without a GC allocator it falls
    //! back to `std::alloc`, and nothing frees the digit arrays, so this proving
    //! arm leaks interpreter-only intermediates. GC integration is separate.

    use majit_rlib::rbigint::RBigInt;

    pub type Big = RBigInt;

    pub fn big_from_i64(v: i64) -> Big {
        RBigInt::fromint(v)
    }

    pub fn big_to_i64(b: &Big) -> Option<i64> {
        b.toint().ok()
    }

    pub fn big_add(a: &Big, b: &Big) -> Big {
        a.add(b)
    }

    pub fn big_sub(a: &Big, b: &Big) -> Big {
        a.sub(b)
    }

    pub fn big_mul(a: &Big, b: &Big) -> Big {
        a.mul(b)
    }

    pub fn big_floor_div(a: &Big, b: &Big) -> Big {
        // RBigInt division is already floored, so arm A's correction is not needed.
        a.div(b)
            .expect("RBigInt division failed: allocation failure")
    }

    pub fn big_floor_mod(a: &Big, b: &Big) -> Big {
        // RBigInt modulo is already floored, so arm A's correction is not needed.
        a.r#mod(b)
            .expect("RBigInt modulo failed: allocation failure")
    }

    pub fn big_ge(a: &Big, b: &Big) -> bool {
        a.ge(b)
    }

    pub fn big_from_dec_str(s: &str) -> Option<Big> {
        RBigInt::fromstr(s, 10, false).ok()
    }

    pub fn big_clone(b: &Big) -> Big {
        b.clone()
    }

    pub fn big_fmt(b: &Big, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", b)
    }
}

#[cfg(not(feature = "runtime-rbigint"))]
pub use conventional::*;
#[cfg(feature = "runtime-rbigint")]
pub use runtime::*;
