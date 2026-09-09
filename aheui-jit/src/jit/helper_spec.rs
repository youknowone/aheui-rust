//! LLBC helpers consumed by the macro-generated interpreter portal.
pub const HELPERS: &[&[&str]] = &[
    &["storage", "linkedlist", "pop_base_known_nonempty"],
    &["storage", "linkedlist", "swap_base_known_two"],
    &["band", "band_add"],
    &["band", "band_sub"],
    &["band", "band_mul"],
    &["band", "band_div"],
    &["band", "band_mod"],
    &["band", "band_cmp"],
    &["band", "band_add_raw"],
    &["band", "band_sub_raw"],
    &["band", "band_mul_raw"],
    &["band", "band_div_raw"],
    &["band", "band_mod_raw"],
    &["band", "band_cmp_raw"],
];
