//! Canonical call/effect specification for aheui interpreter.
//!
//! Data-only module shared by runtime code and `build.rs`.

// Receiver root used by the codewriter to group polymorphic
// Stack/Queue/Port method effects. Matches the `LinkedList` trait
// defined in `aheui_runtime::storage::linkedlist`.
pub const AHEUI_CALL_OWNER_ROOT: &str = "LinkedList";

#[derive(Clone, Copy)]
pub enum CallEffectKind {
    Elidable,
    Residual,
}

#[derive(Clone, Copy)]
pub enum CallTargetSpec {
    Method {
        name: &'static str,
        receiver_root: &'static str,
    },
    FunctionPath(&'static [&'static str]),
}

#[derive(Clone, Copy)]
pub struct CallEffectSpec {
    pub target: CallTargetSpec,
    pub effect: CallEffectKind,
}

pub const AHEUI_CALL_EFFECTS: &[CallEffectSpec] = &[
    // Binops — elidable (pure integer arithmetic)
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "add",
            receiver_root: AHEUI_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Elidable,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "sub",
            receiver_root: AHEUI_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Elidable,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "mul",
            receiver_root: AHEUI_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Elidable,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "div",
            receiver_root: AHEUI_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Elidable,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "modulo",
            receiver_root: AHEUI_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Elidable,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "cmp",
            receiver_root: AHEUI_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Elidable,
    },
    // IO — residual (side effects)
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["output_write_number"]),
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::FunctionPath(&["output_write_utf8"]),
        effect: CallEffectKind::Residual,
    },
    // Storage mutations — residual
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "push",
            receiver_root: AHEUI_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "pop",
            receiver_root: AHEUI_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "dup",
            receiver_root: AHEUI_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
    CallEffectSpec {
        target: CallTargetSpec::Method {
            name: "swap",
            receiver_root: AHEUI_CALL_OWNER_ROOT,
        },
        effect: CallEffectKind::Residual,
    },
];
