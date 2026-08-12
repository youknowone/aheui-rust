// 웹 데모에 포함할 샘플들이 실제로 동작하는지 검증.
import { readFileSync } from "node:fs";
import { interpret } from "./pkg-node/aheui_wasm.js";

const cases = [
  { file: "../snippets/hello-world/hello.puzzlet.aheui", input: "", expect: "안녕하세요?\n" },
  { file: "../snippets/hello-world/hello-world.puzzlet.aheui", input: "", expect: "Hello, world!\n" },
  { file: "../snippets/factorial/factorial.aheui", input: "5\n", expect: "120" },
  { file: "../snippets/fibonacci/fibonacci.codroc.aheui", input: "", expect: "23581321345589144233" },
];

let failed = 0;
for (const { file, input, expect } of cases) {
  try {
    const src = readFileSync(file, "utf8");
    const out = interpret(src, input);
    const ok = out === expect;
    if (ok) console.log(`  PASS: ${file} — ${JSON.stringify(out.slice(0, 50))}`);
    else {
      console.log(`  FAIL: ${file} — expected ${JSON.stringify(expect)}, got ${JSON.stringify(out)}`);
      failed++;
    }
  } catch (e) {
    console.log(`  FAIL: ${file} — ${e.message}`);
    failed++;
  }
}

if (failed > 0) {
  console.log(`\n${failed} samples failed.`);
  process.exitCode = 1;
} else {
  console.log("\nAll samples pass.");
}
