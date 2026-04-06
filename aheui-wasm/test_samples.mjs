// 웹 데모에 포함할 샘플들이 실제로 동작하는지 검증.
import { readFileSync } from "node:fs";
import { interpret } from "./pkg-node/aheui_wasm.js";

const cases = [
  { file: "web/samples/hello.aheui",        input: "", expect: "안녕하세요?\n" },
  { file: "web/samples/hello-world.aheui",  input: "", expect: "Hello, world!\n" },
  { file: "web/samples/factorial.aheui",    input: "5\n", expect: "120" },
  { file: "web/samples/fibonacci.aheui",    input: "10\n", expect: "55" },
];

let failed = 0;
for (const { file, input, expect } of cases) {
  try {
    const src = readFileSync(file, "utf8");
    const out = interpret(src, input);
    const ok = out.includes(expect);
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
