// Runs the demo page's own proof verifiers against vectors made by the Python implementation.
// Usage: node verify_page.mjs index.html vectors.json
import fs from "node:fs";
import vm from "node:vm";

const [htmlPath, vectorPath] = process.argv.slice(2);
const html = fs.readFileSync(htmlPath, "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];
const context = vm.createContext({ crypto: globalThis.crypto, TextEncoder, atob, DOMException, BigInt });
vm.runInContext(`${script}\n;globalThis.page = { verifyInclusion, verifyConsistency, hexToBytes, b45decode };`, context);
const { verifyInclusion, verifyConsistency, hexToBytes, b45decode } = context.page;

const vectors = JSON.parse(fs.readFileSync(vectorPath, "utf8"));
const flip = (bytes) => bytes.map((b, i) => (i === 0 ? b ^ 1 : b));
const failures = [];

for (const [index, size, leaf, path, root] of vectors.inclusion) {
  const proof = path.map(hexToBytes);
  if (!(await verifyInclusion(hexToBytes(leaf), index, size, proof, hexToBytes(root)))) {
    failures.push(`inclusion ${index}/${size} rejected`);
  }
  if (proof.length && (await verifyInclusion(hexToBytes(leaf), index, size, [flip(proof[0]), ...proof.slice(1)], hexToBytes(root)))) {
    failures.push(`tampered inclusion ${index}/${size} accepted`);
  }
}
for (const [first, second, rootFirst, rootSecond, path] of vectors.consistency) {
  const proof = path.map(hexToBytes);
  if (!(await verifyConsistency(first, second, hexToBytes(rootFirst), hexToBytes(rootSecond), proof))) {
    failures.push(`consistency ${first}->${second} rejected`);
  }
  if (proof.length && (await verifyConsistency(first, second, hexToBytes(rootFirst), hexToBytes(rootSecond), [flip(proof[0]), ...proof.slice(1)]))) {
    failures.push(`tampered consistency ${first}->${second} accepted`);
  }
}
for (const [text, hex] of vectors.base45) {
  if (Buffer.from(b45decode(text)).toString("hex") !== hex) failures.push(`base45 ${text}`);
}

console.log(JSON.stringify({ checked: vectors.inclusion.length + vectors.consistency.length, failures }));
