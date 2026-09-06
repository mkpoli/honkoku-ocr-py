// Execute the original TypeScript unchanged using Node's type stripping.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
const root = process.argv[2];
const { ReadingOrderProcessor } = await import(pathToFileURL(resolve(root, 'src/ocr/reading-order.ts')));
const { rawToKoji, rawToPlain } = await import(pathToFileURL(resolve(root, 'src/lib/koji.ts')));
const { cases, texts, repeats } = JSON.parse(readFileSync(0, 'utf8'));
const processor = new ReadingOrderProcessor();
function runOrder() {
  return cases.map(boxes => {
    const lines = boxes.map(([x, y, width, height], id) => ({ x, y, width, height, id }));
    const ranks = Array(lines.length);
    for (const line of processor.orderLines(lines)) ranks[line.id] = line.readingOrder - 1;
    return ranks;
  });
}
function runText() { return texts.map(text => [rawToKoji(text), rawToPlain(text)]); }
function measure(fn) {
  for (let i = 0; i < 3; i++) fn();
  const samples = [];
  let output;
  for (let i = 0; i < repeats; i++) {
    const start = performance.now();
    output = fn();
    samples.push(performance.now() - start);
  }
  return { samples_ms: samples, output };
}
console.log(JSON.stringify({ order: measure(runOrder), text: measure(runText) }));
