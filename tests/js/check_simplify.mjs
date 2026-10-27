// Run with: node tests/js/check_simplify.mjs  (exits non-zero on failure)
import { simplify } from '../../src/doodl/static/stage.js';

const fail = (message) => {
  console.error(message);
  process.exit(1);
};

const line = Array.from({ length: 50 }, (_, i) => ({ x: i / 50, y: 0.5 }));
const flat = simplify(line, 0.001);
if (flat.length !== 2) fail(`collinear points should reduce to the endpoints, got ${flat.length}`);

const across = Array.from({ length: 20 }, (_, i) => ({ x: i / 40, y: 0 }));
const down = Array.from({ length: 20 }, (_, i) => ({ x: 0.5, y: i / 40 }));
const bent = simplify([...across, ...down], 0.001);
if (bent.length !== 3) fail(`an L shape should keep three points, got ${bent.length}`);

const huge = Array.from({ length: 200000 }, (_, i) => ({ x: Math.cos(i / 1000), y: Math.sin(i / 1000) }));
if (simplify(huge, 0.01).length < 10) fail('a long stroke must survive simplification');

if (simplify([{ x: 0, y: 0 }], 0.1).length !== 1) fail('a single point stays a single point');
console.log('ok');
