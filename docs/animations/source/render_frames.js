// node rec_min.js <dir> [scene numbers 1-8]  -> <dir>/frames_<n>/*.png
const { chromium } = require('playwright');
(async () => {
  const d = process.argv[2], only = process.argv.slice(3).map(Number), fps = 25;
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1200, height: 675 }, deviceScaleFactor: 1.6, colorScheme: 'light' });
  await p.goto('file://' + d + '/animation.html'); await p.waitForTimeout(800);
  const durs = await p.evaluate(() => window.__durs);
  const fs = require('fs');
  for (let i = 0; i < durs.length; i++) {
    if (only.length && !only.includes(i + 1)) continue;
    const dir = `${d}/frames_${i + 1}`; fs.rmSync(dir, { recursive: true, force: true }); fs.mkdirSync(dir);
    const n = Math.round((durs[i] + 1.0) * fps);
    for (let f = 0; f < n; f++) {
      await p.evaluate(([i, t]) => window.__render(i, t), [i, Math.min(f / fps, durs[i])]);
      await p.screenshot({ path: `${dir}/f${String(f).padStart(5, '0')}.png`, clip: { x: 0, y: 0, width: 1200, height: 675 } });
    }
    console.log('scene', i + 1, n, 'frames');
  }
  await b.close(); })();
