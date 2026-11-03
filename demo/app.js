// Page logic for the WebGL2 port: camera, feature presets from QuasarFeatures, quality settings,
// and a resolution governor that protects the frame rate.

const RS_M = 1.269e10;
const $ = (id) => document.getElementById(id);
const canvas = $('view');
const hud = $('hud');
const statusEl = $('status');

const VERTEX_SRC = `#version 300 es
void main() {
  // one triangle that covers the screen
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// The camera limits follow the C++ (1e12 m out). The inner limit stops short of its 1e10 m,
// which is inside the horizon, where initRay has no real solution.
const MIN_R = 1.8;
const MAX_R = 1e12 / RS_M;

const PHONE = matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 820;
const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)').matches;

// Quality presets. The C++ runs 32k to 256k steps of fixed length, far beyond what a fragment
// shader can spend per pixel; these are this port's budgets for its longer RK4 steps.
const QUALITY = {
  low: { label: 'Low', scale: 0.5, steps: 90, k: 0.16, maxDpr: 1 },
  medium: { label: 'Medium', scale: 0.75, steps: 150, k: 0.11, maxDpr: 1.25 },
  high: { label: 'High', scale: 1, steps: 260, k: 0.07, maxDpr: 1.5 },
  ultra: { label: 'Ultra', scale: 1, steps: 450, k: 0.04, maxDpr: 2 },
};

// switchToStandard and switchToTON618 in src/black_hole.cpp. The profile changes features only;
// the mass and therefore rs stay those of Sgr A*, as they do in the C++.
const PROFILES = {
  standard: {
    eddington: 0.1, tempPeak: 1e6, multiLayer: false, rotSpeed: 0.1, turbulence: 0.05,
    doppler: false, kerr: false, spin: 0, jets: false, jetAngle: 5, jetBrightness: 1,
  },
  ton618: {
    eddington: 0.7, tempPeak: 1e7, multiLayer: true, rotSpeed: 2, turbulence: 0.3,
    doppler: true, kerr: true, spin: 0.8, jets: true, jetAngle: 6, jetBrightness: 2,
  },
};

const state = {
  profile: PHONE ? 'standard' : 'ton618',
  ...(PHONE ? PROFILES.standard : PROFILES.ton618),
  corrected: true,
  stars: true,
  quality: PHONE ? 'low' : 'medium',
  autoRes: true,
  govern: 1,               // resolution multiplier set by the governor
  radius: 16,
  azimuth: 0.55,
  elevation: 1.32,
};

// --- WebGL ---------------------------------------------------------------------------------

const gl = canvas.getContext('webgl2', {
  alpha: false, antialias: false, depth: false, stencil: false, powerPreference: 'high-performance',
});

function fail(message) {
  statusEl.hidden = false;
  statusEl.textContent = message;
  canvas.setAttribute('aria-label', message);
}

let program = null;
const uniforms = {};
let running = false;

async function buildProgram() {
  statusEl.hidden = false;
  statusEl.textContent = 'Compiling the ray tracing shader...';
  const fragSrc = await (await fetch('geodesic.frag')).text();
  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    return s;
  };
  const vs = compile(gl.VERTEX_SHADER, VERTEX_SRC);
  const fs = compile(gl.FRAGMENT_SHADER, fragSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);

  // Compiling a long ray-marching shader can take seconds on Windows, so don't block the page.
  const parallel = gl.getExtension('KHR_parallel_shader_compile');
  if (parallel) {
    while (!gl.getProgramParameter(prog, parallel.COMPLETION_STATUS_KHR)) {
      await new Promise((r) => setTimeout(r, 50));
      if (gl.isContextLost()) return false;
    }
  }
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error(gl.getShaderInfoLog(fs), gl.getProgramInfoLog(prog));
    fail('This GPU could not compile the ray tracing shader, so the view is unavailable. The source is on GitHub.');
    return false;
  }
  program = prog;
  gl.useProgram(prog);
  const count = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {
    const { name } = gl.getActiveUniform(prog, i);
    uniforms[name] = gl.getUniformLocation(prog, name);
  }
  gl.bindVertexArray(gl.createVertexArray());
  statusEl.hidden = true;
  return true;
}

function cameraBasis() {
  const e = Math.min(Math.max(state.elevation, 0.01), Math.PI - 0.01);
  const pos = [
    state.radius * Math.sin(e) * Math.cos(state.azimuth),
    state.radius * Math.cos(e),
    state.radius * Math.sin(e) * Math.sin(state.azimuth),
  ];
  const fwd = pos.map((x) => -x / state.radius);
  const norm = Math.hypot(fwd[2], fwd[0]);
  const right = [-fwd[2] / norm, 0, fwd[0] / norm];
  const up = [
    right[1] * fwd[2] - right[2] * fwd[1],
    right[2] * fwd[0] - right[0] * fwd[2],
    right[0] * fwd[1] - right[1] * fwd[0],
  ];
  return { pos, fwd, right, up };
}

// --- Canvas sizing -------------------------------------------------------------------------

let dragging = false;
let lastInteraction = -1e9;
const isMoving = () => dragging || performance.now() - lastInteraction < 200;

function applySize() {
  const q = QUALITY[state.quality];
  const dpr = Math.min(window.devicePixelRatio || 1, q.maxDpr);
  const rect = canvas.getBoundingClientRect();
  // The C++ also drops to 0.7x resolution while the camera moves
  const scale = q.scale * (state.autoRes ? state.govern : 1) * (isMoving() ? 0.7 : 1);
  const w = Math.max(64, Math.round(rect.width * dpr * scale));
  const h = Math.max(64, Math.round(rect.height * dpr * scale));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

// --- Frame loop ----------------------------------------------------------------------------

let visible = true;
let raf = 0;
let frames = 0;
let windowStart = 0;
let lastFps = 0;
let lastCost = 0;
const startTime = performance.now();

function setFlag(name, value) { gl.uniform1i(uniforms[name], value ? 1 : 0); }

function draw(now) {
  applySize();
  const { pos, fwd, right, up } = cameraBasis();
  const q = QUALITY[state.quality];
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.uniform3fv(uniforms.uCamPos, pos);
  gl.uniform3fv(uniforms.uCamRight, right);
  gl.uniform3fv(uniforms.uCamUp, up);
  gl.uniform3fv(uniforms.uCamForward, fwd);
  gl.uniform1f(uniforms.uTanHalfFov, Math.tan((60 * Math.PI) / 360));
  gl.uniform1f(uniforms.uAspect, canvas.width / canvas.height);
  gl.uniform2f(uniforms.uResolution, canvas.width, canvas.height);
  gl.uniform1f(uniforms.uTime, REDUCED_MOTION ? 0 : (now - startTime) / 1000);
  setFlag('uMoving', isMoving());
  setFlag('uJitter', !REDUCED_MOTION);
  gl.uniform1f(uniforms.uEddington, state.eddington);
  gl.uniform1f(uniforms.uTempPeak, state.tempPeak);
  gl.uniform1f(uniforms.uRotSpeed, state.rotSpeed);
  gl.uniform1f(uniforms.uTurbulence, state.turbulence);
  gl.uniform1f(uniforms.uSpin, state.spin);
  gl.uniform1f(uniforms.uJetAngle, state.jetAngle);
  gl.uniform1f(uniforms.uJetBrightness, state.jetBrightness);
  setFlag('uMultiLayer', state.multiLayer);
  setFlag('uDoppler', state.doppler);
  setFlag('uKerr', state.kerr);
  setFlag('uJets', state.jets);
  setFlag('uCorrected', state.corrected);
  setFlag('uStars', state.stars);
  gl.uniform1i(uniforms.uMaxSteps, q.steps);
  gl.uniform1f(uniforms.uStepK, q.k);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function frame(now) {
  raf = requestAnimationFrame(frame);
  if (!running) return;
  const elapsed = now - windowStart;
  if (elapsed >= 800) {
    lastFps = (frames * 1000) / elapsed;
    frames = 0;
    windowStart = now;
    governResolution(measureFrameCost(now));
    updateHud();
  } else {
    draw(now);
  }
  frames += 1;
}

// Times one frame on the GPU by forcing it to finish. The frame rate alone can't drive the
// governor, because a browser also throttles requestAnimationFrame for tabs it considers hidden.
const probe = new Uint8Array(4);
function measureFrameCost(now) {
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, probe);
  const started = performance.now();
  draw(now);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, probe);
  lastCost = performance.now() - started;
  return lastCost;
}

function governResolution(costMs) {
  if (!state.autoRes || dragging) return;
  // Over 33 ms a frame is under 30 fps, so render fewer pixels. Add some back when there is room.
  if (costMs > 33) state.govern = Math.max(0.35, state.govern * (costMs > 55 ? 0.7 : 0.85));
  else if (costMs < 14 && state.govern < 1) state.govern = Math.min(1, state.govern * 1.08);
}

function updateHud() {
  const q = QUALITY[state.quality];
  hud.textContent = `${lastFps.toFixed(0)} fps · ${lastCost.toFixed(0)} ms GPU frame · ${canvas.width}×${canvas.height} · up to ${q.steps} steps per ray · ${state.kerr ? `Kerr mode, a/M ${state.spin.toFixed(2)}` : 'Schwarzschild'}`;
}

function start() {
  if (running || !program) return;
  running = visible && !document.hidden;
  if (running) {
    frames = 0;
    windowStart = performance.now();
  }
}

document.addEventListener('visibilitychange', () => {
  running = !document.hidden && visible && !!program;
  windowStart = performance.now();
  frames = 0;
});
new IntersectionObserver((entries) => {
  visible = entries[0].isIntersecting;
  running = visible && !document.hidden && !!program;
  windowStart = performance.now();
  frames = 0;
}).observe(canvas);

// --- Input ---------------------------------------------------------------------------------

const pointers = new Map();
let pinchDistance = 0;

function setRadius(r) {
  state.radius = Math.min(Math.max(r, MIN_R), MAX_R);
  $('zoom').value = String(Math.log(state.radius));
  lastInteraction = performance.now();
}

function orbit(dx, dy) {
  // orbitSpeed 0.01 rad per pixel, as in the C++
  state.azimuth += dx * 0.01;
  state.elevation = Math.min(Math.max(state.elevation - dy * 0.01, 0.01), Math.PI - 0.01);
  lastInteraction = performance.now();
}

canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  dragging = true;
  canvas.focus({ preventScroll: true });
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    pinchDistance = Math.hypot(a[0] - b[0], a[1] - b[1]);
  }
});
canvas.addEventListener('pointermove', (e) => {
  const previous = pointers.get(e.pointerId);
  if (!previous) return;
  const current = [e.clientX, e.clientY];
  pointers.set(e.pointerId, current);
  if (pointers.size === 1) {
    orbit(Math.min(Math.max(current[0] - previous[0], -100), 100), Math.min(Math.max(current[1] - previous[1], -100), 100));
  } else if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
    if (pinchDistance > 0) setRadius(state.radius * (pinchDistance / d));
    pinchDistance = d;
  }
});
const endPointer = (e) => {
  pointers.delete(e.pointerId);
  if (pointers.size === 0) {
    dragging = false;
    lastInteraction = performance.now();
  }
};
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  setRadius(state.radius * Math.exp(e.deltaY * 0.001));
}, { passive: false });
canvas.addEventListener('keydown', (e) => {
  const step = e.shiftKey ? 30 : 10;
  const moves = {
    ArrowLeft: () => orbit(-step, 0), ArrowRight: () => orbit(step, 0),
    ArrowUp: () => orbit(0, -step), ArrowDown: () => orbit(0, step),
    '+': () => setRadius(state.radius * 0.93), '=': () => setRadius(state.radius * 0.93), '-': () => setRadius(state.radius * 1.07),
  };
  if (moves[e.key]) {
    e.preventDefault();
    moves[e.key]();
  }
});

// Shortcut keys from handleFeatureKeys in src/black_hole.cpp. Not while typing in a form control.
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target.matches('input, select, textarea')) return;
  const key = e.key.toLowerCase();
  const toggle = (name) => { state[name] = !state[name]; syncControls(); };
  if (key === 'q') setProfile(state.profile === 'standard' ? 'ton618' : 'standard');
  else if (key === 'k') setKerr(!state.kerr);
  else if (key === 'j') toggle('jets');
  else if (key === 'm') toggle('multiLayer');
  else if (key === 'r') toggle('doppler');
  else if (['1', '2', '3', '4'].includes(key)) setQuality(['low', 'medium', 'high', 'ultra'][Number(key) - 1]);
});

function setProfile(name) {
  Object.assign(state, PROFILES[name], { profile: name });
  syncControls();
}
function setKerr(on) {
  state.kerr = on;
  // Kerr mode at spin 0 is identical to Schwarzschild, so start from the C++'s TON618 spin
  if (on && state.spin === 0) state.spin = PROFILES.ton618.spin;
  syncControls();
}
function setQuality(name) {
  state.quality = name;
  state.govern = 1;
  syncControls();
}

function syncControls() {
  for (const b of document.querySelectorAll('[data-profile]')) b.setAttribute('aria-pressed', String(b.dataset.profile === state.profile));
  for (const b of document.querySelectorAll('[data-metric]')) b.setAttribute('aria-pressed', String((b.dataset.metric === 'kerr') === state.kerr));
  $('jets').checked = state.jets;
  $('multi').checked = state.multiLayer;
  $('doppler').checked = state.doppler;
  $('stars').checked = state.stars;
  $('corrected').checked = state.corrected;
  $('autores').checked = state.autoRes;
  $('quality').value = state.quality;
  $('spin').value = String(state.spin);
  $('spin').disabled = !state.kerr;
  $('spin-out').textContent = state.spin.toFixed(2);
  updateHud();
}

document.querySelectorAll('[data-profile]').forEach((b) => b.addEventListener('click', () => setProfile(b.dataset.profile)));
document.querySelectorAll('[data-metric]').forEach((b) => b.addEventListener('click', () => setKerr(b.dataset.metric === 'kerr')));
$('jets').addEventListener('change', (e) => { state.jets = e.target.checked; });
$('multi').addEventListener('change', (e) => { state.multiLayer = e.target.checked; });
$('doppler').addEventListener('change', (e) => { state.doppler = e.target.checked; });
$('stars').addEventListener('change', (e) => { state.stars = e.target.checked; });
$('corrected').addEventListener('change', (e) => { state.corrected = e.target.checked; });
$('autores').addEventListener('change', (e) => { state.autoRes = e.target.checked; state.govern = 1; });
$('quality').addEventListener('change', (e) => setQuality(e.target.value));
$('spin').addEventListener('input', (e) => {
  state.spin = Number(e.target.value);
  $('spin-out').textContent = state.spin.toFixed(2);
});
$('zoom').min = String(Math.log(MIN_R));
$('zoom').max = String(Math.log(MAX_R));
$('zoom').addEventListener('input', (e) => { setRadius(Math.exp(Number(e.target.value))); });

canvas.addEventListener('webglcontextlost', (e) => {
  e.preventDefault();
  running = false;
  program = null;
  fail('The browser lost the GPU context. Trying to restore it...');
});
canvas.addEventListener('webglcontextrestored', async () => {
  if (await buildProgram()) {
    running = true;
    windowStart = performance.now();
  }
});

// --- Boot ----------------------------------------------------------------------------------

async function boot() {
  $('zoom').value = String(Math.log(state.radius));
  syncControls();
  if (!gl) {
    document.body.classList.add('no-webgl');
    fail('This browser or device does not have WebGL2, which the ray tracer needs. The page below still explains the project, and the source is on GitHub.');
    return;
  }
  if (await buildProgram()) {
    running = visible && !document.hidden;
    windowStart = performance.now();
    raf = requestAnimationFrame(frame);
  }
}

// For checking the page from the browser console.
window.blackHole = { state, cameraBasis, get fps() { return lastFps; }, get frameMs() { return lastCost; } };

boot().catch((err) => {
  console.error(err);
  fail('The ray tracer could not start. The source is on GitHub.');
});
