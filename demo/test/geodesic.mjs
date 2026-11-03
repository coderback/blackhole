// Checks the WebGL port's integrator against the C++/GLSL original, in double precision.
//
// Reference: geodesicRHS / geodesicRHS_Kerr / rk4Step from shaders/geodesic.comp, with its fixed
//            step of 1e7 m and its z polar axis, in units of the Schwarzschild radius.
// Port:      the same right-hand sides under true RK4, step proportional to r, in a per-ray
//            rotated frame so the ray stays away from the coordinate pole.
//
// Run with: node demo/test/geodesic.mjs

const RS_M = 1.269e10;
const D_LAMBDA = 1e7 / RS_M; // the C++ step, in Schwarzschild radii
const { sin, cos, atan2, acos, sqrt, hypot, abs, max, PI } = Math;
const clamp = (x, a, b) => Math.min(Math.max(x, a), b);

function initRay(pos, dir) {
  const r = hypot(...pos);
  const theta = acos(pos[2] / r);
  const phi = atan2(pos[1], pos[0]);
  const [dx, dy, dz] = dir;
  const dr = sin(theta) * cos(phi) * dx + sin(theta) * sin(phi) * dy + cos(theta) * dz;
  const dtheta = (cos(theta) * cos(phi) * dx + cos(theta) * sin(phi) * dy - sin(theta) * dz) / r;
  const dphi = (-sin(phi) * dx + cos(phi) * dy) / (r * sin(theta));
  const f = 1 - 1 / r;
  const dtdL = sqrt((dr * dr) / f + r * r * (dtheta * dtheta + sin(theta) ** 2 * dphi * dphi));
  return { r, theta, phi, dr, dtheta, dphi, E: f * dtdL };
}

// The angular term of the radial equation. The C++ has r * (...). The Schwarzschild Christoffel
// symbol gives (r - rs) * (...), which is what `corrected` switches on.
const angularCoefficient = (r, corrected) => (corrected ? r - 1 : r);

function rhsSchwarzschild(s, E, corrected) {
  const { r, theta, dr, dtheta, dphi } = s;
  const f = 1 - 1 / r;
  const dtdL = E / f;
  return {
    d2r: -(1 / (2 * r * r)) * f * dtdL * dtdL + (1 / (2 * r * r * f)) * dr * dr
      + angularCoefficient(r, corrected) * (dtheta * dtheta + sin(theta) ** 2 * dphi * dphi),
    d2t: (-2 * dr * dtheta) / r + sin(theta) * cos(theta) * dphi * dphi,
    d2p: (-2 * dr * dphi) / r - ((2 * cos(theta)) / sin(theta)) * dtheta * dphi,
  };
}

// The clamps of geodesicRHS_Kerr, in the original z-axis frame
function rhsKerrReference(s, E, spin) {
  const { dr, dtheta, dphi } = s;
  const r = max(s.r, 0.5);
  const theta = clamp(s.theta, 0.01, 3.14159 - 0.01);
  const f = max(1 - 1 / r, 0.01);
  const dtdL = E / f;
  const sinT = sin(theta);
  const cot = abs(sinT) > 0.01 ? cos(theta) / sinT : 0;
  const out = {
    d2r: -(1 / (2 * r * r)) * f * dtdL * dtdL + (1 / (2 * r * r * f)) * dr * dr
      + r * (dtheta * dtheta + sinT * sinT * dphi * dphi),
    d2t: (-2 * dr * dtheta) / r + sinT * cos(theta) * dphi * dphi,
    d2p: (-2 * dr * dphi) / r - 2 * cot * dtheta * dphi,
  };
  if (spin > 0.01 && r > 2) out.d2p += ((0.1 * spin * 0.2) / (r * r * r)) * sinT * sinT * E;
  return out;
}

const velocityVector = (s) => {
  const st = sin(s.theta), ct = cos(s.theta), sp = sin(s.phi), cp = cos(s.phi);
  return [
    s.dr * st * cp + s.r * s.dtheta * ct * cp - s.r * s.dphi * st * sp,
    s.dr * st * sp + s.r * s.dtheta * ct * sp + s.r * s.dphi * st * cp,
    s.dr * ct - s.r * s.dtheta * st,
  ];
};
const norm = (v) => { const n = hypot(...v); return v.map((x) => x / n); };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const angle = (a, b) => acos(clamp(dot(a, b), -1, 1));

// The original integrator: one forward Euler sub-step per iteration (the shader's rk4Step).
function traceReference(pos, dir, rhs, horizon, maxSteps = 6e6, stopR = 60) {
  const s = initRay(pos, dir);
  for (let i = 0; i < maxSteps; i++) {
    if (s.r <= horizon) return { captured: true, steps: i };
    const a = rhs(s, s.E);
    s.r += D_LAMBDA * s.dr;
    s.theta += D_LAMBDA * s.dtheta;
    s.phi += D_LAMBDA * s.dphi;
    s.dr += D_LAMBDA * a.d2r;
    s.dtheta += D_LAMBDA * a.d2t;
    s.dphi += D_LAMBDA * a.d2p;
    if (s.r > stopR && s.dr > 0) return { captured: false, steps: i, dir: norm(velocityVector(s)) };
  }
  return { captured: null, steps: maxSteps };
}

// The port, as written in demo/geodesic.frag.
function tracePort(pos, dir, { k, spin = 0, corrected = false, kerr = false, maxSteps = 400 }) {
  const e1 = norm(pos);
  const e3 = norm(cross(pos, dir));
  const e2 = cross(e3, e1);
  const toLocal = (v) => [dot(v, e1), dot(v, e2), dot(v, e3)];
  const toWorld = (v) => [0, 1, 2].map((i) => v[0] * e1[i] + v[1] * e2[i] + v[2] * e3[i]);
  const s = initRay(toLocal(pos), toLocal(dir));
  const horizon = kerr ? 0.5 * (1 + sqrt(max(1 - spin * spin, 0))) : 1;

  const rhs = (st) => {
    const r = max(st.r, 0.5);
    let f = 1 - 1 / r;
    f = kerr ? max(f, 0.01) : (abs(f) < 1e-3 ? 1e-3 : f);
    const dtdL = s.E / f;
    const sinT = sin(st.theta), cosT = cos(st.theta);
    const out = {
      d2r: -(1 / (2 * r * r)) * f * dtdL * dtdL + (1 / (2 * r * r * f)) * st.dr * st.dr
        + angularCoefficient(r, corrected) * (st.dtheta * st.dtheta + sinT * sinT * st.dphi * st.dphi),
      d2t: (-2 * st.dr * st.dtheta) / r + sinT * cosT * st.dphi * st.dphi,
      d2p: (-2 * st.dr * st.dphi) / r - ((2 * cosT) / sinT) * st.dtheta * st.dphi,
    };
    if (kerr && spin > 0.01 && r > 2) {
      // The frame-drag term acts about the world z axis, as in the original. It is written here
      // through its Cartesian form K sin^2(theta_w) E (-y, x, 0) and projected into this frame.
      const w = toWorld([r * sinT * cos(st.phi), r * sinT * sin(st.phi), r * cosT]);
      const K = (0.1 * spin * 0.2) / (r * r * r);
      const sin2w = (w[0] * w[0] + w[1] * w[1]) / (r * r);
      const a = toLocal([-w[1] * K * sin2w * s.E, w[0] * K * sin2w * s.E, 0]);
      const thetaHat = [cosT * cos(st.phi), cosT * sin(st.phi), -sinT];
      const phiHat = [-sin(st.phi), cos(st.phi), 0];
      out.d2t += dot(a, thetaHat) / r;
      out.d2p += dot(a, phiHat) / (r * sinT);
    }
    return out;
  };
  const stage = (st, acc, h) => ({
    r: st.r + h * st.dr, theta: st.theta + h * st.dtheta, phi: st.phi + h * st.dphi,
    dr: st.dr + h * acc.d2r, dtheta: st.dtheta + h * acc.d2t, dphi: st.dphi + h * acc.d2p,
  });
  let steps = 0;
  for (; steps < maxSteps; steps++) {
    // with the corrected term nothing inside the photon sphere comes back out
    if (s.r <= horizon || (corrected && s.r < 1.5 && s.dr < 0)) return { captured: true, steps };
    if (s.r > 60 && s.dr > 0) return { captured: false, steps, dir: norm(toWorld(velocityVector(s))) };
    const h = k * s.r * (corrected ? 1 : clamp(1.5 * (s.r - 1), 0.08, 1));
    const a1 = rhs(s);
    const s2 = { ...stage(s, a1, h / 2) };
    const a2 = rhs(s2);
    const s3 = { r: s.r + (h / 2) * s2.dr, theta: s.theta + (h / 2) * s2.dtheta, phi: s.phi + (h / 2) * s2.dphi,
      dr: s.dr + (h / 2) * a2.d2r, dtheta: s.dtheta + (h / 2) * a2.d2t, dphi: s.dphi + (h / 2) * a2.d2p };
    const a3 = rhs(s3);
    const s4 = { r: s.r + h * s3.dr, theta: s.theta + h * s3.dtheta, phi: s.phi + h * s3.dphi,
      dr: s.dr + h * a3.d2r, dtheta: s.dtheta + h * a3.d2t, dphi: s.dphi + h * a3.d2p };
    const a4 = rhs(s4);
    s.r += (h / 6) * (s.dr + 2 * s2.dr + 2 * s3.dr + s4.dr);
    s.theta += (h / 6) * (s.dtheta + 2 * s2.dtheta + 2 * s3.dtheta + s4.dtheta);
    s.phi += (h / 6) * (s.dphi + 2 * s2.dphi + 2 * s3.dphi + s4.dphi);
    s.dr += (h / 6) * (a1.d2r + 2 * a2.d2r + 2 * a3.d2r + a4.d2r);
    s.dtheta += (h / 6) * (a1.d2t + 2 * a2.d2t + 2 * a3.d2t + a4.d2t);
    s.dphi += (h / 6) * (a1.d2p + 2 * a2.d2p + 2 * a3.d2p + a4.d2p);
  }
  return { captured: null, steps };
}

// A ray from x = 50 travelling in -x. The offset is tilted away from the z axis so the reference's
// coordinate pole is not on the path.
function setup(b, tilt = 0.7) {
  return { pos: [50, b * cos(tilt), b * sin(tilt)], dir: [-1, 0, 0] };
}

function captureEdge(capturedAt, lo, hi) {
  for (let i = 0; i < 22; i++) {
    const mid = (lo + hi) / 2;
    if (capturedAt(mid)) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

const KS = [0.2, 0.13, 0.08, 0.05];

console.log('1. As in the C++: Euler reference vs port (RK4, step = k * r). Deflection in radians.');
console.log('     b | reference      steps | ' + KS.map((k) => `k=${k}: error, steps`).join(' | '));
for (const b of [1.0, 1.15, 1.3, 1.5, 2, 3, 4, 6, 10, 20]) {
  const { pos, dir } = setup(b);
  const ref = traceReference(pos, dir, (s, E) => rhsSchwarzschild(s, E, false), 1);
  const refAngle = ref.captured ? null : angle(dir, ref.dir);
  const cells = KS.map((k) => {
    const p = tracePort(pos, dir, { k });
    if (p.captured === null) return `over budget (${p.steps})`;
    if (p.captured || ref.captured) return `${p.captured ? 'captured' : 'escaped'}${p.captured === ref.captured ? '' : ' MISMATCH'}, ${p.steps}`;
    return `${(angle(dir, p.dir) - refAngle).toExponential(1)}, ${p.steps}`;
  });
  console.log(`${b.toFixed(2).padStart(6)} | ${(ref.captured ? 'captured' : refAngle.toFixed(4)).padStart(9)} ${String(ref.steps).padStart(9)} | ${cells.join(' | ')}`);
}

console.log('\n2. Edge of the shadow (largest captured impact parameter, in rs)');
const edgeRefVerbatim = captureEdge((b) => { const { pos, dir } = setup(b); return traceReference(pos, dir, (s, E) => rhsSchwarzschild(s, E, false), 1).captured; }, 0.5, 3);
console.log(`   C++ equations, Euler reference: ${edgeRefVerbatim.toFixed(4)}`);
const atEdge = setup(1.74);
console.log(`   steps for a captured ray at b=1.74: reference ${traceReference(atEdge.pos, atEdge.dir, (s, E) => rhsSchwarzschild(s, E, false), 1).steps}, port k=0.13 ${tracePort(atEdge.pos, atEdge.dir, { k: 0.13 }).steps}`);
for (const k of KS) {
  const edge = captureEdge((b) => { const { pos, dir } = setup(b); return tracePort(pos, dir, { k }).captured; }, 0.5, 3);
  console.log(`   C++ equations, port k=${k}: ${edge.toFixed(4)}`);
}
console.log(`   analytic Schwarzschild shadow, 3*sqrt(3)/2: ${((3 * sqrt(3)) / 2).toFixed(4)}`);
for (const k of KS) {
  const edge = captureEdge((b) => { const { pos, dir } = setup(b); return tracePort(pos, dir, { k, corrected: true }).captured === true; }, 1, 4);
  console.log(`   corrected term, port k=${k}: ${edge.toFixed(4)}`);
}

console.log('\n3. Weak-field deflection against the GR value 2 rs / b');
for (const b of [20, 30, 45]) {
  const { pos, dir } = setup(b);
  const verbatim = tracePort(pos, dir, { k: 0.05, maxSteps: 2000 });
  const corrected = tracePort(pos, dir, { k: 0.05, corrected: true, maxSteps: 2000 });
  console.log(`   b=${b}: 2/b=${(2 / b).toFixed(4)}  C++ equations=${angle(dir, verbatim.dir).toFixed(4)}  corrected=${angle(dir, corrected.dir).toFixed(4)}  (both from x=50 to r=60, not from infinity)`);
}

console.log('\n4. Kerr mode, spin 0.8, against the Euler reference of geodesicRHS_Kerr (horizon 0.8 rs)');
for (const b of [0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2, 3, 6]) {
  for (const tilt of [0.7, PI / 2 - 0.05]) {
    const { pos, dir } = setup(b, tilt);
    const ref = traceReference(pos, dir, (s, E) => rhsKerrReference(s, E, 0.8), 0.8);
    const refAngle = ref.captured ? null : angle(dir, ref.dir);
    const p = tracePort(pos, dir, { k: 0.13, spin: 0.8, kerr: true });
    const cell = p.captured || ref.captured || p.captured === null || ref.captured === null
      ? `${p.captured === null ? 'over budget' : p.captured ? 'captured' : 'escaped'}${p.captured === ref.captured ? '' : ' MISMATCH'}`
      : (angle(dir, p.dir) - refAngle).toExponential(1);
    console.log(`   b=${b.toFixed(2)} tilt=${tilt.toFixed(2)} reference=${ref.captured === null ? 'over' : ref.captured ? 'captured' : refAngle.toFixed(4)} port=${cell} steps=${p.steps}`);
  }
}
