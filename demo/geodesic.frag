#version 300 es
// WebGL2 port of shaders/geodesic.comp. Lengths are in units of the Schwarzschild radius, so
// SagA_rs becomes 1 and each equation keeps its form. Where the original needs a length in metres
// (a few colour terms), RS_M converts back. Comments mark every place the port differs.
precision highp float;
precision highp int;

out vec4 fragColor;

uniform vec3 uCamPos, uCamRight, uCamUp, uCamForward;
uniform float uTanHalfFov, uAspect;
uniform bool uMoving;
uniform vec2 uResolution;
uniform float uTime;
uniform bool uJitter;

// Features block of the original
uniform float uEddington, uTempPeak, uRotSpeed, uTurbulence, uSpin, uJetAngle, uJetBrightness;
uniform bool uMultiLayer, uDoppler, uKerr, uJets;
uniform int uMaxSteps;

// Added by the port
uniform float uStepK;     // RK4 step, as a fraction of r
uniform bool uCorrected;  // use (r - rs) in the radial equation instead of the original's r
uniform bool uStars;      // backdrop for escaped rays, the original leaves them black

const float PI = 3.14159265;
const float RS_M = 1.269e10;            // SagA_rs in the original, metres
const float D_LAMBDA = 1e7 / RS_M;      // the original's fixed step, in rs
const float DISK_R1 = 2.2;              // uploadDiskUBO: SagA.r_s * 2.2
const float DISK_R2 = 5.2;              // uploadDiskUBO: SagA.r_s * 5.2
const float ESCAPE_R = 45.0;            // port: past the reference spheres and heading out

// updateObjects: reference spheres at 4e11 m with radius 4e10 m
const vec3 OBJ_POS_0 = vec3(4e11 / RS_M, 0.0, 0.0);
const vec3 OBJ_POS_1 = vec3(0.0, 0.0, 4e11 / RS_M);
const float OBJ_RADIUS = 4e10 / RS_M;

// Port: each ray gets its own polar axis, perpendicular to the plane it starts in. The equations
// are the same in any rotated frame and this keeps theta near pi/2, away from the cot(theta)
// singularity that the original can only afford to cross because its steps are so small.
vec3 e1, e2, e3;
float E;

vec3 toWorld(vec3 q) {
  float st = sin(q.y);
  return q.x * (st * cos(q.z) * e1 + st * sin(q.z) * e2 + cos(q.y) * e3);
}

// geodesicRHS and geodesicRHS_Kerr. q = (r, theta, phi), v = its derivatives, returns the second
// derivatives. Both branches start from the same Schwarzschild terms, as the original does.
vec3 geodesicRHS(vec3 q, vec3 v) {
  float r = q.x;
  float dr = v.x, dtheta = v.y, dphi = v.z;
  float f;
  if (uKerr) {
    r = max(r, 0.5);                    // Stay outside event horizon
    f = max(1.0 - 1.0 / r, 0.01);       // Prevent division by zero
  } else {
    f = 1.0 - 1.0 / r;
    if (abs(f) < 1e-3) f = 1e-3;        // port: single precision needs this guard at f = 0
  }
  float dt_dL = E / f;
  float sinT = sin(q.y), cosT = cos(q.y);
  float angular = dtheta * dtheta + sinT * sinT * dphi * dphi;

  vec3 d2;
  // The original's last term is r * angular. The Schwarzschild Christoffel symbol gives
  // (r - rs) * angular, available through uCorrected.
  d2.x = -(1.0 / (2.0 * r * r)) * f * dt_dL * dt_dL
       + (1.0 / (2.0 * r * r * f)) * dr * dr
       + (uCorrected ? r - 1.0 : r) * angular;
  d2.y = -2.0 * dr * dtheta / r + sinT * cosT * dphi * dphi;
  d2.z = -2.0 * dr * dphi / r - 2.0 * (cosT / sinT) * dtheta * dphi;

  // Lense-Thirring style correction of geodesicRHS_Kerr. In the original it adds
  // 0.1 * (0.2 * spin) * rs / r^3 * sin^2(theta) * E to d2phi, about the z axis, for r > 2 rs.
  // Here that same vector is projected into this ray's frame.
  if (uKerr && uSpin > 0.01 && r > 2.0) {
    float K = 0.1 * (uSpin * 0.2) / (r * r * r);
    vec3 w = toWorld(vec3(r, q.y, q.z));
    float sin2w = dot(w.xy, w.xy) / (r * r);
    vec3 a = K * sin2w * E * vec3(-w.y, w.x, 0.0);
    vec3 aLocal = vec3(dot(a, e1), dot(a, e2), dot(a, e3));
    float cp = cos(q.z), sp = sin(q.z);
    d2.y += dot(aLocal, vec3(cosT * cp, cosT * sp, -sinT)) / r;
    d2.z += dot(aLocal, vec3(-sp, cp, 0.0)) / (r * sinT);
  }
  return d2;
}

// Port: the original's rk4Step takes one forward Euler step per iteration. This is true RK4 on the
// same right-hand side, because a fragment shader has hundreds of steps per ray, not 64,000.
void rk4Step(inout vec3 q, inout vec3 v, float h) {
  vec3 a1 = geodesicRHS(q, v);
  vec3 v2 = v + 0.5 * h * a1;
  vec3 a2 = geodesicRHS(q + 0.5 * h * v, v2);
  vec3 v3 = v + 0.5 * h * a2;
  vec3 a3 = geodesicRHS(q + 0.5 * h * v2, v3);
  vec3 v4 = v + h * a3;
  vec3 a4 = geodesicRHS(q + h * v3, v4);
  q += (h / 6.0) * (v + 2.0 * v2 + 2.0 * v3 + v4);
  v += (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
}

// Port: closest approach over the step, since the steps are far longer than in the original.
bool segmentHitsSphere(vec3 a, vec3 b, vec3 c, float radius) {
  vec3 ab = b - a;
  float t = clamp(dot(c - a, ab) / max(dot(ab, ab), 1e-9), 0.0, 1.0);
  return distance(a + t * ab, c) <= radius;
}

float hash13(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}

// Port: not in the original, whose escaped rays stay black. Makes the lensing visible.
vec3 starfield(vec3 d) {
  vec3 col = vec3(0.0);
  for (int layer = 0; layer < 2; layer++) {
    float scale = layer == 0 ? 60.0 : 130.0;
    vec3 p = d * scale;
    vec3 cell = floor(p);
    float h = hash13(cell + float(layer) * 17.0);
    if (h > 0.93) {
      vec3 offset = vec3(hash13(cell + 1.3), hash13(cell + 2.7), hash13(cell + 4.1)) - 0.5;
      float dist = length(fract(p) - 0.5 - 0.5 * offset);
      float b = smoothstep(0.32, 0.0, dist) * (0.35 + 0.65 * fract(h * 37.0));
      col += b * mix(vec3(1.0, 0.82, 0.68), vec3(0.72, 0.84, 1.0), fract(h * 91.0));
    }
  }
  return col;
}

void main() {
  vec2 pix = gl_FragCoord.xy - 0.5;
  float WIDTH = uResolution.x, HEIGHT = uResolution.y;

  // Initialize ray with anti-aliasing jitter
  float jitterX = 0.0, jitterY = 0.0;
  if (uJitter) {
    jitterX = (sin(uTime * 1.7 + pix.x * 0.1) * 0.5 + 0.5) * 0.8 - 0.4;
    jitterY = (cos(uTime * 2.3 + pix.y * 0.1) * 0.5 + 0.5) * 0.8 - 0.4;
  }
  float u = (2.0 * (pix.x + 0.5 + jitterX) / WIDTH - 1.0) * uAspect * uTanHalfFov;
  float v0 = (1.0 - 2.0 * (pix.y + 0.5 + jitterY) / HEIGHT) * uTanHalfFov;
  vec3 dir = normalize(u * uCamRight - v0 * uCamUp + uCamForward);

  // The per-ray frame: the camera sits at theta = pi/2, phi = 0
  float r0 = length(uCamPos);
  e1 = uCamPos / r0;
  e3 = cross(uCamPos, dir);
  float planeLen = length(e3);
  e3 = planeLen > 1e-5 ? e3 / planeLen : normalize(cross(e1, abs(e1.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  e2 = cross(e3, e1);

  // initRay
  vec3 q = vec3(r0, 0.5 * PI, 0.0);
  vec3 v = vec3(dot(dir, e1), -dot(dir, e3) / r0, dot(dir, e2) / r0);
  float f0 = 1.0 - 1.0 / r0;
  E = f0 * sqrt((v.x * v.x) / f0 + r0 * r0 * (v.y * v.y + v.z * v.z));

  // intercept: Kerr horizon r+ = M + sqrt(M^2 - a^2) with M = rs / 2, or rs
  float horizon = uKerr ? 0.5 * (1.0 + sqrt(max(1.0 - uSpin * uSpin, 0.0))) : 1.0;

  vec4 color = vec4(0.0);
  vec3 prevPos = uCamPos;
  bool hitBlackHole = false, hitDisk = false, hitObject = false, escaped = false;
  int diskLayer = 0;
  vec3 hitPos = vec3(0.0), hitCenter = vec3(0.0), objectColor = vec3(0.0);
  vec4 jetAccumulation = vec4(0.0);

  // Adaptive budget, same factors as the original
  float steps = float(uMaxSteps);
  float adaptiveFactor = 1.0;
  if (r0 > 200.0) adaptiveFactor = 0.4;
  else if (r0 > 100.0) adaptiveFactor = 0.6;
  else if (r0 > 50.0) adaptiveFactor = 0.75;
  else if (r0 > 20.0) adaptiveFactor = 0.85;
  else if (r0 > 10.0) adaptiveFactor = 0.95;
  steps *= adaptiveFactor;
  float complexityFactor = 1.0;
  if (uMultiLayer) complexityFactor += 0.1;
  if (uDoppler) complexityFactor += 0.15;
  if (uKerr) complexityFactor += 0.2;
  if (uJets) complexityFactor += 0.1;
  steps *= min(complexityFactor, 1.4);
  if (uMoving) steps *= 0.7;
  int stepCount = int(steps);

  float jetTan = tan(radians(uJetAngle * 0.5));

  for (int i = 0; i < stepCount; ++i) {
    float r = q.x;
    if (r <= horizon) { hitBlackHole = true; break; }
    // With the corrected term, nothing heading inward inside the photon sphere comes back out.
    if (uCorrected && r < 1.5 && v.x < 0.0) { hitBlackHole = true; break; }

    // Port: step proportional to r in place of the fixed 1e7 m. The original's equations have a
    // 1 / f term that is stiff near r = rs, so the step shrinks there unless uCorrected removes it.
    float h = uStepK * r * (uCorrected ? 1.0 : clamp(1.5 * (r - 1.0), 0.08, 1.0));
    if (uJets) {
      // Port: slow down near the jet so long steps cannot jump across it
      float axisDist = length(prevPos.xz);
      float jetR = max(abs(prevPos.y) * jetTan, 0.1);
      float toJet = max(axisDist - jetR, 3.0 - abs(prevPos.y));
      h = min(h, max(0.3 * uStepK * max(1.0, 0.25 * r), 0.6 * toJet));
    }

    rk4Step(q, v, h);
    vec3 newPos = toWorld(q);

    // interceptObject
    if (segmentHitsSphere(prevPos, newPos, OBJ_POS_0, OBJ_RADIUS)) {
      hitObject = true; hitCenter = OBJ_POS_0; objectColor = vec3(1.0, 1.0, 0.0); hitPos = newPos; break;
    }
    if (segmentHitsSphere(prevPos, newPos, OBJ_POS_1, OBJ_RADIUS)) {
      hitObject = true; hitCenter = OBJ_POS_1; objectColor = vec3(1.0, 0.0, 0.0); hitPos = newPos; break;
    }

    // crossesJetRegion and the energy accumulation that follows it
    if (uJets && abs(newPos.y) > 3.0) {
      float jetAxisDistance = length(newPos.xz);
      float jetHeight = abs(newPos.y);
      float jetRadius = max(jetHeight * jetTan, 0.1);
      if (jetAxisDistance <= jetRadius) {
        float radialDistance = jetAxisDistance / jetRadius;
        float beam_core = 1.0 - smoothstep(0.0, 0.3, radialDistance);
        beam_core = pow(beam_core, 4.0);
        float beam_glow = 1.0 - smoothstep(0.2, 1.0, radialDistance);
        beam_glow = pow(beam_glow, 2.0) * 0.3;
        float beam_intensity = beam_core + beam_glow;
        if (beam_intensity > 0.01) {
          vec3 jetColor = mix(vec3(0.3, 0.8, 1.0), vec3(1.0), beam_core / max(beam_intensity, 0.001));
          // Original: sin(time * 8.0 - jetHeight * 0.0001) with jetHeight in metres, which moves
          // about 1000 radians per step of 1e7 m and so averages to a constant 0.5 along a ray.
          // Sampled at this port's coarser steps it would only flicker, so the mean is used.
          float energy_flow = 0.5;
          jetColor *= 1.0 + energy_flow * 0.3;
          float proximity_boost = 1.0 / (1.0 + jetHeight / 20.0);
          jetColor *= 1.0 + proximity_boost * 2.0;
          // Original adds beam * 0.02 per 1e7 m step. Scaled by this step's length so the total
          // along a path is the same.
          float stepAlpha = beam_intensity * 0.02 * (h / D_LAMBDA);
          jetAccumulation.rgb += jetColor * stepAlpha * uJetBrightness;
          jetAccumulation.a += stepAlpha;
        }
      }
    }

    // crossesMultiLayerDisk. Port: tested at the interpolated crossing point of the y = 0 plane.
    if (prevPos.y * newPos.y < 0.0) {
      vec3 crossing = mix(prevPos, newPos, prevPos.y / (prevPos.y - newPos.y));
      float rr = length(crossing.xz);
      if (uMultiLayer) {
        float layer1_outer = DISK_R1 + (DISK_R2 - DISK_R1) * 0.4;
        float layer2_outer = DISK_R1 + (DISK_R2 - DISK_R1) * 0.7;
        if (rr >= DISK_R1 && rr <= layer1_outer) diskLayer = 1;
        else if (rr > layer1_outer && rr <= layer2_outer) diskLayer = 2;
        else if (rr > layer2_outer && rr <= DISK_R2) diskLayer = 3;
      } else if (rr >= DISK_R1 && rr <= DISK_R2) {
        diskLayer = 1;
      }
      if (diskLayer > 0) { hitDisk = true; hitPos = crossing; break; }
    }
    prevPos = newPos;
    if (r > ESCAPE_R && v.x > 0.0) { escaped = true; break; }
  }

  if (hitDisk) {
    float r_radial = length(hitPos.xz);
    float r_metres = r_radial * RS_M;
    vec3 diskColor = vec3(1.0);
    float opacity = 1.0;

    if (uMultiLayer && diskLayer > 1) {
      if (diskLayer == 2) {
        float layer1_outer = DISK_R1 + (DISK_R2 - DISK_R1) * 0.4;
        float layer2_outer = DISK_R1 + (DISK_R2 - DISK_R1) * 0.7;
        float norm_r = clamp((r_radial - layer1_outer) / (layer2_outer - layer1_outer), 0.0, 1.0);
        diskColor = mix(vec3(1.0, 0.8, 0.3), vec3(1.0, 0.5, 0.1), norm_r);
        float turbulence = sin(uTime * 1.5 + r_metres * 0.1) * uTurbulence;
        diskColor *= (1.0 + turbulence * 0.3);
        opacity = 0.85;
      } else {
        float layer2_outer = DISK_R1 + (DISK_R2 - DISK_R1) * 0.7;
        float norm_r = clamp((r_radial - layer2_outer) / (DISK_R2 - layer2_outer), 0.0, 1.0);
        diskColor = mix(vec3(1.0, 0.5, 0.1), vec3(0.7, 0.1, 0.0), norm_r);
        float clumping = sin(uTime * 0.8 + r_metres * 0.05) * cos(uTime * 1.2 + r_metres * 0.03);
        diskColor *= (1.0 + clumping * uTurbulence * 0.5);
        opacity = 0.7;
      }
    }

    if (diskLayer == 1 || !uMultiLayer) {
      float norm_r = clamp((r_radial - DISK_R1) / (DISK_R2 - DISK_R1), 0.0, 1.0);
      if (uTempPeak > 5e6) {
        vec3 innerYellow = vec3(1.0, 1.0, 0.3), orange = vec3(1.0, 0.7, 0.2), outerRed = vec3(0.7, 0.1, 0.0);
        diskColor = norm_r < 0.5 ? mix(innerYellow, orange, norm_r / 0.5) : mix(orange, outerRed, (norm_r - 0.5) / 0.5);
      } else {
        vec3 innerYellow = vec3(1.0, 0.9, 0.4), orange = vec3(1.0, 0.6, 0.2), outerRed = vec3(0.8, 0.2, 0.1);
        diskColor = norm_r < 0.5 ? mix(innerYellow, orange, norm_r / 0.5) : mix(orange, outerRed, (norm_r - 0.5) / 0.5);
      }
      opacity = 0.9;
    }

    if (uRotSpeed > 0.0 && uDoppler) {
      // kepler_v = sqrt(SagA_rs * 1.5e8 / r), as in the original
      float kepler_v = sqrt(1.5e8 / r_radial);
      float beta = kepler_v / 3e8;
      float phi = atan(hitPos.z, hitPos.x);
      float dopplerFactor = sqrt((1.0 + beta * cos(phi)) / (1.0 - beta * cos(phi)));
      diskColor.r *= pow(dopplerFactor, -0.5);
      diskColor.g *= pow(dopplerFactor, -0.3);
      diskColor.b *= pow(dopplerFactor, 0.2);
      diskColor *= 1.0 + 0.3 * cos(phi) * beta * uRotSpeed;
    }

    if (r_radial > 1.1) {
      float gravRedshift = sqrt(1.0 - 1.0 / r_radial);
      float redshift_z = (1.0 / gravRedshift) - 1.0;
      diskColor.r *= (1.0 + redshift_z * 0.4);
      diskColor.g *= (1.0 + redshift_z * 0.1);
      diskColor.b *= (1.0 - redshift_z * 0.3);
      diskColor *= 0.7 + 0.3 * gravRedshift;
    }

    if (uKerr && uSpin > 0.1) {
      float dragging_factor = clamp(uSpin / r_radial, 0.0, 1.0);
      float phi = atan(hitPos.z, hitPos.x);
      float asymmetry = sin(phi + uTime * dragging_factor * 0.5) * dragging_factor;
      diskColor *= (1.0 + asymmetry * 0.2);
      if (r_radial < 6.0) diskColor *= 1.0 + dragging_factor * 0.5;
    }

    diskColor *= 1.0 + uEddington * 1.2;
    color = vec4(diskColor, opacity);
  } else if (hitBlackHole) {
    color = vec4(0.0, 0.0, 0.0, 1.0);
  } else if (hitObject) {
    vec3 N = normalize(hitPos - hitCenter);
    vec3 V = normalize(uCamPos - hitPos);
    float intensity = 0.1 + 0.9 * max(dot(N, V), 0.0);
    color = vec4(objectColor * intensity, 1.0);
  }

  // Composite accumulated jet energy
  if (!hitBlackHole && jetAccumulation.a > 0.01) {
    float jetAlpha = clamp(jetAccumulation.a * uEddington, 0.0, 0.95);
    vec3 finalJetColor = jetAccumulation.rgb / max(jetAccumulation.a, 0.001);
    color.rgb = color.rgb * (1.0 - jetAlpha) + finalJetColor * jetAlpha;
    color.a = max(color.a, jetAlpha);
  }

  // Tone mapping and grading
  if (color.a > 0.0) {
    vec3 hdrColor = color.rgb;
    float avgBrightness = (hdrColor.r + hdrColor.g + hdrColor.b) / 3.0;
    float exposureAdjust = 1.0;
    if (avgBrightness > 2.0) exposureAdjust = 0.7;
    else if (avgBrightness < 0.1) exposureAdjust = 1.3;
    hdrColor *= exposureAdjust;
    vec3 tonemapped = (hdrColor * (2.51 * hdrColor + 0.03)) / (hdrColor * (2.43 * hdrColor + 0.59) + 0.14);
    tonemapped = pow(max(tonemapped, vec3(0.0)), vec3(1.0 / 2.2));
    if (uTempPeak > 5e6) {
      tonemapped.b *= 1.15;
      tonemapped.r *= 0.95;
    } else {
      tonemapped.r *= 1.05;
      tonemapped.b *= 0.98;
    }
    float contrast = uKerr ? 1.2 : 1.1;
    tonemapped = clamp((tonemapped - 0.5) * contrast + 0.5, 0.0, 1.0);
    vec2 center = pix / vec2(WIDTH, HEIGHT) - 0.5;
    tonemapped *= clamp(1.0 - dot(center, center) * 0.3, 0.7, 1.0);
    color = vec4(tonemapped, color.a);
  }

  vec3 outColor = color.rgb;
  if (uStars && escaped && color.a < 1.0) {
    float st = sin(q.y), ct = cos(q.y), sp = sin(q.z), cp = cos(q.z);
    vec3 local = vec3(
      v.x * st * cp + q.x * v.y * ct * cp - q.x * v.z * st * sp,
      v.x * st * sp + q.x * v.y * ct * sp + q.x * v.z * st * cp,
      v.x * ct - q.x * v.y * st);
    vec3 outDir = normalize(local.x * e1 + local.y * e2 + local.z * e3);
    outColor += starfield(outDir) * (1.0 - color.a);
  }
  fragColor = vec4(outColor, 1.0);
}
