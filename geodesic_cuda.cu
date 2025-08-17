#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <cmath>

#define M_PI 3.14159265358979323846

// Physical constants
const double c = 299792458.0;
const double G = 6.67430e-11;

struct Ray {
    double x, y, z;
    double r, theta, phi;
    double dr, dtheta, dphi;
    double E, L;
};

__device__ void geodesicRHS(const Ray& ray, double rs, double& dr_out, double& dtheta_out, double& dphi_out, double& ddr_out, double& ddtheta_out, double& ddphi_out) {
    double r = ray.r, theta = ray.theta;
    double dr = ray.dr, dtheta = ray.dtheta, dphi = ray.dphi;
    double f = 1.0 - rs / r;
    double dt_dL = ray.E / f;
    dr_out = dr;
    dtheta_out = dtheta;
    dphi_out = dphi;
    ddr_out = - (rs / (2.0 * r*r)) * f * dt_dL * dt_dL
            + (rs / (2.0 * r*r * f)) * dr * dr
            + r * (dtheta*dtheta + sin(theta)*sin(theta)*dphi*dphi);
    ddtheta_out = -2.0*dr*dtheta/r + sin(theta)*cos(theta)*dphi*dphi;
    ddphi_out = -2.0*dr*dphi/r - 2.0*cos(theta)/(sin(theta)) * dtheta * dphi;
}

__device__ void rk4Step(Ray& ray, double dLambda, double rs) {
    double dr1, dtheta1, dphi1, ddr1, ddtheta1, ddphi1;
    double dr2, dtheta2, dphi2, ddr2, ddtheta2, ddphi2;
    double dr3, dtheta3, dphi3, ddr3, ddtheta3, ddphi3;
    double dr4, dtheta4, dphi4, ddr4, ddtheta4, ddphi4;
    geodesicRHS(ray, rs, dr1, dtheta1, dphi1, ddr1, ddtheta1, ddphi1);
    Ray tmp = ray;
    tmp.r      += 0.5 * dLambda * dr1;
    tmp.theta  += 0.5 * dLambda * dtheta1;
    tmp.phi    += 0.5 * dLambda * dphi1;
    tmp.dr     += 0.5 * dLambda * ddr1;
    tmp.dtheta += 0.5 * dLambda * ddtheta1;
    tmp.dphi   += 0.5 * dLambda * ddphi1;
    geodesicRHS(tmp, rs, dr2, dtheta2, dphi2, ddr2, ddtheta2, ddphi2);
    tmp = ray;
    tmp.r      += 0.5 * dLambda * dr2;
    tmp.theta  += 0.5 * dLambda * dtheta2;
    tmp.phi    += 0.5 * dLambda * dphi2;
    tmp.dr     += 0.5 * dLambda * ddr2;
    tmp.dtheta += 0.5 * dLambda * ddtheta2;
    tmp.dphi   += 0.5 * dLambda * ddphi2;
    geodesicRHS(tmp, rs, dr3, dtheta3, dphi3, ddr3, ddtheta3, ddphi3);
    tmp = ray;
    tmp.r      += dLambda * dr3;
    tmp.theta  += dLambda * dtheta3;
    tmp.phi    += dLambda * dphi3;
    tmp.dr     += dLambda * ddr3;
    tmp.dtheta += dLambda * ddtheta3;
    tmp.dphi   += dLambda * ddphi3;
    geodesicRHS(tmp, rs, dr4, dtheta4, dphi4, ddr4, ddtheta4, ddphi4);
    ray.r      += (dLambda / 6.0) * (dr1 + 2*dr2 + 2*dr3 + dr4);
    ray.theta  += (dLambda / 6.0) * (dtheta1 + 2*dtheta2 + 2*dtheta3 + dtheta4);
    ray.phi    += (dLambda / 6.0) * (dphi1 + 2*dphi2 + 2*dphi3 + dphi4);
    ray.dr     += (dLambda / 6.0) * (ddr1 + 2*ddr2 + 2*ddr3 + ddr4);
    ray.dtheta += (dLambda / 6.0) * (ddtheta1 + 2*ddtheta2 + 2*ddtheta3 + ddtheta4);
    ray.dphi   += (dLambda / 6.0) * (ddphi1 + 2*ddphi2 + 2*ddphi3 + ddphi4);
}

__global__ void integrateGeodesics(Ray* rays, int numRays, double dLambda, double rs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRays) {
        for (int i = 0; i < 1000; ++i) {
            rk4Step(rays[idx], dLambda, rs);
        }
    }
}

// Helper vector ops for float3 (defined early so kernels can use them)
__device__ __forceinline__ float3 f3_add(const float3& a, const float3& b){ return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ __forceinline__ float3 f3_scale(const float3& a, float s){ return make_float3(a.x*s, a.y*s, a.z*s); }

__global__ void bhRayKernel(cudaSurfaceObject_t surface,
                            int width, int height,
                            float3 camPos, float3 camFwd, float3 camRight, float3 camUp,
                            float tanHalfFov, float aspect, float rs) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = ( ( (x + 0.5f) / width)  * 2.0f - 1.0f ); // -1..1
    float v = ( ( (y + 0.5f) / height) * 2.0f - 1.0f );
    // v flip so positive v is up
    v = -v;

    float3 dir = camFwd; // camFwd + scalarR*camRight + scalarU*camUp
    float scalarR = u * tanHalfFov * aspect;
    dir.x += scalarR * camRight.x;
    dir.y += scalarR * camRight.y;
    dir.z += scalarR * camRight.z;
    float scalarU = v * tanHalfFov;
    dir.x += scalarU * camUp.x;
    dir.y += scalarU * camUp.y;
    dir.z += scalarU * camUp.z;
    // normalize dir
    float len = rsqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    dir.x *= len; dir.y *= len; dir.z *= len;

    // Ray-sphere intersection (black hole event horizon at origin radius rs)
    float3 O = camPos;
    float3 D = dir;
    float b = O.x*D.x + O.y*D.y + O.z*D.z; // dot(O,D)
    float c = O.x*O.x + O.y*O.y + O.z*O.z - rs*rs;
    float disc = b*b - c; // (since |D|=1)
    bool hit = disc >= 0.0f && (-b - sqrtf(max(disc,0.f))) > 0.0f;

    // Impact parameter ~ |O x D| / |D| ; |D|=1
    float3 crossOD = make_float3(O.y*D.z - O.z*D.y,
                                 O.z*D.x - O.x*D.z,
                                 O.x*D.y - O.y*D.x);
    float impact = sqrtf(crossOD.x*crossOD.x + crossOD.y*crossOD.y + crossOD.z*crossOD.z);

    // Photon ring approx around ~2.6 rs (heuristic visual, real GR differs for camera distance)
    float ringCenter = 2.6f * rs;
    float ringWidth  = 0.35f * rs;
    float ringGlow = expf(- (impact - ringCenter)*(impact - ringCenter) / (2.0f * ringWidth * ringWidth));

    // Background sky gradient based on direction (simple): blend of deep blue and near-black
    float skyT = 0.5f*(dir.y + 1.0f); // 0..1 using vertical component
    float3 low = make_float3(0.05f,0.08f,0.12f);
    float3 high= make_float3(0.10f,0.15f,0.25f);
    float3 skyColor = f3_add(f3_scale(low, (1.0f - skyT)), f3_scale(high, skyT));

    // Ring color (warm)
    float3 ringBase = make_float3(1.0f, 0.75f, 0.25f);
    float3 ringColor = f3_scale(ringBase, 3.0f * ringGlow);

    float3 color;
    if (hit) {
        color = make_float3(0.0f,0.0f,0.0f);
    } else {
        color = f3_add(skyColor, ringColor);
        // tone map simple
        color.x = color.x / (1.0f + color.x);
        color.y = color.y / (1.0f + color.y);
        color.z = color.z / (1.0f + color.z);
    }
    // clamp
    color.x = fminf(fmaxf(color.x,0.0f),1.0f);
    color.y = fminf(fmaxf(color.y,0.0f),1.0f);
    color.z = fminf(fmaxf(color.z,0.0f),1.0f);

    uchar4 out = make_uchar4((unsigned char)(color.x*255.0f),
                             (unsigned char)(color.y*255.0f),
                             (unsigned char)(color.z*255.0f),
                             255);
    surf2Dwrite(out, surface, x * sizeof(uchar4), y);
}

extern "C" void launchCudaRaytracer(cudaArray_t cudaArray, int width, int height,
                                     const float camPos[3], const float camFwd[3], const float camRight[3], const float camUp[3],
                                     float tanHalfFov, float aspect, float rs) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaArray;
    cudaSurfaceObject_t surface = 0;
    cudaCreateSurfaceObject(&surface, &resDesc);

    dim3 block(16,16);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);
    float3 p = make_float3(camPos[0], camPos[1], camPos[2]);
    float3 f = make_float3(camFwd[0], camFwd[1], camFwd[2]);
    float3 r = make_float3(camRight[0], camRight[1], camRight[2]);
    float3 u = make_float3(camUp[0], camUp[1], camUp[2]);
    bhRayKernel<<<grid, block>>>(surface, width, height, p, f, r, u, tanHalfFov, aspect, rs);
    cudaDestroySurfaceObject(surface);
}

#ifdef BUILD_STANDALONE_CUDA
int main() {
    const int numRays = 10;
    double rs = 2.0 * G * 8.54e36 / (c*c); // Example mass
    Ray hostRays[numRays];
    for (int i = 0; i < numRays; ++i) {
        // Example: rays start at different positions along x-axis, all pointing in -z direction
        hostRays[i].x = 1.0e11 + i * 1.0e9;
        hostRays[i].y = 0.0;
        hostRays[i].z = 0.0;
        double r = sqrt(hostRays[i].x * hostRays[i].x + hostRays[i].y * hostRays[i].y + hostRays[i].z * hostRays[i].z);
        hostRays[i].r = r;
        hostRays[i].theta = acos(hostRays[i].z / r);
        hostRays[i].phi = atan2(hostRays[i].y, hostRays[i].x);
        // Direction: all rays point in -z direction
        double dx = 0.0, dy = 0.0, dz = -1.0;
        hostRays[i].dr     = sin(hostRays[i].theta)*cos(hostRays[i].phi)*dx + sin(hostRays[i].theta)*sin(hostRays[i].phi)*dy + cos(hostRays[i].theta)*dz;
        hostRays[i].dtheta = (cos(hostRays[i].theta)*cos(hostRays[i].phi)*dx + cos(hostRays[i].theta)*sin(hostRays[i].phi)*dy - sin(hostRays[i].theta)*dz) / r;
        hostRays[i].dphi   = (-sin(hostRays[i].phi)*dx + cos(hostRays[i].phi)*dy) / (r * sin(hostRays[i].theta));
        hostRays[i].L = r * r * sin(hostRays[i].theta) * hostRays[i].dphi;
        double f = 1.0 - rs / r;
        double dt_dL = sqrt((hostRays[i].dr*hostRays[i].dr)/f + r*r*(hostRays[i].dtheta*hostRays[i].dtheta + sin(hostRays[i].theta)*sin(hostRays[i].theta)*hostRays[i].dphi*hostRays[i].dphi));
        hostRays[i].E = f * dt_dL;
    }
    Ray* deviceRays;
    cudaMalloc(&deviceRays, numRays * sizeof(Ray));
    cudaMemcpy(deviceRays, hostRays, numRays * sizeof(Ray), cudaMemcpyHostToDevice);

    double dLambda = 1e7;
    integrateGeodesics<<<1, numRays>>>(deviceRays, numRays, dLambda, rs);
    cudaDeviceSynchronize();

    cudaMemcpy(hostRays, deviceRays, numRays * sizeof(Ray), cudaMemcpyDeviceToHost);
    cudaFree(deviceRays);

    for (int i = 0; i < numRays; ++i) {
        std::cout << "Ray " << i << ": r = " << hostRays[i].r
                  << ", dr = " << hostRays[i].dr
                  << ", theta = " << hostRays[i].theta
                  << ", phi = " << hostRays[i].phi
                  << ", E = " << hostRays[i].E
                  << ", L = " << hostRays[i].L << std::endl;
    }
    return 0;
}
#endif // BUILD_STANDALONE_CUDA
