#include <cuda_runtime.h>
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
