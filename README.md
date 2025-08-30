# Black Hole Simulation

A real-time, physically accurate black hole visualization system implementing relativistic ray tracing and geodesic computation. Features both Schwarzschild and Kerr black holes with advanced astrophysical effects including gravitational lensing, accretion disks, and relativistic jets.

## Features

### Relativistic Physics
- **Schwarzschild & Kerr Metrics**: Non-rotating and rotating black hole spacetimes
- **Gravitational Lensing**: Light bending around massive objects
- **Frame Dragging**: Lense-Thirring precession effects near rotating black holes
- **Gravitational Redshift**: Spectral shifts due to gravitational time dilation
- **Event Horizon**: Accurate boundary calculations for both metric types

### Accretion Physics
- **Multi-layer Disk Structure**: Temperature gradients from 10⁶-10⁷ K
- **Relativistic Jets**: Synchrotron radiation with configurable opening angles
- **Doppler Beaming**: Relativistic effects from orbital motion
- **Eddington Scaling**: Luminosity based on accretion rate
- **Turbulence Modeling**: Chaotic flow patterns in active galactic nuclei

### Interactive Simulation
- **Real-time Ray Tracing**: GPU-accelerated using OpenGL compute shaders
- **Orbital Camera System**: Smooth navigation around the black hole
- **Quality Presets**: Adaptive performance from 32k-256k ray steps
- **Two Simulation Modes**: 
  - Standard (Sagittarius A*)
  - TON618 Quasar (supermassive active black hole)

## Requirements

- **OpenGL 4.3+** compatible graphics card
- **C++17** compiler (GCC, Clang, or MSVC)
- **CMake 3.21+**

### Dependencies
- **GLEW**: OpenGL extension loading
- **GLFW3**: Window management and input
- **GLM**: Mathematics library for graphics

## Building

```bash
# Clone the repository
git clone <repository-url>
cd space_simulation

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .
```

### Windows (vcpkg)
```bash
# Install dependencies
vcpkg install glew glfw3 glm

# Configure with vcpkg toolchain
cmake .. -DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg.cmake
```

## Usage

### Main Application
```bash
./BlackHole3D
```

### Example Programs
```bash
./Lensing2D      # 2D gravitational lensing
./CPUGeodesic    # CPU-based geodesic computation
./RayTracing     # Basic ray tracer framework
```

## Controls

### Mouse Navigation
- **Left Click + Drag**: Orbit camera around black hole
- **Mouse Wheel**: Zoom in/out
- **Right Click (hold)**: Enable gravity simulation for test objects

### Keyboard Shortcuts

#### Simulation Modes
- **Q**: Toggle between Standard (Sgr A*) and TON618 Quasar modes
- **G**: Toggle gravity simulation for objects

#### Relativistic Effects
- **R**: Toggle relativistic Doppler effects
- **K**: Toggle Kerr metric (rotating black hole)
- **M**: Toggle multi-layer accretion disk
- **J**: Toggle relativistic jets

#### Quality Settings
- **1**: Low quality (32k steps, 0.7x resolution)
- **2**: Medium quality (64k steps, 0.85x resolution)
- **3**: High quality (128k steps, 1.0x resolution)  
- **4**: Ultra quality (256k steps, 1.0x resolution)

#### Help
- **H**: Show control instructions

## Physics Implementation

### Black Hole Parameters
- **Sagittarius A*** (Default): Mass = 8.54×10³⁶ kg, Schwarzschild radius = 1.27×10¹⁰ m
- **Spin Parameter**: 0.0 (Schwarzschild) to 0.998 (near-extremal Kerr)
- **Event Horizon**: r = M + √(M² - a²) for Kerr metric

### Geodesic Integration
- **Numerical Method**: 4th-order Runge-Kutta integration
- **Step Size**: Adaptive based on proximity to singularity
- **Conservation Laws**: Energy (E) and angular momentum (L) preserved
- **Stability**: Numerical safeguards prevent divergence near event horizon

### Accretion Disk Model
- **Inner Edge**: 2.2× Schwarzschild radius
- **Outer Edge**: 5.2× Schwarzschild radius  
- **Temperature Profile**: T ∝ r⁻³/⁴ (standard disk model)
- **Multi-layer Structure**: Hot inner disk, warm middle layer, cool outer torus

### Jet Physics
- **Opening Angle**: 5-6 degrees (configurable)
- **Emission Region**: Starting at 3× Schwarzschild radius
- **Synchrotron Radiation**: Blue-shifted emission from relativistic electrons
- **Energy Transport**: Magnetic field-driven outflows

## Performance

### Optimization Features
- **Adaptive Quality**: Resolution scales with camera distance
- **Motion Detection**: Reduced quality during camera movement
- **GPU Acceleration**: 16×16 compute shader workgroups
- **Memory Efficiency**: Uniform buffer objects for constant data

## Scientific Accuracy

This simulation implements the full Einstein field equations in Boyer-Lindquist coordinates:

```
ds² = -(1-2Mr/ρ²)dt² + (ρ²/Δ)dr² + ρ²dθ² + sin²θ(r²+a²+2Ma²r sin²θ/ρ²)dφ² - (4Mar sin²θ/ρ²)dtdφ
```

Where:
- **ρ² = r² + a²cos²θ**: Kerr coordinate system
- **Δ = r² - 2Mr + a²**: Radial metric function
- **a**: Angular momentum parameter (spin)

## Acknowledgments

- [rossning92/Blackhole](https://github.com/rossning92/Blackhole): OpenGL and C++ black hole simulation framework, providing shader-based visualization and reference implementation of Schwarzschild geodesics.
- [kavan010/black_hole](https://github.com/kavan010/black_hole): C++ GPU-accelerated ray tracing of black holes, with implementations for gravitational lensing and accretion disk rendering.
- [20k.github.io Schwarzschild Tutorial](https://20k.github.io/c++/2024/05/31/schwarzschild.html): Educational series bridging general relativity and C++ implementation, with detailed coverage of geodesics, coordinate systems, and GPU acceleration.