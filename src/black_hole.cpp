#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <fstream>
#include <sstream>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace glm;
using namespace std;
using Clock = std::chrono::high_resolution_clock;

// Performance monitoring variables
double lastPrintTime = 0.0;
int    framesCount   = 0;

// Physical constants
double c = 299792458.0;  // Speed of light (m/s)
double G = 6.67430e-11;  // Gravitational constant (m³/kg·s²)

// Forward declarations and simulation state
struct Ray;
bool Gravity = false;  // Controls whether gravitational forces affect test objects

struct Camera {
    // Camera positioning - maintains orbital view around black hole center
    vec3 target = vec3(0.0f, 0.0f, 0.0f);  // Always focused on black hole
    float radius = 6.34194e10f;             // Distance from black hole (meters)
    float minRadius = 1e10f, maxRadius = 1e12f;  // Zoom limits

    // Spherical coordinates for orbital motion
    float azimuth = 0.0f;      // Horizontal rotation angle
    float elevation = M_PI / 2.0f;  // Vertical angle from pole

    // User interaction sensitivity settings
    float orbitSpeed = 0.01f;   // Mouse sensitivity for orbital movement
    float panSpeed = 0.01f;     // Pan speed (currently unused)
    double zoomSpeed = 25e9f;   // Mouse wheel zoom sensitivity

    // Input state tracking
    bool dragging = false;      // Left mouse button held
    bool panning = false;       // Middle mouse button held (unused)
    bool moving = false;        // Camera is currently being moved
    bool firstMouse = true;     // First mouse movement detection
    double lastX = 0.0, lastY = 0.0;  // Previous mouse position

    // Convert spherical coordinates to Cartesian world position
    vec3 position() const {
        // Prevent singularities at poles by clamping elevation
        float clampedElevation = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);
        
        // Standard spherical to Cartesian conversion
        return vec3(
            radius * sin(clampedElevation) * cos(azimuth),
            radius * cos(clampedElevation),
            radius * sin(clampedElevation) * sin(azimuth)
        );
    }
    void update() {
        // Ensure camera always focuses on black hole at origin
        target = vec3(0.0f, 0.0f, 0.0f);
        
        // Track movement state for adaptive rendering quality
        static double lastMoveTime = 0.0;
        if(dragging | panning) {
            moving = true;
            lastMoveTime = glfwGetTime();
        } else {
            // Wait 200ms after movement stops before enabling high quality
            double timeSinceMove = glfwGetTime() - lastMoveTime;
            moving = timeSinceMove < 0.2;
        }
    }

    void processMouseMove(double x, double y) {
        // Skip processing if not dragging or on first mouse event
        if (firstMouse || !dragging) {
            lastX = x;
            lastY = y;
            firstMouse = false;
            return;
        }
        
        // Calculate mouse movement delta with bounds checking
        float dx = float(x - lastX);
        float dy = float(y - lastY);
        dx = glm::clamp(dx, -100.0f, 100.0f);  // Prevent jumps from alt-tab, etc.
        dy = glm::clamp(dy, -100.0f, 100.0f);

        if (dragging && panning) {
            // Panning intentionally disabled to maintain black hole-centered view
        }
        else if (dragging && !panning) {
            // Apply orbital rotation based on mouse movement
            azimuth   += dx * orbitSpeed;  // Horizontal rotation
            elevation -= dy * orbitSpeed;  // Vertical rotation (inverted for natural feel)
            elevation = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);  // Prevent flipping
        }

        lastX = x;
        lastY = y;
        update();
    }
    void processMouseButton(int button, int action, int mods, GLFWwindow* win) {
        if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_MIDDLE) {
            if (action == GLFW_PRESS) {
                dragging = true;
                // Maintain orbital camera behavior
                panning = false;
                firstMouse = true;
                glfwGetCursorPos(win, &lastX, &lastY);
            } else if (action == GLFW_RELEASE) {
                dragging = false;
                panning = false;
            }
        }
        // Right mouse button toggles gravity simulation for test objects
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS) {
                Gravity = true;   // Enable N-body gravitational simulation
            } else if (action == GLFW_RELEASE) {
                Gravity = false;  // Freeze object positions
            }
        }
    }
    void processScroll(double xoffset, double yoffset) {
        radius -= yoffset * zoomSpeed;
        radius = glm::clamp(radius, minRadius, maxRadius);
        update();
    }
    void processKey(int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS && key == GLFW_KEY_G) {
            Gravity = !Gravity;
            cout << "[INFO] Gravity turned " << (Gravity ? "ON" : "OFF") << endl;
        }
        // Other key handling is implemented in handleFeatureKeys() function
    }
};
Camera camera;

struct BlackHole {
    vec3 position;   // Location in 3D space
    double mass;     // Mass in kilograms
    double radius;   // Physical radius (unused in current implementation)
    double r_s;      // Schwarzschild radius (event horizon)

    BlackHole(vec3 pos, float m) : position(pos), mass(m) {
        // Calculate Schwarzschild radius: rs = 2GM/c²
        r_s = 2.0 * G * mass / (c*c);
    }
    
    // Test if a point is within the event horizon
    bool Intercept(float px, float py, float pz) const {
        double dx = double(px) - double(position.x);
        double dy = double(py) - double(position.y);
        double dz = double(pz) - double(position.z);
        double dist2 = dx * dx + dy * dy + dz * dz;
        return dist2 < r_s * r_s;
    }
};

enum class FeatureProfile { STANDARD, TON618_QUASAR };

// Configuration system for different black hole simulation modes
struct QuasarFeatures {
    FeatureProfile profile = FeatureProfile::STANDARD;
    float eddingtonFraction = 0.1f;    // Fraction of Eddington luminosity (accretion rate)
    float diskTempPeak = 1e6f;         // Peak disk temperature in Kelvin
    float lensingBoost = 1.0f;         // Gravitational lensing enhancement factor
    bool enableMultiLayerDisk = false; // Multi-layer accretion disk structure
    
    // Rendering quality and performance settings
    enum class QualityPreset { LOW, MEDIUM, HIGH, ULTRA };
    QualityPreset quality = QualityPreset::MEDIUM;
    float renderScale = 0.85f;      // Resolution scaling factor
    int maxRaySteps = 64000;        // Maximum ray tracing steps
    bool halfResVolumetrics = true; // Reduce resolution for volumetric effects
    
    // Accretion disk dynamics
    float diskRotationSpeed = 0.0f;     // Keplerian rotation speed multiplier
    float diskTurbulence = 0.0f;        // Magnetorotational instability strength
    bool enableDopplerBeaming = false;   // Relativistic Doppler shift effects
    
    // General relativity parameters
    bool enableKerrMetric = false;      // Use rotating (Kerr) vs static (Schwarzschild) metric
    float spinParameter = 0.0f;         // Dimensionless spin: a/M (0.0-0.998)
    
    // Relativistic jet parameters
    bool enableJets = false;            // Enable bipolar jet emission
    float jetOpeningAngle = 5.0f;       // Half-angle of jet cone in degrees
    float jetBrightness = 1.0f;         // Synchrotron emission intensity
    
    // Enhanced jet parameters
    float jetHelixPitch = 0.0f;        // Helical twist parameter (0 = no helix)
    float jetPrecessionAngle = 0.0f;   // Precession cone half-angle (degrees)
    float jetPrecessionPeriod = 0.0f;  // Precession period in seconds (0 = no precession)
    float jetLorentzFactor = 1.0f;     // Bulk relativistic motion (1 = non-relativistic)
    float jetShockSpeed = 0.0f;        // Speed of shock knots (fraction of c)
    
    // Enhanced disk parameters
    float diskInnerTemp = 1e6f;        // Inner disk temperature (Kelvin)
    float diskTempExponent = -0.75f;   // Temperature profile exponent (Shakura-Sunyaev = -3/4)
    float diskScaleHeight = 0.01f;     // H/R ratio at reference radius
    float diskInstabilityAmp = 0.0f;   // Amplitude of thermal instabilities
    float diskInstabilityFreq = 0.0f;  // Frequency of thermal instabilities (Hz)
    
    // Magnetic field parameters
    float magneticFieldStrength = 0.0f; // Overall B-field strength
    bool enableMagneticField = false;   // Toggle magnetic visualization
    float fieldTurbulence = 0.0f;       // MRI turbulence amplitude
    
    // Photon ring enhancement
    int photonRingOrders = 1;           // Number of image orders to compute
    bool enablePhotonRing = false;      // Toggle enhanced photon ring
    
    // QPO and variability
    float qpoFrequency = 0.0f;          // Quasi-periodic oscillation frequency (Hz)
    float qpoAmplitude = 0.0f;          // QPO brightness modulation amplitude
    float diskWindStrength = 0.0f;      // Disk wind opacity effects
    
    // Configure simulation for TON618-type supermassive quasar
    void switchToTON618() {
        profile = FeatureProfile::TON618_QUASAR;
        eddingtonFraction = 0.7f;      // High accretion rate (70% of Eddington limit)
        diskTempPeak = 1e7f;           // Very hot disk (10 million K)
        enableMultiLayerDisk = true;   // Complex multi-zone disk structure
        diskRotationSpeed = 2.0f;      // Rapid Keplerian rotation
        diskTurbulence = 0.3f;         // Strong MRI turbulence
        enableDopplerBeaming = true;   // Relativistic beaming effects
        enableKerrMetric = true;       // Rapidly rotating black hole
        spinParameter = 0.8f;          // High spin (a/M = 0.8)
        enableJets = true;             // Powerful relativistic jets
        jetOpeningAngle = 6.0f;        // Wider jet opening angle
        jetBrightness = 2.0f;          // Enhanced synchrotron emission
        
        // Enhanced jet physics for TON618
        jetHelixPitch = 5e11f;         // Helical structure with 500 billion meter pitch
        jetPrecessionAngle = 2.0f;     // 2 degree precession cone
        jetPrecessionPeriod = 3600.0f; // 1 hour precession period
        jetLorentzFactor = 15.0f;      // Bulk relativistic motion (Γ = 15)
        jetShockSpeed = 0.8f;          // Shock knots at 80% speed of light
        
        // Enhanced disk physics
        diskInnerTemp = 5e7f;          // 50 million K inner temperature
        diskTempExponent = -0.75f;     // Standard Shakura-Sunyaev profile
        diskScaleHeight = 0.02f;       // Thicker disk for active quasar
        diskInstabilityAmp = 0.2f;     // Strong thermal instabilities
        diskInstabilityFreq = 0.1f;    // 0.1 Hz fluctuations
        
        // Magnetic field effects
        magneticFieldStrength = 1.0f;  // Strong magnetic fields
        enableMagneticField = true;    // Show magnetic structure
        fieldTurbulence = 0.4f;        // Strong MRI turbulence
        
        // Enhanced photon ring
        photonRingOrders = 3;          // Show multiple image orders
        enablePhotonRing = true;       // Enable enhanced lensing
        
        // QPO and variability
        qpoFrequency = 1.0f;           // 1 Hz QPO frequency
        qpoAmplitude = 0.15f;          // 15% brightness modulation
        diskWindStrength = 0.1f;       // Disk wind effects
        
        cout << "[INFO] Switched to TON618 Quasar: full enhanced physics enabled!" << endl;
    }
    
    // Configure simulation for Sagittarius A*-type galactic center black hole
    void switchToStandard() {
        profile = FeatureProfile::STANDARD;
        eddingtonFraction = 0.1f;        // Low accretion rate (10% of Eddington)
        diskTempPeak = 1e6f;            // Moderate disk temperature (1 million K)
        enableMultiLayerDisk = false;    // Simple single-zone disk
        diskRotationSpeed = 0.1f;        // Slow, stable Keplerian rotation
        diskTurbulence = 0.05f;         // Weak turbulence
        enableDopplerBeaming = false;    // Minimal relativistic effects
        enableKerrMetric = false;        // Non-rotating Schwarzschild metric
        spinParameter = 0.0f;           // Zero angular momentum
        enableJets = false;             // No significant jet emission
        jetOpeningAngle = 5.0f;         // Default parameters
        jetBrightness = 1.0f;           // Standard emission
        
        // Reset enhanced parameters to standard values
        jetHelixPitch = 0.0f;          // No helical structure
        jetPrecessionAngle = 0.0f;     // No precession
        jetPrecessionPeriod = 0.0f;    
        jetLorentzFactor = 1.0f;       // Non-relativistic
        jetShockSpeed = 0.0f;          // No shock knots
        
        diskInnerTemp = 1e6f;          // Cooler inner disk
        diskTempExponent = -0.75f;     // Standard profile
        diskScaleHeight = 0.01f;       // Thin disk
        diskInstabilityAmp = 0.0f;     // No instabilities
        diskInstabilityFreq = 0.0f;    
        
        magneticFieldStrength = 0.0f;  // No magnetic fields
        enableMagneticField = false;   
        fieldTurbulence = 0.0f;        
        
        photonRingOrders = 1;          // Basic lensing only
        enablePhotonRing = false;      
        
        qpoFrequency = 0.0f;           // No QPOs
        qpoAmplitude = 0.0f;           
        diskWindStrength = 0.0f;       
        
        cout << "[INFO] Switched to Standard Sgr A*: enhanced features disabled" << endl;
    }
    
    void toggle() {
        if (profile == FeatureProfile::STANDARD) {
            switchToTON618();
        } else {
            switchToStandard();
        }
    }
    
    void setQualityLow() {
        quality = QualityPreset::LOW;
        renderScale = 0.70f;
        maxRaySteps = 32000;    // Increased from 16k
        halfResVolumetrics = true;
        cout << "[INFO] Quality: LOW (0.7x scale, 32k steps)" << endl;
    }
    
    void setQualityMedium() {
        quality = QualityPreset::MEDIUM;
        renderScale = 0.85f;
        maxRaySteps = 64000;    // Increased from 32k
        halfResVolumetrics = true;
        cout << "[INFO] Quality: MEDIUM (0.85x scale, 64k steps)" << endl;
    }
    
    void setQualityHigh() {
        quality = QualityPreset::HIGH;
        renderScale = 1.0f;
        maxRaySteps = 128000;   // High quality
        halfResVolumetrics = false;
        cout << "[INFO] Quality: HIGH (1.0x scale, 128k steps)" << endl;
    }
    
    void setQualityUltra() {
        quality = QualityPreset::ULTRA;
        renderScale = 1.0f;
        maxRaySteps = 256000;   // Ultra-high quality for maximum detail
        halfResVolumetrics = false;
        cout << "[INFO] Quality: ULTRA (1.0x scale, 256k steps) - WARNING: Very demanding!" << endl;
    }
};


// Sagittarius A* - our galaxy's central supermassive black hole
BlackHole SagA(vec3(0.0f, 0.0f, 0.0f), 8.54e36);  // Mass: 4.3 million solar masses
QuasarFeatures features;  // Global simulation configuration

// Data structure for renderable objects in the simulation
struct ObjectData {
    vec4 posRadius; // xyz = position (m), w = radius (m)
    vec4 color;     // rgba color values
    float  mass;    // Mass in kilograms
    vec3 velocity = vec3(0.0f, 0.0f, 0.0f);  // Velocity vector (m/s)
};
vector<ObjectData> objects;  // Collection of all scene objects

// Initialize scene objects for visualization and physics simulation
void updateObjects() {
    objects.clear();
    
    // Add reference spheres to help visualize scale and orientation
    // Yellow sphere at +X axis (400 billion meters from center)
    objects.push_back({ vec4(4e11f, 0.0f, 0.0f, 4e10f), vec4(1,1,0,1), 1.98892e30 });
    // Red sphere at +Z axis  
    objects.push_back({ vec4(0.0f, 0.0f, 4e11f, 4e10f), vec4(1,0,0,1), 1.98892e30 });
    
    // Add invisible mass point representing the central black hole
    // This contributes to spacetime curvature calculations in the grid
    objects.push_back({ 
        vec4(0.0f, 0.0f, 0.0f, 0.1f),  // Tiny radius (effectively invisible)
        vec4(0,0,0,0),                  // Transparent
        static_cast<float>(SagA.mass)   // Full black hole mass
    });
}

// Main rendering and simulation engine
struct Engine {
    // OpenGL rendering resources
    GLuint gridShaderProgram;   // Shader for spacetime grid visualization
    GLFWwindow* window;         // Main application window
    GLuint quadVAO;            // Vertex array for fullscreen quad
    GLuint texture;            // Render target for compute shader
    GLuint shaderProgram;      // Fragment shader for final display
    GLuint computeProgram = 0; // GPU compute shader for ray tracing
    
    // Uniform buffer objects for GPU data
    GLuint cameraUBO = 0;      // Camera parameters
    GLuint diskUBO = 0;        // Accretion disk properties
    GLuint objectsUBO = 0;     // Scene object data
    GLuint featuresUBO = 0;    // Physics feature toggles
    
    // Grid rendering resources
    GLuint gridVAO = 0;        // Vertex array for spacetime grid
    GLuint gridVBO = 0;        // Vertex buffer
    GLuint gridEBO = 0;        // Element buffer for grid lines
    int gridIndexCount = 0;    // Number of grid line indices

    // Window and compute dimensions
    int WIDTH = 800;           // Display window width
    int HEIGHT = 600;          // Display window height
    int COMPUTE_WIDTH  = 200;  // Ray tracing resolution width
    int COMPUTE_HEIGHT = 150;  // Ray tracing resolution height
    
    // Viewport dimensions in physical units (meters)
    float width = 100000000000.0f;   // 100 billion meters
    float height = 75000000000.0f;   // 75 billion meters
    
    Engine() {
        if (!glfwInit()) {
            cerr << "GLFW init failed\n";
            exit(EXIT_FAILURE);
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Black Hole", nullptr, nullptr);
        if (!window) {
            cerr << "Failed to create GLFW window\n";
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;
        GLenum glewErr = glewInit();
        if (glewErr != GLEW_OK) {
            cerr << "Failed to initialize GLEW: "
                << (const char*)glewGetErrorString(glewErr)
                << "\n";
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        cout << "OpenGL " << glGetString(GL_VERSION) << "\n";
        
        // Initialize all shader programs
        this->shaderProgram = CreateShaderProgram();  // Default quad renderer
        gridShaderProgram = CreateShaderProgram("grid.vert", "grid.frag");  // Grid visualization
        computeProgram = CreateComputeProgram("geodesic.comp");  // Main ray tracing
        
        // Set up uniform buffer objects for efficient GPU data transfer
        glGenBuffers(1, &cameraUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
        glBufferData(GL_UNIFORM_BUFFER, 128, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, cameraUBO);  // Matches binding=1 in compute shader

        // Accretion disk parameters buffer
        glGenBuffers(1, &diskUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, diskUBO);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, diskUBO);

        // Scene objects buffer (supports up to 16 objects)
        glGenBuffers(1, &objectsUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
        GLsizeiptr objUBOSize = sizeof(int) + 3 * sizeof(float)    // Count + padding
            + 16 * (sizeof(vec4) + sizeof(vec4))  // Position/radius + color per object
            + 16 * sizeof(float);                 // Mass per object
        glBufferData(GL_UNIFORM_BUFFER, objUBOSize, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, objectsUBO);

        // Physics features configuration buffer
        glGenBuffers(1, &featuresUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, featuresUBO);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 33, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 4, featuresUBO);

        auto result = QuadVAO();
        this->quadVAO = result[0];
        this->texture = result[1];
    }
    // Generate 3D grid that visualizes spacetime curvature around massive objects
    void generateGrid(const vector<ObjectData>& objects) {
        const int gridSize = 25;        // 25x25 grid points
        const float spacing = 1e10f;    // 10 billion meters between grid lines

        vector<vec3> vertices;
        vector<GLuint> indices;

        // Generate grid vertices with curvature effects
        for (int z = 0; z <= gridSize; ++z) {
            for (int x = 0; x <= gridSize; ++x) {
                // Convert grid coordinates to world space
                float worldX = (x - gridSize / 2) * spacing;
                float worldZ = (z - gridSize / 2) * spacing;

                float y = 0.0f;  // Start at flat spacetime

                // Apply gravitational curvature from all massive objects
                for (const auto& obj : objects) {
                    vec3 objPos = vec3(obj.posRadius);
                    double mass = obj.mass;
                    
                    // Calculate Schwarzschild radius and distance to object
                    double r_s = 2.0 * G * mass / (c * c);
                    double dx = worldX - objPos.x;
                    double dz = worldZ - objPos.z;
                    double dist = sqrt(dx * dx + dz * dz);

                    // Apply curvature formula with numerical stability
                    if (dist > r_s) {
                        // Approximate spacetime curvature as vertical displacement
                        double deltaY = 2.0 * sqrt(r_s * (dist - r_s));
                        y += static_cast<float>(deltaY) - 3e10f;  // Offset for visibility
                    } else {
                        // Handle points near or inside event horizon
                        y += 2.0f * static_cast<float>(sqrt(r_s * r_s)) - 3e10f;
                    }
                }

                vertices.emplace_back(worldX, y, worldZ);
            }
        }

        // Create line indices for wireframe rendering
        // Connect each grid point to its neighbors (right and down)
        for (int z = 0; z < gridSize; ++z) {
            for (int x = 0; x < gridSize; ++x) {
                int i = z * (gridSize + 1) + x;
                // Horizontal line to next point
                indices.push_back(i);
                indices.push_back(i + 1);
                // Vertical line to point below
                indices.push_back(i);
                indices.push_back(i + gridSize + 1);
            }
        }

        // Upload geometry to GPU
        if (gridVAO == 0) glGenVertexArrays(1, &gridVAO);
        if (gridVBO == 0) glGenBuffers(1, &gridVBO);
        if (gridEBO == 0) glGenBuffers(1, &gridEBO);

        glBindVertexArray(gridVAO);

        // Upload vertex positions
        glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec3), vertices.data(), GL_DYNAMIC_DRAW);

        // Upload line indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

        // Configure vertex attributes
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)0);

        gridIndexCount = indices.size();
        glBindVertexArray(0);
    }
    void drawGrid(const mat4& viewProj) {
        glUseProgram(gridShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(gridShaderProgram, "viewProj"),
                        1, GL_FALSE, glm::value_ptr(viewProj));
        glBindVertexArray(gridVAO);

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glDrawElements(GL_LINES, gridIndexCount, GL_UNSIGNED_INT, 0);

        glBindVertexArray(0);
        glEnable(GL_DEPTH_TEST);
    }
    void drawFullScreenQuad() {
        glUseProgram(shaderProgram);
        glBindVertexArray(quadVAO);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);

        glDisable(GL_DEPTH_TEST);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);
        glEnable(GL_DEPTH_TEST);
    }
    GLuint CreateShaderProgram(){
        const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        })";

        const char* fragmentShaderSource = R"(
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTexture;
        void main() {
            FragColor = texture(screenTexture, TexCoord);
        })";

        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertexShader);

        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
        glCompileShader(fragmentShader);

        GLuint shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        return shaderProgram;
    };
    GLuint CreateShaderProgram(const char* vertPath, const char* fragPath) {
        auto loadShader = [](const char* path, GLenum type) -> GLuint {
            std::ifstream in(path);
            if (!in.is_open()) {
                std::cerr << "Failed to open shader: " << path << "\n";
                exit(EXIT_FAILURE);
            }
            std::stringstream ss;
            ss << in.rdbuf();
            std::string srcStr = ss.str();
            const char* src = srcStr.c_str();

            GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &src, nullptr);
            glCompileShader(shader);

            GLint success;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                GLint logLen;
                glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
                std::vector<char> log(logLen);
                glGetShaderInfoLog(shader, logLen, nullptr, log.data());
                std::cerr << "Shader compile error (" << path << "):\n" << log.data() << "\n";
                exit(EXIT_FAILURE);
            }
            return shader;
        };

        GLuint vertShader = loadShader(vertPath, GL_VERTEX_SHADER);
        GLuint fragShader = loadShader(fragPath, GL_FRAGMENT_SHADER);

        GLuint program = glCreateProgram();
        glAttachShader(program, vertShader);
        glAttachShader(program, fragShader);
        glLinkProgram(program);

        GLint linkSuccess;
        glGetProgramiv(program, GL_LINK_STATUS, &linkSuccess);
        if (!linkSuccess) {
            GLint logLen;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLen);
            std::vector<char> log(logLen);
            glGetProgramInfoLog(program, logLen, nullptr, log.data());
            std::cerr << "Shader link error:\n" << log.data() << "\n";
            exit(EXIT_FAILURE);
        }

        glDeleteShader(vertShader);
        glDeleteShader(fragShader);

        return program;
    }
    GLuint CreateComputeProgram(const char* path) {
        std::ifstream in(path);
        if(!in.is_open()) {
            std::cerr << "Failed to open compute shader: " << path << "\n";
            exit(EXIT_FAILURE);
        }
        std::stringstream ss;
        ss << in.rdbuf();
        std::string srcStr = ss.str();
        const char* src = srcStr.c_str();

        GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(cs, 1, &src, nullptr);
        glCompileShader(cs);
        GLint ok; 
        glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
        if(!ok) {
            GLint logLen;
            glGetShaderiv(cs, GL_INFO_LOG_LENGTH, &logLen);
            std::vector<char> log(logLen);
            glGetShaderInfoLog(cs, logLen, nullptr, log.data());
            std::cerr << "Compute shader compile error:\n" << log.data() << "\n";
            exit(EXIT_FAILURE);
        }

        GLuint prog = glCreateProgram();
        glAttachShader(prog, cs);
        glLinkProgram(prog);
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if(!ok) {
            GLint logLen;
            glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &logLen);
            std::vector<char> log(logLen);
            glGetProgramInfoLog(prog, logLen, nullptr, log.data());
            std::cerr << "Compute shader link error:\n" << log.data() << "\n";
            exit(EXIT_FAILURE);
        }

        glDeleteShader(cs);
        return prog;
    }
    void dispatchCompute(const Camera& cam) {
        // Adaptive resolution based on quality preset
        int baseW = int(COMPUTE_WIDTH * features.renderScale);
        int baseH = int(COMPUTE_HEIGHT * features.renderScale);
        
        // Further reduce when moving for responsiveness
        int cw = cam.moving ? int(baseW * 0.7f) : baseW;
        int ch = cam.moving ? int(baseH * 0.7f) : baseH;

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D,
                    0,
                    GL_RGBA8,
                    cw,
                    ch,
                    0, GL_RGBA, 
                    GL_UNSIGNED_BYTE, 
                    nullptr);

        glUseProgram(computeProgram);
        uploadCameraUBO(cam);
        uploadDiskUBO();
        uploadObjectsUBO(objects);
        uploadFeaturesUBO();

        glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

        GLuint groupsX = (GLuint)std::ceil(cw / 16.0f);
        GLuint groupsY = (GLuint)std::ceil(ch / 16.0f);
        glDispatchCompute(groupsX, groupsY, 1);

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
    void uploadCameraUBO(const Camera& cam) {
        struct UBOData {
            vec3 pos; float _pad0;
            vec3 right; float _pad1;
            vec3 up; float _pad2;
            vec3 forward; float _pad3;
            float tanHalfFov;
            float aspect;
            bool moving;
            int _pad4;
        } data;
        vec3 fwd = normalize(cam.target - cam.position());
        vec3 up = vec3(0, 1, 0);
        vec3 right = normalize(cross(fwd, up));
        up = cross(right, fwd);

        data.pos = cam.position();
        data.right = right;
        data.up = up;
        data.forward = fwd;
        data.tanHalfFov = tan(radians(60.0f * 0.5f));
        data.aspect = float(WIDTH) / float(HEIGHT);
        data.moving = cam.dragging || cam.panning;

        glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(UBOData), &data);
    }
    void uploadObjectsUBO(const vector<ObjectData>& objs) {
        struct UBOData {
            int   numObjects;
            float _pad0, _pad1, _pad2;        // <-- pad out to 16 bytes
            vec4  posRadius[16];
            vec4  color[16];
            float  mass[16]; 
        } data;

        size_t count = std::min(objs.size(), size_t(16));
        data.numObjects = static_cast<int>(count);

        for (size_t i = 0; i < count; ++i) {
            data.posRadius[i] = objs[i].posRadius;
            data.color[i] = objs[i].color;
            data.mass[i] = objs[i].mass;
        }
        glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(data), &data);
    }
    void uploadDiskUBO() {
        float r1 = SagA.r_s * 2.2f;
        float r2 = SagA.r_s * 5.2f;
        float num = 2.0;
        float thickness = 1e9f;
        float diskData[4] = { r1, r2, num, thickness };

        glBindBuffer(GL_UNIFORM_BUFFER, diskUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(diskData), diskData);
    }
    
    void uploadFeaturesUBO() {
        struct FeaturesData {
            float eddingtonFraction;
            float diskTempPeak;
            float lensingBoost;
            float enableMultiLayerDisk;
            float maxRaySteps;
            float diskRotationSpeed;
            float diskTurbulence;
            float enableDopplerBeaming;
            float enableKerrMetric;
            float spinParameter;
            float enableJets;
            float jetOpeningAngle;
            float jetBrightness;
            float time;
            
            // Enhanced jet parameters
            float jetHelixPitch;
            float jetPrecessionAngle;
            float jetPrecessionPeriod;
            float jetLorentzFactor;
            float jetShockSpeed;
            
            // Enhanced disk parameters
            float diskInnerTemp;
            float diskTempExponent;
            float diskScaleHeight;
            float diskInstabilityAmp;
            float diskInstabilityFreq;
            
            // Magnetic field parameters
            float magneticFieldStrength;
            float enableMagneticField;
            float fieldTurbulence;
            
            // Photon ring enhancement
            float photonRingOrders;
            float enablePhotonRing;
            
            // QPO and variability
            float qpoFrequency;
            float qpoAmplitude;
            float diskWindStrength;
        } data;
        
        // Original parameters
        data.eddingtonFraction = features.eddingtonFraction;
        data.diskTempPeak = features.diskTempPeak;
        data.lensingBoost = features.lensingBoost;
        data.enableMultiLayerDisk = features.enableMultiLayerDisk ? 1.0f : 0.0f;
        data.maxRaySteps = static_cast<float>(features.maxRaySteps);
        data.diskRotationSpeed = features.diskRotationSpeed;
        data.diskTurbulence = features.diskTurbulence;
        data.enableDopplerBeaming = features.enableDopplerBeaming ? 1.0f : 0.0f;
        data.enableKerrMetric = features.enableKerrMetric ? 1.0f : 0.0f;
        data.spinParameter = features.spinParameter;
        data.enableJets = features.enableJets ? 1.0f : 0.0f;
        data.jetOpeningAngle = features.jetOpeningAngle;
        data.jetBrightness = features.jetBrightness;
        data.time = static_cast<float>(glfwGetTime());
        
        // Enhanced jet parameters
        data.jetHelixPitch = features.jetHelixPitch;
        data.jetPrecessionAngle = features.jetPrecessionAngle;
        data.jetPrecessionPeriod = features.jetPrecessionPeriod;
        data.jetLorentzFactor = features.jetLorentzFactor;
        data.jetShockSpeed = features.jetShockSpeed;
        
        // Enhanced disk parameters
        data.diskInnerTemp = features.diskInnerTemp;
        data.diskTempExponent = features.diskTempExponent;
        data.diskScaleHeight = features.diskScaleHeight;
        data.diskInstabilityAmp = features.diskInstabilityAmp;
        data.diskInstabilityFreq = features.diskInstabilityFreq;
        
        // Magnetic field parameters
        data.magneticFieldStrength = features.magneticFieldStrength;
        data.enableMagneticField = features.enableMagneticField ? 1.0f : 0.0f;
        data.fieldTurbulence = features.fieldTurbulence;
        
        // Photon ring enhancement
        data.photonRingOrders = static_cast<float>(features.photonRingOrders);
        data.enablePhotonRing = features.enablePhotonRing ? 1.0f : 0.0f;
        
        // QPO and variability
        data.qpoFrequency = features.qpoFrequency;
        data.qpoAmplitude = features.qpoAmplitude;
        data.diskWindStrength = features.diskWindStrength;
        
        glBindBuffer(GL_UNIFORM_BUFFER, featuresUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(data), &data);
    }
    
    vector<GLuint> QuadVAO(){
        float quadVertices[] = {
            // positions   // texCoords
            -1.0f,  1.0f,  0.0f, 1.0f,  // top left
            -1.0f, -1.0f,  0.0f, 0.0f,  // bottom left
            1.0f, -1.0f,  1.0f, 0.0f,  // bottom right

            -1.0f,  1.0f,  0.0f, 1.0f,  // top left
            1.0f, -1.0f,  1.0f, 0.0f,  // bottom right
            1.0f,  1.0f,  1.0f, 1.0f   // top right
        };
        
        GLuint VAO, VBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D,
                    0,             // mip
                    GL_RGBA8,      // internal format
                    COMPUTE_WIDTH,
                    COMPUTE_HEIGHT,
                    0,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    nullptr);
        vector<GLuint> VAOtexture = {VAO, texture};
        return VAOtexture;
    }
    void renderScene() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(quadVAO);
        // make sure your fragment shader samples from texture unit 0:
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glfwSwapBuffers(window);
        glfwPollEvents();
    };
};
Engine engine;

// Display user interface controls and instructions
void showControls() {
    cout << "\n========== BLACK HOLE SIMULATION CONTROLS ==========" << endl;
    cout << "Mouse Controls:" << endl;
    cout << "  Left Click + Drag    - Orbit camera around black hole" << endl;
    cout << "  Right Click (hold)   - Enable gravity simulation" << endl;
    cout << "  Mouse Wheel          - Zoom in/out" << endl;
    cout << "\nSimulation Modes:" << endl;
    cout << "  Q - Toggle between Standard (Sgr A*) and TON618 Quasar modes" << endl;
    cout << "  G - Toggle gravity simulation for objects" << endl;
    cout << "\nRelativistic Effects:" << endl;
    cout << "  R - Toggle relativistic Doppler effects" << endl;
    cout << "  K - Toggle Kerr metric (rotating black hole)" << endl;
    cout << "  M - Toggle multi-layer accretion disk" << endl;
    cout << "  J - Toggle relativistic jets" << endl;
    cout << "\nQuality Settings:" << endl;
    cout << "  1 - Low quality    (32k steps, 0.7x resolution)" << endl;
    cout << "  2 - Medium quality (64k steps, 0.85x resolution)" << endl;
    cout << "  3 - High quality   (128k steps, 1.0x resolution)" << endl;
    cout << "  4 - Ultra quality  (256k steps, 1.0x resolution)" << endl;
    cout << "\nHelp:" << endl;
    cout << "  H - Show this help menu" << endl;
    cout << "\nFeatures in TON618 Quasar Mode:" << endl;
    cout << "  - Multi-layer accretion disk with temperature gradients" << endl;
    cout << "  - Relativistic jets with synchrotron radiation" << endl;
    cout << "  - Gravitational redshift and time dilation" << endl;
    cout << "  - Doppler beaming and disk rotation effects" << endl;
    cout << "  - Kerr black hole frame dragging" << endl;
    cout << "===================================================\n" << endl;
}

// Process keyboard input for simulation features
void handleFeatureKeys(int key, int action) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_Q) {
            features.toggle();
            updateObjects();
        }
        else if (key == GLFW_KEY_1) {
            features.setQualityLow();
        }
        else if (key == GLFW_KEY_2) {
            features.setQualityMedium();
        }
        else if (key == GLFW_KEY_3) {
            features.setQualityHigh();
        }
        else if (key == GLFW_KEY_4) {
            features.setQualityUltra();
        }
        else if (key == GLFW_KEY_R) {
            features.enableDopplerBeaming = !features.enableDopplerBeaming;
            cout << "[INFO] Relativistic effects: " << (features.enableDopplerBeaming ? "ON" : "OFF") << endl;
        }
        else if (key == GLFW_KEY_K) {
            features.enableKerrMetric = !features.enableKerrMetric;
            cout << "[INFO] Kerr metric (rotating BH): " << (features.enableKerrMetric ? "ON" : "OFF") << endl;
        }
        else if (key == GLFW_KEY_M) {
            features.enableMultiLayerDisk = !features.enableMultiLayerDisk;
            cout << "[INFO] Multi-layer disk: " << (features.enableMultiLayerDisk ? "ON" : "OFF") << endl;
        }
        else if (key == GLFW_KEY_J) {
            features.enableJets = !features.enableJets;
            cout << "[INFO] Relativistic jets: " << (features.enableJets ? "ON" : "OFF") << endl;
        }
        else if (key == GLFW_KEY_H) {
            showControls();
        }
    }
}

void setupCameraCallbacks(GLFWwindow* window) {
    glfwSetWindowUserPointer(window, &camera);

    glfwSetMouseButtonCallback(window, [](GLFWwindow* win, int button, int action, int mods) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processMouseButton(button, action, mods, win);
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* win, double x, double y) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processMouseMove(x, y);
    });

    glfwSetScrollCallback(window, [](GLFWwindow* win, double xoffset, double yoffset) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processScroll(xoffset, yoffset);
    });

    glfwSetKeyCallback(window, [](GLFWwindow* win, int key, int scancode, int action, int mods) {
        Camera* cam = (Camera*)glfwGetWindowUserPointer(win);
        cam->processKey(key, scancode, action, mods);
        handleFeatureKeys(key, action); // Handle QuasarFeatures keys
    });
}


// Main application entry point
int main() {
    // Display user interface controls and instructions at startup
    showControls();
    
    setupCameraCallbacks(engine.window);
    
    // Initialize objects based on default features
    updateObjects();
    
    vector<unsigned char> pixels(engine.WIDTH * engine.HEIGHT * 3);

    auto t0 = Clock::now();
    lastPrintTime = chrono::duration<double>(t0.time_since_epoch()).count();

    double lastTime = glfwGetTime();
    int   renderW  = 800, renderH = 600, numSteps = 80000;
    while (!glfwWindowShouldClose(engine.window)) {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        double now   = glfwGetTime();
        double dt    = now - lastTime;
        lastTime     = now;

        // Gravity
        for (auto& obj : objects) {
            for (auto& obj2 : objects) {
                if (&obj == &obj2) continue;
                 float dx  = obj2.posRadius.x - obj.posRadius.x;
                 float dy = obj2.posRadius.y - obj.posRadius.y;
                 float dz = obj2.posRadius.z - obj.posRadius.z;
                 float distance = sqrt(dx * dx + dy * dy + dz * dz);
                 if (distance > 0) {
                        vector<double> direction = {dx / distance, dy / distance, dz / distance};
                        double Gforce = (G * obj.mass * obj2.mass) / (distance * distance);

                        double acc1 = Gforce / obj.mass;
                        std::vector<double> acc = {direction[0] * acc1, direction[1] * acc1, direction[2] * acc1};
                        if (Gravity) {
                            obj.velocity.x += acc[0];
                            obj.velocity.y += acc[1];
                            obj.velocity.z += acc[2];

                            obj.posRadius.x += obj.velocity.x;
                            obj.posRadius.y += obj.velocity.y;
                            obj.posRadius.z += obj.velocity.z;
                            cout << "velocity: " <<obj.velocity.x<<", " <<obj.velocity.y<<", " <<obj.velocity.z<<endl;
                        }
                    }
            }
        }



        camera.update();
        
        engine.generateGrid(objects);
        mat4 view = lookAt(camera.position(), camera.target, vec3(0,1,0));
        mat4 proj = perspective(radians(60.0f), float(engine.COMPUTE_WIDTH)/engine.COMPUTE_HEIGHT, 1e9f, 1e14f);
        mat4 viewProj = proj * view;
        engine.drawGrid(viewProj);

        glViewport(0, 0, engine.WIDTH, engine.HEIGHT);
        engine.dispatchCompute(camera);
        engine.drawFullScreenQuad();

        framesCount++;
        auto t1 = Clock::now();
        double nowTime = chrono::duration<double>(t1.time_since_epoch()).count();
        if (nowTime - lastPrintTime >= 2.0) {
            double fps = framesCount / (nowTime - lastPrintTime);
            const char* quality = (features.quality == QuasarFeatures::QualityPreset::LOW) ? "LOW" :
                                 (features.quality == QuasarFeatures::QualityPreset::MEDIUM) ? "MEDIUM" : "HIGH";
            const char* profile = (features.profile == FeatureProfile::STANDARD) ? "Sgr A*" : "TON618";
            
            cout << "[PERF] " << fixed << setprecision(1) << fps << " FPS | " 
                 << quality << " quality | " << profile << " profile | "
                 << "Scale: " << features.renderScale << "x" << endl;
            
            framesCount = 0;
            lastPrintTime = nowTime;
        }

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    glfwDestroyWindow(engine.window);
    glfwTerminate();
    return 0;
}
