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

// VARS
double lastPrintTime = 0.0;
int    framesCount   = 0;
double c = 299792458.0;
double G = 6.67430e-11;
struct Ray;
bool Gravity = false;

struct Camera {
    // Center the camera orbit on the black hole at (0, 0, 0)
    vec3 target = vec3(0.0f, 0.0f, 0.0f); // Always look at the black hole center
    float radius = 6.34194e10f;
    float minRadius = 1e10f, maxRadius = 1e12f;

    float azimuth = 0.0f;
    float elevation = M_PI / 2.0f;

    float orbitSpeed = 0.01f;
    float panSpeed = 0.01f;
    double zoomSpeed = 25e9f;

    bool dragging = false;
    bool panning = false;
    bool moving = false; // For compute shader optimization
    bool firstMouse = true; // Track first mouse movement to prevent jumps
    double lastX = 0.0, lastY = 0.0;

    // Calculate camera position in world space
    vec3 position() const {
        float clampedElevation = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);
        // Orbit around (0,0,0) always
        return vec3(
            radius * sin(clampedElevation) * cos(azimuth),
            radius * cos(clampedElevation),
            radius * sin(clampedElevation) * sin(azimuth)
        );
    }
    void update() {
        // Always keep target at black hole center
        target = vec3(0.0f, 0.0f, 0.0f);
        
        static double lastMoveTime = 0.0;
        if(dragging | panning) {
            moving = true;
            lastMoveTime = glfwGetTime();
        } else {
            // Add small delay before switching to high quality to reduce snapping
            double timeSinceMove = glfwGetTime() - lastMoveTime;
            moving = timeSinceMove < 0.2; // Shorter delay for better visual consistency
        }
    }

    void processMouseMove(double x, double y) {
        // Handle first mouse movement to prevent jumps
        if (firstMouse || !dragging) {
            lastX = x;
            lastY = y;
            firstMouse = false;
            return;
        }
        
        float dx = float(x - lastX);
        float dy = float(y - lastY);
        
        // Clamp delta to prevent huge jumps from initialization issues
        dx = glm::clamp(dx, -100.0f, 100.0f);
        dy = glm::clamp(dy, -100.0f, 100.0f);

        if (dragging && panning) {
            // Pan: Shift + Left or Middle Mouse
            // Disable panning to keep camera centered on black hole
        }
        else if (dragging && !panning) {
            // Orbit: Left mouse only
            azimuth   += dx * orbitSpeed;
            elevation -= dy * orbitSpeed;
            elevation = glm::clamp(elevation, 0.01f, float(M_PI) - 0.01f);
        }

        lastX = x;
        lastY = y;
        update();
    }
    void processMouseButton(int button, int action, int mods, GLFWwindow* win) {
        if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_MIDDLE) {
            if (action == GLFW_PRESS) {
                dragging = true;
                // Disable panning so camera always orbits center
                panning = false;
                firstMouse = true; // Reset first mouse flag on new drag
                glfwGetCursorPos(win, &lastX, &lastY);
            } else if (action == GLFW_RELEASE) {
                dragging = false;
                panning = false;
            }
        }
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS) {
                Gravity = true;
            } else if (action == GLFW_RELEASE) {
                Gravity = false;
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
        // Note: QuasarFeatures key handling moved to after class definition
    }
};
Camera camera;

struct BlackHole {
    vec3 position;
    double mass;
    double radius;
    double r_s;

    BlackHole(vec3 pos, float m) : position(pos), mass(m) {r_s = 2.0 * G * mass / (c*c);}
    bool Intercept(float px, float py, float pz) const {
        double dx = double(px) - double(position.x);
        double dy = double(py) - double(position.y);
        double dz = double(pz) - double(position.z);
        double dist2 = dx * dx + dy * dy + dz * dz;
        return dist2 < r_s * r_s;
    }
};

enum class FeatureProfile { STANDARD, TON618_QUASAR };
struct QuasarFeatures {
    FeatureProfile profile = FeatureProfile::STANDARD;
    float eddingtonFraction = 0.1f;    // Standard Sgr A*: low activity
    float diskTempPeak = 1e6f;         // Standard: cooler disk
    float lensingBoost = 1.0f;         // Always physically accurate
    bool enableMultiLayerDisk = false;
    
    // Performance settings
    enum class QualityPreset { LOW, MEDIUM, HIGH, ULTRA };
    QualityPreset quality = QualityPreset::MEDIUM;
    float renderScale = 0.85f;      // Resolution multiplier
    int maxRaySteps = 64000;        // Adaptive based on quality
    bool halfResVolumetrics = true; // For jets/disk
    
    // Rotation and dynamics
    float diskRotationSpeed = 0.0f; // Rotation speed multiplier
    float diskTurbulence = 0.0f;    // Turbulence/chaos factor
    bool enableDopplerBeaming = false; // Relativistic Doppler effects
    
    // New relativistic parameters
    bool enableKerrMetric = false;  // Enable Kerr black hole effects
    float spinParameter = 0.0f;     // Black hole spin (0.0 = Schwarzschild, 0.998 = maximum)
    
    // Jet parameters
    bool enableJets = false;        // Enable relativistic jets
    float jetOpeningAngle = 5.0f;   // Jet opening angle in degrees
    float jetBrightness = 1.0f;     // Jet brightness multiplier
    
    void switchToTON618() {
        profile = FeatureProfile::TON618_QUASAR;
        eddingtonFraction = 0.7f;      // Active quasar: high accretion
        diskTempPeak = 1e7f;           // Hot, bright disk
        enableMultiLayerDisk = true;
        diskRotationSpeed = 2.0f;      // Fast rotation like active quasar
        diskTurbulence = 0.3f;         // Chaotic accretion patterns
        enableDopplerBeaming = true;   // Relativistic effects
        enableKerrMetric = true;       // Enable rotating black hole physics
        spinParameter = 0.8f;          // High spin parameter for active quasar
        enableJets = true;             // Enable relativistic jets
        jetOpeningAngle = 6.0f;        // Slightly wider jets for quasar
        jetBrightness = 2.0f;          // Brighter jets for active system
        cout << "[INFO] Switched to TON618 Quasar: enhanced disk and jet physics!" << endl;
    }
    
    void switchToStandard() {
        profile = FeatureProfile::STANDARD;
        eddingtonFraction = 0.1f;
        diskTempPeak = 1e6f;
        enableMultiLayerDisk = false;
        diskRotationSpeed = 0.1f;      // Slow, stable rotation
        diskTurbulence = 0.05f;        // Minimal turbulence
        enableDopplerBeaming = false;  // No relativistic effects
        enableKerrMetric = false;      // Classic Schwarzschild (non-rotating)
        spinParameter = 0.0f;          // No spin
        enableJets = false;            // No jets in standard mode
        jetOpeningAngle = 5.0f;        // Default values
        jetBrightness = 1.0f;          // Default values
        cout << "[INFO] Switched to Standard Sgr A*: classic Schwarzschild black hole" << endl;
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


BlackHole SagA(vec3(0.0f, 0.0f, 0.0f), 8.54e36); // Sagittarius A black hole
QuasarFeatures features; // Global features instance

struct ObjectData {
    vec4 posRadius; // xyz = position, w = radius
    vec4 color;     // rgb = color, a = unused
    float  mass;
    vec3 velocity = vec3(0.0f, 0.0f, 0.0f); // Initial velocity
};
vector<ObjectData> objects;

// Function to update objects based on current features
void updateObjects() {
    // Reset to base objects (excluding jets)
    objects.clear();
    
    // Add standard objects (distant reference spheres)
    objects.push_back({ vec4(4e11f, 0.0f, 0.0f, 4e10f), vec4(1,1,0,1), 1.98892e30 });
    objects.push_back({ vec4(0.0f, 0.0f, 4e11f, 4e10f), vec4(1,0,0,1), 1.98892e30 });
    
    // Add black hole for grid curvature calculation (invisible to raytracer)
    objects.push_back({ 
        vec4(0.0f, 0.0f, 0.0f, 0.1f), // Very small radius so raytracer ignores it
        vec4(0,0,0,0), // Transparent/invisible color  
        static_cast<float>(SagA.mass) // Full mass for grid curvature
    });
    
    // Jets removed - focusing on disk physics
}

struct Engine {
    GLuint gridShaderProgram;
    // -- Quad & Texture render -- //
    GLFWwindow* window;
    GLuint quadVAO;
    GLuint texture;
    GLuint shaderProgram;
    GLuint computeProgram = 0;
    // -- UBOs -- //
    GLuint cameraUBO = 0;
    GLuint diskUBO = 0;
    GLuint objectsUBO = 0;
    GLuint featuresUBO = 0;
    // -- grid mess vars -- //
    GLuint gridVAO = 0;
    GLuint gridVBO = 0;
    GLuint gridEBO = 0;
    int gridIndexCount = 0;

    int WIDTH = 800;  // Window width
    int HEIGHT = 600; // Window height
    int COMPUTE_WIDTH  = 200;   // Compute resolution width
    int COMPUTE_HEIGHT = 150;  // Compute resolution height
    float width = 100000000000.0f; // Width of the viewport in meters
    float height = 75000000000.0f; // Height of the viewport in meters
    
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
        this->shaderProgram = CreateShaderProgram();
        gridShaderProgram = CreateShaderProgram("grid.vert", "grid.frag");

        computeProgram = CreateComputeProgram("geodesic.comp");
        glGenBuffers(1, &cameraUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
        glBufferData(GL_UNIFORM_BUFFER, 128, nullptr, GL_DYNAMIC_DRAW); // alloc ~128 bytes
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, cameraUBO); // binding = 1 matches shader

        glGenBuffers(1, &diskUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, diskUBO);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW); // 3 values + 1 padding
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, diskUBO); // binding = 2 matches compute shader

        glGenBuffers(1, &objectsUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
        // allocate space for 16 objects: 
        // sizeof(int) + padding + 16×(vec4 posRadius + vec4 color)
        GLsizeiptr objUBOSize = sizeof(int) + 3 * sizeof(float)
            + 16 * (sizeof(vec4) + sizeof(vec4))
            + 16 * sizeof(float); // 16 floats for mass
        glBufferData(GL_UNIFORM_BUFFER, objUBOSize, nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 3, objectsUBO);  // binding = 3 matches shader

        glGenBuffers(1, &featuresUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, featuresUBO);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * 14, nullptr, GL_DYNAMIC_DRAW); // features data (14 parameters)
        glBindBufferBase(GL_UNIFORM_BUFFER, 4, featuresUBO); // binding = 4 for features

        auto result = QuadVAO();
        this->quadVAO = result[0];
        this->texture = result[1];
    }
    void generateGrid(const vector<ObjectData>& objects) {
        const int gridSize = 25;
        const float spacing = 1e10f;  // tweak this

        vector<vec3> vertices;
        vector<GLuint> indices;

        for (int z = 0; z <= gridSize; ++z) {
            for (int x = 0; x <= gridSize; ++x) {
                float worldX = (x - gridSize / 2) * spacing;
                float worldZ = (z - gridSize / 2) * spacing;

                float y = 0.0f;

                // ✅ Warp grid using Schwarzschild geometry
                for (const auto& obj : objects) {
                    vec3 objPos = vec3(obj.posRadius);
                    double mass = obj.mass;
                    double radius = obj.posRadius.w;

                    double r_s = 2.0 * G * mass / (c * c);
                    double dx = worldX - objPos.x;
                    double dz = worldZ - objPos.z;
                    double dist = sqrt(dx * dx + dz * dz);

                    // prevent sqrt of negative or divide-by-zero (inside or at the black hole center)
                    if (dist > r_s) {
                        double deltaY = 2.0 * sqrt(r_s * (dist - r_s));
                        y += static_cast<float>(deltaY) - 3e10f;
                    } else {
                        // 🔴 For points inside or at r_s: make it dip down sharply
                        y += 2.0f * static_cast<float>(sqrt(r_s * r_s)) - 3e10f;  // or add a deep pit
                    }
                }

                vertices.emplace_back(worldX, y, worldZ);
            }
        }

        // 🧩 Add indices for GL_LINE rendering
        for (int z = 0; z < gridSize; ++z) {
            for (int x = 0; x < gridSize; ++x) {
                int i = z * (gridSize + 1) + x;
                indices.push_back(i);
                indices.push_back(i + 1);

                indices.push_back(i);
                indices.push_back(i + gridSize + 1);
            }
        }

        // 🔌 Upload to GPU
        if (gridVAO == 0) glGenVertexArrays(1, &gridVAO);
        if (gridVBO == 0) glGenBuffers(1, &gridVBO);
        if (gridEBO == 0) glGenBuffers(1, &gridEBO);

        glBindVertexArray(gridVAO);

        glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec3), vertices.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0); // location = 0
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
        glUseProgram(shaderProgram); // fragment + vertex shader
        glBindVertexArray(quadVAO);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(shaderProgram, "screenTexture"), 0);

        glDisable(GL_DEPTH_TEST);  // draw as background
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);  // 2 triangles
        glEnable(GL_DEPTH_TEST);
    }
    GLuint CreateShaderProgram(){
        const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;  // Changed to vec2
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);  // Explicit z=0
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

        // vertex shader
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertexShader);

        // fragment shader
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
        // 1) read GLSL source
        std::ifstream in(path);
        if(!in.is_open()) {
            std::cerr << "Failed to open compute shader: " << path << "\n";
            exit(EXIT_FAILURE);
        }
        std::stringstream ss;
        ss << in.rdbuf();
        std::string srcStr = ss.str();
        const char* src = srcStr.c_str();

        // 2) compile
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

        // 3) link
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

        // 1) reallocate the texture if needed
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D,
                    0,                // mip
                    GL_RGBA8,         // internal format
                    cw,               // width
                    ch,               // height
                    0, GL_RGBA, 
                    GL_UNSIGNED_BYTE, 
                    nullptr);

        // 2) bind compute program & UBOs
        glUseProgram(computeProgram);
        uploadCameraUBO(cam);
        uploadDiskUBO();
        uploadObjectsUBO(objects);
        uploadFeaturesUBO();

        // 3) bind it as image unit 0
        glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

        // 4) dispatch grid
        GLuint groupsX = (GLuint)std::ceil(cw / 16.0f);
        GLuint groupsY = (GLuint)std::ceil(ch / 16.0f);
        glDispatchCompute(groupsX, groupsY, 1);

        // 5) sync
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
        vec3 up = vec3(0, 1, 0); // y axis is up, so disk is in x-z plane
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

        // Upload
        glBindBuffer(GL_UNIFORM_BUFFER, objectsUBO);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(data), &data);
    }
    void uploadDiskUBO() {
        // disk
        float r1 = SagA.r_s * 2.2f;    // inner radius just outside the event horizon
        float r2 = SagA.r_s * 5.2f;   // outer radius of the disk
        float num = 2.0;               // number of rays
        float thickness = 1e9f;          // padding for std140 alignment
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
        } data;
        
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

// Display controls help
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

// Handle QuasarFeatures key bindings
void handleFeatureKeys(int key, int action) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_Q) {
            cout << "[DEBUG] Q key pressed - switching profiles..." << endl;
            features.toggle();
            updateObjects(); // Update objects when profile changes
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


// -- MAIN -- //
int main() {
    // Display controls help at startup
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
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // optional, but good practice
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        double now   = glfwGetTime();
        double dt    = now - lastTime;   // seconds since last frame
        lastTime     = now;

        // Gravity
        for (auto& obj : objects) {
            for (auto& obj2 : objects) {
                if (&obj == &obj2) continue; // skip self-interaction
                 float dx  = obj2.posRadius.x - obj.posRadius.x;
                 float dy = obj2.posRadius.y - obj.posRadius.y;
                 float dz = obj2.posRadius.z - obj.posRadius.z;
                 float distance = sqrt(dx * dx + dy * dy + dz * dz);
                 if (distance > 0) {
                        vector<double> direction = {dx / distance, dy / distance, dz / distance};
                        //distance *= 1000;
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



        // ---------- CAMERA UPDATE ------------- //
        camera.update(); // Update camera state and moving flag
        
        // ---------- GRID ------------- //
        // 2) rebuild grid mesh on CPU
        engine.generateGrid(objects);
        // 5) overlay the bent grid
        mat4 view = lookAt(camera.position(), camera.target, vec3(0,1,0));
        mat4 proj = perspective(radians(60.0f), float(engine.COMPUTE_WIDTH)/engine.COMPUTE_HEIGHT, 1e9f, 1e14f);
        mat4 viewProj = proj * view;
        engine.drawGrid(viewProj);

        // ---------- RUN RAYTRACER ------------- //
        glViewport(0, 0, engine.WIDTH, engine.HEIGHT);
        engine.dispatchCompute(camera);
        engine.drawFullScreenQuad();

        // 6) FPS counter with performance info
        framesCount++;
        auto t1 = Clock::now();
        double nowTime = chrono::duration<double>(t1.time_since_epoch()).count();
        if (nowTime - lastPrintTime >= 2.0) { // Every 2 seconds
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

        // 7) present to screen
        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    glfwDestroyWindow(engine.window);
    glfwTerminate();
    return 0;
}
