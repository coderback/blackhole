#version 330 core

// Fragment shader for spacetime grid rendering
// Outputs semi-transparent gray lines to visualize spacetime curvature
out vec4 FragColor;

void main() {
    // Render grid lines as semi-transparent gray
    // Alpha blending allows visibility of objects behind the grid
    FragColor = vec4(0.5, 0.5, 0.5, 0.7);  // 50% gray with 70% opacity
}
