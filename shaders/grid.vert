#version 330 core

// Vertex shader for spacetime curvature grid visualization
// Input: 3D vertex positions representing curved spacetime geometry
layout(location = 0) in vec3 aPos;

// Combined view-projection matrix from camera
uniform mat4 viewProj;

void main() {
    // Transform curved spacetime vertex to clip space
    gl_Position = viewProj * vec4(aPos, 1.0);
}
