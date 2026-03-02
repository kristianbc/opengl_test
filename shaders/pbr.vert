#version 450 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUV;
layout (location = 3) in vec4 aTangent;

uniform mat4 uModel;
uniform mat4 uViewProj;

out VS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 uv;
    vec3 tangent;
    float tangentSign;
} vs;

void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    mat3 nmat = transpose(inverse(mat3(uModel)));

    vs.worldPos = world.xyz;
    vs.normal = normalize(nmat * aNormal);
    vs.uv = aUV;
    vs.tangent = normalize(mat3(uModel) * aTangent.xyz);
    vs.tangentSign = aTangent.w;

    gl_Position = uViewProj * world;
}
