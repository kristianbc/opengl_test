#version 450 core

in VS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 uv;
    vec3 tangent;
    float tangentSign;
} fs;

out vec4 FragColor;

struct Material {
    vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float occlusionStrength;
};

uniform Material uMat;
uniform vec3 uCameraPos;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform int uHasTangents;
uniform int uDebugFlatShade;

uniform sampler2D uBaseColorTex;
uniform sampler2D uMetallicRoughnessTex;
uniform sampler2D uNormalTex;
uniform sampler2D uOcclusionTex;

uniform int uHasBaseColorTex;
uniform int uHasMetallicRoughnessTex;
uniform int uHasNormalTex;
uniform int uHasOcclusionTex;
uniform int uDebugShowAlbedo;
uniform int uDisableNormalTex;
uniform int uDisableOcclusionTex;
uniform int uFlipNormalGreen;
uniform int uWireframeMode;
uniform vec3 uWireframeColor;

const float PI = 3.14159265359;

vec3 SRGBToLinear(vec3 c) {
    return pow(c, vec3(2.2));
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / max(denom, 1e-6);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return num / max(denom, 1e-6);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

vec3 BuildNormal() {
    vec3 N = normalize(fs.normal);
    if (length(N) < 1e-6) {
        return vec3(0.0, 1.0, 0.0);
    }
    if (uHasNormalTex == 0 || uHasTangents == 0 || uDisableNormalTex == 1) {
        return N;
    }

    vec3 T = vec3(0.0);
    vec3 B = vec3(0.0);
    bool validTbn = false;

    vec3 tangentRaw = fs.tangent - N * dot(fs.tangent, N);
    if (length(tangentRaw) >= 1e-6) {
        vec3 bitangentRaw = cross(N, tangentRaw);
        if (length(bitangentRaw) >= 1e-6) {
            T = normalize(tangentRaw);
            B = normalize(bitangentRaw) * fs.tangentSign;
            validTbn = true;
        }
    }

    // Fallback for meshes with broken/degenerate tangents on only part of the surface.
    if (!validTbn) {
        vec3 dp1 = dFdx(fs.worldPos);
        vec3 dp2 = dFdy(fs.worldPos);
        vec2 duv1 = dFdx(fs.uv);
        vec2 duv2 = dFdy(fs.uv);

        vec3 t = dp1 * duv2.y - dp2 * duv1.y;
        vec3 b = dp2 * duv1.x - dp1 * duv2.x;
        if (length(t) < 1e-6 || length(b) < 1e-6) {
            return N;
        }
        T = normalize(t - N * dot(N, t));
        B = normalize(b - N * dot(N, b));
    }

    mat3 TBN = mat3(T, B, N);

    vec3 normalSample = texture(uNormalTex, fs.uv).xyz * 2.0 - 1.0;
    if (uFlipNormalGreen == 1) {
        normalSample.y = -normalSample.y;
    }
    if (length(normalSample) < 1e-6) {
        return N;
    }
    normalSample.xy *= uMat.normalScale;
    return normalize(TBN * normalSample);
}

void main() {
    if (uWireframeMode == 1) {
        FragColor = vec4(uWireframeColor, 1.0);
        return;
    }

    vec4 baseColor = uMat.baseColorFactor;
    if (uHasBaseColorTex == 1) {
        vec4 sampled = texture(uBaseColorTex, fs.uv);
        sampled.rgb = SRGBToLinear(sampled.rgb);
        baseColor *= sampled;
    }

    float metallic = clamp(uMat.metallicFactor, 0.0, 1.0);
    float roughness = clamp(uMat.roughnessFactor, 0.045, 1.0);

    if (uHasMetallicRoughnessTex == 1) {
        vec4 mr = texture(uMetallicRoughnessTex, fs.uv);
        roughness *= mr.g;
        metallic *= mr.b;
    }

    vec3 N = BuildNormal();
    if (uDebugFlatShade == 1) {
        float ndotl = max(dot(N, normalize(-uLightDir)), 0.0);
        vec3 debugColor = mix(vec3(0.12, 0.6, 1.0), vec3(1.0, 0.55, 0.12), clamp(ndotl, 0.0, 1.0));
        FragColor = vec4(debugColor, 1.0);
        return;
    }
    if (uDebugShowAlbedo == 1) {
        FragColor = vec4(pow(baseColor.rgb, vec3(1.0 / 2.2)), baseColor.a);
        return;
    }
    vec3 V = normalize(uCameraPos - fs.worldPos);
    vec3 L = normalize(-uLightDir);
    vec3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);

    vec3 albedo = baseColor.rgb;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 1e-4;
    vec3 specular = numerator / denominator;

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    vec3 diffuse = kD * albedo / PI;
    vec3 direct = (diffuse + specular) * uLightColor * NdotL;

    float ao = 1.0;
    if (uHasOcclusionTex == 1 && uDisableOcclusionTex == 0) {
        float occ = texture(uOcclusionTex, fs.uv).r;
        ao = mix(1.0, occ, clamp(uMat.occlusionStrength, 0.0, 1.0));
    }
    ao = max(ao, 0.28);
    float up = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 hemiAmbient = mix(vec3(0.05, 0.055, 0.065), vec3(0.15, 0.16, 0.17), up);
    vec3 ambient = albedo * hemiAmbient * ao;
    vec3 color = ambient + direct;

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, baseColor.a);
}
