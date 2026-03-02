#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <windows.h>
#include <stb_image.h>

#include <tiny_gltf.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

constexpr int kWindowWidth = 1920;
constexpr int kWindowHeight = 1080;
constexpr float kMouseSensitivity = 0.09f;
constexpr float kBaseMoveSpeed = 5.5f;

struct Shader {
    GLuint id = 0;

    static std::string ReadTextFile(const fs::path& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open shader: " + path.string());
        }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        return content;
    }

    static GLuint Compile(GLenum type, const std::string& source) {
        const char* src = source.c_str();
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);

        GLint ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            GLint logLen = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
            std::string log(logLen, '\0');
            glGetShaderInfoLog(shader, logLen, nullptr, log.data());
            throw std::runtime_error("Shader compile error:\n" + log);
        }
        return shader;
    }

    void Load(const fs::path& vertexPath, const fs::path& fragmentPath) {
        const auto vertexSrc = ReadTextFile(vertexPath);
        const auto fragmentSrc = ReadTextFile(fragmentPath);

        const GLuint vs = Compile(GL_VERTEX_SHADER, vertexSrc);
        const GLuint fs = Compile(GL_FRAGMENT_SHADER, fragmentSrc);

        id = glCreateProgram();
        glAttachShader(id, vs);
        glAttachShader(id, fs);
        glLinkProgram(id);

        GLint ok = 0;
        glGetProgramiv(id, GL_LINK_STATUS, &ok);
        if (!ok) {
            GLint logLen = 0;
            glGetProgramiv(id, GL_INFO_LOG_LENGTH, &logLen);
            std::string log(logLen, '\0');
            glGetProgramInfoLog(id, logLen, nullptr, log.data());
            glDeleteShader(vs);
            glDeleteShader(fs);
            throw std::runtime_error("Shader link error:\n" + log);
        }

        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    void Use() const { glUseProgram(id); }

    GLint Uniform(const char* name) const { return glGetUniformLocation(id, name); }

    ~Shader() {
        if (id != 0) {
            glDeleteProgram(id);
        }
    }
};

struct Camera {
    glm::vec3 position {0.0f, 1.2f, 3.5f};
    float yaw = -90.0f;
    float pitch = -8.0f;

    glm::vec3 Forward() const {
        const float yawRad = glm::radians(yaw);
        const float pitchRad = glm::radians(pitch);
        return glm::normalize(glm::vec3(
            std::cos(yawRad) * std::cos(pitchRad),
            std::sin(pitchRad),
            std::sin(yawRad) * std::cos(pitchRad)
        ));
    }

    glm::mat4 View() const {
        const glm::vec3 forward = Forward();
        return glm::lookAt(position, position + forward, glm::vec3(0.0f, 1.0f, 0.0f));
    }
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec4 tangent;
};

struct GpuPrimitive {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei indexCount = 0;
    GLsizei vertexCount = 0;
    int materialIndex = -1;
    bool hasTangents = false;
    glm::vec3 localMin {0.0f};
    glm::vec3 localMax {0.0f};

    GpuPrimitive() = default;
    GpuPrimitive(const GpuPrimitive&) = delete;
    GpuPrimitive& operator=(const GpuPrimitive&) = delete;

    GpuPrimitive(GpuPrimitive&& other) noexcept
        : vao(other.vao),
          vbo(other.vbo),
          ebo(other.ebo),
          indexCount(other.indexCount),
          vertexCount(other.vertexCount),
          materialIndex(other.materialIndex),
          hasTangents(other.hasTangents),
          localMin(other.localMin),
          localMax(other.localMax) {
        other.vao = 0;
        other.vbo = 0;
        other.ebo = 0;
        other.indexCount = 0;
        other.vertexCount = 0;
        other.materialIndex = -1;
        other.hasTangents = false;
    }

    GpuPrimitive& operator=(GpuPrimitive&& other) noexcept {
        if (this == &other) return *this;
        if (ebo) glDeleteBuffers(1, &ebo);
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);

        vao = other.vao;
        vbo = other.vbo;
        ebo = other.ebo;
        indexCount = other.indexCount;
        vertexCount = other.vertexCount;
        materialIndex = other.materialIndex;
        hasTangents = other.hasTangents;
        localMin = other.localMin;
        localMax = other.localMax;

        other.vao = 0;
        other.vbo = 0;
        other.ebo = 0;
        other.indexCount = 0;
        other.vertexCount = 0;
        other.materialIndex = -1;
        other.hasTangents = false;
        return *this;
    }

    ~GpuPrimitive() {
        if (ebo) glDeleteBuffers(1, &ebo);
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
    }
};

struct GpuTexture {
    GLuint id = 0;

    GpuTexture() = default;
    GpuTexture(const GpuTexture&) = delete;
    GpuTexture& operator=(const GpuTexture&) = delete;

    GpuTexture(GpuTexture&& other) noexcept : id(other.id) {
        other.id = 0;
    }

    GpuTexture& operator=(GpuTexture&& other) noexcept {
        if (this == &other) return *this;
        if (id) glDeleteTextures(1, &id);
        id = other.id;
        other.id = 0;
        return *this;
    }

    ~GpuTexture() {
        if (id) glDeleteTextures(1, &id);
    }
};

struct PbrMaterial {
    glm::vec4 baseColorFactor {1.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    float normalScale = 1.0f;
    float occlusionStrength = 1.0f;

    int baseColorTexture = -1;
    int metallicRoughnessTexture = -1;
    int normalTexture = -1;
    int occlusionTexture = -1;
    bool doubleSided = false;
};

struct DrawCommand {
    int primitive = -1;
    glm::mat4 transform {1.0f};
    int materialIndex = -1;
};

struct SceneModel {
    std::vector<GpuPrimitive> primitives;
    std::vector<GpuTexture> textures;
    std::vector<PbrMaterial> materials;
    std::vector<DrawCommand> drawCommands;
};

static Camera g_camera;
static bool g_firstMouse = true;
static double g_lastMouseX = 0.0;
static double g_lastMouseY = 0.0;
static int g_fbWidth = kWindowWidth;
static int g_fbHeight = kWindowHeight;
static bool g_debugConsoleVisible = false;
static bool g_f11WasDown = false;
static bool g_f12WasDown = false;
static bool g_f9WasDown = false;
static bool g_f8WasDown = false;
static bool g_f7WasDown = false;
static bool g_f6WasDown = false;
static bool g_f5WasDown = false;
static bool g_f4WasDown = false;
static bool g_pWasDown = false;
static bool g_debugSpamPaused = false;
static bool g_forceDebugFlatShade = false;
static bool g_forceShowAlbedo = false;
static bool g_disableNormalMap = false;
static bool g_disableOcclusionMap = false;
static bool g_flipNormalGreen = false;
static bool g_wireframeMode = false;
static glm::vec3 g_wireframeColor {0.06f, 1.0f, 0.18f};
static bool g_statisticsEnabled = false;
static int g_selectedDrawCommand = -1;
static bool g_lmbWasDown = false;
static glm::vec3 g_lightDir {-0.55f, -1.0f, -0.35f};
static glm::vec3 g_lightColor {5.0f, 4.8f, 4.6f};
static HWND g_debugWindow = nullptr;
static HWND g_debugEdit = nullptr;
static HWND g_debugInput = nullptr;
static WNDPROC g_debugInputOldProc = nullptr;
static HWND g_statsOverlay = nullptr;
static HBRUSH g_debugBrush = nullptr;
static HFONT g_debugFont = nullptr;
static HFONT g_statsFont = nullptr;
static std::wstring g_statsOverlayText;
static std::vector<std::string> g_startupSummaryLines;
static std::string g_lastCommandHint;

struct SceneBounds {
    glm::vec3 min {0.0f};
    glm::vec3 max {0.0f};
    bool valid = false;
};

static std::wstring Utf8ToWide(const std::string& text) {
    if (text.empty()) return L"";
    const int needed = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (needed <= 0) return L"";
    std::wstring out(static_cast<size_t>(needed - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, out.data(), needed);
    return out;
}

static std::string WideToUtf8(const std::wstring& text) {
    if (text.empty()) return {};
    const int needed = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 0) return {};
    std::string out(static_cast<size_t>(needed - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, out.data(), needed, nullptr, nullptr);
    return out;
}

static void DebugLog(const std::string& text) {
    if (!g_debugEdit) return;
    std::wstring w = Utf8ToWide(text) + L"\r\n";
    const int len = GetWindowTextLengthW(g_debugEdit);
    SendMessageW(g_debugEdit, EM_SETSEL, static_cast<WPARAM>(len), static_cast<LPARAM>(len));
    SendMessageW(g_debugEdit, EM_REPLACESEL, FALSE, reinterpret_cast<LPARAM>(w.c_str()));
    const int newLen = GetWindowTextLengthW(g_debugEdit);
    if (newLen > 200000) {
        SendMessageW(g_debugEdit, EM_SETSEL, 0, 100000);
        SendMessageW(g_debugEdit, EM_REPLACESEL, FALSE, reinterpret_cast<LPARAM>(L""));
    }
    SendMessageW(g_debugEdit, EM_SCROLLCARET, 0, 0);
}

static void StartupSummaryLog(const std::string& text) {
    std::cout << text << "\n";
    g_startupSummaryLines.push_back(text);
    DebugLog(text);
}

static const char* GlDebugSourceString(GLenum source) {
    switch (source) {
        case GL_DEBUG_SOURCE_API: return "API";
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM: return "WindowSystem";
        case GL_DEBUG_SOURCE_SHADER_COMPILER: return "ShaderCompiler";
        case GL_DEBUG_SOURCE_THIRD_PARTY: return "ThirdParty";
        case GL_DEBUG_SOURCE_APPLICATION: return "Application";
        case GL_DEBUG_SOURCE_OTHER: return "Other";
        default: return "UnknownSource";
    }
}

static const char* GlDebugTypeString(GLenum type) {
    switch (type) {
        case GL_DEBUG_TYPE_ERROR: return "Error";
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: return "DeprecatedBehavior";
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: return "UndefinedBehavior";
        case GL_DEBUG_TYPE_PORTABILITY: return "Portability";
        case GL_DEBUG_TYPE_PERFORMANCE: return "Performance";
        case GL_DEBUG_TYPE_MARKER: return "Marker";
        case GL_DEBUG_TYPE_PUSH_GROUP: return "PushGroup";
        case GL_DEBUG_TYPE_POP_GROUP: return "PopGroup";
        case GL_DEBUG_TYPE_OTHER: return "Other";
        default: return "UnknownType";
    }
}

static const char* GlDebugSeverityString(GLenum severity) {
    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH: return "HIGH";
        case GL_DEBUG_SEVERITY_MEDIUM: return "MEDIUM";
        case GL_DEBUG_SEVERITY_LOW: return "LOW";
        case GL_DEBUG_SEVERITY_NOTIFICATION: return "NOTIFY";
        default: return "UNKNOWN";
    }
}

static void APIENTRY OpenGlDebugCallback(
    GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei,
    const GLchar* message,
    const void*)
{
    if (!message) return;
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

    const std::string line =
        std::string("[gl-debug][") + GlDebugSeverityString(severity) +
        "][" + GlDebugTypeString(type) +
        "][" + GlDebugSourceString(source) + "] id=" + std::to_string(id) + " " + message;
    std::cerr << line << "\n";
    DebugLog(line);
    g_startupSummaryLines.push_back(line);
}

static void SetupOpenGlDebugOutput() {
    GLint flags = 0;
    glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    const bool debugContext = ((flags & GL_CONTEXT_FLAG_DEBUG_BIT) != 0);
    StartupSummaryLog(std::string("[debug] GL context debug bit: ") + (debugContext ? "ON" : "OFF"));

    if (glDebugMessageCallback == nullptr || glDebugMessageControl == nullptr) {
        StartupSummaryLog("[debug] KHR_debug extension unavailable.");
        return;
    }
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(OpenGlDebugCallback, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
    StartupSummaryLog("[debug] OpenGL debug callback registered.");
}

static const char* GlErrorToString(GLenum err) {
    switch (err) {
        case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
        case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
        default: return "GL_UNKNOWN_ERROR";
    }
}

static void DrainGlErrors(const char* context, bool startup = false) {
    bool hadError = false;
    for (GLenum err = glGetError(); err != GL_NO_ERROR; err = glGetError()) {
        hadError = true;
        const std::string line = std::string("[glerr] ") + context + " -> " + GlErrorToString(err) + " (" + std::to_string(err) + ")";
        if (startup) {
            StartupSummaryLog(line);
        } else {
            DebugLog(line);
        }
    }
    if (startup && !hadError) {
        StartupSummaryLog(std::string("[glerr] ") + context + " -> GL_NO_ERROR");
    }
}

static void DumpStartupSummaryToDebugWindow() {
    if (!g_debugConsoleVisible) return;
    DebugLog("========== Startup Summary (F9) ==========");
    for (const auto& line : g_startupSummaryLines) {
        DebugLog(line);
    }
    DebugLog("==========================================");
}

static std::string Trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) --e;
    return s.substr(b, e - b);
}

static void ShowCommandHints(const std::string& inputRaw) {
    const std::string input = Trim(inputRaw);
    if (input.empty() || input[0] != '/') {
        g_lastCommandHint.clear();
        return;
    }

    std::string hint;
    if (input == "/" || input == "/h" || input == "/he" || input == "/hel" || input == "/help") {
        hint = "[cmd] /help, /light x y z, /wireframe ..., /statistics ...";
    } else if (input.rfind("/l", 0) == 0) {
        hint = "[cmd] /light x y z  (sets sun direction, normalized)";
    } else if (input.rfind("/w", 0) == 0) {
        hint = "[cmd] /wireframe [on|off|toggle|color r g b]";
    } else if (input.rfind("/s", 0) == 0) {
        hint = "[cmd] /statistics [on|off|toggle]";
    } else {
        hint = "[cmd] Unknown command prefix. Try: /help";
    }

    if (hint != g_lastCommandHint) {
        g_lastCommandHint = hint;
        DebugLog(hint);
    }
}

static void ExecuteConsoleCommand(const std::string& raw) {
    const std::string cmd = Trim(raw);
    g_lastCommandHint.clear();
    if (cmd.empty()) return;
    if (cmd[0] != '/') {
        DebugLog("[cmd] Commands must start with '/'. Try: /help");
        return;
    }

    std::istringstream iss(cmd);
    std::string op;
    iss >> op;

    if (op == "/help") {
        DebugLog("[cmd] Available commands:");
        DebugLog("[cmd]   /help");
        DebugLog("[cmd]   /light x y z");
        DebugLog("[cmd]   /wireframe [on|off|toggle|color r g b]");
        DebugLog("[cmd]   /statistics [on|off|toggle]");
        return;
    }

    if (op == "/light") {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        if (!(iss >> x >> y >> z)) {
            DebugLog("[cmd] Usage: /light x y z");
            DebugLog("[cmd] Current: /light " + std::to_string(g_lightDir.x) + " " +
                     std::to_string(g_lightDir.y) + " " + std::to_string(g_lightDir.z));
            return;
        }
        const glm::vec3 v(x, y, z);
        const float len = glm::length(v);
        if (len < 1e-5f) {
            DebugLog("[cmd] /light rejected: direction cannot be zero.");
            return;
        }
        g_lightDir = v / len;
        DebugLog("[cmd] Light direction set to (" +
                 std::to_string(g_lightDir.x) + ", " +
                 std::to_string(g_lightDir.y) + ", " +
                 std::to_string(g_lightDir.z) + ")");
        return;
    }

    if (op == "/wireframe") {
        std::string arg;
        if (!(iss >> arg) || arg == "toggle") {
            g_wireframeMode = !g_wireframeMode;
        } else if (arg == "color") {
            float r = 0.0f;
            float g = 0.0f;
            float b = 0.0f;
            if (!(iss >> r >> g >> b)) {
                DebugLog("[cmd] Usage: /wireframe color r g b");
                return;
            }
            if (r > 1.0f || g > 1.0f || b > 1.0f) {
                r /= 255.0f;
                g /= 255.0f;
                b /= 255.0f;
            }
            g_wireframeColor = glm::clamp(glm::vec3(r, g, b), glm::vec3(0.0f), glm::vec3(1.0f));
            DebugLog("[cmd] Wireframe color set to (" +
                     std::to_string(g_wireframeColor.r) + ", " +
                     std::to_string(g_wireframeColor.g) + ", " +
                     std::to_string(g_wireframeColor.b) + ")");
            return;
        } else if (arg == "on" || arg == "1" || arg == "true") {
            g_wireframeMode = true;
        } else if (arg == "off" || arg == "0" || arg == "false") {
            g_wireframeMode = false;
        } else {
            DebugLog("[cmd] Usage: /wireframe [on|off|toggle|color r g b]");
            return;
        }
        DebugLog(std::string("[cmd] Wireframe mode ") + (g_wireframeMode ? "enabled." : "disabled."));
        return;
    }

    if (op == "/statistics") {
        std::string arg;
        if (!(iss >> arg) || arg == "toggle") {
            g_statisticsEnabled = !g_statisticsEnabled;
        } else if (arg == "on" || arg == "1" || arg == "true") {
            g_statisticsEnabled = true;
        } else if (arg == "off" || arg == "0" || arg == "false") {
            g_statisticsEnabled = false;
        } else {
            DebugLog("[cmd] Usage: /statistics [on|off|toggle]");
            return;
        }
        DebugLog(std::string("[cmd] Statistics overlay ") + (g_statisticsEnabled ? "enabled." : "disabled."));
        return;
    }

    DebugLog("[cmd] Unknown command: " + op + " (try /help)");
}

static LRESULT CALLBACK DebugInputProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_KEYDOWN && wParam == VK_RETURN) {
        const int len = GetWindowTextLengthW(hwnd);
        std::wstring wtext(static_cast<size_t>(std::max(len, 0)) + 1, L'\0');
        if (len > 0) {
            GetWindowTextW(hwnd, wtext.data(), len + 1);
        }
        wtext.resize(static_cast<size_t>(std::max(len, 0)));
        SetWindowTextW(hwnd, L"");
        ExecuteConsoleCommand(WideToUtf8(wtext));
        return 0;
    }
    if (msg == WM_CHAR && wParam == '/') {
        ShowCommandHints("/");
    }
    if (g_debugInputOldProc) {
        return CallWindowProcW(g_debugInputOldProc, hwnd, msg, wParam, lParam);
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static LRESULT CALLBACK DebugWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    constexpr int kLogEditId = 1001;
    constexpr int kInputEditId = 1002;
    switch (msg) {
        case WM_CREATE: {
            g_debugEdit = CreateWindowExW(
                WS_EX_CLIENTEDGE, L"EDIT", L"",
                WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_LEFT | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
                0, 0, 100, 100, hwnd, reinterpret_cast<HMENU>(static_cast<intptr_t>(kLogEditId)), GetModuleHandleW(nullptr), nullptr);
            g_debugInput = CreateWindowExW(
                WS_EX_CLIENTEDGE, L"EDIT", L"",
                WS_CHILD | WS_VISIBLE | ES_LEFT | ES_AUTOHSCROLL | WS_TABSTOP,
                0, 0, 100, 28, hwnd, reinterpret_cast<HMENU>(static_cast<intptr_t>(kInputEditId)), GetModuleHandleW(nullptr), nullptr);
            if (g_debugEdit) {
                g_debugFont = CreateFontW(
                    18, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                    DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                    FIXED_PITCH | FF_MODERN, L"Consolas");
                SendMessageW(g_debugEdit, WM_SETFONT, reinterpret_cast<WPARAM>(g_debugFont), TRUE);
                if (g_debugInput) {
                    SendMessageW(g_debugInput, WM_SETFONT, reinterpret_cast<WPARAM>(g_debugFont), TRUE);
                    g_debugInputOldProc = reinterpret_cast<WNDPROC>(
                        SetWindowLongPtrW(g_debugInput, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(DebugInputProc)));
                }
            }
            return 0;
        }
        case WM_SIZE: {
            const int width = LOWORD(lParam);
            const int height = HIWORD(lParam);
            const int inputHeight = 30;
            if (g_debugEdit) {
                MoveWindow(g_debugEdit, 0, 0, width, std::max(1, height - inputHeight), TRUE);
            }
            if (g_debugInput) {
                MoveWindow(g_debugInput, 0, std::max(0, height - inputHeight), width, inputHeight, TRUE);
            }
            return 0;
        }
        case WM_COMMAND: {
            const int id = LOWORD(wParam);
            const int code = HIWORD(wParam);
            if (id == kInputEditId && code == EN_CHANGE && g_debugInput) {
                const int len = GetWindowTextLengthW(g_debugInput);
                std::wstring wtext(static_cast<size_t>(std::max(len, 0)) + 1, L'\0');
                if (len > 0) {
                    GetWindowTextW(g_debugInput, wtext.data(), len + 1);
                }
                wtext.resize(static_cast<size_t>(std::max(len, 0)));
                ShowCommandHints(WideToUtf8(wtext));
            }
            return 0;
        }
        case WM_CTLCOLOREDIT: {
            HDC hdc = reinterpret_cast<HDC>(wParam);
            SetTextColor(hdc, RGB(255, 255, 255));
            SetBkColor(hdc, RGB(0, 0, 0));
            return reinterpret_cast<LRESULT>(g_debugBrush);
        }
        case WM_CLOSE:
            DestroyWindow(hwnd);
            return 0;
        case WM_DESTROY:
            if (g_debugFont) {
                DeleteObject(g_debugFont);
                g_debugFont = nullptr;
            }
            g_debugEdit = nullptr;
            g_debugInput = nullptr;
            g_debugInputOldProc = nullptr;
            g_debugWindow = nullptr;
            g_debugConsoleVisible = false;
            return 0;
        default:
            return DefWindowProcW(hwnd, msg, wParam, lParam);
    }
}

static void PumpDebugWindowMessages() {
    MSG msg {};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

static void ToggleDebugConsole() {
    if (!g_debugConsoleVisible) {
        if (!g_debugBrush) {
            g_debugBrush = CreateSolidBrush(RGB(0, 0, 0));
        }
        static bool classRegistered = false;
        if (!classRegistered) {
            WNDCLASSW wc {};
            wc.lpfnWndProc = DebugWndProc;
            wc.hInstance = GetModuleHandleW(nullptr);
            wc.lpszClassName = L"AdvancedRenderingDebugWindow";
            wc.hbrBackground = g_debugBrush;
            wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
            RegisterClassW(&wc);
            classRegistered = true;
        }
        g_debugWindow = CreateWindowExW(
            0,
            L"AdvancedRenderingDebugWindow",
            L"Advanced Rendering Debug Console",
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT, CW_USEDEFAULT, 980, 680,
            nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
        if (g_debugWindow) {
            g_debugConsoleVisible = true;
            DebugLog("[debug] Debug window enabled (F11 toggle, F12 report).");
            DebugLog("[cmd] Type /help in the bottom input box.");
            DumpStartupSummaryToDebugWindow();
            ShowWindow(g_debugWindow, SW_SHOWNORMAL);
            SetForegroundWindow(g_debugWindow);
            if (g_debugInput) {
                SetFocus(g_debugInput);
            }
        }
    } else {
        if (g_debugWindow) {
            DestroyWindow(g_debugWindow);
        }
        g_debugConsoleVisible = false;
    }
}

static void PrintSceneDebugReport(const SceneModel& scene) {
    DebugLog("========== Scene Debug Report ==========");
    DebugLog("Primitives: " + std::to_string(scene.primitives.size()));
    DebugLog("Materials:  " + std::to_string(scene.materials.size()));
    DebugLog("Textures:   " + std::to_string(scene.textures.size()));
    DebugLog("Draw cmds:  " + std::to_string(scene.drawCommands.size()));
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        const auto& m = scene.materials[i];
        DebugLog("Material[" + std::to_string(i) + "] "
                 "baseTex=" + std::to_string(m.baseColorTexture) +
                 " mrTex=" + std::to_string(m.metallicRoughnessTexture) +
                 " normalTex=" + std::to_string(m.normalTexture) +
                 " occTex=" + std::to_string(m.occlusionTexture) +
                 " metallic=" + std::to_string(m.metallicFactor) +
                 " roughness=" + std::to_string(m.roughnessFactor) +
                 " doubleSided=" + std::to_string(m.doubleSided ? 1 : 0));
    }
    for (size_t i = 0; i < scene.primitives.size(); ++i) {
        const auto& p = scene.primitives[i];
        DebugLog("Prim[" + std::to_string(i) + "] "
                 "indices=" + std::to_string(p.indexCount) +
                 " mat=" + std::to_string(p.materialIndex) +
                 " tangents=" + std::to_string(p.hasTangents ? 1 : 0) +
                 " localMin=(" + std::to_string(p.localMin.x) + "," + std::to_string(p.localMin.y) + "," + std::to_string(p.localMin.z) + ")" +
                 " localMax=(" + std::to_string(p.localMax.x) + "," + std::to_string(p.localMax.y) + "," + std::to_string(p.localMax.z) + ")");
    }
    DebugLog("========================================");
}

static GLuint UploadTextureFromFile(const fs::path& path, bool srgb) {
    int w = 0;
    int h = 0;
    int comp = 0;
    stbi_uc* pixels = stbi_load(path.string().c_str(), &w, &h, &comp, 0);
    if (!pixels) {
        throw std::runtime_error("Failed to load fallback texture: " + path.string());
    }

    GLenum format = GL_RGBA;
    GLint internalFormat = srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8;
    if (comp == 1) {
        format = GL_RED;
        internalFormat = GL_R8;
    } else if (comp == 2) {
        format = GL_RG;
        internalFormat = GL_RG8;
    } else if (comp == 3) {
        format = GL_RGB;
        internalFormat = srgb ? GL_SRGB8 : GL_RGB8;
    } else if (comp == 4) {
        format = GL_RGBA;
        internalFormat = srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8;
    }

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, w, h, 0, format, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
#ifdef GL_TEXTURE_MAX_ANISOTROPY_EXT
    if (GLAD_GL_EXT_texture_filter_anisotropic) {
        float maxAniso = 1.0f;
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, std::min(8.0f, maxAniso));
    }
#endif
    glBindTexture(GL_TEXTURE_2D, 0);
    stbi_image_free(pixels);
    return tex;
}

static std::optional<fs::path> FindTextureByNameContains(const fs::path& dir, const std::string& tokenA, const std::string& tokenB = "") {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        return std::nullopt;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const bool hasA = (name.find(tokenA) != std::string::npos);
        const bool hasB = tokenB.empty() || (name.find(tokenB) != std::string::npos);
        const bool isPng = (entry.path().extension() == ".png" || entry.path().extension() == ".PNG");
        if (hasA && hasB && isPng) {
            return entry.path();
        }
    }
    return std::nullopt;
}

static void AttachFallbackTextures(SceneModel& scene, const fs::path& gltfPath) {
    const fs::path texDir = gltfPath.parent_path() / "texture";
    const auto base = FindTextureByNameContains(texDir, "basecolor");
    const auto normal = FindTextureByNameContains(texDir, "normal");
    auto orm = FindTextureByNameContains(texDir, "occlusionroughnessmetallic");
    if (!orm) {
        orm = FindTextureByNameContains(texDir, "roughness", "metallic");
    }

    if (!base && !normal && !orm) {
        StartupSummaryLog("[debug] No fallback textures found in: " + texDir.string());
        return;
    }

    int baseIdx = -1;
    int normalIdx = -1;
    int ormIdx = -1;

    if (base) {
        GpuTexture t;
        t.id = UploadTextureFromFile(*base, true);
        baseIdx = static_cast<int>(scene.textures.size());
        scene.textures.push_back(std::move(t));
        StartupSummaryLog("[debug] Fallback baseColor loaded: " + base->string());
    }
    if (normal) {
        GpuTexture t;
        t.id = UploadTextureFromFile(*normal, false);
        normalIdx = static_cast<int>(scene.textures.size());
        scene.textures.push_back(std::move(t));
        StartupSummaryLog("[debug] Fallback normal loaded: " + normal->string());
    }
    if (orm) {
        GpuTexture t;
        t.id = UploadTextureFromFile(*orm, false);
        ormIdx = static_cast<int>(scene.textures.size());
        scene.textures.push_back(std::move(t));
        StartupSummaryLog("[debug] Fallback ORM loaded: " + orm->string());
    }

    for (auto& m : scene.materials) {
        if (m.baseColorTexture < 0 && baseIdx >= 0) m.baseColorTexture = baseIdx;
        if (m.normalTexture < 0 && normalIdx >= 0) m.normalTexture = normalIdx;
        if (m.metallicRoughnessTexture < 0 && ormIdx >= 0) m.metallicRoughnessTexture = ormIdx;
        if (m.occlusionTexture < 0 && ormIdx >= 0) m.occlusionTexture = ormIdx;
    }
}

static SceneBounds ComputeSceneBounds(const SceneModel& scene) {
    SceneBounds out;
    glm::vec3 bmin(1e30f);
    glm::vec3 bmax(-1e30f);
    for (const auto& cmd : scene.drawCommands) {
        if (cmd.primitive < 0 || cmd.primitive >= static_cast<int>(scene.primitives.size())) continue;
        const auto& p = scene.primitives[cmd.primitive];
        const glm::vec3 mn = p.localMin;
        const glm::vec3 mx = p.localMax;
        const std::array<glm::vec3, 8> corners = {{
            {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z}, {mn.x, mx.y, mn.z}, {mx.x, mx.y, mn.z},
            {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z}, {mn.x, mx.y, mx.z}, {mx.x, mx.y, mx.z}
        }};
        for (const auto& c : corners) {
            const glm::vec3 wc = glm::vec3(cmd.transform * glm::vec4(c, 1.0f));
            bmin = glm::min(bmin, wc);
            bmax = glm::max(bmax, wc);
        }
    }
    if (bmin.x <= bmax.x && bmin.y <= bmax.y && bmin.z <= bmax.z) {
        out.min = bmin;
        out.max = bmax;
        out.valid = true;
    }
    return out;
}

static bool MatrixIsFinite(const glm::mat4& m) {
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            if (!std::isfinite(m[c][r])) {
                return false;
            }
        }
    }
    return true;
}

struct VisibilityProbe {
    int insideNdc = 0;
    int inFront = 0;
};

static VisibilityProbe ProbePrimitiveVisibility(const GpuPrimitive& p, const glm::mat4& model, const glm::mat4& viewProj) {
    const glm::vec3 mn = p.localMin;
    const glm::vec3 mx = p.localMax;
    const std::array<glm::vec3, 8> corners = {{
        {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z}, {mn.x, mx.y, mn.z}, {mx.x, mx.y, mn.z},
        {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z}, {mn.x, mx.y, mx.z}, {mx.x, mx.y, mx.z}
    }};

    VisibilityProbe out {};
    for (const auto& c : corners) {
        const glm::vec4 clip = viewProj * (model * glm::vec4(c, 1.0f));
        if (clip.w > 0.0f) {
            ++out.inFront;
        }
        if (std::abs(clip.w) > 1e-6f) {
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            if (ndc.x >= -1.0f && ndc.x <= 1.0f &&
                ndc.y >= -1.0f && ndc.y <= 1.0f &&
                ndc.z >= -1.0f && ndc.z <= 1.0f) {
                ++out.insideNdc;
            }
        }
    }
    return out;
}

static void SetCameraLookAt(const glm::vec3& eye, const glm::vec3& target) {
    g_camera.position = eye;
    const glm::vec3 d = glm::normalize(target - eye);
    g_camera.yaw = glm::degrees(std::atan2(d.z, d.x));
    g_camera.pitch = glm::degrees(std::asin(std::clamp(d.y, -1.0f, 1.0f)));
}

static void ComputeWorldBoundsForDraw(const GpuPrimitive& p, const glm::mat4& model, glm::vec3& outMin, glm::vec3& outMax) {
    const glm::vec3 mn = p.localMin;
    const glm::vec3 mx = p.localMax;
    const std::array<glm::vec3, 8> corners = {{
        {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z}, {mn.x, mx.y, mn.z}, {mx.x, mx.y, mn.z},
        {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z}, {mn.x, mx.y, mx.z}, {mx.x, mx.y, mx.z}
    }};
    outMin = glm::vec3(1e30f);
    outMax = glm::vec3(-1e30f);
    for (const auto& c : corners) {
        const glm::vec3 wc = glm::vec3(model * glm::vec4(c, 1.0f));
        outMin = glm::min(outMin, wc);
        outMax = glm::max(outMax, wc);
    }
}

static bool RayIntersectAabb(const glm::vec3& rayOrigin, const glm::vec3& rayDir, const glm::vec3& bmin, const glm::vec3& bmax, float& tHit) {
    float tmin = 0.0f;
    float tmax = 1e30f;
    for (int i = 0; i < 3; ++i) {
        if (std::abs(rayDir[i]) < 1e-8f) {
            if (rayOrigin[i] < bmin[i] || rayOrigin[i] > bmax[i]) return false;
            continue;
        }
        const float invD = 1.0f / rayDir[i];
        float t0 = (bmin[i] - rayOrigin[i]) * invD;
        float t1 = (bmax[i] - rayOrigin[i]) * invD;
        if (t0 > t1) std::swap(t0, t1);
        tmin = std::max(tmin, t0);
        tmax = std::min(tmax, t1);
        if (tmax < tmin) return false;
    }
    tHit = tmin;
    return true;
}

static int PickDrawCommand(
    const SceneModel& scene,
    const glm::vec3& rayOrigin,
    const glm::vec3& rayDir)
{
    int best = -1;
    float bestT = 1e30f;
    for (size_t i = 0; i < scene.drawCommands.size(); ++i) {
        const auto& cmd = scene.drawCommands[i];
        if (cmd.primitive < 0 || cmd.primitive >= static_cast<int>(scene.primitives.size())) continue;
        const auto& prim = scene.primitives[cmd.primitive];
        glm::vec3 wmin, wmax;
        ComputeWorldBoundsForDraw(prim, cmd.transform, wmin, wmax);
        float t = 0.0f;
        if (RayIntersectAabb(rayOrigin, rayDir, wmin, wmax, t) && t < bestT) {
            bestT = t;
            best = static_cast<int>(i);
        }
    }
    return best;
}

static LRESULT CALLBACK StatsOverlayProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_ERASEBKGND:
            return 1;
        case WM_PAINT: {
            PAINTSTRUCT ps {};
            HDC hdc = BeginPaint(hwnd, &ps);
            RECT rc {};
            GetClientRect(hwnd, &rc);
            HBRUSH black = CreateSolidBrush(RGB(0, 0, 0));
            FillRect(hdc, &rc, black);
            DeleteObject(black);
            SetBkMode(hdc, TRANSPARENT);
            SetTextColor(hdc, RGB(255, 255, 255));
            if (g_statsFont) {
                SelectObject(hdc, g_statsFont);
            }
            DrawTextW(hdc, g_statsOverlayText.c_str(), -1, &rc, DT_LEFT | DT_TOP | DT_WORDBREAK | DT_NOPREFIX);
            EndPaint(hwnd, &ps);
            return 0;
        }
        default:
            return DefWindowProcW(hwnd, msg, wParam, lParam);
    }
}

static void EnsureStatsOverlay(GLFWwindow* window) {
    if (g_statsOverlay) return;
    const HWND mainHwnd = glfwGetWin32Window(window);
    if (!mainHwnd) return;

    static bool clsRegistered = false;
    if (!clsRegistered) {
        WNDCLASSW wc {};
        wc.lpfnWndProc = StatsOverlayProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"AdvancedRenderingStatsOverlay";
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        RegisterClassW(&wc);
        clsRegistered = true;
    }

    g_statsOverlay = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW,
        L"AdvancedRenderingStatsOverlay",
        L"",
        WS_POPUP,
        0, 0, 640, 140,
        mainHwnd,
        nullptr,
        GetModuleHandleW(nullptr),
        nullptr);

    if (!g_statsOverlay) return;

    SetLayeredWindowAttributes(g_statsOverlay, RGB(0, 0, 0), 0, LWA_COLORKEY);
    if (!g_statsFont) {
        g_statsFont = CreateFontW(
            18, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
            FIXED_PITCH | FF_MODERN, L"Consolas");
    }
}

static void UpdateStatsOverlay(GLFWwindow* window, const SceneModel& scene) {
    if (g_debugConsoleVisible) {
        if (g_statsOverlay) {
            ShowWindow(g_statsOverlay, SW_HIDE);
        }
        return;
    }
    if (!g_statisticsEnabled) {
        if (g_statsOverlay) {
            ShowWindow(g_statsOverlay, SW_HIDE);
        }
        return;
    }
    EnsureStatsOverlay(window);
    if (!g_statsOverlay) return;
    ShowWindow(g_statsOverlay, SW_SHOWNOACTIVATE);

    std::string line = "No selection. Click object (LMB) to select.";
    if (g_selectedDrawCommand >= 0 && g_selectedDrawCommand < static_cast<int>(scene.drawCommands.size())) {
        const auto& cmd = scene.drawCommands[g_selectedDrawCommand];
        if (cmd.primitive >= 0 && cmd.primitive < static_cast<int>(scene.primitives.size())) {
            const auto& prim = scene.primitives[cmd.primitive];
            const int tris = static_cast<int>(prim.indexCount / 3);
            line = "Selected draw=" + std::to_string(g_selectedDrawCommand) +
                   " prim=" + std::to_string(cmd.primitive) +
                   " verts=" + std::to_string(prim.vertexCount) +
                   " tris=" + std::to_string(tris) +
                   " indices=" + std::to_string(prim.indexCount) +
                   " mat=" + std::to_string(std::max(cmd.materialIndex, 0));
        }
    }
    const HWND mainHwnd = glfwGetWin32Window(window);
    RECT wr {};
    GetWindowRect(mainHwnd, &wr);
    const int ow = 760;
    const int oh = 140;
    SetWindowPos(g_statsOverlay, HWND_TOP, wr.left + 12, wr.top + 42, ow, oh, SWP_NOACTIVATE | SWP_SHOWWINDOW);

    const std::wstring text = Utf8ToWide(line);
    if (text != g_statsOverlayText) {
        g_statsOverlayText = text;
        InvalidateRect(g_statsOverlay, nullptr, FALSE);
    }
}

static GLenum ToWrap(int wrapMode) {
    switch (wrapMode) {
        case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE: return GL_CLAMP_TO_EDGE;
        case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT: return GL_MIRRORED_REPEAT;
        default: return GL_REPEAT;
    }
}

static GLenum ToMagFilter(int filter) {
    switch (filter) {
        case TINYGLTF_TEXTURE_FILTER_NEAREST:
        case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
        case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
            return GL_NEAREST;
        default:
            return GL_LINEAR;
    }
}

static GLenum ToMinFilter(int filter) {
    switch (filter) {
        case TINYGLTF_TEXTURE_FILTER_NEAREST: return GL_NEAREST;
        case TINYGLTF_TEXTURE_FILTER_LINEAR: return GL_LINEAR;
        case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST: return GL_NEAREST_MIPMAP_NEAREST;
        case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST: return GL_LINEAR_MIPMAP_NEAREST;
        case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR: return GL_NEAREST_MIPMAP_LINEAR;
        case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
        default:
            return GL_LINEAR_MIPMAP_LINEAR;
    }
}

static int GetNumComponents(int type) {
    switch (type) {
        case TINYGLTF_TYPE_SCALAR: return 1;
        case TINYGLTF_TYPE_VEC2: return 2;
        case TINYGLTF_TYPE_VEC3: return 3;
        case TINYGLTF_TYPE_VEC4: return 4;
        case TINYGLTF_TYPE_MAT2: return 4;
        case TINYGLTF_TYPE_MAT3: return 9;
        case TINYGLTF_TYPE_MAT4: return 16;
        default: throw std::runtime_error("Unsupported accessor type.");
    }
}

static int GetComponentSize(int componentType) {
    switch (componentType) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: return 1;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: return 2;
        case TINYGLTF_COMPONENT_TYPE_INT:
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        case TINYGLTF_COMPONENT_TYPE_FLOAT: return 4;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE: return 8;
        default: throw std::runtime_error("Unsupported component type.");
    }
}

static const unsigned char* AccessorDataPtr(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
    const auto& view = model.bufferViews.at(accessor.bufferView);
    const auto& buffer = model.buffers.at(view.buffer);
    return buffer.data.data() + view.byteOffset + accessor.byteOffset;
}

static size_t AccessorStride(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
    const auto& view = model.bufferViews.at(accessor.bufferView);
    const size_t stride = accessor.ByteStride(view);
    if (stride != 0) {
        return stride;
    }
    return static_cast<size_t>(GetNumComponents(accessor.type) * GetComponentSize(accessor.componentType));
}

static glm::mat4 NodeTransform(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        glm::mat4 m(1.0f);
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                m[c][r] = static_cast<float>(node.matrix[c * 4 + r]);
            }
        }
        return m;
    }

    glm::vec3 translation(0.0f);
    if (node.translation.size() == 3) {
        translation = glm::vec3(
            static_cast<float>(node.translation[0]),
            static_cast<float>(node.translation[1]),
            static_cast<float>(node.translation[2])
        );
    }

    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
    if (node.rotation.size() == 4) {
        rotation = glm::quat(
            static_cast<float>(node.rotation[3]),
            static_cast<float>(node.rotation[0]),
            static_cast<float>(node.rotation[1]),
            static_cast<float>(node.rotation[2])
        );
    }

    glm::vec3 scale(1.0f);
    if (node.scale.size() == 3) {
        scale = glm::vec3(
            static_cast<float>(node.scale[0]),
            static_cast<float>(node.scale[1]),
            static_cast<float>(node.scale[2])
        );
    }

    return glm::translate(glm::mat4(1.0f), translation) * glm::toMat4(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

static void BuildNodeDrawList(
    const tinygltf::Model& model,
    int nodeIndex,
    const glm::mat4& parent,
    const std::vector<std::vector<int>>& meshToPrimitives,
    std::vector<DrawCommand>& out)
{
    const auto& node = model.nodes.at(nodeIndex);
    const glm::mat4 world = parent * NodeTransform(node);
    if (!MatrixIsFinite(world)) {
        StartupSummaryLog("[debug] WARNING: Non-finite node transform at node " + std::to_string(nodeIndex));
    }

    if (node.mesh >= 0 && node.mesh < static_cast<int>(meshToPrimitives.size())) {
        for (const int primId : meshToPrimitives[node.mesh]) {
            DrawCommand cmd;
            cmd.primitive = primId;
            cmd.transform = world;
            out.push_back(cmd);
        }
    }

    for (const int child : node.children) {
        BuildNodeDrawList(model, child, world, meshToPrimitives, out);
    }
}

static GLuint UploadTexture(const tinygltf::Model& model, const tinygltf::Texture& texture) {
    if (texture.source < 0 || texture.source >= static_cast<int>(model.images.size())) {
        throw std::runtime_error("Texture references invalid image.");
    }

    const tinygltf::Image& image = model.images.at(texture.source);
    if (image.image.empty()) {
        throw std::runtime_error("Texture image data is empty.");
    }

    GLint internalFormat = GL_RGBA8;
    GLenum format = GL_RGBA;

    if (image.component == 1) {
        internalFormat = GL_R8;
        format = GL_RED;
    } else if (image.component == 2) {
        internalFormat = GL_RG8;
        format = GL_RG;
    } else if (image.component == 3) {
        internalFormat = GL_RGB8;
        format = GL_RGB;
    } else if (image.component == 4) {
        internalFormat = GL_RGBA8;
        format = GL_RGBA;
    } else {
        throw std::runtime_error("Unsupported image component count.");
    }

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        internalFormat,
        image.width,
        image.height,
        0,
        format,
        GL_UNSIGNED_BYTE,
        image.image.data());

    int samplerIndex = texture.sampler;
    GLenum wrapS = GL_REPEAT;
    GLenum wrapT = GL_REPEAT;
    GLenum minFilter = GL_LINEAR_MIPMAP_LINEAR;
    GLenum magFilter = GL_LINEAR;

    if (samplerIndex >= 0 && samplerIndex < static_cast<int>(model.samplers.size())) {
        const auto& sampler = model.samplers.at(samplerIndex);
        wrapS = ToWrap(sampler.wrapS);
        wrapT = ToWrap(sampler.wrapT);
        minFilter = ToMinFilter(sampler.minFilter);
        magFilter = ToMagFilter(sampler.magFilter);
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, static_cast<GLint>(wrapS));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, static_cast<GLint>(wrapT));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(minFilter));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(magFilter));

    glGenerateMipmap(GL_TEXTURE_2D);

#ifdef GL_TEXTURE_MAX_ANISOTROPY_EXT
    if (GLAD_GL_EXT_texture_filter_anisotropic) {
        float maxAniso = 1.0f;
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, std::min(8.0f, maxAniso));
    }
#endif

    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

static std::vector<uint32_t> ReadIndices(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
    if (accessor.type != TINYGLTF_TYPE_SCALAR) {
        throw std::runtime_error("Index accessor must be scalar.");
    }

    const unsigned char* src = AccessorDataPtr(model, accessor);
    const size_t stride = AccessorStride(model, accessor);

    std::vector<uint32_t> indices(accessor.count);

    for (size_t i = 0; i < accessor.count; ++i) {
        const unsigned char* p = src + i * stride;
        switch (accessor.componentType) {
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                indices[i] = *reinterpret_cast<const uint8_t*>(p);
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                indices[i] = *reinterpret_cast<const uint16_t*>(p);
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                indices[i] = *reinterpret_cast<const uint32_t*>(p);
                break;
            default:
                throw std::runtime_error("Unsupported index component type.");
        }
    }
    return indices;
}

static std::vector<float> ReadFloatAttribute(const tinygltf::Model& model, const tinygltf::Accessor& accessor, int expectedComponents) {
    if (accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        throw std::runtime_error("Only float vertex attributes are supported.");
    }

    const int comps = GetNumComponents(accessor.type);
    if (comps != expectedComponents) {
        throw std::runtime_error("Unexpected attribute component count.");
    }

    const unsigned char* src = AccessorDataPtr(model, accessor);
    const size_t stride = AccessorStride(model, accessor);
    std::vector<float> data(accessor.count * expectedComponents);

    for (size_t i = 0; i < accessor.count; ++i) {
        const float* p = reinterpret_cast<const float*>(src + i * stride);
        for (int c = 0; c < expectedComponents; ++c) {
            data[i * expectedComponents + c] = p[c];
        }
    }

    return data;
}

static SceneModel LoadScene(const fs::path& gltfPath) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string warnings;
    std::string errors;

    const bool isBinary = (gltfPath.extension() == ".glb");
    bool ok = false;
    if (isBinary) {
        ok = loader.LoadBinaryFromFile(&model, &errors, &warnings, gltfPath.string());
    } else {
        ok = loader.LoadASCIIFromFile(&model, &errors, &warnings, gltfPath.string());
    }

    if (!warnings.empty()) {
        std::cerr << "[tinygltf warning] " << warnings << "\n";
    }
    if (!errors.empty()) {
        std::cerr << "[tinygltf error] " << errors << "\n";
    }
    if (!ok) {
        throw std::runtime_error("Failed to load glTF: " + gltfPath.string());
    }

    SceneModel scene;

    scene.textures.reserve(model.textures.size());
    for (const auto& tex : model.textures) {
        GpuTexture gpuTex;
        gpuTex.id = UploadTexture(model, tex);
        scene.textures.push_back(std::move(gpuTex));
    }

    scene.materials.reserve(std::max<size_t>(1, model.materials.size()));
    if (model.materials.empty()) {
        scene.materials.emplace_back();
    } else {
        for (const auto& mat : model.materials) {
            PbrMaterial m;
            if (mat.pbrMetallicRoughness.baseColorFactor.size() == 4) {
                m.baseColorFactor = glm::vec4(
                    static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[0]),
                    static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[1]),
                    static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[2]),
                    static_cast<float>(mat.pbrMetallicRoughness.baseColorFactor[3])
                );
            }
            m.metallicFactor = static_cast<float>(mat.pbrMetallicRoughness.metallicFactor);
            m.roughnessFactor = static_cast<float>(mat.pbrMetallicRoughness.roughnessFactor);
            m.normalScale = static_cast<float>(mat.normalTexture.scale);
            m.occlusionStrength = static_cast<float>(mat.occlusionTexture.strength);

            m.baseColorTexture = mat.pbrMetallicRoughness.baseColorTexture.index;
            m.metallicRoughnessTexture = mat.pbrMetallicRoughness.metallicRoughnessTexture.index;
            m.normalTexture = mat.normalTexture.index;
            m.occlusionTexture = mat.occlusionTexture.index;
            m.doubleSided = mat.doubleSided;
            if (m.baseColorFactor.r <= 0.001f && m.baseColorFactor.g <= 0.001f && m.baseColorFactor.b <= 0.001f) {
                m.baseColorFactor = glm::vec4(1.0f);
            }

            scene.materials.push_back(m);
        }
    }

    std::vector<std::vector<int>> meshToPrimitives(model.meshes.size());

    for (size_t meshIndex = 0; meshIndex < model.meshes.size(); ++meshIndex) {
        const auto& mesh = model.meshes.at(meshIndex);

        for (const auto& primitive : mesh.primitives) {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                continue;
            }

            auto posIt = primitive.attributes.find("POSITION");
            auto nrmIt = primitive.attributes.find("NORMAL");
            auto uvIt = primitive.attributes.find("TEXCOORD_0");
            if (posIt == primitive.attributes.end() || nrmIt == primitive.attributes.end() || uvIt == primitive.attributes.end()) {
                throw std::runtime_error("Primitive missing required POSITION/NORMAL/TEXCOORD_0.");
            }

            const tinygltf::Accessor& posAcc = model.accessors.at(posIt->second);
            const tinygltf::Accessor& nrmAcc = model.accessors.at(nrmIt->second);
            const tinygltf::Accessor& uvAcc = model.accessors.at(uvIt->second);

            const auto positions = ReadFloatAttribute(model, posAcc, 3);
            const auto normals = ReadFloatAttribute(model, nrmAcc, 3);
            const auto uvs = ReadFloatAttribute(model, uvAcc, 2);

            std::vector<float> tangents;
            bool hasTangents = false;
            auto tanIt = primitive.attributes.find("TANGENT");
            if (tanIt != primitive.attributes.end()) {
                tangents = ReadFloatAttribute(model, model.accessors.at(tanIt->second), 4);
                hasTangents = true;
            }

            if (primitive.indices < 0) {
                throw std::runtime_error("Primitive without indices is not supported.");
            }
            const auto indices = ReadIndices(model, model.accessors.at(primitive.indices));

            const size_t vertexCount = posAcc.count;
            std::vector<Vertex> vertices(vertexCount);
            glm::vec3 localMin(1e30f);
            glm::vec3 localMax(-1e30f);
            uint32_t maxIndex = 0;
            for (size_t i = 0; i < vertexCount; ++i) {
                vertices[i].position = glm::vec3(
                    positions[i * 3 + 0],
                    positions[i * 3 + 1],
                    positions[i * 3 + 2]
                );
                localMin = glm::min(localMin, vertices[i].position);
                localMax = glm::max(localMax, vertices[i].position);
                vertices[i].normal = glm::normalize(glm::vec3(
                    normals[i * 3 + 0],
                    normals[i * 3 + 1],
                    normals[i * 3 + 2]
                ));
                vertices[i].uv = glm::vec2(
                    uvs[i * 2 + 0],
                    uvs[i * 2 + 1]
                );
                if (hasTangents) {
                    vertices[i].tangent = glm::vec4(
                        tangents[i * 4 + 0],
                        tangents[i * 4 + 1],
                        tangents[i * 4 + 2],
                        tangents[i * 4 + 3]
                    );
                } else {
                    vertices[i].tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
                }
            }
            for (const uint32_t idx : indices) {
                maxIndex = std::max(maxIndex, idx);
            }
            if (!indices.empty() && maxIndex >= vertexCount) {
                throw std::runtime_error(
                    "Index buffer out of range for primitive: maxIndex=" + std::to_string(maxIndex) +
                    " vertexCount=" + std::to_string(vertexCount));
            }

            GpuPrimitive gpuPrim;
            gpuPrim.hasTangents = hasTangents;
            gpuPrim.materialIndex = primitive.material >= 0 ? primitive.material : 0;
            gpuPrim.indexCount = static_cast<GLsizei>(indices.size());
            gpuPrim.vertexCount = static_cast<GLsizei>(vertices.size());
            gpuPrim.localMin = localMin;
            gpuPrim.localMax = localMax;

            glGenVertexArrays(1, &gpuPrim.vao);
            glGenBuffers(1, &gpuPrim.vbo);
            glGenBuffers(1, &gpuPrim.ebo);

            glBindVertexArray(gpuPrim.vao);
            glBindBuffer(GL_ARRAY_BUFFER, gpuPrim.vbo);
            glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)), vertices.data(), GL_STATIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuPrim.ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indices.size() * sizeof(uint32_t)), indices.data(), GL_STATIC_DRAW);

            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, position)));

            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));

            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, uv)));

            glEnableVertexAttribArray(3);
            glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, tangent)));

            glBindVertexArray(0);

            const int primId = static_cast<int>(scene.primitives.size());
            meshToPrimitives[meshIndex].push_back(primId);
            scene.primitives.push_back(std::move(gpuPrim));
        }
    }

    const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (sceneIndex >= 0 && sceneIndex < static_cast<int>(model.scenes.size())) {
        const tinygltf::Scene& activeScene = model.scenes.at(sceneIndex);
        for (int rootNode : activeScene.nodes) {
            BuildNodeDrawList(model, rootNode, glm::mat4(1.0f), meshToPrimitives, scene.drawCommands);
        }
    }

    for (auto& cmd : scene.drawCommands) {
        if (cmd.primitive >= 0 && cmd.primitive < static_cast<int>(scene.primitives.size())) {
            cmd.materialIndex = scene.primitives[cmd.primitive].materialIndex;
        }
    }

    std::sort(scene.drawCommands.begin(), scene.drawCommands.end(), [](const DrawCommand& a, const DrawCommand& b) {
        return a.materialIndex < b.materialIndex;
    });

    AttachFallbackTextures(scene, gltfPath);

    StartupSummaryLog("Loaded glTF: \"" + gltfPath.string() + "\"");
    StartupSummaryLog("  Mesh primitives: " + std::to_string(scene.primitives.size()));
    StartupSummaryLog("  Materials:       " + std::to_string(scene.materials.size()));
    StartupSummaryLog("  Textures:        " + std::to_string(scene.textures.size()));
    StartupSummaryLog("  Draw commands:   " + std::to_string(scene.drawCommands.size()));

    return scene;
}

static void FramebufferSizeCallback(GLFWwindow*, int width, int height) {
    g_fbWidth = std::max(width, 1);
    g_fbHeight = std::max(height, 1);
    glViewport(0, 0, g_fbWidth, g_fbHeight);
}

static void MouseCallback(GLFWwindow*, double xpos, double ypos) {
    if (g_firstMouse) {
        g_lastMouseX = xpos;
        g_lastMouseY = ypos;
        g_firstMouse = false;
        return;
    }

    const double dx = xpos - g_lastMouseX;
    const double dy = g_lastMouseY - ypos;
    g_lastMouseX = xpos;
    g_lastMouseY = ypos;

    g_camera.yaw += static_cast<float>(dx * kMouseSensitivity);
    g_camera.pitch += static_cast<float>(dy * kMouseSensitivity);
    g_camera.pitch = std::clamp(g_camera.pitch, -89.0f, 89.0f);
}

static void ProcessKeyboard(GLFWwindow* window, float dt) {
    float speed = kBaseMoveSpeed;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        speed *= 2.0f;
    }

    const glm::vec3 forward = g_camera.Forward();
    const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    const glm::vec3 up(0.0f, 1.0f, 0.0f);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) g_camera.position += forward * speed * dt;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) g_camera.position -= forward * speed * dt;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) g_camera.position -= right * speed * dt;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) g_camera.position += right * speed * dt;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) g_camera.position -= up * speed * dt;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) g_camera.position += up * speed * dt;
}

int main(int argc, char** argv) {
    try {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW.");
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

        GLFWwindow* window = glfwCreateWindow(kWindowWidth, kWindowHeight, "Advanced Rendering Engine GL", nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("Failed to create window.");
        }

        glfwMakeContextCurrent(window);
        glfwSetFramebufferSizeCallback(window, FramebufferSizeCallback);
        glfwSetCursorPosCallback(window, MouseCallback);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSwapInterval(1);

        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
            glfwDestroyWindow(window);
            glfwTerminate();
            throw std::runtime_error("Failed to initialize GLAD.");
        }
        SetupOpenGlDebugOutput();

        glViewport(0, 0, kWindowWidth, kWindowHeight);
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glEnable(GL_MULTISAMPLE);

        const fs::path launchDir = fs::current_path();
        const fs::path exeDir = fs::absolute(fs::path(argv[0])).parent_path();

        const auto ResolveExisting = [](std::initializer_list<fs::path> candidates) -> fs::path {
            for (const auto& p : candidates) {
                if (!p.empty() && fs::exists(p)) {
                    return p;
                }
            }
            return {};
        };

        const fs::path vertPath = ResolveExisting({
            launchDir / "shaders" / "pbr.vert",
            exeDir / "shaders" / "pbr.vert",
            exeDir.parent_path() / "shaders" / "pbr.vert",
            exeDir.parent_path().parent_path() / "shaders" / "pbr.vert"
        });
        const fs::path fragPath = ResolveExisting({
            launchDir / "shaders" / "pbr.frag",
            exeDir / "shaders" / "pbr.frag",
            exeDir.parent_path() / "shaders" / "pbr.frag",
            exeDir.parent_path().parent_path() / "shaders" / "pbr.frag"
        });
        if (vertPath.empty() || fragPath.empty()) {
            throw std::runtime_error("Could not locate shaders/pbr.vert and shaders/pbr.frag.");
        }

        Shader pbrShader;
        pbrShader.Load(vertPath, fragPath);
        DrainGlErrors("after shader load", true);

        const fs::path modelPath = (argc >= 2)
            ? fs::path(argv[1])
            : ResolveExisting({
                launchDir / "models" / "umyvadlo.glb",
                launchDir / "assets" / "model.gltf",
                exeDir / "models" / "umyvadlo.glb",
                exeDir.parent_path() / "models" / "umyvadlo.glb",
                exeDir.parent_path().parent_path() / "models" / "umyvadlo.glb"
            });
        if (modelPath.empty()) {
            throw std::runtime_error("No model argument provided and default model not found.");
        }
        SceneModel scene = LoadScene(modelPath);
        DrainGlErrors("after scene load", true);
        const SceneBounds sceneBounds = ComputeSceneBounds(scene);
        if (sceneBounds.valid) {
            const glm::vec3 center = (sceneBounds.min + sceneBounds.max) * 0.5f;
            const glm::vec3 extent = sceneBounds.max - sceneBounds.min;
            const float radius = std::max(0.2f, glm::length(extent) * 0.5f);
            const glm::vec3 eye = center + glm::vec3(0.35f * radius, 0.20f * radius, 2.4f * radius);
            SetCameraLookAt(eye, center);
            StartupSummaryLog("[debug] Auto-framed camera to bounds center=(" +
                              std::to_string(center.x) + "," + std::to_string(center.y) + "," + std::to_string(center.z) +
                              ") radius=" + std::to_string(radius));
        } else {
            StartupSummaryLog("[debug] Scene bounds invalid, using default camera.");
        }
        PrintSceneDebugReport(scene);
        DrainGlErrors("after scene debug report", true);

        pbrShader.Use();
        glUniform1i(pbrShader.Uniform("uBaseColorTex"), 0);
        glUniform1i(pbrShader.Uniform("uMetallicRoughnessTex"), 1);
        glUniform1i(pbrShader.Uniform("uNormalTex"), 2);
        glUniform1i(pbrShader.Uniform("uOcclusionTex"), 3);
        glUniform1i(pbrShader.Uniform("uDebugFlatShade"), 0);
        glUniform1i(pbrShader.Uniform("uDebugShowAlbedo"), 0);
        glUniform1i(pbrShader.Uniform("uDisableNormalTex"), 0);
        glUniform1i(pbrShader.Uniform("uDisableOcclusionTex"), 0);
        glUniform1i(pbrShader.Uniform("uFlipNormalGreen"), 0);
        glUniform1i(pbrShader.Uniform("uWireframeMode"), 0);
        glUniform3fv(pbrShader.Uniform("uWireframeColor"), 1, &g_wireframeColor[0]);

        GLuint samplesQuery = 0;
        glGenQueries(1, &samplesQuery);
        bool queryInFlight = false;
        unsigned int lastSamplesPassed = std::numeric_limits<unsigned int>::max();
        bool emittedVisibilityProbe = false;

        float lastTime = static_cast<float>(glfwGetTime());
        float fpsTimer = 0.0f;
        int fpsFrames = 0;
        float fps = 0.0f;
        int lastMaterial = -99999;

        while (!glfwWindowShouldClose(window)) {
            const float now = static_cast<float>(glfwGetTime());
            const float dt = std::max(0.0001f, now - lastTime);
            lastTime = now;

            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            const bool f11Down = (glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS);
            if (f11Down && !g_f11WasDown) {
                ToggleDebugConsole();
            }
            g_f11WasDown = f11Down;
            const bool f12Down = (glfwGetKey(window, GLFW_KEY_F12) == GLFW_PRESS);
            if (f12Down && !g_f12WasDown) {
                if (!g_debugConsoleVisible) {
                    ToggleDebugConsole();
                }
                PrintSceneDebugReport(scene);
            }
            g_f12WasDown = f12Down;
            const bool f9Down = (glfwGetKey(window, GLFW_KEY_F9) == GLFW_PRESS);
            if (f9Down && !g_f9WasDown) {
                if (!g_debugConsoleVisible) {
                    ToggleDebugConsole();
                }
                DumpStartupSummaryToDebugWindow();
            }
            g_f9WasDown = f9Down;
            const bool f8Down = (glfwGetKey(window, GLFW_KEY_F8) == GLFW_PRESS);
            if (f8Down && !g_f8WasDown) {
                g_forceDebugFlatShade = !g_forceDebugFlatShade;
                if (g_debugConsoleVisible) {
                    DebugLog(std::string("[debug] Flat-shade override ") + (g_forceDebugFlatShade ? "enabled (F8)." : "disabled (F8)."));
                }
                StartupSummaryLog(std::string("[debug] Flat-shade override ") + (g_forceDebugFlatShade ? "enabled (F8)." : "disabled (F8)."));
            }
            g_f8WasDown = f8Down;
            const bool f7Down = (glfwGetKey(window, GLFW_KEY_F7) == GLFW_PRESS);
            if (f7Down && !g_f7WasDown) {
                g_forceShowAlbedo = !g_forceShowAlbedo;
                StartupSummaryLog(std::string("[debug] Albedo-only mode ") + (g_forceShowAlbedo ? "enabled (F7)." : "disabled (F7)."));
            }
            g_f7WasDown = f7Down;
            const bool f6Down = (glfwGetKey(window, GLFW_KEY_F6) == GLFW_PRESS);
            if (f6Down && !g_f6WasDown) {
                g_disableNormalMap = !g_disableNormalMap;
                StartupSummaryLog(std::string("[debug] Normal map ") + (g_disableNormalMap ? "disabled (F6)." : "enabled (F6)."));
            }
            g_f6WasDown = f6Down;
            const bool f5Down = (glfwGetKey(window, GLFW_KEY_F5) == GLFW_PRESS);
            if (f5Down && !g_f5WasDown) {
                g_disableOcclusionMap = !g_disableOcclusionMap;
                StartupSummaryLog(std::string("[debug] Occlusion map ") + (g_disableOcclusionMap ? "disabled (F5)." : "enabled (F5)."));
            }
            g_f5WasDown = f5Down;
            const bool f4Down = (glfwGetKey(window, GLFW_KEY_F4) == GLFW_PRESS);
            if (f4Down && !g_f4WasDown) {
                g_flipNormalGreen = !g_flipNormalGreen;
                StartupSummaryLog(std::string("[debug] Normal Y (green) ") + (g_flipNormalGreen ? "flipped (F4)." : "default (F4)."));
            }
            g_f4WasDown = f4Down;
            const bool pDown = (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS);
            if (pDown && !g_pWasDown) {
                g_debugSpamPaused = !g_debugSpamPaused;
                if (g_debugConsoleVisible) {
                    DebugLog(std::string("[debug] Spam ") + (g_debugSpamPaused ? "paused (P)." : "resumed (P)."));
                }
            }
            g_pWasDown = pDown;

            ProcessKeyboard(window, dt);
            fpsTimer += dt;
            ++fpsFrames;
            if (fpsTimer >= 1.0f) {
                fps = static_cast<float>(fpsFrames) / fpsTimer;
                fpsFrames = 0;
                fpsTimer = 0.0f;
            }

            glClearColor(0.03f, 0.05f, 0.08f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glPolygonMode(GL_FRONT_AND_BACK, g_wireframeMode ? GL_LINE : GL_FILL);

            const glm::mat4 proj = glm::perspective(glm::radians(65.0f), static_cast<float>(g_fbWidth) / static_cast<float>(g_fbHeight), 0.05f, 800.0f);
            const glm::mat4 view = g_camera.View();
            const glm::mat4 viewProj = proj * view;

            const bool lmbDown = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
            if (lmbDown && !g_lmbWasDown) {
                double mx = static_cast<double>(g_fbWidth) * 0.5;
                double my = static_cast<double>(g_fbHeight) * 0.5;
                const int cursorMode = glfwGetInputMode(window, GLFW_CURSOR);
                if (cursorMode != GLFW_CURSOR_DISABLED) {
                    glfwGetCursorPos(window, &mx, &my);
                }
                const float x = (2.0f * static_cast<float>(mx) / static_cast<float>(std::max(g_fbWidth, 1))) - 1.0f;
                const float y = 1.0f - (2.0f * static_cast<float>(my) / static_cast<float>(std::max(g_fbHeight, 1)));
                const glm::mat4 invVP = glm::inverse(viewProj);
                glm::vec4 pNear = invVP * glm::vec4(x, y, -1.0f, 1.0f);
                glm::vec4 pFar = invVP * glm::vec4(x, y, 1.0f, 1.0f);
                pNear /= std::max(std::abs(pNear.w), 1e-6f);
                pFar /= std::max(std::abs(pFar.w), 1e-6f);
                const glm::vec3 ro = g_camera.position;
                const glm::vec3 rd = glm::normalize(glm::vec3(pFar - pNear));
                g_selectedDrawCommand = PickDrawCommand(scene, ro, rd);
                if (g_selectedDrawCommand >= 0) {
                    DebugLog("[pick] selected draw command " + std::to_string(g_selectedDrawCommand));
                } else {
                    DebugLog("[pick] no hit");
                }
            }
            g_lmbWasDown = lmbDown;

            pbrShader.Use();
            glUniformMatrix4fv(pbrShader.Uniform("uViewProj"), 1, GL_FALSE, &viewProj[0][0]);
            glUniform3fv(pbrShader.Uniform("uCameraPos"), 1, &g_camera.position[0]);
            glUniform3fv(pbrShader.Uniform("uLightDir"), 1, &g_lightDir[0]);
            glUniform3fv(pbrShader.Uniform("uLightColor"), 1, &g_lightColor[0]);
            glUniform1i(pbrShader.Uniform("uDebugFlatShade"), g_forceDebugFlatShade ? 1 : 0);
            glUniform1i(pbrShader.Uniform("uDebugShowAlbedo"), g_forceShowAlbedo ? 1 : 0);
            glUniform1i(pbrShader.Uniform("uDisableNormalTex"), g_disableNormalMap ? 1 : 0);
            glUniform1i(pbrShader.Uniform("uDisableOcclusionTex"), g_disableOcclusionMap ? 1 : 0);
            glUniform1i(pbrShader.Uniform("uFlipNormalGreen"), g_flipNormalGreen ? 1 : 0);
            glUniform1i(pbrShader.Uniform("uWireframeMode"), g_wireframeMode ? 1 : 0);
            glUniform3fv(pbrShader.Uniform("uWireframeColor"), 1, &g_wireframeColor[0]);

            if (!emittedVisibilityProbe) {
                for (size_t i = 0; i < scene.drawCommands.size(); ++i) {
                    const auto& cmd = scene.drawCommands[i];
                    if (cmd.primitive < 0 || cmd.primitive >= static_cast<int>(scene.primitives.size())) continue;
                    const auto& prim = scene.primitives[cmd.primitive];
                    const VisibilityProbe vp = ProbePrimitiveVisibility(prim, cmd.transform, viewProj);
                    StartupSummaryLog("[debug] Visibility probe draw[" + std::to_string(i) + "] inFrontCorners=" +
                                      std::to_string(vp.inFront) + "/8 insideNdcCorners=" + std::to_string(vp.insideNdc) + "/8");
                }
                emittedVisibilityProbe = true;
            }

            if (queryInFlight) {
                glGetQueryObjectuiv(samplesQuery, GL_QUERY_RESULT, &lastSamplesPassed);
                queryInFlight = false;
            }
            glBeginQuery(GL_SAMPLES_PASSED, samplesQuery);
            queryInFlight = true;

            for (const auto& cmd : scene.drawCommands) {
                const auto& prim = scene.primitives[cmd.primitive];
                const int matIndex = std::clamp(cmd.materialIndex, 0, static_cast<int>(scene.materials.size()) - 1);

                if (matIndex != lastMaterial) {
                    const auto& mat = scene.materials[matIndex];

                    glUniform4fv(pbrShader.Uniform("uMat.baseColorFactor"), 1, &mat.baseColorFactor[0]);
                    glUniform1f(pbrShader.Uniform("uMat.metallicFactor"), mat.metallicFactor);
                    glUniform1f(pbrShader.Uniform("uMat.roughnessFactor"), mat.roughnessFactor);
                    glUniform1f(pbrShader.Uniform("uMat.normalScale"), mat.normalScale);
                    glUniform1f(pbrShader.Uniform("uMat.occlusionStrength"), mat.occlusionStrength);

                    const auto bindTexOrZero = [&](int unit, int idx) {
                        glActiveTexture(GL_TEXTURE0 + unit);
                        if (idx >= 0 && idx < static_cast<int>(scene.textures.size())) {
                            glBindTexture(GL_TEXTURE_2D, scene.textures[idx].id);
                            return 1;
                        }
                        glBindTexture(GL_TEXTURE_2D, 0);
                        return 0;
                    };

                    glUniform1i(pbrShader.Uniform("uHasBaseColorTex"), bindTexOrZero(0, mat.baseColorTexture));
                    glUniform1i(pbrShader.Uniform("uHasMetallicRoughnessTex"), bindTexOrZero(1, mat.metallicRoughnessTexture));
                    glUniform1i(pbrShader.Uniform("uHasNormalTex"), bindTexOrZero(2, mat.normalTexture));
                    glUniform1i(pbrShader.Uniform("uHasOcclusionTex"), bindTexOrZero(3, mat.occlusionTexture));

                    lastMaterial = matIndex;
                }

                glUniformMatrix4fv(pbrShader.Uniform("uModel"), 1, GL_FALSE, &cmd.transform[0][0]);
                glUniform1i(pbrShader.Uniform("uHasTangents"), prim.hasTangents ? 1 : 0);

                glBindVertexArray(prim.vao);
                glDrawElements(GL_TRIANGLES, prim.indexCount, GL_UNSIGNED_INT, nullptr);
            }
            glEndQuery(GL_SAMPLES_PASSED);

            if (g_debugConsoleVisible && !g_debugSpamPaused) {
                DebugLog(
                    "[dbg] fps=" + std::to_string(fps) +
                    " dt_ms=" + std::to_string(dt * 1000.0f) +
                    " cam=(" + std::to_string(g_camera.position.x) + ", " + std::to_string(g_camera.position.y) + ", " + std::to_string(g_camera.position.z) + ")" +
                    " yaw=" + std::to_string(g_camera.yaw) +
                    " pitch=" + std::to_string(g_camera.pitch) +
                    " draws=" + std::to_string(scene.drawCommands.size()) +
                    " prims=" + std::to_string(scene.primitives.size()) +
                    " mats=" + std::to_string(scene.materials.size()) +
                    " samplesPassed=" + (lastSamplesPassed == std::numeric_limits<unsigned int>::max() ? std::string("n/a") : std::to_string(lastSamplesPassed)) +
                    " viewport=" + std::to_string(g_fbWidth) + "x" + std::to_string(g_fbHeight));
            }
            if (fpsTimer == 0.0f && lastSamplesPassed != std::numeric_limits<unsigned int>::max()) {
                DrainGlErrors("frame", false);
            }

            glfwSwapBuffers(window);
            UpdateStatsOverlay(window, scene);
            glfwPollEvents();
            PumpDebugWindowMessages();
        }

        if (samplesQuery) {
            glDeleteQueries(1, &samplesQuery);
        }

        if (g_statsOverlay) {
            DestroyWindow(g_statsOverlay);
            g_statsOverlay = nullptr;
            g_statsOverlayText.clear();
        }
        if (g_statsFont) {
            DeleteObject(g_statsFont);
            g_statsFont = nullptr;
        }

        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
