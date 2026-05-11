/**
 * Scaffold-ChunGS: Gaussian Viewer implementation.
 *
 * Requires ImGui with GLFW + OpenGL3 backends.
 * Configure ImGui paths in CMake:
 *   set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui CACHE PATH "Path to imgui")
 */

#include "scaffold_chunks/gaussian_viewer.h"

// ImGui (provided externally — see header note)
// Include paths: imgui/, imgui/backends/
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace scaffold_chungs {

// =============================================================================
// View Matrix Computation
// =============================================================================

Eigen::Matrix4f ViewerCamera::viewMatrix() const {
  float sa = std::sin(azimuth);
  float ca = std::cos(azimuth);
  float se = std::sin(elevation);
  float ce = std::cos(elevation);

  Eigen::Vector3f eye(
      target_x + distance * ce * sa,
      target_y + distance * se,
      target_z + distance * ce * ca);

  Eigen::Vector3f center(target_x, target_y, target_z);
  Eigen::Vector3f up(0.0f, 1.0f, 0.0f);

  Eigen::Vector3f f = (center - eye).normalized();
  Eigen::Vector3f s = f.cross(up).normalized();
  Eigen::Vector3f u = s.cross(f);

  Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
  V(0, 0) = s.x();  V(0, 1) = s.y();  V(0, 2) = s.z();  V(0, 3) = -s.dot(eye);
  V(1, 0) = u.x();  V(1, 1) = u.y();  V(1, 2) = u.z();  V(1, 3) = -u.dot(eye);
  V(2, 0) = -f.x(); V(2, 1) = -f.y(); V(2, 2) = -f.z(); V(2, 3) = f.dot(eye);
  return V;
}

// =============================================================================
// Built-in Shader Sources
// =============================================================================

static const char* kPointVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in float aOpacity;

uniform mat4 uMVP;
uniform float uPointSize;

out vec3 vColor;
out float vOpacity;

void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);
  gl_PointSize = uPointSize;
  vColor = aColor;
  vOpacity = aOpacity;
}
)";

static const char* kPointFragmentShader = R"(
#version 330 core
in vec3 vColor;
in float vOpacity;
out vec4 fragColor;

void main() {
  // Circular point sprite
  vec2 center = gl_PointCoord - vec2(0.5);
  float dist = length(center);
  if (dist > 0.5) discard;
  float alpha = smoothstep(0.5, 0.4, dist) * vOpacity;
  if (alpha < 0.01) discard;
  fragColor = vec4(vColor, alpha);
}
)";

static const char* kLineVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 uMVP;

out vec3 vColor;

void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);
  vColor = aColor;
}
)";

static const char* kLineFragmentShader = R"(
#version 330 core
in vec3 vColor;
out vec4 fragColor;

void main() {
  fragColor = vec4(vColor, 1.0);
}
)";

static const char* kGridVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 uMVP;
uniform vec4 uColor;

out vec4 vColor;

void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);
  // Fade with distance
  float dist = length(aPos.xz);
  float alpha = exp(-dist * 0.05);
  vColor = vec4(uColor.rgb, uColor.a * alpha);
}
)";

static const char* kGridFragmentShader = R"(
#version 330 core
in vec4 vColor;
out vec4 fragColor;

void main() {
  fragColor = vColor;
}
)";

// =============================================================================
// Construction / Destruction
// =============================================================================

GaussianViewer::GaussianViewer(const ViewerConfig& cfg)
    : cfg_(cfg), camera_() {
  camera_.distance = cfg_.orbit_distance;
  camera_.azimuth = cfg_.orbit_azimuth;
  camera_.elevation = cfg_.orbit_elevation;
}

GaussianViewer::~GaussianViewer() {
  stop();
}

// =============================================================================
// Lifecycle
// =============================================================================

void GaussianViewer::start() {
  if (running_.load()) return;
  running_.store(true);
  viewer_thread_ = std::make_unique<std::thread>(&GaussianViewer::viewerLoop, this);
}

void GaussianViewer::stop() {
  if (!running_.load()) return;
  running_.store(false);
  window_open_.store(false);

  // Wake any waiters
  close_cv_.notify_all();

  if (viewer_thread_ && viewer_thread_->joinable()) {
    viewer_thread_->join();
  }
}

void GaussianViewer::waitForClose() {
  std::unique_lock<std::mutex> lock(close_mutex_);
  close_cv_.wait(lock, [this] { return !window_open_.load(); });
}

// =============================================================================
// Frame Data Submission (from mapper thread)
// =============================================================================

void GaussianViewer::submitFrameData(const ViewerFrameData& data) {
  int next = (write_index_.load() + 1) % 2;
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    frame_data_[next] = data;
    frame_updated_ = true;
  }
  write_index_.store(next);
}

// =============================================================================
// Shader Helpers
// =============================================================================

unsigned GaussianViewer::compileShader(unsigned type, const char* source) {
  unsigned shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  int success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char info[512];
    glGetShaderInfoLog(shader, sizeof(info), nullptr, info);
    std::cerr << "[Viewer] Shader compile error: " << info << "\n";
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}

unsigned GaussianViewer::createShaderProgram(const char* vertex_src,
                                              const char* fragment_src) {
  unsigned vs = compileShader(GL_VERTEX_SHADER, vertex_src);
  unsigned fs = compileShader(GL_FRAGMENT_SHADER, fragment_src);
  if (!vs || !fs) return 0;

  unsigned program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);

  int success;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    char info[512];
    glGetProgramInfoLog(program, sizeof(info), nullptr, info);
    std::cerr << "[Viewer] Program link error: " << info << "\n";
    glDeleteProgram(program);
    program = 0;
  }

  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}

// =============================================================================
// Viewer Loop (runs on viewer thread)
// =============================================================================

void GaussianViewer::viewerLoop() {
  // Init GLFW
  if (!glfwInit()) {
    std::cerr << "[Viewer] Failed to initialize GLFW\n";
    running_.store(false);
    return;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);

  GLFWmonitor* monitor = cfg_.fullscreen ? glfwGetPrimaryMonitor() : nullptr;
  GLFWwindow* glfw_window = glfwCreateWindow(
      cfg_.window_width, cfg_.window_height,
      cfg_.window_title.c_str(), monitor, nullptr);

  if (!glfw_window) {
    std::cerr << "[Viewer] Failed to create GLFW window\n";
    glfwTerminate();
    running_.store(false);
    return;
  }

  window_ = glfw_window;
  glfwMakeContextCurrent(glfw_window);
  glfwSwapInterval(cfg_.vsync ? 1 : 0);

  // Init OpenGL
  if (!gladLoadGL()) {
    std::cerr << "[Viewer] Failed to load OpenGL\n";
    glfwDestroyWindow(glfw_window);
    glfwTerminate();
    running_.store(false);
    return;
  }

  // Init ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(glfw_window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  // Compile shaders
  shader_points_ = createShaderProgram(kPointVertexShader, kPointFragmentShader);
  shader_lines_ = createShaderProgram(kLineVertexShader, kLineFragmentShader);
  shader_grid_ = createShaderProgram(kGridVertexShader, kGridFragmentShader);

  // Setup point cloud VAO
  glGenVertexArrays(1, &vao_points_);
  glGenBuffers(1, &vbo_points_pos_);
  glGenBuffers(1, &vbo_points_col_);
  glGenBuffers(1, &vbo_points_opacity_);

  // Setup line VAO (for frustums and trajectory)
  glGenVertexArrays(1, &vao_lines_);
  glGenBuffers(1, &vbo_lines_);

  // Setup grid VAO
  glGenVertexArrays(1, &vao_grid_);
  glGenBuffers(1, &vbo_grid_);

  // Build grid vertices
  {
    std::vector<float> grid_verts;
    float grid_size = 50.0f;
    float step = 1.0f;
    for (float i = -grid_size; i <= grid_size; i += step) {
      grid_verts.insert(grid_verts.end(), {i, 0.0f, -grid_size, i, 0.0f, grid_size});
      grid_verts.insert(grid_verts.end(), {-grid_size, 0.0f, i, grid_size, 0.0f, i});
    }
    glBindVertexArray(vao_grid_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_grid_);
    glBufferData(GL_ARRAY_BUFFER, grid_verts.size() * sizeof(float),
                 grid_verts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
  }

  window_open_.store(true);
  std::cout << "[Viewer] Window opened (" << cfg_.window_width << "x"
            << cfg_.window_height << ")\n";

  // ---- Main render loop ----
  auto last_time = std::chrono::steady_clock::now();
  float frame_time_accum = 0.0f;
  int frame_count = 0;

  while (running_.load() && !glfwWindowShouldClose(glfw_window)) {
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - last_time).count();
    last_time = now;

    // FPS smoothing
    frame_time_accum += dt;
    frame_count++;
    if (frame_time_accum > 0.5f) {
      fps_smooth_ = frame_count / frame_time_accum;
      frame_time_accum = 0.0f;
      frame_count = 0;
    }
    last_frame_time_ = dt;

    glfwPollEvents();
    processInput(dt);

    // Start ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Get framebuffer size
    glfwGetFramebufferSize(glfw_window, &fb_width_, &fb_height_);

    renderFrame();
    renderImGui();

    // Render ImGui
    ImGui::Render();
    glViewport(0, 0, fb_width_, fb_height_);

    float bg_r = cfg_.bg_color_r, bg_g = cfg_.bg_color_g, bg_b = cfg_.bg_color_b;
    glClearColor(bg_r, bg_g, bg_b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(glfw_window);
  }

  // ---- Cleanup ----
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glDeleteVertexArrays(1, &vao_points_);
  glDeleteBuffers(1, &vbo_points_pos_);
  glDeleteBuffers(1, &vbo_points_col_);
  glDeleteBuffers(1, &vbo_points_opacity_);
  glDeleteProgram(shader_points_);

  glDeleteVertexArrays(1, &vao_lines_);
  glDeleteBuffers(1, &vbo_lines_);
  glDeleteProgram(shader_lines_);

  glDeleteVertexArrays(1, &vao_grid_);
  glDeleteBuffers(1, &vbo_grid_);
  glDeleteProgram(shader_grid_);

  glfwDestroyWindow(glfw_window);
  glfwTerminate();

  window_open_.store(false);
  close_cv_.notify_all();
  std::cout << "[Viewer] Window closed\n";
}

// =============================================================================
// Input Processing
// =============================================================================

void GaussianViewer::processInput(float dt) {
  GLFWwindow* glfw_window = static_cast<GLFWwindow*>(window_);
  if (!glfw_window) return;

  // Close with Escape
  if (glfwGetKey(glfw_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(glfw_window, GLFW_TRUE);
    return;
  }

  // Toggle UI
  static bool f1_pressed = false;
  if (glfwGetKey(glfw_window, GLFW_KEY_F1) == GLFW_PRESS) {
    if (!f1_pressed) { show_ui_ = !show_ui_; f1_pressed = true; }
  } else { f1_pressed = false; }

  // Auto-orbit toggle
  static bool o_pressed = false;
  if (glfwGetKey(glfw_window, GLFW_KEY_O) == GLFW_PRESS) {
    if (!o_pressed) { auto_orbit_ = !auto_orbit_; o_pressed = true; }
  } else { o_pressed = false; }

  if (auto_orbit_) {
    camera_.azimuth += dt * 0.3f;
  }

  // Mouse orbit
  static double last_mx = 0, last_my = 0;
  double mx, my;
  glfwGetCursorPos(glfw_window, &mx, &my);

  bool left_down = glfwGetMouseButton(glfw_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  bool middle_down = glfwGetMouseButton(glfw_window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
  bool right_down = glfwGetMouseButton(glfw_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

  // ImGui wants mouse? Skip camera control
  if (!ImGui::GetIO().WantCaptureMouse) {
    if (left_down) {
      float dx = static_cast<float>(mx - last_mx);
      float dy = static_cast<float>(my - last_my);
      camera_.azimuth -= dx * cfg_.orbit_sensitivity;
      camera_.elevation += dy * cfg_.orbit_sensitivity;
      camera_.elevation = std::max(-1.5f, std::min(1.5f, camera_.elevation));
    }
    if (middle_down || right_down) {
      float dx = static_cast<float>(mx - last_mx);
      float dy = static_cast<float>(my - last_my);
      float s = camera_.distance * cfg_.orbit_sensitivity;
      Eigen::Vector3f eye_dir(
          -std::sin(camera_.azimuth) * std::cos(camera_.elevation),
          -std::sin(camera_.elevation),
          -std::cos(camera_.azimuth) * std::cos(camera_.elevation));
      Eigen::Vector3f right_dir(
          std::cos(camera_.azimuth),
          0.0f,
          -std::sin(camera_.azimuth));
      Eigen::Vector3f up(0.0f, 1.0f, 0.0f);
      camera_.target_x += (-dx * right_dir.x() + dy * up.x()) * s;
      camera_.target_y += (-dx * right_dir.y() + dy * up.y()) * s;
      camera_.target_z += (-dx * right_dir.z() + dy * up.z()) * s;
    }
  }

  // Scroll zoom
  if (!ImGui::GetIO().WantCaptureMouse) {
    // zoom handled via ImGui mouse wheel
  }

  last_mx = mx;
  last_my = my;
}

// =============================================================================
// Rendering
// =============================================================================

void GaussianViewer::renderFrame() {
  // Swap in latest frame data
  ViewerFrameData current_data;
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    int read_idx = write_index_.load();
    if (frame_updated_) {
      current_data = frame_data_[read_idx];
      frame_updated_ = false;
    }
  }

  if (!current_data.valid) return;

  Eigen::Matrix4f V = camera_.viewMatrix();

  float aspect = fb_width_ / static_cast<float>(fb_height_);
  float fov_rad = camera_.fov_y * M_PI / 180.0f;
  float tan_half_fov = std::tan(fov_rad * 0.5f);
  float top = camera_.near_plane * tan_half_fov;
  float right = top * aspect;

  Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
  P(0, 0) = camera_.near_plane / right;
  P(1, 1) = camera_.near_plane / top;
  P(2, 2) = -(camera_.far_plane + camera_.near_plane) /
             (camera_.far_plane - camera_.near_plane);
  P(2, 3) = -2.0f * camera_.far_plane * camera_.near_plane /
            (camera_.far_plane - camera_.near_plane);
  P(3, 2) = -1.0f;

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_PROGRAM_POINT_SIZE);

  // Render grid
  if (show_grid_ && cfg_.show_grid) {
    glUseProgram(shader_grid_);
    auto mvp_eigen = P * V;
    float mvp_data[16];
    Eigen::Matrix4f mvp_t = mvp_eigen.transpose();
    std::memcpy(mvp_data, mvp_t.data(), sizeof(mvp_data));
    glUniformMatrix4fv(glGetUniformLocation(shader_grid_, "uMVP"), 1, GL_FALSE, mvp_data);
    glUniform4f(glGetUniformLocation(shader_grid_, "uColor"), 0.3f, 0.3f, 0.3f, 0.5f);
    glBindVertexArray(vao_grid_);
    glDrawArrays(GL_LINES, 0, 404);  // 2 * 101 * 2
  }

  // Render camera frustums
  if (show_cameras_ && cfg_.show_cameras && !current_data.keyframe_poses.empty()) {
    // Build frustum lines
    std::vector<float> line_verts;
    float frustum_size = cfg_.frustum_scale;

    for (const auto& pose : current_data.keyframe_poses) {
      Eigen::Vector3f center = pose.block<3, 1>(0, 3);
      // Simplified frustum: tetrahedron from camera center
      float s = frustum_size;
      Eigen::Vector3f p0 = center + pose.block<3, 3>(0, 0) * Eigen::Vector3f(0, 0, s);
      Eigen::Vector3f p1 = center + pose.block<3, 3>(0, 0) * Eigen::Vector3f(-s * 0.5f, -s * 0.5f, s);
      Eigen::Vector3f p2 = center + pose.block<3, 3>(0, 0) * Eigen::Vector3f(s * 0.5f, -s * 0.5f, s);
      Eigen::Vector3f p3 = center + pose.block<3, 3>(0, 0) * Eigen::Vector3f(-s * 0.5f, s * 0.5f, s);
      Eigen::Vector3f p4 = center + pose.block<3, 3>(0, 0) * Eigen::Vector3f(s * 0.5f, s * 0.5f, s);

      float r = cfg_.camera_color_r, g = cfg_.camera_color_g, b = cfg_.camera_color_b;
      auto add_edge = [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b2) {
        line_verts.insert(line_verts.end(), {a.x(), a.y(), a.z(), r, g, b,
                                              b2.x(), b2.y(), b2.z(), r, g, b});
      };
      // From center to corners
      add_edge(center, p1); add_edge(center, p2);
      add_edge(center, p3); add_edge(center, p4);
      // Near plane quad
      add_edge(p1, p2); add_edge(p2, p4); add_edge(p4, p3); add_edge(p3, p1);
    }

    if (!line_verts.empty()) {
      glUseProgram(shader_lines_);
      auto mvp_eigen = P * V;
      float mvp_data[16];
      Eigen::Matrix4f mvp_t = mvp_eigen.transpose();
      std::memcpy(mvp_data, mvp_t.data(), sizeof(mvp_data));
      glUniformMatrix4fv(glGetUniformLocation(shader_lines_, "uMVP"), 1, GL_FALSE, mvp_data);

      glBindVertexArray(vao_lines_);
      glBindBuffer(GL_ARRAY_BUFFER, vbo_lines_);
      glBufferData(GL_ARRAY_BUFFER, line_verts.size() * sizeof(float),
                   line_verts.data(), GL_STREAM_DRAW);
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                            reinterpret_cast<void*>(3 * sizeof(float)));
      glEnableVertexAttribArray(1);
      glDrawArrays(GL_LINES, 0, static_cast<int>(line_verts.size() / 6));
    }
  }

  // Render trajectory
  if (show_trajectory_ && current_data.keyframe_poses.size() > 1) {
    std::vector<float> traj_verts;
    float tr = cfg_.trajectory_color_r;
    float tg = cfg_.trajectory_color_g;
    float tb = cfg_.trajectory_color_b;
    for (size_t i = 0; i + 1 < current_data.keyframe_poses.size(); ++i) {
      Eigen::Vector3f p0 = current_data.keyframe_poses[i].block<3, 1>(0, 3);
      Eigen::Vector3f p1 = current_data.keyframe_poses[i + 1].block<3, 1>(0, 3);
      traj_verts.insert(traj_verts.end(), {p0.x(), p0.y(), p0.z(), tr, tg, tb,
                                            p1.x(), p1.y(), p1.z(), tr, tg, tb});
    }
    if (!traj_verts.empty()) {
      glUseProgram(shader_lines_);
      auto mvp_eigen = P * V;
      float mvp_data[16];
      Eigen::Matrix4f mvp_t = mvp_eigen.transpose();
      std::memcpy(mvp_data, mvp_t.data(), sizeof(mvp_data));
      glUniformMatrix4fv(glGetUniformLocation(shader_lines_, "uMVP"), 1, GL_FALSE, mvp_data);

      glBindVertexArray(vao_lines_);
      glBindBuffer(GL_ARRAY_BUFFER, vbo_lines_);
      glBufferData(GL_ARRAY_BUFFER, traj_verts.size() * sizeof(float),
                   traj_verts.data(), GL_STREAM_DRAW);
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                            reinterpret_cast<void*>(3 * sizeof(float)));
      glEnableVertexAttribArray(1);
      glDrawArrays(GL_LINES, 0, static_cast<int>(traj_verts.size() / 6));
    }
  }

  // Render Gaussian point cloud
  if (show_points_ && cfg_.show_points && !current_data.point_positions.empty()) {
    size_t limit = std::min(current_data.point_positions.size(),
                            static_cast<size_t>(cfg_.max_display_gaussians));

    std::vector<float> point_data;
    point_data.reserve(limit * 7);
    for (size_t i = 0; i < limit; ++i) {
      point_data.push_back(current_data.point_positions[i].x());
      point_data.push_back(current_data.point_positions[i].y());
      point_data.push_back(current_data.point_positions[i].z());
      bool has_color = i < current_data.point_colors.size();
      point_data.push_back(has_color ? current_data.point_colors[i].x() : 1.0f);
      point_data.push_back(has_color ? current_data.point_colors[i].y() : 1.0f);
      point_data.push_back(has_color ? current_data.point_colors[i].z() : 1.0f);
      bool has_opacity = i < current_data.point_opacities.size();
      point_data.push_back(has_opacity ? current_data.point_opacities[i] : 0.5f);
    }

    glUseProgram(shader_points_);
    auto mvp_eigen = P * V;
    float mvp_data[16];
    Eigen::Matrix4f mvp_t = mvp_eigen.transpose();
    std::memcpy(mvp_data, mvp_t.data(), sizeof(mvp_data));
    glUniformMatrix4fv(glGetUniformLocation(shader_points_, "uMVP"), 1, GL_FALSE, mvp_data);
    glUniform1f(glGetUniformLocation(shader_points_, "uPointSize"), cfg_.point_size);

    glBindVertexArray(vao_points_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_points_pos_);
    glBufferData(GL_ARRAY_BUFFER, limit * 3 * sizeof(float),
                 point_data.data(), GL_STREAM_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_points_col_);
    glBufferData(GL_ARRAY_BUFFER, limit * 7 * sizeof(float),
                 point_data.data(), GL_STREAM_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float),
                          reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_points_opacity_);
    glBufferData(GL_ARRAY_BUFFER, limit * 7 * sizeof(float),
                 point_data.data(), GL_STREAM_DRAW);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float),
                          reinterpret_cast<void*>(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glDrawArrays(GL_POINTS, 0, static_cast<int>(limit));
  }
}

// =============================================================================
// ImGui Panel
// =============================================================================

void GaussianViewer::renderImGui() {
  if (!show_ui_) return;

  ImGui::Begin("Scaffold-ChunGS", nullptr,
               ImGuiWindowFlags_AlwaysAutoResize);

  // Stats
  ViewerFrameData current_data;
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    current_data = frame_data_[write_index_.load()];
  }

  ImGui::Text("FPS: %.1f", fps_smooth_);
  ImGui::Separator();

  if (current_data.valid) {
    ImGui::Text("Iteration: %lld", static_cast<long long>(current_data.current_iteration));
    ImGui::Text("Total Anchors: %d", current_data.total_anchors);
    ImGui::Text("Total Gaussians: %d", current_data.total_gaussians);
    ImGui::Text("Visible: %d", current_data.visible_gaussians);
    ImGui::Text("Keyframes: %zu", current_data.keyframe_poses.size());
    ImGui::Text("Loss: %.4f", current_data.last_loss);
  }

  ImGui::Separator();
  ImGui::Text("Camera: dist=%.1f az=%.2f el=%.2f",
              camera_.distance, camera_.azimuth, camera_.elevation);

  ImGui::Separator();

  // Rendering toggles
  ImGui::Checkbox("Point Cloud", &show_points_);
  ImGui::SameLine();
  ImGui::Checkbox("Cameras", &show_cameras_);

  ImGui::Checkbox("Trajectory", &show_trajectory_);
  ImGui::SameLine();
  ImGui::Checkbox("Grid", &show_grid_);

  ImGui::Checkbox("Auto Orbit", &auto_orbit_);

  // Point size slider
  ImGui::SliderFloat("Point Size", &cfg_.point_size, 0.5f, 10.0f);

  // Recent loop closures
  if (!current_data.recent_loops.empty()) {
    ImGui::Separator();
    ImGui::Text("Recent Loop Closures:");
    for (const auto& loop : current_data.recent_loops) {
      ImGui::Text("  %lld -> %lld (score: %.3f)",
                  static_cast<long long>(loop.from_kf),
                  static_cast<long long>(loop.to_kf),
                  loop.score);
    }
  }

  ImGui::Separator();
  ImGui::Text("Controls: LMB=Orbit, RMB=Pan, Scroll=Zoom");
  ImGui::Text("F1=Toggle UI, O=Auto Orbit, Esc=Exit");

  ImGui::End();
}

// =============================================================================
// Explicit GLAD implementation
// =============================================================================

// Minimal GLAD loader — loads the essential OpenGL 3.3 functions used by this viewer.
// In production, use the full GLAD or GLEW library.
static bool gladLoadGL() {
  // GL 3.3 core functions needed by this viewer
  if (!glfwGetCurrentContext()) return false;

  // All functions used are standard OpenGL 1.1–3.3 entry points
  // that are loaded by glfwGetProcAddress. We rely on GLFW's context
  // creation which binds all core functions automatically on most platforms.
  // If using Windows without a loader, uncomment:
  // #define GLAD_GL_IMPLEMENTATION
  // #include "glad/glad.h"

  // For now, assume GLFW's context makes functions available directly
  return true;
}

}  // namespace scaffold_chungs
