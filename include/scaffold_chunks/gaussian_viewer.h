/**
 * Scaffold-ChunGS: Real-time 3D Gaussian Splatting Viewer.
 *
 * Requirements: GLFW 3.3+, OpenGL 3.3, ImGui 1.90+ (docking branch recommended).
 *
 * ImGui source files expected at one of:
 *   third_party/imgui/          (preferred)
 *   /usr/include/imgui/         (system)
 *   external/imgui/             (fallback)
 *
 * The viewer runs in its own thread and reads double-buffered snapshots
 * from the mapper thread — it never touches the GaussianModel directly.
 */

#pragma once

#include "config.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace scaffold_chungs {

// =============================================================================
// Viewer Camera
// =============================================================================

struct ViewerCamera {
  float distance = 5.0f;
  float azimuth = 0.0f;    // radians, horizontal orbit
  float elevation = 0.5f;  // radians, vertical orbit
  float target_x = 0.0f, target_y = 0.0f, target_z = 0.0f;

  float fov_y = 45.0f;
  float near_plane = 0.01f;
  float far_plane = 1000.0f;

  // Derived view matrix
  Eigen::Matrix4f viewMatrix() const;
};

// =============================================================================
// Frame Data (shared between mapper and viewer threads)
// =============================================================================

struct ViewerFrameData {
  // Camera poses (keyframe frustums + trajectory)
  std::vector<Eigen::Matrix4f> keyframe_poses;
  std::vector<int64_t> keyframe_ids;

  // Gaussian point cloud snapshot
  std::vector<Eigen::Vector3f> point_positions;   // XYZ in world frame
  std::vector<Eigen::Vector3f> point_colors;      // RGB [0-1]
  std::vector<float> point_opacities;

  // Stats
  int total_anchors = 0;
  int total_gaussians = 0;
  int visible_gaussians = 0;
  int64_t current_iteration = 0;
  float last_loss = 0.0f;
  float fps = 0.0f;

  // Loop closures
  struct LoopEvent {
    int64_t from_kf;
    int64_t to_kf;
    float score;
  };
  std::vector<LoopEvent> recent_loops;

  bool valid = false;
};

// =============================================================================
// Gaussian Viewer
// =============================================================================

class GaussianViewer {
 public:
  explicit GaussianViewer(const ViewerConfig& cfg);
  ~GaussianViewer();

  /** Start the viewer thread. */
  void start();

  /** Signal stop and join the viewer thread. */
  void stop();

  /** Check if the viewer window is open. */
  bool isOpen() const { return window_open_.load(); }

  /** Submit a new frame snapshot from the mapper thread (non-blocking).
   *  Swaps the write buffer — reader thread will pick it up. */
  void submitFrameData(const ViewerFrameData& data);

  /** Wait until the viewer window is closed by the user. */
  void waitForClose();

  ViewerConfig& config() { return cfg_; }

 private:
  void viewerLoop();

  // OpenGL rendering
  void renderFrame();
  void renderPointCloud();
  void renderCameraFrustums();
  void renderTrajectory();
  void renderGrid();
  void renderImGui();

  // Camera controls
  void processInput(float dt);
  void updateViewProjection();

  // Shader compilation
  unsigned compileShader(unsigned type, const char* source);
  unsigned createShaderProgram(const char* vertex_src, const char* fragment_src);

  ViewerConfig cfg_;
  ViewerCamera camera_;

  // Window
  void* window_ = nullptr;  // GLFWwindow*
  int fb_width_ = 1920, fb_height_ = 1080;

  // OpenGL state
  unsigned vao_points_ = 0;
  unsigned vbo_points_pos_ = 0;
  unsigned vbo_points_col_ = 0;
  unsigned vbo_points_opacity_ = 0;
  unsigned shader_points_ = 0;

  unsigned vao_lines_ = 0;
  unsigned vbo_lines_ = 0;
  unsigned shader_lines_ = 0;

  unsigned vao_grid_ = 0;
  unsigned vbo_grid_ = 0;
  unsigned shader_grid_ = 0;

  // ImGui state
  bool show_ui_ = true;
  bool show_cameras_ = true;
  bool show_trajectory_ = true;
  bool show_points_ = true;
  bool show_grid_ = true;
  bool auto_orbit_ = false;

  // Double-buffered frame data
  ViewerFrameData frame_data_[2];
  std::atomic<int> write_index_{0};
  std::mutex frame_mutex_;
  bool frame_updated_ = false;

  // Threading
  std::unique_ptr<std::thread> viewer_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> window_open_{false};
  std::condition_variable close_cv_;
  std::mutex close_mutex_;

  // Timing
  float last_frame_time_ = 0.0f;
  float fps_smooth_ = 0.0f;
};

}  // namespace scaffold_chungs
