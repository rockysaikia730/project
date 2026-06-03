#ifndef LBM_CUDA_HH
#define LBM_CUDA_HH

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/// Memory layout for the nine distribution slabs.
///   SoA: f[i * (nx*ny) + (y*nx + x)]   -- coalesces (adjacent x -> adjacent addr)
///   AoS: f[(y*nx + x) * 9 + i]         -- strided by 9 across threads
enum class Layout { SoA, AoS };

inline Layout layout_from_string(const std::string & s)
{
  return (s == "aos" || s == "AoS" || s == "AOS") ? Layout::AoS : Layout::SoA;
}

/**
 * Single-GPU D2Q9 BGK lattice Boltzmann solver (CUDA, double precision).
 *
 * One CUDA thread per lattice cell, 2D bounds-checked grid. The nine
 * distributions are stored either SoA or AoS, selected at construction; the
 * kernels are templated on the layout so the choice costs nothing at runtime.
 *
 * The device holds two distribution buffers (ping-pong for streaming), the
 * solid mask, and a tiny probe scratch. A host mirror of the field is kept
 * for snapshot output; call sync_to_host() before the host-side accessors.
 */
class LBM_CUDA
{
public:
  static constexpr int Q = 9;

  // Host copies of the D2Q9 constants (device copies live in __constant__
  // memory inside lbm.cu). Used by host-side init/accessors/output.
  static const int    cx[Q];
  static const int    cy[Q];
  static const int    opp[Q];
  static const double w[Q];

  LBM_CUDA(std::size_t nx, std::size_t ny,
           double u_in, double Re,
           double cyl_x, double cyl_y, double cyl_r,
           Layout layout, int block_x, int block_y);
  ~LBM_CUDA();

  void add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r);
  void initialize();
  void step();

  /// Enable the fused collide-stream kernel (one DRAM pass, 144 B/cell)
  /// instead of the separate collide+bounce+stream kernels (288 B/cell).
  /// Uses bounce-back-on-link at solid walls. SoA only.
  void set_fused(bool f) { fused_ = f; }
  bool fused() const { return fused_; }

  /// Copy the device field into the host mirror so the accessors below read
  /// up-to-date data. Needed before write_snapshot.
  void sync_to_host();

  // Host-side macroscopic accessors (read the host mirror).
  double rho      (std::size_t x, std::size_t y) const;
  double ux       (std::size_t x, std::size_t y) const;
  double uy       (std::size_t x, std::size_t y) const;
  double vorticity(std::size_t x, std::size_t y) const;
  bool   is_solid (std::size_t x, std::size_t y) const;

  /// Compute (ux, uy) at one cell on the device and copy the two doubles back.
  /// Cheap enough to call every step for the probe; skip in timing runs.
  void probe(std::size_t x, std::size_t y, double & ux_out, double & uy_out);

  std::size_t nx()   const { return nx_; }
  std::size_t ny()   const { return ny_; }
  double      tau()  const { return tau_; }
  double      u_in() const { return u_in_; }
  Layout      layout() const { return layout_; }

private:
  std::size_t idx(std::size_t x, std::size_t y) const { return y * nx_ + x; }
  std::size_t hfidx(int i, std::size_t x, std::size_t y) const;  // host, layout-aware
  void mark_obstacle(double c_x, double c_y, double r);
  void upload_solid_and_init();

  // Layout-dispatched launchers (defined in lbm.cu).
  template <Layout L> void step_impl();
  template <Layout L> void init_impl();
  template <Layout L> void probe_impl(std::size_t x, std::size_t y);

  std::size_t nx_, ny_;
  double      u_in_, tau_;
  Layout      layout_;
  int         bx_, by_;
  bool        fused_ = false;

  // Device buffers.
  double  * d_f_     = nullptr;   ///< 9 * nx * ny doubles
  double  * d_ftmp_  = nullptr;   ///< streaming scratch
  uint8_t * d_solid_ = nullptr;   ///< nx * ny bytes
  double  * d_probe_ = nullptr;   ///< 2 doubles (ux, uy)

  // Host mirrors.
  std::vector<double>  h_f_;
  std::vector<uint8_t> h_solid_;
};

#endif  // LBM_CUDA_HH
