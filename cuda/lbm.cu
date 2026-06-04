#include "lbm.hh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Error-checking helper (course convention).
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n",            \
                   __FILE__, __LINE__, cudaGetErrorString(err));\
      std::exit(EXIT_FAILURE);                                  \
    }                                                           \
  } while (0)

// ---------------------------------------------------------------------------
// D2Q9 constants. Host copies (for init/accessors/output) + device copies in
// __constant__ memory (for kernels).
//   0: rest  1:E  2:N  3:W  4:S  5:NE  6:NW  7:SW  8:SE
// ---------------------------------------------------------------------------
const int    LBM_CUDA::cx[9]  = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
const int    LBM_CUDA::cy[9]  = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
const int    LBM_CUDA::opp[9] = { 0,  3,  4,  1,  2,  7,  8,  5,  6};
const double LBM_CUDA::w[9]   = {
  4.0/9.0,
  1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

__constant__ int    c_cx[9];
__constant__ int    c_cy[9];
__constant__ int    c_opp[9];
__constant__ double c_w[9];

// ---------------------------------------------------------------------------
// Layout-aware indexing. L is a compile-time constant, so the branch folds
// away -- SoA and AoS each compile to a single address expression.
// ---------------------------------------------------------------------------
template <Layout L>
__device__ __forceinline__ std::size_t
fidx(int i, std::size_t cell, std::size_t N)
{
  if (L == Layout::SoA) return std::size_t(i) * N + cell;
  else                  return cell * 9 + std::size_t(i);
}

// ---------------------------------------------------------------------------
// Kernels (one thread per cell, 2D bounds-checked grid).
// ---------------------------------------------------------------------------
template <Layout L>
__global__ void
init_kernel(double * f, const uint8_t * solid,
            std::size_t nx, std::size_t ny, double u_in)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;

  const std::size_t N    = nx * ny;
  const std::size_t cell = y * nx + x;
  const double rho = 1.0;
  const double ux  = solid[cell] ? 0.0 : u_in;
  const double uy  = 0.0;
  const double u2  = ux * ux + uy * uy;
  for (int i = 0; i < 9; ++i) {
    const double cu  = c_cx[i] * ux + c_cy[i] * uy;
    f[fidx<L>(i, cell, N)] =
        c_w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
  }
}

template <Layout L>
__global__ void
collide_kernel(double * f, const uint8_t * solid,
               std::size_t nx, std::size_t ny, double inv_tau)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;

  const std::size_t N    = nx * ny;
  const std::size_t cell = y * nx + x;
  if (solid[cell]) return;                 // bounce-back handles solids

  double fi[9], rho = 0.0, mx = 0.0, my = 0.0;
  for (int i = 0; i < 9; ++i) {
    fi[i] = f[fidx<L>(i, cell, N)];
    rho  += fi[i];
    mx   += c_cx[i] * fi[i];
    my   += c_cy[i] * fi[i];
  }
  const double ux = (rho > 0.0) ? mx / rho : 0.0;
  const double uy = (rho > 0.0) ? my / rho : 0.0;
  const double u2 = ux * ux + uy * uy;
  for (int i = 0; i < 9; ++i) {
    const double cu  = c_cx[i] * ux + c_cy[i] * uy;
    const double feq = c_w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    f[fidx<L>(i, cell, N)] = fi[i] - inv_tau * (fi[i] - feq);
  }
}

template <Layout L>
__global__ void
bounce_back_kernel(double * f, const uint8_t * solid,
                   std::size_t nx, std::size_t ny)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;

  const std::size_t N    = nx * ny;
  const std::size_t cell = y * nx + x;
  if (!solid[cell]) return;

  // Swap each pair of opposite directions: (1,3) (2,4) (5,7) (6,8).
  #pragma unroll
  for (int p = 0; p < 4; ++p) {
    const int i = (p == 0) ? 1 : (p == 1) ? 2 : (p == 2) ? 5 : 6;
    const int j = c_opp[i];
    const std::size_t a = fidx<L>(i, cell, N);
    const std::size_t b = fidx<L>(j, cell, N);
    const double tmp = f[a]; f[a] = f[b]; f[b] = tmp;
  }
}

template <Layout L>
__global__ void
stream_kernel(const double * f, double * ftmp,
              std::size_t nx, std::size_t ny)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;

  const std::size_t N    = nx * ny;
  const std::size_t cell = y * nx + x;
  // Pull: ftmp[i, x, y] = f[i, x - cx, y - cy], else keep current value.
  for (int i = 0; i < 9; ++i) {
    const long sx = long(x) - c_cx[i];
    const long sy = long(y) - c_cy[i];
    if (sx >= 0 && sx < long(nx) && sy >= 0 && sy < long(ny)) {
      const std::size_t src = std::size_t(sy) * nx + std::size_t(sx);
      ftmp[fidx<L>(i, cell, N)] = f[fidx<L>(i, src, N)];
    } else {
      ftmp[fidx<L>(i, cell, N)] = f[fidx<L>(i, cell, N)];
    }
  }
}

// Fused collide + stream in one DRAM pass (pull scheme, post-collision
// storage, halfway bounce-back-on-link). Reads the 9 distributions arriving
// at this cell from its neighbours (reflecting at solid/boundary links),
// computes the moments (BGK conserves rho and momentum, so moments of the
// post-collision state equal those of the pre-collision state), relaxes, and
// writes the 9 post-collision values. 9 reads + 9 writes = 144 B/cell.
template <Layout L>
__global__ void
fused_kernel(const double * f_old, double * f_new, const uint8_t * solid,
             std::size_t nx, std::size_t ny, double inv_tau)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;

  const std::size_t N    = nx * ny;
  const std::size_t cell = y * nx + x;

  if (solid[cell]) {
    // Solid cells take no part in the fluid update (reflection happens at the
    // neighbouring fluid cells). Copy through so the buffer stays well-defined.
    for (int i = 0; i < 9; ++i)
      f_new[fidx<L>(i, cell, N)] = f_old[fidx<L>(i, cell, N)];
    return;
  }

  double fi[9], rho = 0.0, mx = 0.0, my = 0.0;
  for (int i = 0; i < 9; ++i) {
    const long sx = long(x) - c_cx[i];
    const long sy = long(y) - c_cy[i];
    double val;
    if (sx >= 0 && sx < long(nx) && sy >= 0 && sy < long(ny) &&
        !solid[std::size_t(sy) * nx + std::size_t(sx)]) {
      val = f_old[fidx<L>(i, std::size_t(sy) * nx + std::size_t(sx), N)];
    } else {
      // bounce-back-on-link: reflect the opposite population at this cell.
      val = f_old[fidx<L>(c_opp[i], cell, N)];
    }
    fi[i] = val;
    rho += val; mx += c_cx[i] * val; my += c_cy[i] * val;
  }

  const double ux = (rho > 0.0) ? mx / rho : 0.0;
  const double uy = (rho > 0.0) ? my / rho : 0.0;
  const double u2 = ux * ux + uy * uy;
  for (int i = 0; i < 9; ++i) {
    const double cu  = c_cx[i] * ux + c_cy[i] * uy;
    const double feq = c_w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    f_new[fidx<L>(i, cell, N)] = fi[i] - inv_tau * (fi[i] - feq);
  }
}

template <Layout L>
__global__ void
inlet_kernel(double * f, const uint8_t * solid,
             std::size_t nx, std::size_t ny, double u_in)
{
  const std::size_t y = blockIdx.x * blockDim.x + threadIdx.x;  // 1D over column
  if (y >= ny) return;
  const std::size_t x = 0;
  const std::size_t N    = nx * ny;
  const std::size_t cell = y * nx + x;
  if (solid[cell]) return;

  const double rho = 1.0, ux = u_in, uy = 0.0, u2 = ux * ux;
  for (int i = 0; i < 9; ++i) {
    const double cu  = c_cx[i] * ux + c_cy[i] * uy;
    f[fidx<L>(i, cell, N)] =
        c_w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
  }
}

template <Layout L>
__global__ void
outlet_kernel(double * f, std::size_t nx, std::size_t ny)
{
  const std::size_t y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= ny) return;
  const std::size_t N    = nx * ny;
  const std::size_t dst  = y * nx + (nx - 1);
  const std::size_t src  = y * nx + (nx - 2);
  for (int i = 0; i < 9; ++i)
    f[fidx<L>(i, dst, N)] = f[fidx<L>(i, src, N)];
}

template <Layout L>
__global__ void
probe_kernel(const double * f, std::size_t nx, std::size_t ny,
             std::size_t px, std::size_t py, double * out)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const std::size_t N    = nx * ny;
  const std::size_t cell = py * nx + px;
  double rho = 0.0, mx = 0.0, my = 0.0;
  for (int i = 0; i < 9; ++i) {
    const double fi = f[fidx<L>(i, cell, N)];
    rho += fi; mx += c_cx[i] * fi; my += c_cy[i] * fi;
  }
  out[0] = (rho > 0.0) ? mx / rho : 0.0;
  out[1] = (rho > 0.0) ? my / rho : 0.0;
}

// ---------------------------------------------------------------------------
// Host class.
// ---------------------------------------------------------------------------
LBM_CUDA::LBM_CUDA(std::size_t nx, std::size_t ny,
                   double u_in, double Re,
                   double cyl_x, double cyl_y, double cyl_r,
                   Layout layout, int block_x, int block_y)
  : nx_(nx), ny_(ny), u_in_(u_in), tau_(0.0),
    layout_(layout), bx_(block_x), by_(block_y),
    h_solid_(nx * ny, 0)   // h_f_ is allocated lazily in sync_to_host()
{
  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;

  // Build the solid mask on the host: no-slip top/bottom walls + cylinder.
  for (std::size_t x = 0; x < nx_; ++x) {
    h_solid_[idx(x, 0)]       = 1;
    h_solid_[idx(x, ny_ - 1)] = 1;
  }
  mark_obstacle(cyl_x, cyl_y, cyl_r);

  // Device allocations.
  CUDA_CHECK(cudaMalloc(&d_f_,     9 * nx_ * ny_ * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_ftmp_,  9 * nx_ * ny_ * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_solid_, nx_ * ny_ * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_probe_, 2 * sizeof(double)));

  // Upload the D2Q9 constants to __constant__ memory.
  CUDA_CHECK(cudaMemcpyToSymbol(c_cx,  cx,  9 * sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_cy,  cy,  9 * sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_opp, opp, 9 * sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_w,   w,   9 * sizeof(double)));
}

LBM_CUDA::~LBM_CUDA()
{
  cudaFree(d_f_);
  cudaFree(d_ftmp_);
  cudaFree(d_solid_);
  cudaFree(d_probe_);
}

void
LBM_CUDA::add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r)
{
  if (cyl2_r > 0.0) mark_obstacle(cyl2_x, cyl2_y, cyl2_r);
}

void
LBM_CUDA::mark_obstacle(double c_x, double c_y, double r)
{
  const double r2 = r * r;
  for (std::size_t y = 0; y < ny_; ++y)
    for (std::size_t x = 0; x < nx_; ++x) {
      const double dx = double(x) - c_x, dy = double(y) - c_y;
      if (dx * dx + dy * dy <= r2) h_solid_[idx(x, y)] = 1;
    }
}

void
LBM_CUDA::initialize()
{
  // Solid mask must be on the device before the init kernel reads it.
  CUDA_CHECK(cudaMemcpy(d_solid_, h_solid_.data(),
                        nx_ * ny_ * sizeof(uint8_t), cudaMemcpyHostToDevice));
  if (layout_ == Layout::SoA) init_impl<Layout::SoA>();
  else                        init_impl<Layout::AoS>();
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <Layout L>
void
LBM_CUDA::init_impl()
{
  const dim3 block(bx_, by_);
  const dim3 grid((nx_ + bx_ - 1) / bx_, (ny_ + by_ - 1) / by_);
  init_kernel<L><<<grid, block>>>(d_f_, d_solid_, nx_, ny_, u_in_);
  CUDA_CHECK(cudaGetLastError());
}

void
LBM_CUDA::step()
{
  if (layout_ == Layout::SoA) step_impl<Layout::SoA>();
  else                        step_impl<Layout::AoS>();
}

template <Layout L>
void
LBM_CUDA::step_impl()
{
  const dim3 block(bx_, by_);
  const dim3 grid((nx_ + bx_ - 1) / bx_, (ny_ + by_ - 1) / by_);
  const double inv_tau = 1.0 / tau_;

  if (fused_) {
    // One DRAM pass: collide + stream + bounce-back-on-link.
    fused_kernel<L><<<grid, block>>>(d_f_, d_ftmp_, d_solid_, nx_, ny_, inv_tau);
    std::swap(d_f_, d_ftmp_);
  } else {
    collide_kernel<L><<<grid, block>>>(d_f_, d_solid_, nx_, ny_, inv_tau);
    bounce_back_kernel<L><<<grid, block>>>(d_f_, d_solid_, nx_, ny_);
    stream_kernel<L><<<grid, block>>>(d_f_, d_ftmp_, nx_, ny_);
    std::swap(d_f_, d_ftmp_);
  }

  const int tcol = 128;
  const int gcol = int((ny_ + tcol - 1) / tcol);
  inlet_kernel<L><<<gcol, tcol>>>(d_f_, d_solid_, nx_, ny_, u_in_);
  outlet_kernel<L><<<gcol, tcol>>>(d_f_, nx_, ny_);
  // No per-step sync: kernels in the default stream run in order. The caller
  // synchronizes once around the timed loop.
}

void
LBM_CUDA::probe(std::size_t x, std::size_t y, double & ux_out, double & uy_out)
{
  if (layout_ == Layout::SoA) probe_impl<Layout::SoA>(x, y);
  else                        probe_impl<Layout::AoS>(x, y);
  double tmp[2];
  CUDA_CHECK(cudaMemcpy(tmp, d_probe_, 2 * sizeof(double), cudaMemcpyDeviceToHost));
  ux_out = tmp[0];
  uy_out = tmp[1];
}

template <Layout L>
void
LBM_CUDA::probe_impl(std::size_t x, std::size_t y)
{
  probe_kernel<L><<<1, 1>>>(d_f_, nx_, ny_, x, y, d_probe_);
  CUDA_CHECK(cudaGetLastError());
}

void
LBM_CUDA::sync_to_host()
{
  CUDA_CHECK(cudaDeviceSynchronize());
  // Allocate the host mirror on first use (timing runs never call this, so
  // large grids avoid a multi-GB host allocation that can exceed --mem).
  if (h_f_.size() != 9 * nx_ * ny_) h_f_.resize(9 * nx_ * ny_);
  CUDA_CHECK(cudaMemcpy(h_f_.data(), d_f_,
                        9 * nx_ * ny_ * sizeof(double), cudaMemcpyDeviceToHost));
}

std::size_t
LBM_CUDA::hfidx(int i, std::size_t x, std::size_t y) const
{
  const std::size_t N = nx_ * ny_, cell = y * nx_ + x;
  return (layout_ == Layout::SoA) ? std::size_t(i) * N + cell : cell * 9 + std::size_t(i);
}

double
LBM_CUDA::rho(std::size_t x, std::size_t y) const
{
  double r = 0.0;
  for (int i = 0; i < 9; ++i) r += h_f_[hfidx(i, x, y)];
  return r;
}

double
LBM_CUDA::ux(std::size_t x, std::size_t y) const
{
  double r = 0.0, m = 0.0;
  for (int i = 0; i < 9; ++i) { const double fi = h_f_[hfidx(i, x, y)]; r += fi; m += cx[i] * fi; }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM_CUDA::uy(std::size_t x, std::size_t y) const
{
  double r = 0.0, m = 0.0;
  for (int i = 0; i < 9; ++i) { const double fi = h_f_[hfidx(i, x, y)]; r += fi; m += cy[i] * fi; }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM_CUDA::vorticity(std::size_t x, std::size_t y) const
{
  if (x == 0 || x == nx_ - 1 || y == 0 || y == ny_ - 1) return 0.0;
  return 0.5 * ((uy(x + 1, y) - uy(x - 1, y)) - (ux(x, y + 1) - ux(x, y - 1)));
}

bool
LBM_CUDA::is_solid(std::size_t x, std::size_t y) const
{
  return h_solid_[idx(x, y)] != 0;
}
