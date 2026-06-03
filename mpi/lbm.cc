#include "lbm.hh"

#include <algorithm>

// D2Q9 constants. Indexing:
//   0: rest    5: NE
//   1: E       6: NW
//   2: N       7: SW
//   3: W       8: SE
//   4: S
const int LBM_MPI::cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
const int LBM_MPI::cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};

const double LBM_MPI::w[9] = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

const int LBM_MPI::opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

LBM_MPI::LBM_MPI(std::size_t nx_global, std::size_t ny,
                 double u_in, double Re,
                 double cyl_x, double cyl_y, double cyl_r,
                 MPI_Comm comm)
  : nx_global_(nx_global), ny_(ny),
    nx_local_(0), x_start_(0),
    u_in_(u_in), tau_(0.0),
    comm_(comm), rank_(0), psize_(1),
    left_(MPI_PROC_NULL), right_(MPI_PROC_NULL)
{
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &psize_);

  // 1D decomposition along x: first (nx % P) ranks get one extra column.
  const std::size_t base = nx_global_ / std::size_t(psize_);
  const std::size_t rem  = nx_global_ % std::size_t(psize_);
  nx_local_ = base + (std::size_t(rank_) < rem ? 1 : 0);
  x_start_  = std::size_t(rank_) * base + std::min(std::size_t(rank_), rem);

  // Neighbours in a non-periodic 1D chain.
  left_  = (rank_ > 0)            ? rank_ - 1 : MPI_PROC_NULL;
  right_ = (rank_ < psize_ - 1)   ? rank_ + 1 : MPI_PROC_NULL;

  const std::size_t nx_p = nx_padded();
  f_   .assign(9 * nx_p * ny_, 0.0);
  ftmp_.assign(9 * nx_p * ny_, 0.0);
  solid_.assign(nx_p * ny_, 0);

  send_left_ .assign(9 * ny_, 0.0);
  send_right_.assign(9 * ny_, 0.0);
  recv_left_ .assign(9 * ny_, 0.0);
  recv_right_.assign(9 * ny_, 0.0);

  // tau from Re and cylinder diameter (global units).
  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;

  // No-slip top and bottom walls: every rank marks its local y=0 and
  // y=ny-1 rows as solid, including the ghost columns (harmless).
  for (std::size_t x_loc = 0; x_loc < nx_p; ++x_loc) {
    solid_[idx(x_loc, 0)]        = 1;
    solid_[idx(x_loc, ny_ - 1)]  = 1;
  }

  mark_obstacle(cyl_x, cyl_y, cyl_r);
}

void
LBM_MPI::add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r)
{
  if (cyl2_r > 0.0) mark_obstacle(cyl2_x, cyl2_y, cyl2_r);
}

void
LBM_MPI::mark_obstacle(double c_x, double c_y, double r)
{
  // Walk over local interior cells, compare to the *global* circle.
  const double r2 = r * r;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x_loc = 1; x_loc <= nx_local_; ++x_loc) {
      const double x_glob = double(x_start_ + x_loc - 1);
      const double dx = x_glob - c_x;
      const double dy = double(y) - c_y;
      if (dx * dx + dy * dy <= r2) solid_[idx(x_loc, y)] = 1;
    }
  }
}

void
LBM_MPI::initialize()
{
  const std::size_t nx_p = nx_padded();
  const std::size_t N    = nx_p * ny_;

  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x_loc = 0; x_loc < nx_p; ++x_loc) {
      const std::size_t k = idx(x_loc, y);
      const double rho = 1.0;
      const double ux  = solid_[k] ? 0.0 : u_in_;
      const double uy  = 0.0;
      const double u2  = ux * ux + uy * uy;
      for (int i = 0; i < Q; ++i) {
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        f_[i * N + k] = feq;
      }
    }
  }
}

void
LBM_MPI::step()
{
  collide();
  bounce_back();
  exchange_halos();
  stream();
  apply_inlet();
  apply_outlet();
}

void
LBM_MPI::collide()
{
  const std::size_t nx_p    = nx_padded();
  const std::size_t N       = nx_p * ny_;
  const double      inv_tau = 1.0 / tau_;

  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x_loc = 1; x_loc <= nx_local_; ++x_loc) {
      const std::size_t k = y * nx_p + x_loc;
      double rho = 0.0, mx = 0.0, my = 0.0;
      for (int i = 0; i < Q; ++i) {
        const double fi = f_[i * N + k];
        rho += fi;
        mx  += cx[i] * fi;
        my  += cy[i] * fi;
      }
      const double ux = (rho > 0.0) ? mx / rho : 0.0;
      const double uy = (rho > 0.0) ? my / rho : 0.0;
      const double u2 = ux * ux + uy * uy;

      for (int i = 0; i < Q; ++i) {
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        f_[i * N + k] += -inv_tau * (f_[i * N + k] - feq);
      }
    }
  }
}

void
LBM_MPI::bounce_back()
{
  const std::size_t nx_p = nx_padded();
  const std::size_t N    = nx_p * ny_;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x_loc = 1; x_loc <= nx_local_; ++x_loc) {
      const std::size_t k = y * nx_p + x_loc;
      if (!solid_[k]) continue;
      std::swap(f_[1 * N + k], f_[3 * N + k]);
      std::swap(f_[2 * N + k], f_[4 * N + k]);
      std::swap(f_[5 * N + k], f_[7 * N + k]);
      std::swap(f_[6 * N + k], f_[8 * N + k]);
    }
  }
}

void
LBM_MPI::exchange_halos()
{
  const std::size_t nx_p = nx_padded();
  const std::size_t N    = nx_p * ny_;

  // Pack: rightmost interior column (x_loc = nx_local_) -> send_right_
  //        leftmost interior column  (x_loc = 1)         -> send_left_
  for (int i = 0; i < Q; ++i) {
    for (std::size_t y = 0; y < ny_; ++y) {
      send_right_[i * ny_ + y] = f_[i * N + y * nx_p + nx_local_];
      send_left_ [i * ny_ + y] = f_[i * N + y * nx_p + 1];
    }
  }

  // Send right edge to right neighbour, receive its left edge into our right ghost.
  MPI_Sendrecv(send_right_.data(), int(9 * ny_), MPI_DOUBLE, right_, 0,
               recv_right_.data(), int(9 * ny_), MPI_DOUBLE, right_, 0,
               comm_, MPI_STATUS_IGNORE);

  // Send left edge to left neighbour, receive its right edge into our left ghost.
  MPI_Sendrecv(send_left_.data(),  int(9 * ny_), MPI_DOUBLE, left_,  0,
               recv_left_.data(),  int(9 * ny_), MPI_DOUBLE, left_,  0,
               comm_, MPI_STATUS_IGNORE);

  // Unpack into ghost columns. With MPI_PROC_NULL the recv buffer is unchanged;
  // we still write whatever it holds into the ghost, which is fine because
  // inlet/outlet overwrite the affected interior column after stream().
  for (int i = 0; i < Q; ++i) {
    for (std::size_t y = 0; y < ny_; ++y) {
      f_[i * N + y * nx_p + (nx_local_ + 1)] = recv_right_[i * ny_ + y];
      f_[i * N + y * nx_p + 0]               = recv_left_ [i * ny_ + y];
    }
  }
}

void
LBM_MPI::stream()
{
  const std::size_t nx_p = nx_padded();
  const std::size_t N    = nx_p * ny_;

  for (int i = 0; i < Q; ++i) {
    for (std::size_t y = 0; y < ny_; ++y) {
      for (std::size_t x_loc = 1; x_loc <= nx_local_; ++x_loc) {
        const long sx = long(x_loc) - cx[i];
        const long sy = long(y)     - cy[i];
        // sx is always within [0, nx_p) because x_loc in [1, nx_local_] and
        // cx in {-1, 0, 1}, so we never need to test it. sy can go out at the
        // physical top/bottom; mirror the serial behaviour there.
        if (sy >= 0 && sy < long(ny_)) {
          ftmp_[i * N + y * nx_p + x_loc] = f_[i * N + std::size_t(sy) * nx_p + std::size_t(sx)];
        } else {
          ftmp_[i * N + y * nx_p + x_loc] = f_[i * N + y * nx_p + x_loc];
        }
      }
    }
  }
  f_.swap(ftmp_);
}

void
LBM_MPI::apply_inlet()
{
  // Only rank 0 owns the global inlet column (global x = 0, local x = 1).
  if (rank_ != 0) return;

  const std::size_t nx_p = nx_padded();
  const std::size_t N    = nx_p * ny_;
  const std::size_t x_loc = 1;
  for (std::size_t y = 0; y < ny_; ++y) {
    if (solid_[idx(x_loc, y)]) continue;
    const double rho = 1.0;
    const double ux  = u_in_;
    const double uy  = 0.0;
    const double u2  = ux * ux + uy * uy;
    for (int i = 0; i < Q; ++i) {
      const double cu  = cx[i] * ux + cy[i] * uy;
      f_[i * N + idx(x_loc, y)] = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }
  }
}

void
LBM_MPI::apply_outlet()
{
  // Only rank (P-1) owns the global outlet column (global x = nx-1).
  if (rank_ != psize_ - 1) return;
  if (nx_local_ < 2) return;   // pathological: outlet needs two adjacent interior columns

  const std::size_t nx_p = nx_padded();
  const std::size_t N    = nx_p * ny_;
  const std::size_t x_dst = nx_local_;       // last interior column, global x = nx-1
  const std::size_t x_src = nx_local_ - 1;   // second-to-last interior column
  for (std::size_t y = 0; y < ny_; ++y) {
    for (int i = 0; i < Q; ++i) {
      f_[i * N + idx(x_dst, y)] = f_[i * N + idx(x_src, y)];
    }
  }
}

double
LBM_MPI::rho(std::size_t x_loc, std::size_t y) const
{
  const std::size_t N = nx_padded() * ny_;
  double r = 0.0;
  for (int i = 0; i < Q; ++i) r += f_[i * N + idx(x_loc, y)];
  return r;
}

double
LBM_MPI::ux(std::size_t x_loc, std::size_t y) const
{
  const std::size_t N = nx_padded() * ny_;
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[i * N + idx(x_loc, y)];
    r += fi;
    m += cx[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM_MPI::uy(std::size_t x_loc, std::size_t y) const
{
  const std::size_t N = nx_padded() * ny_;
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[i * N + idx(x_loc, y)];
    r += fi;
    m += cy[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM_MPI::vorticity(std::size_t x_loc, std::size_t y) const
{
  // Local vorticity using the ghost columns; correct as long as x_loc has
  // both x_loc-1 and x_loc+1 available in this rank's padded array. The
  // global-edge case (rank 0 at x_loc=1 or rank P-1 at x_loc=nx_local) is
  // handled at the output stage (we recompute vorticity after gather).
  if (y == 0 || y == ny_ - 1) return 0.0;
  return 0.5 * ((uy(x_loc + 1, y) - uy(x_loc - 1, y))
              - (ux(x_loc, y + 1) - ux(x_loc, y - 1)));
}

bool
LBM_MPI::is_solid(std::size_t x_loc, std::size_t y) const
{
  return solid_[idx(x_loc, y)] != 0;
}
