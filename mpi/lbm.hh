#ifndef LBM_MPI_HH
#define LBM_MPI_HH

#include <cstddef>
#include <cstdint>
#include <vector>
#include <mpi.h>

/**
 * 2D D2Q9 BGK lattice Boltzmann solver, MPI version.
 *
 * Domain split into 1D slabs along x. Each rank owns nx_local interior
 * columns of width ny, plus one ghost column on each side. Local storage
 * is therefore (nx_local + 2) * ny doubles per direction.
 *
 * Local x-indexing convention:
 *   x_loc = 0              -> left ghost   (from left neighbour)
 *   x_loc = 1 .. nx_local  -> interior     (owned by this rank)
 *   x_loc = nx_local + 1   -> right ghost  (from right neighbour)
 *
 * Global x of local cell x_loc is  x_start_ + x_loc - 1.
 */
class LBM_MPI
{
public:
  static constexpr int Q = 9;

  static const int    cx[Q];
  static const int    cy[Q];
  static const double w[Q];
  static const int    opp[Q];

  LBM_MPI(std::size_t nx_global, std::size_t ny,
          double u_in, double Re,
          double cyl_x, double cyl_y, double cyl_r,
          MPI_Comm comm);

  void add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r);
  void initialize();
  void step();

  // Macroscopic accessors. (x_loc, y) are local coordinates including
  // ghost columns, but in practice callers query interior cells only.
  double rho       (std::size_t x_loc, std::size_t y) const;
  double ux        (std::size_t x_loc, std::size_t y) const;
  double uy        (std::size_t x_loc, std::size_t y) const;
  double vorticity (std::size_t x_loc, std::size_t y) const;
  bool   is_solid  (std::size_t x_loc, std::size_t y) const;

  std::size_t nx_global() const { return nx_global_; }
  std::size_t ny()        const { return ny_; }
  std::size_t nx_local()  const { return nx_local_; }
  std::size_t nx_padded() const { return nx_local_ + 2; }
  std::size_t x_start()   const { return x_start_; }
  double      tau()       const { return tau_; }
  double      u_in()      const { return u_in_; }
  int         rank()      const { return rank_; }
  int         psize()     const { return psize_; }

  bool        owns_global_x(std::size_t x_glob) const {
    return x_glob >= x_start_ && x_glob < x_start_ + nx_local_;
  }
  std::size_t local_x_of(std::size_t x_glob) const {
    return x_glob - x_start_ + 1;
  }

private:
  std::size_t idx (std::size_t x_loc, std::size_t y) const {
    return y * nx_padded() + x_loc;
  }
  std::size_t fidx(int i, std::size_t x_loc, std::size_t y) const {
    return i * nx_padded() * ny_ + idx(x_loc, y);
  }

  void mark_obstacle (double c_x, double c_y, double r);
  void collide       ();
  void bounce_back   ();
  void exchange_halos();
  void stream        ();
  void apply_inlet   ();
  void apply_outlet  ();

  std::size_t nx_global_, ny_;
  std::size_t nx_local_;
  std::size_t x_start_;        ///< global x of local interior column x_loc=1
  double u_in_, tau_;

  std::vector<double>  f_, ftmp_;
  std::vector<uint8_t> solid_;

  MPI_Comm comm_;
  int rank_, psize_;
  int left_, right_;

  // Halo pack/unpack buffers, sized 9 * ny.
  std::vector<double> send_left_, send_right_;
  std::vector<double> recv_left_, recv_right_;
};

#endif  // LBM_MPI_HH
