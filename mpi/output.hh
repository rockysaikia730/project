#ifndef LBM_MPI_OUTPUT_HH
#define LBM_MPI_OUTPUT_HH

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <mpi.h>

class LBM_MPI;

/**
 * MPI HDF5+XDMF writer. Every rank computes its local (rho, ux, uy) field
 * slabs and ships them to rank 0 via MPI_Gatherv. Rank 0 reconstructs the
 * global fields, computes vorticity by central differences, and writes
 * one HDF5 file per snapshot plus a temporal XDMF index identical in
 * format to the serial writer.
 *
 * Output is only meant for the single validation run. Disable in scaling
 * studies by passing `every=0`.
 */
class XDMFWriterMPI
{
public:
  XDMFWriterMPI(const std::string & prefix,
                std::size_t nx_global, std::size_t ny,
                MPI_Comm comm);

  /// Write the static solid mask (once). Call before the first snapshot.
  void write_mask(const LBM_MPI & solver);

  /// Append a snapshot at simulation time t.
  void write_snapshot(const LBM_MPI & solver, double t);

private:
  void write_root_xdmf() const;

  std::string  prefix_;
  std::string  basename_;
  std::size_t  nx_, ny_;
  bool         has_mask_;
  std::vector<double> times_;

  MPI_Comm comm_;
  int rank_, psize_;
};

#endif  // LBM_MPI_OUTPUT_HH
