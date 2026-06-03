#ifndef LBM_CUDA_OUTPUT_HH
#define LBM_CUDA_OUTPUT_HH

#include <cstddef>
#include <string>
#include <vector>

class LBM_CUDA;

/**
 * HDF5 + XDMF snapshot writer for the CUDA solver. Identical output format to
 * the serial writer, so viz/viz.py and ParaView read it unchanged. The caller
 * must LBM_CUDA::sync_to_host() before each write so the host mirror is fresh.
 */
class XDMFWriter
{
public:
  XDMFWriter(const std::string & prefix, std::size_t nx, std::size_t ny);

  void write_mask(const LBM_CUDA & solver);
  void write_snapshot(const LBM_CUDA & solver, double t);

private:
  void write_root_xdmf() const;

  std::string  prefix_;
  std::string  basename_;
  std::size_t  nx_, ny_;
  bool         has_mask_;
  std::vector<double> times_;
};

#endif  // LBM_CUDA_OUTPUT_HH
