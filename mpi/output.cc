#include "output.hh"
#include "lbm.hh"

#include <hdf5.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

void
write_h5_double(const std::string & fname,
                const std::string & dset,
                const std::vector<double> & data,
                std::size_t nx, std::size_t ny,
                bool truncate)
{
  hid_t file = truncate
    ? H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
    : H5Fopen  (fname.c_str(), H5F_ACC_RDWR,  H5P_DEFAULT);
  if (file < 0) {
    std::cerr << "Could not open HDF5 file " << fname << "\n";
    return;
  }
  hsize_t dims[2] = { hsize_t(ny), hsize_t(nx) };
  hid_t   space   = H5Screate_simple(2, dims, nullptr);
  hid_t   dataset = H5Dcreate2(file, dset.c_str(), H5T_IEEE_F64LE,
                               space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(dataset);
  H5Sclose(space);
  H5Fclose(file);
}

void
write_h5_uint8(const std::string & fname,
               const std::string & dset,
               const std::vector<std::uint8_t> & data,
               std::size_t nx, std::size_t ny)
{
  hid_t file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0) {
    std::cerr << "Could not create HDF5 file " << fname << "\n";
    return;
  }
  hsize_t dims[2] = { hsize_t(ny), hsize_t(nx) };
  hid_t   space   = H5Screate_simple(2, dims, nullptr);
  hid_t   dataset = H5Dcreate2(file, dset.c_str(), H5T_STD_U8LE,
                               space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(dataset);
  H5Sclose(space);
  H5Fclose(file);
}

std::string
strip_dir(const std::string & p)
{
  const auto pos = p.find_last_of("/\\");
  return (pos == std::string::npos) ? p : p.substr(pos + 1);
}

std::string
parent_dir(const std::string & p)
{
  const auto pos = p.find_last_of("/\\");
  return (pos == std::string::npos) ? std::string{} : p.substr(0, pos);
}

void
mkdir_p(const std::string & path)
{
  if (path.empty()) return;
  std::string acc;
  for (char c : path) {
    if ((c == '/' || c == '\\') && !acc.empty()) ::mkdir(acc.c_str(), 0755);
    acc.push_back(c);
  }
  if (!acc.empty()) ::mkdir(acc.c_str(), 0755);
}

std::string
indexed_suffix(std::size_t i)
{
  std::ostringstream os;
  os << std::setw(6) << std::setfill('0') << i;
  return os.str();
}

}  // namespace

XDMFWriterMPI::XDMFWriterMPI(const std::string & prefix,
                             std::size_t nx_global, std::size_t ny,
                             MPI_Comm comm)
  : prefix_(prefix), basename_(strip_dir(prefix)),
    nx_(nx_global), ny_(ny), has_mask_(false),
    comm_(comm), rank_(0), psize_(1)
{
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &psize_);

  if (rank_ == 0) mkdir_p(parent_dir(prefix_));
}

void
XDMFWriterMPI::write_mask(const LBM_MPI & solver)
{
  // Pack local mask in (x, y)-blocked form: nx_local consecutive cells per row.
  // The gather then interleaves rows from each rank, producing a row-major
  // (ny rows of nx_global columns) global array. Same convention used below
  // for the macroscopic fields.
  const std::size_t nx_loc = solver.nx_local();
  std::vector<std::uint8_t> local(nx_loc * ny_);
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x_loc = 1; x_loc <= nx_loc; ++x_loc) {
      local[y * nx_loc + (x_loc - 1)] = solver.is_solid(x_loc, y) ? 1 : 0;
    }
  }

  // Because each rank's local block is (ny rows of nx_local), but the gather
  // concatenates contiguous arrays, we need to gather per-row and stitch on
  // rank 0 -- otherwise the layout would be wrong (rank-major instead of
  // row-major). Do the stitching here.
  std::vector<std::uint8_t> global;
  if (rank_ == 0) global.assign(nx_ * ny_, 0);

  // Per-row gather: counts are nx_local per rank, displacements are running
  // sums into the row.
  std::vector<int> row_counts(psize_), row_displs(psize_);
  std::size_t base = nx_ / std::size_t(psize_);
  std::size_t rem  = nx_ % std::size_t(psize_);
  std::size_t off = 0;
  for (int r = 0; r < psize_; ++r) {
    const std::size_t nl = base + (std::size_t(r) < rem ? 1 : 0);
    row_counts[r] = int(nl);
    row_displs[r] = int(off);
    off += nl;
  }
  std::vector<std::uint8_t> row_local(nx_loc);
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t i = 0; i < nx_loc; ++i)
      row_local[i] = local[y * nx_loc + i];
    MPI_Gatherv(row_local.data(), int(nx_loc), MPI_UINT8_T,
                rank_ == 0 ? global.data() + y * nx_ : nullptr,
                row_counts.data(), row_displs.data(), MPI_UINT8_T,
                0, comm_);
  }

  if (rank_ == 0) {
    write_h5_uint8(prefix_ + "_mask.h5", "solid", global, nx_, ny_);
    has_mask_ = true;
    write_root_xdmf();
  }
}

void
XDMFWriterMPI::write_snapshot(const LBM_MPI & solver, double t)
{
  const std::size_t nx_loc = solver.nx_local();

  // Build local row-blocked fields.
  std::vector<double> rho_local (nx_loc * ny_);
  std::vector<double> ux_local  (nx_loc * ny_);
  std::vector<double> uy_local  (nx_loc * ny_);
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x_loc = 1; x_loc <= nx_loc; ++x_loc) {
      const std::size_t k = y * nx_loc + (x_loc - 1);
      rho_local[k] = solver.rho(x_loc, y);
      ux_local [k] = solver.ux (x_loc, y);
      uy_local [k] = solver.uy (x_loc, y);
    }
  }

  // Per-row gather (same pattern as write_mask).
  std::vector<int> row_counts(psize_), row_displs(psize_);
  const std::size_t base = nx_ / std::size_t(psize_);
  const std::size_t rem  = nx_ % std::size_t(psize_);
  std::size_t off = 0;
  for (int r = 0; r < psize_; ++r) {
    const std::size_t nl = base + (std::size_t(r) < rem ? 1 : 0);
    row_counts[r] = int(nl);
    row_displs[r] = int(off);
    off += nl;
  }

  std::vector<double> rho_g, ux_g, uy_g, vor_g;
  if (rank_ == 0) {
    rho_g.assign(nx_ * ny_, 0.0);
    ux_g .assign(nx_ * ny_, 0.0);
    uy_g .assign(nx_ * ny_, 0.0);
    vor_g.assign(nx_ * ny_, 0.0);
  }

  std::vector<double> row_local(nx_loc);
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t i = 0; i < nx_loc; ++i) row_local[i] = rho_local[y * nx_loc + i];
    MPI_Gatherv(row_local.data(), int(nx_loc), MPI_DOUBLE,
                rank_ == 0 ? rho_g.data() + y * nx_ : nullptr,
                row_counts.data(), row_displs.data(), MPI_DOUBLE, 0, comm_);

    for (std::size_t i = 0; i < nx_loc; ++i) row_local[i] = ux_local[y * nx_loc + i];
    MPI_Gatherv(row_local.data(), int(nx_loc), MPI_DOUBLE,
                rank_ == 0 ? ux_g.data() + y * nx_ : nullptr,
                row_counts.data(), row_displs.data(), MPI_DOUBLE, 0, comm_);

    for (std::size_t i = 0; i < nx_loc; ++i) row_local[i] = uy_local[y * nx_loc + i];
    MPI_Gatherv(row_local.data(), int(nx_loc), MPI_DOUBLE,
                rank_ == 0 ? uy_g.data() + y * nx_ : nullptr,
                row_counts.data(), row_displs.data(), MPI_DOUBLE, 0, comm_);
  }

  if (rank_ == 0) {
    // Compute vorticity by central differences over the gathered global field.
    for (std::size_t y = 1; y + 1 < ny_; ++y) {
      for (std::size_t x = 1; x + 1 < nx_; ++x) {
        const std::size_t k = y * nx_ + x;
        const double duy_dx = uy_g[k + 1]    - uy_g[k - 1];
        const double dux_dy = ux_g[k + nx_]  - ux_g[k - nx_];
        vor_g[k] = 0.5 * (duy_dx - dux_dy);
      }
    }

    const std::size_t step_idx = times_.size();
    const std::string fname    = prefix_ + "_" + indexed_suffix(step_idx) + ".h5";
    write_h5_double(fname, "rho",       rho_g, nx_, ny_, /*truncate=*/true);
    write_h5_double(fname, "ux",        ux_g,  nx_, ny_, /*truncate=*/false);
    write_h5_double(fname, "uy",        uy_g,  nx_, ny_, /*truncate=*/false);
    write_h5_double(fname, "vorticity", vor_g, nx_, ny_, /*truncate=*/false);

    times_.push_back(t);
    write_root_xdmf();
  }
}

void
XDMFWriterMPI::write_root_xdmf() const
{
  // Same XDMF layout as the serial writer.
  const std::string xmf_path = prefix_ + ".xdmf";
  std::ofstream xmf(xmf_path);
  if (!xmf) {
    std::cerr << "Could not open " << xmf_path << "\n";
    return;
  }

  xmf << "<?xml version=\"1.0\" ?>\n"
      << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
      << "<Xdmf Version=\"3.0\">\n"
      << "  <Domain>\n"
      << "    <Grid Name=\"LBM\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";

  for (std::size_t i = 0; i < times_.size(); ++i) {
    const std::string tag = indexed_suffix(i);
    const std::string h5  = basename_ + "_" + tag + ".h5";

    xmf << "      <Grid Name=\"step_" << tag << "\" GridType=\"Uniform\">\n"
        << "        <Time Value=\"" << times_[i] << "\"/>\n"
        << "        <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\""
        << ny_ << " " << nx_ << "\"/>\n"
        << "        <Geometry GeometryType=\"ORIGIN_DXDY\">\n"
        << "          <DataItem Format=\"XML\" Dimensions=\"2\">0.0 0.0</DataItem>\n"
        << "          <DataItem Format=\"XML\" Dimensions=\"2\">1.0 1.0</DataItem>\n"
        << "        </Geometry>\n";

    auto write_attr = [&](const char * name, const char * dset) {
      xmf << "        <Attribute Name=\"" << name
          << "\" AttributeType=\"Scalar\" Center=\"Node\">\n"
          << "          <DataItem Format=\"HDF\" Dimensions=\""
          << ny_ << " " << nx_ << "\" NumberType=\"Float\" Precision=\"8\">"
          << h5 << ":/" << dset
          << "</DataItem>\n"
          << "        </Attribute>\n";
    };
    write_attr("rho",       "rho");
    write_attr("ux",        "ux");
    write_attr("uy",        "uy");
    write_attr("vorticity", "vorticity");

    if (has_mask_) {
      xmf << "        <Attribute Name=\"solid\" AttributeType=\"Scalar\" Center=\"Node\">\n"
          << "          <DataItem Format=\"HDF\" Dimensions=\""
          << ny_ << " " << nx_ << "\" NumberType=\"UChar\">"
          << basename_ << "_mask.h5:/solid"
          << "</DataItem>\n"
          << "        </Attribute>\n";
    }

    xmf << "      </Grid>\n";
  }

  xmf << "    </Grid>\n"
      << "  </Domain>\n"
      << "</Xdmf>\n";
}
