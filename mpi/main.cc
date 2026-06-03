#include "lbm.hh"
#include "output.hh"

#include <mpi.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

using Args = std::unordered_map<std::string, std::string>;

Args
parse_args(int argc, char ** argv)
{
  Args kv;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto eq = a.find('=');
    if (eq == std::string::npos) {
      std::cerr << "Bad argument '" << a << "' (expected key=value)\n";
      std::exit(1);
    }
    kv[a.substr(0, eq)] = a.substr(eq + 1);
  }
  return kv;
}

template <typename T>
T
get(const Args & kv, const std::string & key, T def)
{
  auto it = kv.find(key);
  if (it == kv.end()) return def;
  std::istringstream iss(it->second);
  T v; iss >> v;
  return v;
}

std::string
get_string(const Args & kv, const std::string & key, const std::string & def)
{
  auto it = kv.find(key);
  return (it == kv.end()) ? def : it->second;
}

}  // namespace

int
main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0, psize = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);

  const Args kv = parse_args(argc, argv);

  // Grid + physics.
  const std::size_t nx    = get<std::size_t>(kv, "nx",    800);
  const std::size_t ny    = get<std::size_t>(kv, "ny",    400);
  const double      Re    = get<double>     (kv, "re",    100.0);
  const double      u_in  = get<double>     (kv, "u_in",  0.05);
  const std::size_t steps = get<std::size_t>(kv, "steps", 60000);

  // Cylinders.
  const double cx0 = get<double>(kv, "cyl_x", double(nx) * 0.25);
  const double cy0 = get<double>(kv, "cyl_y", double(ny) * 0.50);
  const double cr0 = get<double>(kv, "cyl_r", double(ny) * 0.025);
  const double cx1 = get<double>(kv, "cyl2_x", -1.0);
  const double cy1 = get<double>(kv, "cyl2_y", -1.0);
  const double cr1 = get<double>(kv, "cyl2_r", -1.0);

  // Output and probe.
  const std::size_t every     = get<std::size_t>(kv, "every", 0);
  const std::string out_pref  = get_string(kv, "out", "out/lbm");
  const std::string probe_csv = get_string(kv, "probe", "probe.csv");
  const std::size_t px = get<std::size_t>(kv, "probe_x", std::size_t(cx0 + 8.0 * cr0));
  const std::size_t py = get<std::size_t>(kv, "probe_y", std::size_t(cy0));

  LBM_MPI solver(nx, ny, u_in, Re, cx0, cy0, cr0, MPI_COMM_WORLD);
  if (cr1 > 0.0) solver.add_second_cylinder(cx1, cy1, cr1);
  solver.initialize();

  if (rank == 0) {
    std::cout << "LBM 2D D2Q9 BGK (MPI, 1D slab)\n"
              << "  grid          : " << nx << " x " << ny << "\n"
              << "  ranks         : " << psize << "\n"
              << "  Re            : " << Re << "\n"
              << "  u_in          : " << u_in << "\n"
              << "  tau           : " << solver.tau() << "\n"
              << "  cylinder      : (" << cx0 << ", " << cy0
              << "), r = " << cr0 << "\n";
    if (cr1 > 0.0)
      std::cout << "  cylinder #2   : (" << cx1 << ", " << cy1
                << "), r = " << cr1 << "\n";
    std::cout << "  steps         : " << steps << "\n"
              << "  output every  : " << every << " (0 = off)\n"
              << "  output prefix : " << out_pref << "\n"
              << "  probe at      : (" << px << ", " << py << ")\n"
              << "  probe csv     : " << probe_csv << "\n";
  }

  // Probe: a single rank owns (px, py). The others skip the per-step write.
  const bool probe_owner = (px < nx) && solver.owns_global_x(px) && (py < ny);
  std::ofstream probe;
  if (probe_owner && probe_csv != "/dev/null") {
    probe.open(probe_csv);
    probe << "step,ux,uy\n";
  }
  const std::size_t probe_x_local = probe_owner ? solver.local_x_of(px) : 0;

  // HDF5 output (rank-0 gather). Only used when every > 0.
  XDMFWriterMPI writer(out_pref, nx, ny, MPI_COMM_WORLD);
  if (every > 0) writer.write_mask(solver);

  // Time the main loop with MPI_Wtime; barrier on both ends and reduce-max.
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();

  for (std::size_t step = 1; step <= steps; ++step) {
    solver.step();

    if (probe_owner && probe.is_open()) {
      probe << step << ','
            << solver.ux(probe_x_local, py) << ','
            << solver.uy(probe_x_local, py) << '\n';
    }

    if (every > 0 && step % every == 0) {
      writer.write_snapshot(solver, double(step));
      if (rank == 0) std::cout << "\r  step " << step << " / " << steps << std::flush;
    }
  }
  if (every > 0 && rank == 0) std::cout << "\n";

  MPI_Barrier(MPI_COMM_WORLD);
  const double dt_local = MPI_Wtime() - t0;
  double dt_max = 0.0;
  MPI_Reduce(&dt_local, &dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    const double mlups = double(nx) * double(ny) * double(steps) / dt_max / 1.0e6;
    std::cout << "Wall time : " << dt_max << " s\n"
              << "MLUPS     : " << mlups << "\n";
    // Single-line CSV marker for the scaling-sweep parser:
    //   CSV: nx,ny,np,steps,wall_time_s,mlups
    std::cout << "CSV: " << nx << "," << ny << "," << psize << ","
              << steps << "," << dt_max << "," << mlups << "\n";
  }

  MPI_Finalize();
  return 0;
}
