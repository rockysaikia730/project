#include "lbm.hh"
#include "output.hh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n",            \
                   __FILE__, __LINE__, cudaGetErrorString(err));\
      std::exit(EXIT_FAILURE);                                  \
    }                                                           \
  } while (0)

namespace {

using Args = std::unordered_map<std::string, std::string>;

Args parse_args(int argc, char ** argv)
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
T get(const Args & kv, const std::string & key, T def)
{
  auto it = kv.find(key);
  if (it == kv.end()) return def;
  std::istringstream iss(it->second);
  T v; iss >> v;
  return v;
}

std::string get_string(const Args & kv, const std::string & key, const std::string & def)
{
  auto it = kv.find(key);
  return (it == kv.end()) ? def : it->second;
}

}  // namespace

int
main(int argc, char ** argv)
{
  const Args kv = parse_args(argc, argv);

  const std::size_t nx    = get<std::size_t>(kv, "nx",    800);
  const std::size_t ny    = get<std::size_t>(kv, "ny",    400);
  const double      Re    = get<double>     (kv, "re",    100.0);
  const double      u_in  = get<double>     (kv, "u_in",  0.05);
  const std::size_t steps = get<std::size_t>(kv, "steps", 60000);

  const double cx0 = get<double>(kv, "cyl_x", double(nx) * 0.25);
  const double cy0 = get<double>(kv, "cyl_y", double(ny) * 0.50);
  const double cr0 = get<double>(kv, "cyl_r", double(ny) * 0.025);
  const double cx1 = get<double>(kv, "cyl2_x", -1.0);
  const double cy1 = get<double>(kv, "cyl2_y", -1.0);
  const double cr1 = get<double>(kv, "cyl2_r", -1.0);

  const std::size_t every     = get<std::size_t>(kv, "every", 0);
  const std::string out_pref  = get_string(kv, "out", "out/lbm");
  const std::string probe_csv = get_string(kv, "probe", "probe.csv");
  const std::size_t px = get<std::size_t>(kv, "probe_x", std::size_t(cx0 + 8.0 * cr0));
  const std::size_t py = get<std::size_t>(kv, "probe_y", std::size_t(cy0));

  // Block geometry + layout.
  const int    bx     = get<int>(kv, "block_x", 32);
  const int    by     = get<int>(kv, "block_y", 8);
  const Layout layout = layout_from_string(get_string(kv, "layout", "soa"));
  const bool   fused  = get<int>(kv, "fused", 0) != 0;

  LBM_CUDA solver(nx, ny, u_in, Re, cx0, cy0, cr0, layout, bx, by);
  solver.set_fused(fused);
  if (cr1 > 0.0) solver.add_second_cylinder(cx1, cy1, cr1);
  solver.initialize();

  const char * lname = (layout == Layout::SoA) ? "SoA" : "AoS";
  std::cout << "LBM 2D D2Q9 BGK (CUDA, V100)\n"
            << "  grid          : " << nx << " x " << ny << "\n"
            << "  Re            : " << Re << "\n"
            << "  u_in          : " << u_in << "\n"
            << "  tau           : " << solver.tau() << "\n"
            << "  layout        : " << lname << "\n"
            << "  kernel        : " << (fused ? "fused collide-stream" : "separate") << "\n"
            << "  block         : " << bx << " x " << by << "\n"
            << "  steps         : " << steps << "\n"
            << "  output every  : " << every << " (0 = off)\n"
            << "  probe at      : (" << px << ", " << py << ")\n"
            << "  probe csv     : " << probe_csv << "\n";

  const bool do_probe = (probe_csv != "/dev/null") && (px < nx) && (py < ny);
  std::ofstream probe;
  if (do_probe) { probe.open(probe_csv); probe << "step,ux,uy\n"; }

  XDMFWriter writer(out_pref, nx, ny);
  if (every > 0) { solver.sync_to_host(); writer.write_mask(solver); }

  // Time the step loop with CUDA events; exclude the one-time H2D done in
  // the constructor and any D2H done for output. Probe/output runs are not
  // used for the headline MLUPS (run with probe=/dev/null every=0).
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(t0));

  for (std::size_t step = 1; step <= steps; ++step) {
    solver.step();

    if (do_probe) {
      double ux, uy;
      solver.probe(px, py, ux, uy);   // tiny D2H; only in validation runs
      probe << step << ',' << ux << ',' << uy << '\n';
    }
    if (every > 0 && step % every == 0) {
      solver.sync_to_host();
      writer.write_snapshot(solver, double(step));
      std::cout << "\r  step " << step << " / " << steps << std::flush;
    }
  }

  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  if (every > 0) std::cout << "\n";

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  const double dt    = ms / 1.0e3;
  const double mlups = double(nx) * double(ny) * double(steps) / dt / 1.0e6;

  std::cout << "Wall time : " << dt << " s\n"
            << "MLUPS     : " << mlups << "\n";
  // CSV marker for the sweep parsers:
  //   CSV: nx,ny,block_x,block_y,layout,steps,wall_time_s,mlups
  std::cout << "CSV: " << nx << "," << ny << "," << bx << "," << by << ","
            << lname << "," << steps << "," << dt << "," << mlups << "\n";

  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
  return 0;
}
