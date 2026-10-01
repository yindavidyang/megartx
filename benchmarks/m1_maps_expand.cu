// Manual standalone experiment. Never imported or selected by the model runtime.
#include "../kernels/m1_maps_expand.cuh"
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace megartx::experimental;
namespace fs = std::filesystem;
using Bytes = std::vector<unsigned char>;
void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
Bytes read(fs::path path, std::size_t size) {
  if (!fs::is_regular_file(path) || fs::file_size(path) != size)
    throw std::runtime_error("fixture extent mismatch");
  Bytes out(size);
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(out.data()), size);
  if (!stream) throw std::runtime_error("fixture read failed");
  return out;
}
void write(fs::path path, Bytes const& bytes) {
  if (fs::exists(path)) throw std::runtime_error("output already exists");
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<char const*>(bytes.data()), bytes.size());
  if (!stream) throw std::runtime_error("output write failed");
}
std::size_t scratch = 0;
struct Buffer {
  unsigned char* storage = nullptr;
  std::size_t size;
  explicit Buffer(std::size_t n) : size(n) {
    check(cudaMalloc(&storage, n + 64));
    scratch += n + 64;
    reset();
  }
  ~Buffer() { if (storage) cudaFree(storage); }
  Buffer(Buffer const&) = delete;
  unsigned char* data() const { return storage + 32; }
  void reset() {
    check(cudaMemset(storage, 0xA7, 32));
    check(cudaMemset(data(), 0xD3, size));
    check(cudaMemset(data() + size, 0xB6, 32));
  }
  void upload(unsigned char const* bytes) { check(cudaMemcpy(data(), bytes, size, cudaMemcpyHostToDevice)); }
  Bytes snapshot(bool guards = false) const {
    Bytes raw(size + 64);
    check(cudaMemcpy(raw.data(), storage, raw.size(), cudaMemcpyDeviceToHost));
    if (!std::all_of(raw.begin(), raw.begin() + 32, [](auto x) { return x == 0xA7; }) ||
        !std::all_of(raw.end() - 32, raw.end(), [](auto x) { return x == 0xB6; }))
      throw std::runtime_error("buffer guard corrupted");
    if (guards) return raw;
    return Bytes(raw.begin() + 32, raw.end() - 32);
  }
};
struct Inputs {
  Buffer ids{32}, weights{32}, aq{kAQBytes}, sf{128 * kSFBlocks};
  void upload(Bytes const& raw) {
    ids.upload(raw.data()); weights.upload(raw.data() + 32);
    aq.upload(raw.data() + 64); sf.upload(raw.data() + 64 + kAQBytes);
  }
  Bytes snapshot() const {
    Bytes out;
    for (auto b : {&ids, &weights, &aq, &sf}) {
      auto bytes = b->snapshot(); out.insert(out.end(), bytes.begin(), bytes.end());
    }
    return out;
  }
};
struct Outputs {
  Buffer rank{32}, sorted{32}, offsets{129 * 8}, aq{8 * kAQBytes}, weights{32}, sf{kSFBytes};
  M1Buffers bind(Inputs const& i) {
    return {reinterpret_cast<int const*>(i.ids.data()),
      reinterpret_cast<std::uint32_t const*>(i.weights.data()), i.aq.data(), i.sf.data(),
      reinterpret_cast<int*>(rank.data()), reinterpret_cast<int*>(sorted.data()),
      reinterpret_cast<std::int64_t*>(offsets.data()), aq.data(),
      reinterpret_cast<std::uint32_t*>(weights.data()), sf.data()};
  }
  Bytes snapshot() const {
    Bytes out;
    for (auto b : {&rank, &sorted, &offsets, &aq, &weights}) {
      auto bytes = b->snapshot(); out.insert(out.end(), bytes.begin(), bytes.end());
    }
    auto scales = sf.snapshot(true); out.insert(out.end(), scales.begin(), scales.end());
    return out;
  }
};
void headroom() {
  std::size_t free, total;
  check(cudaMemGetInfo(&free, &total));
  if (free < (std::size_t{2} << 30)) throw std::runtime_error("GPU headroom guard");
  std::ifstream mem("/proc/meminfo");
  std::string key, rest;
  std::size_t kb = 0;
  while (mem >> key) {
    if (key == "MemAvailable:") { mem >> kb; break; }
    std::getline(mem, rest);
  }
  if (kb < (std::size_t{8} << 20)) throw std::runtime_error("host headroom guard");
}
void validate_row(Bytes const& raw) {
  std::array<int, 8> ids;
  std::memcpy(ids.data(), raw.data(), 32);
  for (int i = 0; i < 8; ++i) {
    if (ids[i] < 0 || ids[i] >= 128) throw std::runtime_error("invalid route ID");
    for (int j = 0; j < i; ++j) if (ids[i] == ids[j]) throw std::runtime_error("duplicate route ID");
  }
}
void negative_routes(Inputs& in, Outputs& a, Outputs& b, Bytes original, cudaStream_t stream) {
  auto before_a = a.snapshot(), before_b = b.snapshot();
  for (int bad : {-1, 128, 0}) {
    auto raw = original;
    if (bad == 0) std::copy(raw.begin() + 4, raw.begin() + 8, raw.begin());
    else std::memcpy(raw.data(), &bad, 4);
    in.upload(raw);
    check(launch(a.bind(in), stream, true)); check(launch(b.bind(in), stream, false));
    check(cudaStreamSynchronize(stream));
    if (a.snapshot() != before_a || b.snapshot() != before_b || in.snapshot() != raw)
      throw std::runtime_error("invalid route mutated storage");
  }
  in.upload(original);
  for (int m : {2, 4, 8})
    if (launch(a.bind(in), stream, true, m) != cudaErrorInvalidValue)
      throw std::runtime_error("unsupported M accepted");
  if (launch(a.bind(in), stream, true, 1, 2815) != cudaErrorInvalidValue ||
      launch(a.bind(in), stream, true, 1, 2816, 127) != cudaErrorInvalidValue ||
      launch(a.bind(in), stream, true, 1, 2816, 128, 7) != cudaErrorInvalidValue)
    throw std::runtime_error("unsupported geometry accepted");
  auto invalid = a.bind(in); invalid.aq += 1;
  if (launch(invalid, stream, true) != cudaErrorInvalidValue)
    throw std::runtime_error("unaligned input accepted");
  invalid = a.bind(in); invalid.expanded_sf = nullptr;
  if (launch(invalid, stream, true) != cudaErrorInvalidValue)
    throw std::runtime_error("null output accepted");
  if (a.snapshot() != before_a || b.snapshot() != before_b || in.snapshot() != original)
    throw std::runtime_error("rejected geometry mutated storage");
}
double measure(M1Buffers args, cudaStream_t stream, bool fused, int iterations) {
  cudaEvent_t begin, end;
  check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
  for (int i = 0; i < 32; ++i) check(launch(args, stream, fused));
  check(cudaEventRecord(begin, stream));
  for (int i = 0; i < iterations; ++i) check(launch(args, stream, fused));
  check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
  float ms = 0;
  check(cudaEventElapsedTime(&ms, begin, end));
  check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
  return ms * 1000.0 / iterations;
}
void array_json(std::vector<double> const& values) {
  std::cout << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
  std::cout << ']';
}
int run(int argc, char** argv) {
  static_assert(sizeof(int) == 4 && sizeof(std::int64_t) == 8);
  if (argc != 4 || (std::string(argv[1]) != "correctness" && std::string(argv[1]) != "timing"))
    throw std::runtime_error("usage: binary correctness|timing fixtures NEW_OUTPUT");
  bool timing = std::string(argv[1]) == "timing";
  fs::path fixtures = argv[2], output = argv[3];
  if (fs::exists(output)) throw std::runtime_error("output must be new");
  fs::create_directory(output);
  check(cudaSetDevice(0)); headroom();
  cudaFuncAttributes attributes;
  check(cudaFuncGetAttributes(&attributes, m1_maps_expand));
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  try {
    Inputs input;
    Outputs fused, control;
    headroom();
    if (scratch > (8 << 20)) throw std::runtime_error("scratch budget exceeded");
    constexpr std::size_t input_bytes = 24000, output_bytes = 2896040;
    for (int index = 0; index < (timing ? 1 : 3); ++index) {
      auto raw = read(fixtures / ("case" + std::to_string(index) + ".input"), input_bytes);
      auto expected = read(fixtures / ("case" + std::to_string(index) + ".expected"), output_bytes);
      validate_row(raw); input.upload(raw);
      check(launch(fused.bind(input), stream, true)); check(launch(control.bind(input), stream, false));
      check(cudaStreamSynchronize(stream));
      auto a = fused.snapshot(), b = control.snapshot(), after = input.snapshot();
      if (a != expected || b != expected || after != raw) throw std::runtime_error("independent oracle mismatch");
      if (!timing) {
        a.insert(a.end(), after.begin(), after.end()); b.insert(b.end(), after.begin(), after.end());
        write(output / ("case" + std::to_string(index) + ".fused"), a);
        write(output / ("case" + std::to_string(index) + ".control"), b);
      }
    }
    auto first = read(fixtures / "case0.input", input_bytes);
    negative_routes(input, fused, control, first, stream);
    std::vector<double> fused_us, control_us;
    if (timing) {
      for (int round = 0; round < 9; ++round) {
        headroom();
        if (round % 2 == 0) {
          fused_us.push_back(measure(fused.bind(input), stream, true, 512));
          control_us.push_back(measure(control.bind(input), stream, false, 512));
        } else {
          control_us.push_back(measure(control.bind(input), stream, false, 512));
          fused_us.push_back(measure(fused.bind(input), stream, true, 512));
        }
      }
      check(cudaStreamSynchronize(stream));
      auto expected = read(fixtures / "case0.expected", output_bytes);
      if (fused.snapshot() != expected || control.snapshot() != expected || input.snapshot() != first)
        throw std::runtime_error("post-timing oracle mismatch");
    }
    std::cout << "{\"scope\":\"isolated M1 maps/AQ/SF; development control\",\"oracle_match\":true,"
              << "\"cases\":" << (timing ? 1 : 3) << ",\"scratch_bytes\":" << scratch
              << ",\"fused_threads\":256,\"fused_registers_per_thread\":" << attributes.numRegs
              << ",\"fused_static_shared_bytes\":" << attributes.sharedSizeBytes
              << ",\"fused_local_bytes_per_thread\":" << attributes.localSizeBytes
              << ",\"binary_version\":" << attributes.binaryVersion
              << ",\"negative_routes\":3,\"unsupported_shapes_rejected\":true,"
              << "\"iterations_per_batch\":512,\"warmups_per_batch\":32,\"fused_us\":";
    array_json(fused_us); std::cout << ",\"two_launch_control_us\":"; array_json(control_us);
    std::cout << ",\"candidate_selectable\":false,\"full_model_qualified\":false}\n";
    check(cudaStreamSynchronize(stream));
  } catch (...) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); throw; }
  check(cudaStreamDestroy(stream));
  return 0;
}
int main(int argc, char** argv) {
  try { return run(argc, argv); }
  catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
