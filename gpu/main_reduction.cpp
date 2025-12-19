#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <sys/time.h>
#include <unistd.h>

constexpr double PI = 3.14159265358979323846;

// Simple HIP error check
#define HIP_CHECK(call)                                                     \
  do {                                                                     \
    hipError_t err = call;                                                 \
    if (err != hipSuccess) {                                               \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,         \
              hipGetErrorString(err));                                     \
      MPI_Abort(MPI_COMM_WORLD, -1);                                       \
    }                                                                      \
  } while (0)

// ------------------------ Timing helper ------------------------
double get_time() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

// ------------------------ Analytic solution & RHS (3D) ------------------------
__host__ __device__
double j_exact(double x, double y, double z) {
  return sin(2.0 * PI * x) * sin(2.0 * PI * y) * sin(2.0 * PI * z);
}

__host__ __device__
double f_rhs(double x, double y, double z) {
  double u = j_exact(x, y, z);
  return -12.0 * PI * PI * u;
}

// ------------------------ Indexing helper ------------------------
__host__ __device__
inline int idx3D(int i, int j, int k_local, int Nx, int Ny) {
  return k_local * (Nx * Ny) + i * Ny + j;
}

// ------------------------ 3D Jacobi kernel ------------------------
__global__
void jacobi3d_kernel(const double *u_old, double *u_new, const double *f,
                     int Nx, int Ny, int local_Nz_with_halo,
                     int k_min, int k_max, int k_start_global,
                     int Nz_global, double dx2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k_local = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= Nx || j >= Ny) return;
  if (k_local < k_min || k_local > k_max) return;

  int idx = idx3D(i, j, k_local, Nx, Ny);
  int k_global = k_start_global + (k_local - 1);

  if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1 ||
      k_global == 0 || k_global == Nz_global - 1) {
    u_new[idx] = u_old[idx];
    return;
  }

  int idx_xm = idx3D(i - 1, j, k_local, Nx, Ny);
  int idx_xp = idx3D(i + 1, j, k_local, Nx, Ny);
  int idx_ym = idx3D(i, j - 1, k_local, Nx, Ny);
  int idx_yp = idx3D(i, j + 1, k_local, Nx, Ny);
  int idx_zm = idx3D(i, j, k_local - 1, Nx, Ny);
  int idx_zp = idx3D(i, j, k_local + 1, Nx, Ny);

  u_new[idx] = (u_old[idx_xm] + u_old[idx_xp] +
                u_old[idx_ym] + u_old[idx_yp] +
                u_old[idx_zm] + u_old[idx_zp] - dx2 * f[idx]) / 6.0;
}

__device__ __forceinline__ double warp_reduce_sum(double v) {
  // Warp-level reduction using shuffle down
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset, warpSize);
  }
  return v;
}

// ------------------------ Fused error + residual block partials ------------------------
// Produces one error and one residual partial per block; no atomics on the full grid.
__global__
void fused_err_res_partials(const double *u_num, const double *u_true,
                            const double *f,
                            double *block_err, double *block_res,
                            int Nx, int Ny, int local_Nz_with_halo,
                            int k_start_global, int Nz_global,
                            double dx2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k_local = blockIdx.z * blockDim.z + threadIdx.z;

  double err_val = 0.0;
  double res_val = 0.0;

  if (i < Nx && j < Ny && k_local < local_Nz_with_halo) {
    int k_global = k_start_global + (k_local - 1);
    bool is_interior = (k_local >= 1 && k_local <= local_Nz_with_halo - 2) &&
                       (i > 0 && i < Nx - 1) && (j > 0 && j < Ny - 1) &&
                       (k_global > 0 && k_global < Nz_global - 1);
    if (is_interior) {
      int idx    = idx3D(i,     j,     k_local,     Nx, Ny);
      double uval = u_num[idx];
      double diff = uval - u_true[idx];
      err_val = diff * diff;

      int idx_xm = idx3D(i - 1, j,     k_local,     Nx, Ny);
      int idx_xp = idx3D(i + 1, j,     k_local,     Nx, Ny);
      int idx_ym = idx3D(i,     j - 1, k_local,     Nx, Ny);
      int idx_yp = idx3D(i,     j + 1, k_local,     Nx, Ny);
      int idx_zm = idx3D(i,     j,     k_local - 1, Nx, Ny);
      int idx_zp = idx3D(i,     j,     k_local + 1, Nx, Ny);

      double sum_nb = u_num[idx_xm] + u_num[idx_xp]
                    + u_num[idx_ym] + u_num[idx_yp]
                    + u_num[idx_zm] + u_num[idx_zp];
      double r = sum_nb - 6.0 * uval - dx2 * f[idx];
      res_val = r * r;
    }
  }

  const int tid = ((threadIdx.z * blockDim.y) + threadIdx.y) * blockDim.x + threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp_id = tid / warpSize;

  err_val = warp_reduce_sum(err_val);
  res_val = warp_reduce_sum(res_val);

  __shared__ double warp_err[32];
  __shared__ double warp_res[32];
  if (lane == 0) {
    warp_err[warp_id] = err_val;
    warp_res[warp_id] = res_val;
  }
  __syncthreads();

  if (warp_id == 0) {
    const int block_threads = blockDim.x * blockDim.y * blockDim.z;
    const int nwarps = (block_threads + warpSize - 1) / warpSize;
    double block_err_sum = (lane < nwarps) ? warp_err[lane] : 0.0;
    double block_res_sum = (lane < nwarps) ? warp_res[lane] : 0.0;
    block_err_sum = warp_reduce_sum(block_err_sum);
    block_res_sum = warp_reduce_sum(block_res_sum);
    if (lane == 0) {
      int block_linear = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
      block_err[block_linear] = block_err_sum;
      block_res[block_linear] = block_res_sum;
    }
  }
}

// ------------------------ Final reduction of block partials ------------------------
__global__
void reduce_partials(const double *partials, int n, double *out_sum) {
  extern __shared__ double sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double v = (idx < n) ? partials[idx] : 0.0;
  sdata[tid] = v;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sdata[tid] += sdata[tid + stride];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out_sum, sdata[0]);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Problem size (can be overridden: Nx Ny Nz)
  int Nx = 256, Ny = 256, Nz_global = 256;
  if (argc >= 4) {
    Nx = std::atoi(argv[1]);
    Ny = std::atoi(argv[2]);
    Nz_global = std::atoi(argv[3]);
  }

  int max_iter = 50000;
  double tol = 1e-6;
  int sample_every = 100;

  double dx = 1.0 / (Nx - 1);
  double dy = 1.0 / (Ny - 1);
  double dz = 1.0 / (Nz_global - 1);
  double dx2 = dx * dx;

  if (Nz_global % size != 0 && rank == 0) {
    printf("WARNING: Nz_global not divisible by MPI size; assuming even split.\n");
  }
  int local_Nz = Nz_global / size;
  int k_start_global = rank * local_Nz;
  int local_Nz_with_halo = local_Nz + 2;

  size_t plane_size = static_cast<size_t>(Nx) * Ny;
  size_t local_size = plane_size * local_Nz_with_halo;

  // Host arrays
  double *h_u_init = (double *)calloc(local_size, sizeof(double));
  double *h_u_true = (double *)calloc(local_size, sizeof(double));
  double *h_f = (double *)malloc(local_size * sizeof(double));

  // Initialize slabs
  for (int k_local = 1; k_local <= local_Nz; ++k_local) {
    int k_global = k_start_global + (k_local - 1);
    double z = k_global * dz;
    for (int i = 0; i < Nx; ++i) {
      double x = i * dx;
      for (int j = 0; j < Ny; ++j) {
        double y = j * dy;
        int idx = idx3D(i, j, k_local, Nx, Ny);
        double ut = j_exact(x, y, z);
        double ff = f_rhs(x, y, z);
        h_u_true[idx] = ut;
        h_f[idx] = ff;
        bool boundary = (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1 ||
                         k_global == 0 || k_global == Nz_global - 1);
        h_u_init[idx] = boundary ? ut : 0.0;
      }
    }
  }

  // Device allocations
  double *d_u_old, *d_u_new, *d_f, *d_u_true, *d_err_sum;
  double *d_res_sum;
  double *d_block_err = nullptr, *d_block_res = nullptr;
  HIP_CHECK(hipMalloc(&d_u_old, local_size * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_u_new, local_size * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_f, local_size * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_u_true, local_size * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_err_sum, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_res_sum, sizeof(double)));

  HIP_CHECK(hipMemcpy(d_u_old, h_u_init, local_size * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_u_new, h_u_init, local_size * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_f, h_f, local_size * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_u_true, h_u_true, local_size * sizeof(double), hipMemcpyHostToDevice));

  dim3 blockDim(8, 8, 4);
  dim3 gridDim((Nx + blockDim.x - 1) / blockDim.x,
               (Ny + blockDim.y - 1) / blockDim.y,
               (local_Nz_with_halo + blockDim.z - 1) / blockDim.z);

  hipEvent_t start_event, stop_event;
  hipEventCreate(&start_event);
  hipEventCreate(&stop_event);

  // allocate per-block partial arrays for fused reduction
  size_t num_blocks = static_cast<size_t>(gridDim.x) * gridDim.y * gridDim.z;
  HIP_CHECK(hipMalloc(&d_block_err, num_blocks * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_block_res, num_blocks * sizeof(double)));

  int up = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
  int down = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
  MPI_Request reqs[4];

  double error = 1.0;
  int iter = 0;
  double global_start = get_time();

  if (rank == 0) {
    printf("MPI+HIP Jacobi with on-GPU error reduction: %dx%dx%d, %d ranks\n",
           Nx, Ny, Nz_global, size);
  }

  while (error > tol && iter < max_iter) {
    iter++;

    // Halo exchange on u_old
    double *bottom_halo = d_u_old + 0 * plane_size;
    double *bottom_send = d_u_old + 1 * plane_size;
    double *top_send = d_u_old + local_Nz * plane_size;
    double *top_halo = d_u_old + (local_Nz + 1) * plane_size;

    MPI_Irecv(bottom_halo, plane_size, MPI_DOUBLE, down, 0, comm, &reqs[0]);
    MPI_Irecv(top_halo, plane_size, MPI_DOUBLE, up, 1, comm, &reqs[1]);
    MPI_Isend(bottom_send, plane_size, MPI_DOUBLE, down, 1, comm, &reqs[2]);
    MPI_Isend(top_send, plane_size, MPI_DOUBLE, up, 0, comm, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // Jacobi update
    hipEventRecord(start_event);
    jacobi3d_kernel<<<gridDim, blockDim>>>(d_u_old, d_u_new, d_f, Nx, Ny,
                                           local_Nz_with_halo, 1, local_Nz,
                                           k_start_global, Nz_global, dx2);
    HIP_CHECK(hipGetLastError());
    hipDeviceSynchronize();
    hipEventRecord(stop_event);
    hipEventSynchronize(stop_event);
    float elapsed_ms;
    hipEventElapsedTime(&elapsed_ms, start_event, stop_event);

    std::swap(d_u_old, d_u_new);

    // Periodic error check fully on GPU
    if (iter % sample_every == 0 || iter == max_iter) {
      // zero device scalars
      HIP_CHECK(hipMemset(d_err_sum, 0, sizeof(double)));
      HIP_CHECK(hipMemset(d_res_sum, 0, sizeof(double)));

      // fused block partials
      fused_err_res_partials<<<gridDim, blockDim>>>(d_u_old, d_u_true, d_f,
                       d_block_err, d_block_res,
                       Nx, Ny, local_Nz_with_halo,
                       k_start_global, Nz_global,
                       dx2);
      HIP_CHECK(hipGetLastError());

      // final reduction over block partials
      int red_block = 256;
      int red_grid = (num_blocks + red_block - 1) / red_block;
      size_t shmem = red_block * sizeof(double);
      reduce_partials<<<red_grid, red_block, shmem>>>(d_block_err, num_blocks, d_err_sum);
      reduce_partials<<<red_grid, red_block, shmem>>>(d_block_res, num_blocks, d_res_sum);
      HIP_CHECK(hipGetLastError());

      double h_err_sum = 0.0;
      double h_res_sum = 0.0;
      HIP_CHECK(hipMemcpy(&h_err_sum, d_err_sum, sizeof(double), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(&h_res_sum, d_res_sum, sizeof(double), hipMemcpyDeviceToHost));

      double global_err_sum = 0.0;
      double global_res_sum = 0.0;
      MPI_Allreduce(&h_err_sum, &global_err_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(&h_res_sum, &global_res_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
      double N_interior = (double)(Nx - 2) * (Ny - 2) * (Nz_global - 2);
      error = std::sqrt(global_err_sum / N_interior);
      double residual = std::sqrt(global_res_sum / N_interior);

      if (rank == 0) {
        printf("Iter %d: error = %e, residual = %e, GPU time = %f ms\n",
               iter, error, residual, elapsed_ms);
      }
    }
  }

  double global_end = get_time();
  if (rank == 0) {
    printf("Finished at iter %d, error %e, wall time %f s\n",
           iter, error, global_end - global_start);
  }

  hipFree(d_u_old);
  hipFree(d_u_new);
  hipFree(d_f);
  hipFree(d_u_true);
  hipFree(d_err_sum);
  hipFree(d_res_sum);
  hipFree(d_block_err);
  hipFree(d_block_res);
  free(h_u_init);
  free(h_u_true);
  free(h_f);

  MPI_Finalize();

  char host[256];
  gethostname(host, sizeof(host));
  int pid = (int)getpid();
  printf("[rank %d/%d] host=%s pid=%d\n", rank, size, host, pid);
  fflush(stdout);
  
  return 0;
}
