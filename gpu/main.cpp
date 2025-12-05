#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <sys/time.h>

// ------------------------ Timing helper ------------------------
double get_time() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

// ------------------------ Analytic solution & RHS (3D) ------------------------
// j(x,y,z) = sin(2πx) * sin(2πy) * sin(2πz)
__host__ __device__
double j_exact(double x, double y, double z) {
  return sin(2.0 * M_PI * x) * sin(2.0 * M_PI * y) * sin(2.0 * M_PI * z);
}

// f = ∇^2 j = -12π^2 * j
__host__ __device__
double f_rhs(double x, double y, double z) {
  double u = j_exact(x, y, z);
  return -12.0 * M_PI * M_PI * u;
}

// ------------------------ Indexing helper ------------------------
// Flatten (i,j,k_local) into 1D index
// Layout: planes in k_local, then i, then j: idx = k*(Nx*Ny) + i*Ny + j
__host__ __device__
inline int idx3D(int i, int j, int k_local, int Nx, int Ny) {
  return k_local * (Nx * Ny) + i * Ny + j;
}

// ------------------------ 3D Jacobi kernel (local slab with halos) ------------------------
__global__
void jacobi3d_kernel(const double *u_old,
                     double *u_new,
                     const double *f,
                     int Nx, int Ny, int local_Nz_with_halo,
                     int k_min, int k_max,           // local z-range to update
                     int k_start_global,             // first global k handled by this rank
                     int Nz_global,
                     double dx2)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k_local = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= Nx || j >= Ny) return;
  if (k_local < k_min || k_local > k_max) return;

  int idx = idx3D(i, j, k_local, Nx, Ny);

  // Map local k (1..local_Nz) to global k (0..Nz_global-1)
  int k_global = k_start_global + (k_local - 1);

  // Dirichlet physical boundaries: keep u fixed
  if (i == 0 || i == Nx - 1 ||
      j == 0 || j == Ny - 1 ||
      k_global == 0 || k_global == Nz_global - 1) {
    u_new[idx] = u_old[idx];
    return;
  }

  int idx_xm = idx3D(i - 1, j,     k_local,     Nx, Ny);
  int idx_xp = idx3D(i + 1, j,     k_local,     Nx, Ny);
  int idx_ym = idx3D(i,     j - 1, k_local,     Nx, Ny);
  int idx_yp = idx3D(i,     j + 1, k_local,     Nx, Ny);
  int idx_zm = idx3D(i,     j,     k_local - 1, Nx, Ny);
  int idx_zp = idx3D(i,     j,     k_local + 1, Nx, Ny);

  u_new[idx] = (u_old[idx_xm] + u_old[idx_xp]
              + u_old[idx_ym] + u_old[idx_yp]
              + u_old[idx_zm] + u_old[idx_zp]
              - dx2 * f[idx]) / 6.0;
}

// ------------------------ Error kernel (3D, local slab) ------------------------
__global__
void error_kernel(const double *u_num,
                  const double *u_true,
                  double *err2,
                  int Nx, int Ny, int local_Nz_with_halo,
                  int k_start_global, int Nz_global)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k_local = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= Nx || j >= Ny || k_local >= local_Nz_with_halo) return;

  int k_global = k_start_global + (k_local - 1);

  // Only measure error on interior (physical) points (excluding global boundaries & halos)
  bool is_interior =
      (k_local >= 1 && k_local <= local_Nz_with_halo - 2) &&
      (i > 0 && i < Nx - 1) &&
      (j > 0 && j < Ny - 1) &&
      (k_global > 0 && k_global < Nz_global - 1);

  int idx = idx3D(i, j, k_local, Nx, Ny);
  if (!is_interior) {
    err2[idx] = 0.0;
    return;
  }

  double diff = u_num[idx] - u_true[idx];
  err2[idx] = diff * diff;
}

// ------------------------ Main: MPI + HIP, single kernel per iteration ------------------------
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Global problem size (adjust as needed)
  int Nx = 128;
  int Ny = 128;
  int Nz_global = 128;

  int max_iter = 50000;
  double tol = 1e-6;

  double dx = 1.0 / (Nx - 1);
  double dy = 1.0 / (Ny - 1);
  double dz = 1.0 / (Nz_global - 1);
  double dx2 = dx * dx;  // assuming dx = dy = dz

  // 1D slab decomposition in z
  if (Nz_global % size != 0 && rank == 0) {
    printf("WARNING: Nz_global not divisible by MPI size; "
           "this code assumes perfect divisibility.\n");
  }
  int local_Nz = Nz_global / size;          // owned planes per rank
  int k_start_global = rank * local_Nz;     // first global k this rank owns
  int local_Nz_with_halo = local_Nz + 2;    // +2 for bottom and top halos

  size_t plane_size = (size_t)Nx * Ny;
  size_t local_size = plane_size * local_Nz_with_halo;

  // Host arrays
  double *h_u_init   = (double *)calloc(local_size, sizeof(double));
  double *h_u_true   = (double *)calloc(local_size, sizeof(double));
  double *h_f        = (double *)malloc(local_size * sizeof(double));
  double *h_err2     = (double *)calloc(local_size, sizeof(double));

  // Initialize u, u_true, and f on this rank's slab (k_local = 1..local_Nz)
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
        h_f[idx]      = ff;

        bool is_boundary =
            (i == 0 || i == Nx - 1 ||
             j == 0 || j == Ny - 1 ||
             k_global == 0 || k_global == Nz_global - 1);

        if (is_boundary) {
          h_u_init[idx] = ut;   // Dirichlet boundary
        } else {
          h_u_init[idx] = 0.0;  // interior initial guess
        }
      }
    }
  }
  // Halos (k_local = 0 and local_Nz+1) remain 0; they will be filled by halo exchange.

  // Device allocations
  double *d_u_old, *d_u_new, *d_f, *d_u_true, *d_err2;
  hipMalloc(&d_u_old,  local_size * sizeof(double));
  hipMalloc(&d_u_new,  local_size * sizeof(double));
  hipMalloc(&d_f,      local_size * sizeof(double));
  hipMalloc(&d_u_true, local_size * sizeof(double));
  hipMalloc(&d_err2,   local_size * sizeof(double));

  // Copy initial data to device
  hipMemcpy(d_u_old,  h_u_init, local_size * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_u_new,  h_u_init, local_size * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_f,      h_f,      local_size * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_u_true, h_u_true, local_size * sizeof(double), hipMemcpyHostToDevice);

  // HIP launch configuration
  dim3 blockDim(8, 8, 4);
  dim3 gridDim(
      (Nx + blockDim.x - 1) / blockDim.x,
      (Ny + blockDim.y - 1) / blockDim.y,
      (local_Nz_with_halo + blockDim.z - 1) / blockDim.z
  );

  // HIP events for timing GPU work
  hipEvent_t start_event, stop_event;
  hipEventCreate(&start_event);
  hipEventCreate(&stop_event);

  // MPI neighbors in z
  int up   = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
  int down = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;

  MPI_Request reqs[4];

  double error = 1.0;
  int iter = 0;

  if (rank == 0) {
    printf("Starting MPI+HIP 3D Jacobi (single kernel/iter): %dx%dx%d, %d ranks\n",
           Nx, Ny, Nz_global, size);
  }

  double global_start = get_time();

  while (error > tol && iter < max_iter) {
    iter++;

    // ---------------- Halo exchange on u_old (blocking overall, but non-blocking calls) ----------------
    double *bottom_halo = d_u_old + 0            * plane_size;      // k_local = 0
    double *bottom_send = d_u_old + 1            * plane_size;      // k_local = 1
    double *top_send    = d_u_old + local_Nz     * plane_size;      // k_local = local_Nz
    double *top_halo    = d_u_old + (local_Nz+1) * plane_size;      // k_local = local_Nz+1

    // NOTE: assumes GPU-aware MPI; if not, use host staging buffers.
    MPI_Irecv(bottom_halo, plane_size, MPI_DOUBLE, down, 0, comm, &reqs[0]);
    MPI_Irecv(top_halo,    plane_size, MPI_DOUBLE, up,   1, comm, &reqs[1]);
    MPI_Isend(bottom_send, plane_size, MPI_DOUBLE, down, 1, comm, &reqs[2]);
    MPI_Isend(top_send,    plane_size, MPI_DOUBLE, up,   0, comm, &reqs[3]);

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // ---------------- GPU: single Jacobi update over all owned planes ----------------
    hipEventRecord(start_event);

    int k_min = 1;
    int k_max = local_Nz;  // update all real planes; halos are 0 and local_Nz+1

    jacobi3d_kernel<<<gridDim, blockDim>>>(
        d_u_old, d_u_new, d_f,
        Nx, Ny, local_Nz_with_halo,
        k_min, k_max,
        k_start_global, Nz_global,
        dx2
    );
    hipDeviceSynchronize();

    hipEventRecord(stop_event);
    hipEventSynchronize(stop_event);
    float elapsed_ms;
    hipEventElapsedTime(&elapsed_ms, start_event, stop_event);

    // ---------------- Swap buffers ----------------
    std::swap(d_u_old, d_u_new);

    // ---------------- Compute error every N iterations ----------------
    if (iter % 100 == 0 || iter == max_iter) {
      // GPU: pointwise squared error
      error_kernel<<<gridDim, blockDim>>>(
          d_u_old, d_u_true, d_err2,
          Nx, Ny, local_Nz_with_halo,
          k_start_global, Nz_global
      );
      hipDeviceSynchronize();

      // Copy local err2 to host, sum, then MPI_Allreduce
      hipMemcpy(h_err2, d_err2, local_size * sizeof(double),
                hipMemcpyDeviceToHost);

      double local_err_sum = 0.0;
      for (int k_local = 1; k_local <= local_Nz; ++k_local) {
        int k_global = k_start_global + (k_local - 1);
        if (k_global == 0 || k_global == Nz_global - 1) continue;
        for (int i = 1; i < Nx - 1; ++i) {
          for (int j = 1; j < Ny - 1; ++j) {
            int idx = idx3D(i, j, k_local, Nx, Ny);
            local_err_sum += h_err2[idx];
          }
        }
      }

      double global_err_sum = 0.0;
      MPI_Allreduce(&local_err_sum, &global_err_sum, 1,
                    MPI_DOUBLE, MPI_SUM, comm);

      double N_interior = (double)(Nx - 2) * (Ny - 2) * (Nz_global - 2);
      error = std::sqrt(global_err_sum / N_interior);

      if (rank == 0) {
        printf("Iter %d: Error = %e, GPU time = %f ms\n",
               iter, error, elapsed_ms);
      }
    }
  }

  double global_end = get_time();
  if (rank == 0) {
    printf("Finished at iteration %d with error %e, wall time %f s\n",
           iter, error, global_end - global_start);
  }

  // Cleanup
  hipFree(d_u_old);
  hipFree(d_u_new);
  hipFree(d_f);
  hipFree(d_u_true);
  hipFree(d_err2);
  free(h_u_init);
  free(h_u_true);
  free(h_f);
  free(h_err2);

  MPI_Finalize();
  return 0;
}
