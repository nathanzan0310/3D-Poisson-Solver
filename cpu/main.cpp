#include <cmath>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Maps 3D indices (i, j, k) into 1D index for flattened array
inline std::size_t idx(int i, int j, int k,
                       int Nx, int Ny, int Nz_local_with_halo) {
    // layout: k fastest across slabs
    return static_cast<std::size_t>(k) * Ny * Nx +
           static_cast<std::size_t>(j) * Nx +
           static_cast<std::size_t>(i);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Global grid size (can be overridden by argv)
    int Nx = 64, Ny = 64, Nz_global = 64;
    if (argc >= 4) {
        Nx = std::atoi(argv[1]);
        Ny = std::atoi(argv[2]);
        Nz_global = std::atoi(argv[3]);
    }

    // Mode numbers for analytic solution
    int n = 2, m = 2, k_mode = 2;
    const double PI = std::acos(-1.0);
    const double Lx = 1.0, Ly = 1.0, Lz = 1.0;

    // Grid spacing in each direction
    double dx = Lx / (Nx - 1);
    double dy = Ly / (Ny - 1);
    double dz = Lz / (Nz_global - 1);

    // We’ll assume equal spacing for the Jacobi formula simplicity:
    // u_new = (sum_neighbors - h^2 f) / 6
    std::ofstream csv;
    if (rank == 0) {
        if (std::fabs(dx - dy) > 1e-12 || std::fabs(dx - dz) > 1e-12) {
            std::cerr << "Warning: dx,dy,dz differ; code assumes dx=dy=dz.\n";
        }
        csv.open("poisson_stats.csv");
        csv << "iter,residual_l2,error_l2\n";
    }
    // record error/residual every N iterations
    int sample_every = 1; 

    // Finite difference grid spacing (assuming uniform)
    double h = dx;
    double h2 = h * h;

    // 1D slab decomposition along z (global index k = 0..Nz_global-1)
    int base = Nz_global / size;
    int rem  = Nz_global % size;

    int Nz_local = base + (rank < rem ? 1 : 0); // interior planes per rank
    int k_start = rank * base + std::min(rank, rem); // global k for local k=0
    int k_end   = k_start + Nz_local - 1;            // inclusive

    // Add 2 halo planes: local k indices [0..Nz_local+1]
    int Nz_local_with_halo = Nz_local + 2;

    std::size_t Nloc = static_cast<std::size_t>(Nx) * Ny * Nz_local_with_halo;

    // Current numerical approximation of solution to Poisson
    std::vector<double> u(Nloc, 0.0);
    
    // Next iteration in Jacobi method
    std::vector<double> u_new(Nloc, 0.0);
    
    // RHS of poisson equation
    std::vector<double> f(Nloc, 0.0); 

    // Precompute lambda for analytic solution:
    double lambda = (n * n + m * m + k_mode * k_mode) * PI * PI;

    auto j_exact = [&](double x, double y, double z) {
    return std::sin(n * PI * x) * std::cos(m * PI * y) * std::sin(k_mode * PI * z);
    };

    for (int kk = 0; kk < Nz_local; ++kk) {
        int k_global = k_start + kk;
        double z = k_global * dz;
        int k_local = kk + 1; // shift for halo (0 and Nz_local+1 are halos)

        for (int j = 0; j < Ny; ++j) {
            double y = j * dy;
            for (int i = 0; i < Nx; ++i) {
                double x = i * dx;
                std::size_t id = idx(i, j, k_local, Nx, Ny, Nz_local_with_halo);

                double jval = j_exact(x, y, z);

                // Detect physical domain boundary
                bool is_boundary =
                    (i == 0 || i == Nx - 1 ||
                    j == 0 || j == Ny - 1 ||
                    k_global == 0 || k_global == Nz_global - 1);

                if (is_boundary) {
                    // Dirichlet BC: solution equals analytic j_exact on the boundary
                    u[id]     = jval;
                    u_new[id] = jval;
                    // f[id] is not used at boundary in our Jacobi update, so we can set it to 0
                    f[id] = 0.0;
                } else {
                    u[id]     = 0.0; // initial guess for interior
                    u_new[id] = 0.0;

                    // Compute discrete Laplacian of j_exact at this interior point:
                    //   Δ_h j ≈ (j_ip - 2*j + j_im)/dx^2 + ...
                    double x_ip = (i + 1) * dx;
                    double x_im = (i - 1) * dx;
                    double y_jp = (j + 1) * dy;
                    double y_jm = (j - 1) * dy;
                    double z_kp = (k_global + 1) * dz;
                    double z_km = (k_global - 1) * dz;

                    double j_ip = j_exact(x_ip, y,    z);
                    double j_im = j_exact(x_im, y,    z);
                    double j_jp = j_exact(x,    y_jp, z);
                    double j_jm = j_exact(x,    y_jm, z);
                    double j_kp = j_exact(x,    y,    z_kp);
                    double j_km = j_exact(x,    y,    z_km);

                    double lap_discrete =
                        (j_ip - 2.0 * jval + j_im) / (dx * dx) +
                        (j_jp - 2.0 * jval + j_jm) / (dy * dy) +
                        (j_kp - 2.0 * jval + j_km) / (dz * dz);

                    // Right-hand side f is chosen so that
                    //   -Δ_h j_exact = f  (discrete manufactured solution)
                    f[id] = -lap_discrete;
                }
            }
        }
    }


    // Jacobi parameters
    int max_iters = 5000;
    double tol = 1e-8;
    int print_every = 100;

    if (rank == 0) {
        std::cout << "Global grid: " << Nx << " x " << Ny << " x " << Nz_global << "\n";
        std::cout << "Ranks: " << size << "\n";
        std::cout << "Local Nz (rank " << rank << "): " << Nz_local << "\n";
    }

    // Buffers (planes) are contiguous: Nx*Ny doubles per plane
    int plane_size = Nx * Ny;

    double t_start = MPI_Wtime();
    int last_iter = 0;

    //  Jacobi main iteration loop
    //     Exchange halo planes with neighbors using MPI
    //     Perform Jacobi update on interior points using OpenMP parallel
    //     Track max update per iteration for convergence
    //     Periodically compute residual and L2 error vs analytic solution
    for (int iter = 0; iter < max_iters; ++iter) {
        // Halo exchange (non-blocking) on u
        MPI_Request reqs[4];
        int req_count = 0;

        // Halo exchange with non-blocking MPI send/recv
        //   Exchange the outermost interior planes with neighbor ranks.
        //   Fill k_local=0 and k_local=Nz_local+1 halo planes.
        // send/recv with rank-1 (bottom neighbor)
        if (rank > 0) {
            // Receive bottom halo (k_local = 0) from rank-1
            MPI_Irecv(&u[idx(0, 0, 0, Nx, Ny, Nz_local_with_halo)],
                      plane_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
                      &reqs[req_count++]);

            // Send first interior plane (k_local = 1) to rank-1
            MPI_Isend(&u[idx(0, 0, 1, Nx, Ny, Nz_local_with_halo)],
                      plane_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }

        // send/recv with rank+1 (top neighbor)
        if (rank < size - 1) {
            // Receive top halo (k_local = Nz_local+1) from rank+1
            MPI_Irecv(&u[idx(0, 0, Nz_local + 1, Nx, Ny, Nz_local_with_halo)],
                      plane_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD,
                      &reqs[req_count++]);

            // Send last interior plane (k_local = Nz_local) to rank+1
            MPI_Isend(&u[idx(0, 0, Nz_local, Nx, Ny, Nz_local_with_halo)],
                      plane_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }

        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }

        // Jacobi update on interior points
        double local_max_update = 0.0;

        #pragma omp parallel for collapse(3) reduction(max:local_max_update)
        for (int kk = 1; kk <= Nz_local; ++kk) { // local planes with data
            
            for (int j = 1; j < Ny - 1; ++j) {
                for (int i = 1; i < Nx - 1; ++i) {
                    int k_global = k_start + (kk - 1);
                    bool is_boundary =
                        (k_global == 0 || k_global == Nz_global - 1 ||
                         i == 0 || i == Nx - 1 ||
                         j == 0 || j == Ny - 1);

                    if (is_boundary) {
                        // Keep Dirichlet boundaries fixed
                        continue;
                    }

                    std::size_t id = idx(i, j, kk, Nx, Ny, Nz_local_with_halo);

                    double sum_neighbors =
                        u[idx(i + 1, j,     kk,     Nx, Ny, Nz_local_with_halo)] +
                        u[idx(i - 1, j,     kk,     Nx, Ny, Nz_local_with_halo)] +
                        u[idx(i,     j + 1, kk,     Nx, Ny, Nz_local_with_halo)] +
                        u[idx(i,     j - 1, kk,     Nx, Ny, Nz_local_with_halo)] +
                        u[idx(i,     j,     kk + 1, Nx, Ny, Nz_local_with_halo)] +
                        u[idx(i,     j,     kk - 1, Nx, Ny, Nz_local_with_halo)];

                    double u_new_val = (sum_neighbors + h2 * f[id]) / 6.0;
                    double diff = std::fabs(u_new_val - u[id]);
                    if (diff > local_max_update) local_max_update = diff;

                    u_new[id] = u_new_val;
                }
            }
        }

        // swap old and new solutions so u now holds the updated values
        u.swap(u_new);

        // Global max update for convergence check across all ranks
        double global_max_update = 0.0;
        MPI_Allreduce(&local_max_update, &global_max_update, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0 && (iter % print_every == 0)) {
            std::cout << "Iter " << iter
                      << ", global max update = " << global_max_update << "\n";
        }

        bool do_sample = (iter % sample_every == 0) || (global_max_update < tol);

        if (do_sample) {
            // compute residual and error on this iteration

            double local_res2 = 0.0;
            double local_err2 = 0.0;
            long long local_points = 0;

            #pragma omp parallel for collapse(3) reduction(+:local_res2, local_err2, local_points)
            for (int kk = 1; kk <= Nz_local; ++kk) {
                for (int j = 1; j < Ny - 1; ++j) {
                    for (int i = 1; i < Nx - 1; ++i) {
                        int k_global = k_start + (kk - 1);
                        bool is_boundary =
                            (i == 0 || i == Nx - 1 ||
                            j == 0 || j == Ny - 1 ||
                            k_global == 0 || k_global == Nz_global - 1);
                        if (is_boundary) continue; // PDE applies only to interior

                        std::size_t id = idx(i, j, kk, Nx, Ny, Nz_local_with_halo);

                        // residual: r = f + Δ_h u
                        double sum_neighbors =
                            u[idx(i + 1, j,     kk,     Nx, Ny, Nz_local_with_halo)] +
                            u[idx(i - 1, j,     kk,     Nx, Ny, Nz_local_with_halo)] +
                            u[idx(i,     j + 1, kk,     Nx, Ny, Nz_local_with_halo)] +
                            u[idx(i,     j - 1, kk,     Nx, Ny, Nz_local_with_halo)] +
                            u[idx(i,     j,     kk + 1, Nx, Ny, Nz_local_with_halo)] +
                            u[idx(i,     j,     kk - 1, Nx, Ny, Nz_local_with_halo)];

                        double lap_h = (sum_neighbors - 6.0 * u[id]) / h2;
                        double r = f[id] + lap_h;  // because -Δ_h u = f ⇒ Δ_h u + f = 0

                        local_res2 += r * r;

                        // error vs analytic solution at this grid point
                        double x = i * dx;
                        double y = j * dy;
                        double z = k_global * dz;
                        double exact = j_exact(x, y, z);
                        double diff = u[id] - exact;
                        local_err2 += diff * diff;

                        local_points += 1;
                    }
                }
            }

            double global_res2 = 0.0;
            double global_err2 = 0.0;
            long long global_points = 0;

            MPI_Allreduce(&local_res2, &global_res2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_err2, &global_err2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_points, &global_points, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

            double res_l2 = std::sqrt(global_res2 / static_cast<double>(global_points));
            double err_l2 = std::sqrt(global_err2 / static_cast<double>(global_points));

            if (rank == 0) {
                csv << iter << "," << res_l2 << "," << err_l2 << "\n";
                // optional: csv.flush();
            }
        }


        if (global_max_update < tol) {
            if (rank == 0) {
                std::cout << "Converged at iter " << iter
                          << ", global max update = " << global_max_update << "\n";
            }
            last_iter = iter;
            break;
        }

        if (iter == max_iters - 1) {
            last_iter = iter;
        }
    }

    // Compute global L2 error and max pointwise error vs analytic solution
    // over interior points only
    double local_err2 = 0.0;
    double local_max_err = 0.0;
    std::size_t local_points = 0;

    #pragma omp parallel for collapse(3) reduction(+:local_err2, local_points) reduction(max:local_max_err)
    for (int kk = 1; kk <= Nz_local; ++kk) {
        
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                int k_global = k_start + (kk - 1);

                bool is_interior =
                    (i > 0 && i < Nx - 1) &&
                    (j > 0 && j < Ny - 1) &&
                    (k_global > 0 && k_global < Nz_global - 1);

                if (!is_interior) continue;

                double x = i * dx;
                double y = j * dy;
                double z = k_global * dz;
                std::size_t id = idx(i, j, kk, Nx, Ny, Nz_local_with_halo);

                double exact = j_exact(x, y, z);
                double diff  = u[id] - exact;

                local_err2 += diff * diff;
                local_points += 1;

                if (std::fabs(diff) > local_max_err)
                    local_max_err = std::fabs(diff);
            }
        }
    }

    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    // Number of iterations actually performed (iter 0..last_iter inclusive)
    int num_iters = last_iter + 1;

    // Global interior points
    long long N_int = (long long)(Nx - 2) * (Ny - 2) * (Nz_global - 2);

    // Roofline model numbers
    double flops_per_point  = 11.0;
    double bytes_per_point  = 72.0;
    double total_flops      = flops_per_point * (double)N_int * (double)num_iters;
    double total_bytes      = bytes_per_point * (double)N_int * (double)num_iters;
    double intensity        = flops_per_point / bytes_per_point; // ≈ 0.15
    double gflops           = total_flops / (elapsed * 1e9);
    double bandwidth_GBps   = total_bytes / (elapsed * 1e9);

    double global_err2 = 0.0;
    double global_max_err = 0.0;
    long long global_points = 0;

    MPI_Allreduce(&local_err2, &global_err2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_points, &global_points, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        double l2_err = std::sqrt(global_err2 / static_cast<double>(global_points));
        std::cout << "Global L2 error vs analytic solution: " << l2_err << "\n";
        std::cout << "Global max pointwise error: " << global_max_err << "\n";


        std::cout << "Jacobi timing:\n";
        std::cout << "  iterations  = " << num_iters << "\n";
        std::cout << "  elapsed     = " << elapsed << " s\n";
        std::cout << "  GFLOP/s     = " << gflops << "\n";
        std::cout << "  GB/s        = " << bandwidth_GBps << "\n";
        std::cout << "  intensity   = " << intensity << " FLOPs/byte\n";


        std::ofstream roof("roofline_jacobi.csv");
        // Simple single-row file; you can run with different sizes/ranks and append.
        roof << "Nx,Ny,Nz_global,ranks,threads,num_iters,elapsed_s,"
                "total_flops,total_bytes,intensity,GFLOPs,GBps\n";
#ifdef _OPENMP
    int threads = omp_get_max_threads();
#else
    int threads = 1;
#endif
        roof << Nx << "," << Ny << "," << Nz_global << ","
            << size << "," << threads << ","
            << num_iters << "," << elapsed << ","
            << total_flops << "," << total_bytes << ","
            << intensity << "," << gflops << "," << bandwidth_GBps << "\n";
    }

    if (rank == 0 && csv.is_open()) {
        csv.close();
    }
    MPI_Finalize();
    return 0;
}
