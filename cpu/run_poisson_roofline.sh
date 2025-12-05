#!/bin/bash
set -euo pipefail

# 1) Build STREAM if needed
if [ ! -x ./stream ]; then
    echo "[INFO] Building STREAM benchmark..."
    if [ ! -f stream.c ]; then
        wget -q https://www.cs.virginia.edu/stream/FTP/Code/stream.c
    fi
    gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=80000000 stream.c -o stream
fi

# 2) Measure peak memory bandwidth with STREAM (Triad)
CORES=$(nproc)
export OMP_NUM_THREADS=$CORES
echo "[INFO] Running STREAM with OMP_NUM_THREADS=${OMP_NUM_THREADS} ..."
STREAM_OUT=$(./stream | tee stream_last.log)

TRIAD_MBPS=$(echo "$STREAM_OUT" | awk '/Triad:/ {print $2}')
if [ -z "$TRIAD_MBPS" ]; then
    echo "[ERROR] Could not parse Triad bandwidth from STREAM output."
    exit 1
fi

# Convert MB/s -> GB/s (decimal GB)
B_PEAK_GBPS=$(echo "$TRIAD_MBPS / 1000.0" | bc -l)
echo "[INFO] Measured peak memory bandwidth (Triad): ${B_PEAK_GBPS} GB/s"

# 3) Estimate peak compute (GFLOP/s) from cpu MHz and cores (16 DP FLOPs/cycle)
CPU_MHZ=$(grep "cpu MHz" /proc/cpuinfo | head -1 | awk '{print $4}')
if [ -z "$CPU_MHZ" ]; then
    echo "[ERROR] Could not read cpu MHz from /proc/cpuinfo."
    exit 1
fi

FLOPS_PER_CYCLE=16  # DP: 256-bit AVX2, 2 FMA units
CPU_GHZ=$(echo "$CPU_MHZ / 1000.0" | bc -l)

PEAK_GFLOPS=$(echo "$CORES * $CPU_GHZ * $FLOPS_PER_CYCLE" | bc -l)

echo "[INFO] Estimated peak compute: ${PEAK_GFLOPS} GFLOP/s"
echo

# 4) Run the MPI Poisson solver
echo "[INFO] Running MPI Poisson solver ..."
if [ -n "${SLURM_NTASKS-}" ]; then
    # Inside a Slurm allocation: use prun
    prun ./main
else
    # Fallback: local mpirun
    mpirun -np 1 ./main
fi

echo
echo "[INFO] Poisson solver finished. Generating roofline plot with peak GFLOPS=${PEAK_GFLOPS} and peak BW=${B_PEAK_GBPS} ..."

# 5) Call Python roofline script with measured peaks
python3 plot_roofline.py \
    --peak-gflops "${PEAK_GFLOPS}" \
    --peak-bw "${B_PEAK_GBPS}"

echo "[INFO] Roofline plot saved as roofline_poisson.png"
