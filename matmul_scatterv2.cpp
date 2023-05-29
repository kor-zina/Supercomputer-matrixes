#include <iostream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include <random>
#include "vectors_and_matrices/array_types.hpp"

#include "mpi.h"

using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;

void fill_random(vec<double> x, double xmin, double xmax, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(xmin, xmax);
    for (ptrdiff_t i = 0; i < x.length(); i++)
    {
        x(i) = dist(rng);
    }
}

void fill_random(matrix<double> x, double dispersion, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0, dispersion);
    for (ptrdiff_t i = 0; i < x.length(); i++)
    {
        x(i) = dist(rng);
    }
}

matrix<double> matmul_ikj(int *sendcounts, int *displs, matrix<double> a, matrix<double> b, MPI_Comm comm)
{
    int rowa = a.nrows();
    int rowb = b.nrows();
    int cola = a.ncols();
    int colb = b.ncols();
    ptrdiff_t i, j, k;

    int myrank, comm_size;

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &myrank);

    matrix<double> c(rowa, colb);

    for (i = 0; i < c.length(); i++)
    {
        c(i) = 0;
    }

    for (i = 0; i < rowa; i++)
    {
        for (k = 0; k < rowb; k++)
        {
            double a_ik = a(i, k);
            for (j = 0; j < colb; j++)
            {
                c(i, j) += a_ik * b(k, j);
            }
        }
    }
    return c;
}

// read an integer number from stdin into `n`
void read_integer(int *n, int rank, MPI_Comm comm)
{
    if (rank == 0)
    {
        std::cin >> *n;
    }

    MPI_Bcast(n, 1, MPI_INT, 0, comm);
}

// scatter matrix `source` over processes in communicator `comm` from `root`
void scatter_matrix(matrix<double> source, matrix<double> dest, int root, MPI_Comm comm)
{
    int n = dest.ncols(), m_each = dest.nrows();

    double *src_ptr = source.raw_ptr(), *dest_ptr = dest.raw_ptr();
    MPI_Scatter(src_ptr, m_each * n, MPI_DOUBLE, dest_ptr, m_each * n, MPI_DOUBLE, root, comm);
}

void scatterv_matrix(int *sendcounts, int *displs, matrix<double> source, matrix<double> dest, int root, MPI_Comm comm)
{
    double *src_ptr = source.raw_ptr(), *dest_ptr = dest.raw_ptr();
    int myrank;
    MPI_Comm_rank(comm, &myrank);
    MPI_Scatterv(src_ptr, sendcounts, displs, MPI_DOUBLE, dest_ptr, sendcounts[myrank], MPI_DOUBLE, root, comm);
}

int main(int argc, char *argv[])
{
    int n;

    int myrank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    read_integer(&n, myrank, MPI_COMM_WORLD);

    int nrows_base = n / world_size, nrows_rem = n % world_size;
    int displs[world_size], sendcounts[world_size];
    displs[0] = 0;
    for (int i = 0; i < world_size - 1; i++)
    {
        sendcounts[i] = nrows_base * n;
        if (i < nrows_rem)
            sendcounts[i] += n;
        displs[i + 1] = displs[i] + sendcounts[i];
    }
    sendcounts[world_size - 1] = n * nrows_base;

    matrix<double> a(sendcounts[myrank] / n, n), b(n, n);
    // generate matrix on rank 0 (for simplicity)
    if (myrank == 0)
    {
        matrix<double> a_all(n, n);
        fill_random(a_all, 1.0, 9876);
        scatterv_matrix(sendcounts, displs, a_all, a, 0, MPI_COMM_WORLD);

        fill_random(b, 1.0, 9877);
        MPI_Bcast(b.raw_ptr(), b.length(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        scatterv_matrix(sendcounts, displs, a, a, 0, MPI_COMM_WORLD);
        MPI_Bcast(b.raw_ptr(), b.length(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double t0 = MPI_Wtime();

    matrix<double> c = matmul_ikj(sendcounts, displs, a, b, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    if (myrank == world_size - 1)
    {
        std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
                  << "Timing: " << t1 - t0 << " sec\n"
                  << "Answer[n, n] = " << c(c.length() - 1)
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
