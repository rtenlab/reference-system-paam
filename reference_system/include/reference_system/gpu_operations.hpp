#ifndef GPU_OPERATIONS_HPP
#define GPU_OPERATIONS_HPP

#include <functional>
#include <memory>
#include <thread>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h> //might need cmakelist update
#include <sys/time.h>
#include <unistd.h> //might need cmakelist update
#include <chrono>
#include <cinttypes>
#include <fstream>
#include <iostream>
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <boost/lexical_cast.hpp>

// void run_sgemm(float *A_h, float *B_h, float *C_h, size_t A_sz, size_t B_sz, size_t C_sz, unsigned int matArow, unsigned int matAcol, unsigned int matBrow, unsigned int matBcol);
// void run_hist(unsigned int* in_h, unsigned int* bins_h, unsigned int num_elements, unsigned int num_bins);

class gemm_operator
{
public:
    gemm_operator();
    ~gemm_operator();
    //void run_sgemm(float *A_h, float *B_h, float *C_h, size_t A_sz, size_t B_sz, size_t C_sz, unsigned int matArow, unsigned int matAcol, unsigned int matBrow, unsigned int matBcol);
    void init_sgemm();
    void gemm_wrapper();
    float *A_h, *B_h, *C_h;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    size_t A_sz, B_sz, C_sz;
};

void gemm_operator::init_sgemm()
{
    matArow = 500;
    matAcol = matBrow = 500;
    matBcol = 500;

    A_sz = matArow * matAcol;
    B_sz = matBrow * matBcol;
    C_sz = matArow * matBcol;

    A_h = (float *)malloc(sizeof(float) * A_sz);
    for (unsigned int i = 0; i < A_sz; i++)
    {
        A_h[i] = (rand() % 100) / 100.00;
    }

    B_h = (float *)malloc(sizeof(float) * B_sz);
    for (unsigned int i = 0; i < B_sz; i++)
    {
        B_h[i] = (rand() % 100) / 100.00;
    }

    C_h = (float *)malloc(sizeof(float) * C_sz);
}
gemm_operator::gemm_operator()
{
    init_sgemm();
}
gemm_operator::~gemm_operator()
{
    free(A_h);
    free(B_h);
    free(C_h);
}
void di_gemm(gemm_operator* gemm_op);
void gemm_operator::gemm_wrapper(void){
    di_gemm(this);
}
#endif // GPU_OPERATIONS_HPP