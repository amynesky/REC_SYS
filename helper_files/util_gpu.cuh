  #ifndef UTIL_GPU_H_
#define UTIL_GPU_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <iterator>
#include <string>
#include <algorithm>
#include <sys/time.h>
#include <boost/algorithm/string.hpp>
#include <boost/random.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>
#include <thrust/sort.h>

 
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <float.h> 
#endif

/* Using updated (v2) interfaces to cublas and cusparse */
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "helper_cuda.h"
#include "cusolverSp.h"
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
//#include "helper_cusolver.h"
#include "cusolverDn.h"


#include "util.h"
 

#ifndef CPU_ONLY  // GPU


#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *getCudaErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
            
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}
#endif

#ifdef CURAND_H_
// cuRAND API errors
static const char *curandGetErrorString(curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_SUCCESS:
            return "CURAND_STATUS_SUCCESS";

        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";

        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";

        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";

        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";

        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";

        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";

        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

#ifdef CUSOLVER_COMMON_H_
//cuSOLVER API errors
static const char *getCuSolverErrorEnum(cusolverStatus_t error)
{
   switch(error)
   {
       case CUSOLVER_STATUS_SUCCESS:
           return "CUSOLVER_STATUS_SUCCESS";
       case CUSOLVER_STATUS_NOT_INITIALIZED:
           return "CUSOLVER_STATUS_NOT_INITIALIZED";
       case CUSOLVER_STATUS_ALLOC_FAILED:
           return "CUSOLVER_STATUS_ALLOC_FAILED";
       case CUSOLVER_STATUS_INVALID_VALUE:
           return "CUSOLVER_STATUS_INVALID_VALUE";
       case CUSOLVER_STATUS_ARCH_MISMATCH:
           return "CUSOLVER_STATUS_ARCH_MISMATCH";
       case CUSOLVER_STATUS_MAPPING_ERROR:
           return "CUSOLVER_STATUS_MAPPING_ERROR";
       case CUSOLVER_STATUS_EXECUTION_FAILED:
           return "CUSOLVER_STATUS_EXECUTION_FAILED";
       case CUSOLVER_STATUS_INTERNAL_ERROR:
           return "CUSOLVER_STATUS_INTERNAL_ERROR";
       case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
           return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
       case CUSOLVER_STATUS_NOT_SUPPORTED :
           return "CUSOLVER_STATUS_NOT_SUPPORTED ";
       case CUSOLVER_STATUS_ZERO_PIVOT:
           return "CUSOLVER_STATUS_ZERO_PIVOT";
       case CUSOLVER_STATUS_INVALID_LICENSE:
           return "CUSOLVER_STATUS_INVALID_LICENSE";
    }

    return "<unknown>";

}
#endif

//#ifdef CUSPARSE_COMMON_H_
//cuSPARSE API errors
static const char *getCuSparseErrorEnum(cusparseStatus_t error)
{
   switch(error)
   {
       case CUSPARSE_STATUS_SUCCESS:
           return "CUSPARSE_STATUS_SUCCESS";
       case CUSPARSE_STATUS_NOT_INITIALIZED:
           return "CUSPARSE_STATUS_NOT_INITIALIZED";
       case CUSPARSE_STATUS_ALLOC_FAILED:
           return "CUSPARSE_STATUS_ALLOC_FAILED";
       case CUSPARSE_STATUS_INVALID_VALUE:
           return "CUSPARSE_STATUS_INVALID_VALUE";
       case CUSPARSE_STATUS_ARCH_MISMATCH:
           return "CUSPARSE_STATUS_ARCH_MISMATCH";
       case CUSPARSE_STATUS_MAPPING_ERROR:
           return "CUSPARSE_STATUS_MAPPING_ERROR";
       case CUSPARSE_STATUS_EXECUTION_FAILED:
           return "CUSPARSE_STATUS_EXECUTION_FAILED";
       case CUSPARSE_STATUS_INTERNAL_ERROR:
           return "CUSPARSE_STATUS_INTERNAL_ERROR";
       case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
           return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
       case CUSPARSE_STATUS_ZERO_PIVOT:
           return "CUSPARSE_STATUS_ZERO_PIVOT";
    }

    return "<unknown>";

}
//#endif

 // CUDA: various checks for different function calls.
 #define CUDA_CHECK(condition) \
   /* Code block avoids redefinition of cudaError_t error */ \
   do { \
     cudaError_t error = condition; \
     ABORT_IF_NEQ(error, cudaSuccess, " " << cudaGetErrorString(error)); \
   } while (0)
 
 #define CUBLAS_CHECK(condition) \
   do { \
     cublasStatus_t status = condition; \
     ABORT_IF_NEQ(status, CUBLAS_STATUS_SUCCESS, " " << getCudaErrorEnum(status)); \
   } while (0)
 
 #define CURAND_CHECK(condition) \
   do { \
     curandStatus_t status = condition; \
     ABORT_IF_NEQ(status, CURAND_STATUS_SUCCESS, " "<< curandGetErrorString(status)); \
   } while (0)

 #define CUSOLVER_CHECK(condition) \
   do { \
     cusolverStatus_t status = condition; \
     ABORT_IF_NEQ(status, CUSOLVER_STATUS_SUCCESS, " "<<getCuSolverErrorEnum(status)); \
   } while (0)

 #define CUSPARSE_CHECK(condition) \
   do { \
     cusparseStatus_t status = condition; \
     ABORT_IF_NEQ(status, CUSPARSE_STATUS_SUCCESS, " "<<getCuSparseErrorEnum(status)); \
   } while (0)
 
 // CUDA: grid stride looping
 #define CUDA_KERNEL_LOOP(i, n) \
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

template < typename Dtype>
bool gpu_isBad(const Dtype* A, long long int size);

//============================================================================================
// Device information utilities
//============================================================================================

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */
int getDeviceVersion(void);
size_t getDeviceMemory(void);
#if defined(__cplusplus)
}
#endif /* __cplusplus */


//============================================================================================
// Status Check
//============================================================================================


/* For debugging */
void printCusparseStatus(cusparseStatus_t status);

/* For debugging */
void printCublasStatus(cublasStatus_t status);

void checkCublasBatchedStatus(cublasStatus_t * batchStatus, int num_blocks);

//============================================================================================
// Prints and Saves
//============================================================================================


template < typename Dtype>
void print_gpu_mtx_entries(const Dtype* array, int lda, int sda, bool transpose = 0);

template < typename Dtype>
void print_gpu_array_entries(const Dtype* array, int count);

template <typename Dtype>
void save_device_mtx_to_file(const Dtype* A_dev, int lda, int sda, std::string title, bool transpose = false);

template <typename Dtype>
void save_device_array_to_file(const Dtype* A_dev, int count, std::string title);

template <typename Dtype>
void append_device_array_to_file(const Dtype* A_dev, int count, std::string title);

template <typename Dtype>
void save_device_arrays_side_by_side_to_file(const Dtype* A_dev, const Dtype* B_dev, int count, std::string title);


void save_device_arrays_side_by_side_to_file(const int* A_dev, const float* B_dev, int count, std::string title);

template <typename Dtype>
void save_device_arrays_side_by_side_to_file(const Dtype* A_dev, const Dtype* B_dev, const Dtype* C_dev, int count, std::string title);

void save_device_arrays_side_by_side_to_file(const int* A_dev, const int* B_dev, const float* C_dev, int count, std::string title);

template <typename Dtype>
void save_device_arrays_side_by_side_to_file(const Dtype* A_dev, const Dtype* B_dev, 
                                             const Dtype* C_dev, const Dtype* D_dev, 
                                             int count, std::string title);

//============================================================================================
// Add and Scale
//============================================================================================

template <typename Dtype>
void gpu_add_scalar(const long long int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void gpu_scale(cublasHandle_t dn_handle, const long long int n, const Dtype alpha, Dtype *x);

template <typename Dtype>
void gpu_scale(cublasHandle_t dn_handle, const long long int n, const Dtype alpha, const Dtype *x, Dtype *y);

//============================================================================================
// CURAND INITIALIZATION
//============================================================================================

template < typename Dtype>
void gpu_get_rand_bools(const long long int n,  Dtype* x, float probability_of_success);

void gpu_get_rand_groups(const long long int n,  int* x, float* probability_of_success, const int num_groups);

template < typename Dtype>
void gpu_reverse_bools(const long long int n,  Dtype* x);

template <typename Dtype>
void gpu_rng_uniform(cublasHandle_t handle, const long long int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void gpu_rng_gaussian(const long long int n, const Dtype mu, const Dtype sigma,
                            Dtype* r);

template <typename Dtype>
void gpu_shuffle_array(cublasHandle_t handle, const long long int n,  Dtype* x);

template <typename Dtype>
void gpu_sort_index_by_max(cublasHandle_t handle, const long long int n,  Dtype* x, Dtype* indicies);

template <typename Dtype>
void gpu_sort_index_by_max(cublasHandle_t handle, const long long int rows, const long long int cols,  
                            Dtype* x, Dtype* indicies);


//============================================================================================
// template wrappers for cuda functions and classic math
//============================================================================================

template<typename Dtype>
void gpu_set_all(Dtype* x, const long long int n, const Dtype alpha);

void gpu_set_as_index(int* x, const long long int n);

template<typename Dtype>
void gpu_set_as_index(Dtype* x, const long long int rows, const long long int cols);

template<typename Dtype>
void gpu_set_as_index_host(Dtype* x_host, const long long int rows, const long long int cols);

template<typename Dtype>
Dtype gpu_abs_max(const long long int n, const Dtype* x); 

template<typename Dtype>
Dtype gpu_min(const long long int n, const Dtype* x); 

template<typename Dtype>
Dtype gpu_max(const long long int n, const Dtype* x); 

template<typename Dtype>
Dtype gpu_range(const long long int n, const Dtype* x); 

template<typename Dtype>
Dtype gpu_sum(const long long int n,  const Dtype* x);

template <typename Dtype>
Dtype gpu_norm(cublasHandle_t dn_handle, const long long int n, const Dtype* x);

template<typename Dtype>
Dtype gpu_expected_value(const long long int n,  const Dtype* x);

template<typename Dtype>
Dtype gpu_expected_abs_value(const long long int n,  const Dtype* x);

template<typename Dtype>
Dtype gpu_variance(const long long int n, const Dtype* x);

template<typename Dtype>
Dtype gpu_expected_dist_two_guassian(cublasHandle_t dn_handle, const long long int n);

template<typename Dtype>
Dtype gpu_sum_of_squares(const long long int n, const Dtype* x);

template <typename Dtype>
void gpu_dot(cublasHandle_t dn_handle, const long long int n, const Dtype* x, const Dtype* y, Dtype* out);

template <typename Dtype>
void gpu_asum(cublasHandle_t dn_handle, const long long int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void swap_ordering(const long long int rows, const long long int cols, Dtype *A, const bool row_major_ordering);


template < typename Dtype>
cublasStatus_t transpose(cublasHandle_t handle,
                         long long int m, long long int n, const Dtype *A, Dtype *result);

template < typename Dtype>
void transpose_in_place(cublasHandle_t handle, const long long int lda, const long long int sda, Dtype *A);

template <typename Dtype>
void gpu_gemv(cublasHandle_t dn_handle, const bool TransA, const int M, const int N,
                  const Dtype alpha, const Dtype* A, const Dtype* x, const int inc_x, 
                  const Dtype beta, Dtype* y, const int inc_y);

template <typename Dtype>
void gpu_hadamard(const long long int n, const Dtype* A, Dtype* B );

template <typename Dtype>
void gpu_axpy(cublasHandle_t dn_handle, const long long int N, const Dtype alpha, const Dtype* X,
                    Dtype* Y);

template <typename Dtype>
void gpu_axpby(cublasHandle_t dn_handle, const long long int N, const Dtype alpha, const Dtype* X,
                     const Dtype beta, Dtype* Y);

template <typename Dtype>
void gpu_axpby(cublasHandle_t dn_handle, const long long int N, const Dtype alpha, const Dtype* X,
                     const Dtype beta, const Dtype* Y, Dtype* result);

template<typename Dtype>
Dtype gpu_sum_of_squares_of_diff(cublasHandle_t dn_handle, const long long int n, const Dtype* x, Dtype* y);

template < typename Dtype>
cublasStatus_t cublasXgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
                           int m, int n, int k, Dtype *alpha, const Dtype *A, int lda,
                           Dtype *B, int ldb, Dtype *beta, Dtype *C, int ldc);

template <typename Dtype>
void gpu_gemm(cublasHandle_t dn_handle, const bool TransA,
              const bool TransB, const int M, const int N, const int K,
              const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
              Dtype* C);

template <typename Dtype>
void gpu_noisey_gemm(cublasHandle_t dn_handle, const bool TransA,
                    const bool TransB, const int M, const int N, const int K,
                    const Dtype alpha, const Dtype* A, const Dtype range, const Dtype beta,
                    Dtype* C);


template < typename Dtype>
cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                                Dtype *alpha, const Dtype *Aarray[], int lda,
                                const Dtype *Barray[], int ldb, Dtype *beta,
                                Dtype *Carray[], int ldc, int batchCount);

template < typename Dtype>
void cublasXSparsegemm(cusparseHandle_t handle, bool TransA, bool TransB,
                                    int m, int n, int k, int nnz, Dtype *alpha, const cusparseMatDescr_t descrA, 
                                    const Dtype *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                    Dtype *B, int ldb, Dtype *beta, Dtype *C, int ldc);


template < typename Dtype>
void cublasCSCSparseXgemm(cusparseHandle_t handle,int m, int n, int k,
                       int nnz, Dtype *alpha, const Dtype *A, int lda,
                       const Dtype *cscValB, const int *cscColPtrB, const int *cscRowIndB, const Dtype *beta, 
                       Dtype *C, int ldc);



//============================================================================================
// me-made math
//============================================================================================




int count_testing_data(const int nnz_data,  const int* coo_data, bool *testing_bools);

void count_each_group_from_coo(const int num_groups, const int* group_indicies, 
                               const int nnz_data,  const int* coo_data, int* group_sizes);

int count_nnz_bools(const int n,  const bool *testing_bools) ;

void count_each_group(const int n,  const int* group_indicies, int* group_sizes, const int num_groups);

void gpu_split_data(const int* csr_format_ratingsMtx_userID_dev,
                        const int* coo_format_ratingsMtx_itemID_dev,
                        const float* coo_format_ratingsMtx_rating_dev, 
                        const int ratings_rows, const bool *testing_bools,
                        int* csr_format_ratingsMtx_userID_dev_training,
                        int* coo_format_ratingsMtx_itemID_dev_training,
                        float* coo_format_ratingsMtx_rating_dev_training,
                        const int ratings_rows_training,
                        int* csr_format_ratingsMtx_userID_dev_testing,
                        int* coo_format_ratingsMtx_itemID_dev_testing,
                        float* coo_format_ratingsMtx_rating_dev_testing,
                        const int ratings_rows_testing);

void gpu_split_data(const int* csr_format_ratingsMtx_userID_dev,
                        const int* coo_format_ratingsMtx_itemID_dev,
                        const float* coo_format_ratingsMtx_rating_dev, 
                        const int ratings_rows, const int *group_indicies,
                        int** csr_format_ratingsMtx_userID_dev_by_group,
                        int** coo_format_ratingsMtx_itemID_dev_by_group,
                        float** coo_format_ratingsMtx_rating_dev_by_group,
                        const int* ratings_rows_by_group);



void collect_user_means(float* user_means,float* user_var, const long long int ratings_rows,
                        const int* csr_format_ratingsMtx_userID_dev,
                        const float* coo_format_ratingsMtx_rating_dev);

void collect_user_means(float* user_means_training,float* user_var_training, const long long int ratings_rows_training,
                        const int* csr_format_ratingsMtx_userID_dev_training,
                        const float* coo_format_ratingsMtx_rating_dev_training,
                        float* user_means_testing,float* user_var_testing, const long long int ratings_rows_testing,
                        const int* csr_format_ratingsMtx_userID_dev_testing,
                        const float* coo_format_ratingsMtx_rating_dev_testing,
                        float* user_means_GA,float* user_var_GA, const long long int ratings_rows_GA,
                        const int* csr_format_ratingsMtx_userID_dev_GA,
                        const float* coo_format_ratingsMtx_rating_dev_GA);


void gpu_fill_training_mtx(const long long int ratings_rows_training, 
                        const long long int ratings_cols_training, 
                        const bool row_major_ordering,
                        const int* csr_format_ratingsMtx_userID_dev_training,
                        const int* coo_format_ratingsMtx_itemID_dev_training,
                        const float* coo_format_ratingsMtx_rating_dev_training,
                        float* full_training_ratings_mtx);

void gpu_supplement_training_mtx_with_content_based(const long long int ratings_rows_training, 
                                                    const long long int ratings_cols_training, 
                                                    const bool row_major_ordering,
                                                    const int* csr_format_ratingsMtx_userID_dev_training,
                                                    const int* coo_format_ratingsMtx_itemID_dev_training,
                                                    const float* coo_format_ratingsMtx_rating_dev_training,
                                                    float* full_training_ratings_mtx,
                                                    const int* csr_format_keyWordMtx_itemID_dev,
                                                    const int* coo_format_keyWordMtx_keyWord_dev);

void center_ratings(const float* user_means, const float* user_var, 
                    const int ratings_rows, const int num_entries,
                    const int* csr_format_ratingsMtx_userID_dev,
                    const float* coo_format_ratingsMtx_rating_dev,
                    float* coo_format_ratingsMtx_row_centered_rating_dev,
                    const float val_when_var_is_zero);

void center_rows(const long long int rows, const long long int cols, 
                 float* X, const float val_when_var_is_zero, float* user_means,  float* user_var);

void get_cosine_similarity(const long long int ratings_rows, const long long int num_entries,
                          const int* csr_format_ratingsMtx_userID_dev,
                          const int* coo_format_ratingsMtx_itemID_dev,
                          const float* coo_format_ratingsMtx_rating_dev,
                          float* cosine_similarity);

void get_cosine_similarity_host(const long long int ratings_rows, 
    const int* csr_format_ratingsMtx_userID_dev,
    const int* coo_format_ratingsMtx_itemID_dev,
    const float* coo_format_ratingsMtx_rating_dev,
    float* cosine_similarity_host);

void get_cosine_similarity_host_experiment(const long long int ratings_rows, 
    const int* csr_format_ratingsMtx_userID_dev,
    const int* coo_format_ratingsMtx_itemID_dev,
    const float* coo_format_ratingsMtx_rating_dev,
    float* cosine_similarity_host);

void gpu_copy(const int M, const int N,  const float* x, 
                    const float* row_indicies, float* y);

void gpu_copy(const int N,  const int* host_x, float* dev_x);

template <typename Dtype>
void gpu_permute(Dtype* A, const int* P, int rows, int cols, bool permute_rows);

void gpu_shuffle_mtx_rows_or_cols(cublasHandle_t dn_handle, const long long int M, const long long int N,  
                                  bool row_major_ordering, float* x, bool shuffle_rows);

void gpu_normalize_mtx_rows_or_cols(const long long int M, const long long int N,  
                                  bool row_major_ordering, float* x, bool normalize_rows);

template <typename Dtype>
void gpu_div_US_in_SVD(const long long int m, const long long int num_latent_factors,
                        Dtype* U, const Dtype* S, const bool right_divide_by_S);

template <typename Dtype>
void gpu_mult_US_in_SVD(const long long int m, const long long int num_latent_factors,
                        Dtype* U, const Dtype* S, const bool right_multiply_by_S);

template <typename Dtype>
void get_num_latent_factors(cublasHandle_t dn_handle, const long long int m, Dtype* S, 
                                  long long int* num_latent_factors, const Dtype percent);

template <typename Dtype>
void preserve_first_m_rows(const long long int old_lda, const long long int new_lda, const long long int sda,
                                  Dtype* V) ;

template <typename Dtype>
void gpu_orthogonal_decomp(cublasHandle_t handle, cusolverDnHandle_t dn_solver_handle,
                          const long long int m, const long long int n, 
                          long long int* num_latent_factors, const Dtype percent,
                          Dtype* A, Dtype* U, Dtype* V); 

// solve A*x = b by LU with partial pivoting
template <typename Dtype>
int linearSolverLU(
    cusolverDnHandle_t handle,
    int n,
    const Dtype *Acopy,
    int lda,
    const Dtype *b,
    Dtype *x);

// solve A*x = b by QR
template <typename Dtype>
int linearSolverQR(
    cublasHandle_t cublasHandle,
    cusolverDnHandle_t handle,
    int n,
    const Dtype *Acopy,
    int lda,
    const Dtype *b,
    Dtype *x);

int gpu_get_num_entries_in_rows(const int first_row, const int last_row, const int* csr);

int gpu_get_first_coo_index(const int first_row, const int* csr);

void sparse_error(const int rows, const int cols, const float* dense_mtx_A, 
                  const int* csr_rows_B, const int* coo_cols_B,
                  const float* coo_entries_B, float* coo_errors, const int num_sparse_entries,
                  float* coo_A);

template <typename Dtype>
void gpu_spXdense_MMM_check(const cublasHandle_t dn_handle, const bool TransA, const bool TransB, 
                              const int m, const int n, const int k, const int first_ind,
                              const Dtype alpha,
                              const Dtype *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                              const Dtype *B, const Dtype beta, Dtype *C);

template <typename Dtype>
void gpu_spXdense_MMM(const cusparseHandle_t handle, const bool TransA, const bool TransB, 
                              const int m, const int n, const int k, const int nnz, const int first_ind,
                              const Dtype *alpha, const cusparseMatDescr_t descrA, 
                              const Dtype *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                              const Dtype *B, const int ldb, const Dtype *beta, Dtype *C, const int ldc, bool Debug);

template <typename Dtype>
void gpu_R_error(const cublasHandle_t dn_handle, const cusparseHandle_t sp_handle, const cusparseMatDescr_t sp_descr,
                        const int batch_size_testing, const int batch_size_CU, const int num_latent_factors, const int ratings_cols,
                        const int nnz, const int first_coo_ind, const bool compress, 
                        Dtype* testing_entries, Dtype* coo_testing_errors, const Dtype testing_fraction,
                        const Dtype *coo_format_ratingsMtx_rating_dev_testing, 
                        const int *csr_format_ratingsMtx_userID_dev_testing_batch, 
                        const int *coo_format_ratingsMtx_itemID_dev_testing,
                        const Dtype *V, Dtype *U_testing, Dtype *R_testing, 
                        std::string name, float training_rate, float regularization_constant);



template <typename Dtype>
void gpu_R_error_training(const cublasHandle_t dn_handle, const cusparseHandle_t sp_handle, const cusparseMatDescr_t sp_descr,
                                const int batch_size_training, const int batch_size_CU, const int num_latent_factors, const int ratings_cols,
                                const int nnz, const int first_coo_ind, const bool compress, Dtype* coo_errors, 
                                const Dtype *coo_format_ratingsMtx_rating_dev_training, 
                                const int *csr_format_ratingsMtx_userID_dev_training_batch, 
                                const int *coo_format_ratingsMtx_itemID_dev_training,
                                const Dtype *V, Dtype *U_training, Dtype *R_training, 
                                Dtype training_rate, Dtype regularization_constant);


template <typename Dtype>
void gpu_R_error_testing(const cublasHandle_t dn_handle, const cusparseHandle_t sp_handle, const cusparseMatDescr_t sp_descr,
                        const int batch_size_testing, const int batch_size_CU, const int num_latent_factors, const int ratings_cols,
                        const int nnz, const int first_coo_ind, const bool compress, 
                        Dtype* testing_entries, Dtype* coo_testing_errors, const Dtype testing_fraction,
                        const Dtype *coo_format_ratingsMtx_rating_dev_testing, 
                        const int *csr_format_ratingsMtx_userID_dev_testing_batch, 
                        const int *coo_format_ratingsMtx_itemID_dev_testing,
                        const Dtype *V, Dtype *U_testing, Dtype *R_testing, 
                        float training_rate, float regularization_constant,
                        float* testing_error_on_training_entries, float* testing_error_on_testing_entries, 
                        long long int* total_iterations);




#endif  // !CPU_ONLY


#endif //UTIL_GPU_H_