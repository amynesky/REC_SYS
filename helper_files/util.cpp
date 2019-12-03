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
#include <boost/random/uniform_real.hpp>
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>
#include <thrust/sort.h>
#include <bitset>
#ifdef _OPENMP
#include <omp.h>
#endif


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
#include "util_gpu.cuh"

using namespace Eigen;

typedef boost::minstd_rand base_generator_type;


bool too_big(const long long int li){
  if (li >= (long long int)INT_MIN && li <= (long long int)INT_MAX) {
    return false;
  } else {
    return true;
  }
}

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

//============================================================================================
// manipulate memory
//============================================================================================


template <typename Dtype>
void host_copy(const long long int N, const Dtype* X, Dtype* Y) 
{
  if (X != Y) {
    memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
  }

}

template void host_copy<int>(const long long int N, const int* X, int* Y);
template void host_copy<unsigned int>(const long long int N, const unsigned int* X, unsigned int* Y);
template void host_copy<float>(const long long int N, const float* X, float* Y);
template void host_copy<double>(const long long int N, const double* X, double* Y);

template <typename Dtype>
void copy_device_mtx_into_host_submtx(const int M, const int N, const Dtype* X, Dtype* Y, const int inc_Y)
{
  bool Debug = false;
  std::string blank = "";
  if(Debug){
    LOG("copy_device_mtx_into_host_submtx called");
    LOG("X is on the device "<<M<<" by "<<N<<" in memory = "<<M * N);
    LOG("Y is on host "<<inc_Y<<" by "<<N<<" in memory = "<<inc_Y * N);
    LOG("M : "<< M);
    LOG("N : "<< N);
    LOG("inc_Y : "<< inc_Y);
    print_gpu_mtx_entries<Dtype>(X, M, N, "X", 0, strPreamble(blank));
  }

  if(inc_Y == M){
    checkCudaErrors(cudaMemcpy(Y,  X,  M * N * sizeof(Dtype), cudaMemcpyDeviceToHost));
    if(Debug){
      checkCudaErrors(cudaDeviceSynchronize());
      print_host_mtx<Dtype>(Y, inc_Y, N, "Y", 0, strPreamble(blank));
      checkCudaErrors(cudaDeviceSynchronize());
      print_gpu_mtx_entries<Dtype>(X, M, N, "X", 0, strPreamble(blank));
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }else{
    int nthreads = 1;
    #ifdef _OPENMP
      int nProcessors = omp_get_max_threads();
      nthreads = (int)std::min(nProcessors, N);
      if(Debug){
        LOG("copy_device_mtx_into_host_submtx done");
        LOG("nthreads : "<< nthreads);
      }
      omp_set_num_threads(nthreads);
      omp_lock_t printlock;
      omp_init_lock(&printlock);
    #endif
    #pragma omp parallel shared(nthreads)
    {
      int th_id = 0;
      #ifdef _OPENMP
        th_id = omp_get_thread_num();
      #endif
      for(long long int j = (long long int)th_id; j < (long long int)N; j += (long long int)nthreads){
      //for(long long int j = (long long int)0; j < (long long int)N; j+=(long long int)1){
        long long int y_skip = j * (long long int)inc_Y;
        long long int x_skip = j * (long long int)M;
        if(Debug){
          omp_set_lock(&printlock);
          LOG("th_id : "<< th_id);
          LOG("j : "<< j);
          LOG("N : "<< N);
          LOG("y_skip : "<< y_skip);
          LOG("y_skip  + M: "<< y_skip + M <<" <= "<<inc_Y * N);
          LOG("x_skip : "<< x_skip);
          LOG("x_skip  + M: "<< x_skip + M <<" <= "<<M * N);
        }
        Dtype* Y_temp = Y + y_skip;
        const Dtype* X_temp = X + x_skip;
        checkCudaErrors(cudaMemcpy(Y_temp,  X_temp,  M * sizeof(Dtype), cudaMemcpyDeviceToHost));
        if(Debug){
          checkCudaErrors(cudaDeviceSynchronize());
          print_host_mtx<Dtype>(Y, inc_Y, N, "Y", 0, strPreamble(blank));
          omp_unset_lock(&printlock);
        }
      }
    }
  }
  if(Debug){
    LOG("copy_device_mtx_into_host_submtx done");
  }
}

template void copy_device_mtx_into_host_submtx<float>(const int M, const int N, const float* X, float* Y, const int inc_Y);

//============================================================================================
// math functions
//============================================================================================

template <>
int cpu_asum<int>(const int n, const int* x) {
  int s = 0;
  for (int k = 0; k < n; ++k){
    s += abs(x[k]);
    //LOG(INFO) << x[k];
  };
  return s;
}

template <>
float cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
float cpu_sum<float>(const long long int n,  const float* x) 
{
  if(too_big(n) ) {ABORT_IF_EQ(0, 1,"Long long long int too big");}
  float sum = thrust::reduce(thrust::host, x, x + n, (float)0., thrust::plus<float>());
  return sum;
}

// square<Dtype> computes the square of a number f(x) -> x*x 
template <typename Dtype> 
struct square_ 
{ 
  Dtype operator()(const Dtype& x) const {
   return x * x; 
 } 
};

template <>
float cpu_sum_of_squares<float>(const long long int n, const float* x) {
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  square_<float> unary_op; 
  thrust::plus<float> binary_op; 
  float init = 0; 
  // compute norm 
  float s = thrust::transform_reduce(thrust::host, x, x+n, unary_op, init, binary_op) ; 
  return s/* /(float)n*/;
}

template <>
void cpu_scal<float>(const long long int N, const float alpha, float *X) {
  bool Debug = false;
  if(Debug) {
    LOG("INT_MAX : "<<INT_MAX);
    LOG("N : "<<N);
    LOG("(int)N : "<<(int)N);
    LOG("alpha : "<<alpha);
  }
  if(too_big(N) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  //cblas_sscal((int)N, alpha, X, 1);
  for (long long int k = (long long int)0; k < N; ++k){
    X[k] = X[k] * alpha;
  };
}

template <>
void cpu_scal<double>(const long long int N, const double alpha, double *X) {
  if(too_big(N) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  cblas_dscal((int)N, alpha, X, 1);
}


int gcd(int a, int b) 
{
  return b == 0 ? a : gcd(b, a % b);
}


template <typename Dtype>
void cpu_permute(Dtype* A, const int* P, const long long int rows, const long long int cols, bool permute_rows) 
{
  bool Debug = false;
  // A is rows by cols stored in col major ordering
  // this function permutes in place
  if(Debug) LOG("cpu_permute called");
  if(permute_rows){
    if(Debug) LOG("permute_rows is true") ;
    //pvt is an array length rows
    int nthreads = 1;
    #ifdef _OPENMP
      int nProcessors = omp_get_max_threads();
      nthreads = (int)std::min((long long int)nProcessors, cols);
      omp_set_num_threads(nthreads);
    #endif
    #pragma omp parallel shared(nthreads)
    {
      int th_id = 0;
      #ifdef _OPENMP
        th_id = omp_get_thread_num();
      #endif
      for(long long int j = (long long int)th_id; j < cols; j += (long long int)nthreads){
      //for(long long int  j = 0; j <cols; j+=(long long int)1) {
        long long int  ind=(long long int)0;
        Dtype temp = (Dtype)0;

        for(long long int i = (long long int)0; i < rows - (long long int)1; i+=(long long int)1){
          // get next index
          ind = P[i];
          while(ind<i)
            ind = P[ind];

          // swap elements in array
          temp = A[i + j * rows];
          A[i + j * rows] = A[ind + j * rows];
          A[ind + j * rows] = temp;
        };
      };
    }
  } else{
    if(Debug) LOG("permute_rows is false") ;
    //pvt is an array length cols
    int nthreads = 1;
    #ifdef _OPENMP
      int nProcessors = omp_get_max_threads();
      nthreads = (int)std::min((long long int)nProcessors, rows);
      omp_set_num_threads(nthreads);
    #endif
    #pragma omp parallel shared(nthreads)
    {
      int th_id = 0;
      #ifdef _OPENMP
        th_id = omp_get_thread_num();
      #endif
      for(long long int i = (long long int)th_id; i < rows; i += (long long int)nthreads){
      //for(long long int i = (long long int)0; i<rows; i+=(long long int)1) {
        long long int ind = (long long int)0;
        Dtype temp = (Dtype)0.0;

        for(long long int j=(long long int)0; j < cols - (long long int)1; j+=(long long int)1){
          // get next index
          ind = P[j];
          while(ind<j)
            ind = P[ind];

          // swap elements in array
          if(Debug) LOG("i + j * rows: "<<i<<" + "<<j<<" * "<<rows<<" = "<<i + j * rows) ;
          if(Debug) LOG("i + ind * rows: "<<i<<" + "<<ind<<" * "<<rows<<" = "<<i + ind * rows) ;
          temp = A[i + j * rows];
          A[i + j * rows] = A[i + ind * rows];
          A[i + ind * rows] = temp;
        };
      };
    }
  }
}
template void cpu_permute<float>(float* A, const int* P, const long long int rows, const long long int cols, bool permute_rows) ;
template void cpu_permute<double>(double* a, const int* pvt,const long long int  rows,const long long int  cols, bool direction);


// Non-square matrix transpose of matrix of size r x c and base address A 
template <>
void MatrixInplaceTranspose<float>(float *A, int r, int c, bool row_major_ordering) 
{ //ABORT_IF_NEQ(0, 1, "Function Not Supported Yet");

  // int HASH_SIZE = 128;

  // int size = r*c - 1; 
  // float t; // holds element to be replaced, eventually becomes next element to move 
  // int next; // location of 't' to be moved 
  // int cycleBegin; // holds start of cycle 
  // int i; // iterator 
  // bitset<HASH_SIZE> b; // hash to mark moved elements 

  // b.reset(); 
  // b[0] = b[size] = 1; 
  // i = 1; // Note that A[0] and A[size-1] won't move 
  // while (i < size) 
  // { 
  //     cycleBegin = i; 
  //     t = A[i]; 
  //     do
  //     { 
  //         // Input matrix [r x c] 
  //         // Output matrix  
  //         // i_new = (i*r)%(N-1) 
  //         next = (i*r)%size; 
  //         swap(A[next], t); 
  //         b[i] = 1; 
  //         i = next; 
  //     } 
  //     while (i != cycleBegin); 

  //     // Get Next Move (what about querying random location?) 
  //     for (i = 1; i < size && b[i]; i++) 
  //         ; 
  //     std::cout << endl; 
  // } 
  if(row_major_ordering){
    mkl_simatcopy('R', 'T', r, c, (float)1.0, A, c, r);
  }else{
    mkl_simatcopy('C', 'T', r, c, (float)1.0, A, r, c);
  }
} 

template <>
void cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  /*
  y := a*x + b*y
  where:

  a and b are scalars

  x and y are vectors each with n elements.
  */
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  /*
  y := a*x + b*y
  where:

  a and b are scalars

  x and y are vectors each with n elements.
  */
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <typename Dtype>
Dtype cpu_abs_max(long long int n, Dtype* X){
  Dtype max_ = std::abs(X[0]);
  for(long long int i = (long long int)1; i < n; i+=(long long int)1){
    Dtype temp = std::abs(X[i]);
    if(temp > max_){
      max_ = temp;
    }
  }
  return max_;
}

template float cpu_abs_max<float>(long long int n, float* X);
template int cpu_abs_max<int>(long long int n, int* X);

template<>
void cpu_gemm<float>(const bool TransA,
                     const bool TransB, const int M, const int N, const int K,
                     const float alpha, const float* A, const float* B, const float beta,
                     float* C) 
{
  // M, N, K
  //M number of rows of matrix op(A) and C.
  //N is number of columns of matrix op(B) and C.]
  //K is number of rows of op(B) and columns of op(A).

  // op(A) is M by K
  // op(B) is K by N
  // C is M by N
  // cublasDgemm(handle, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) 
  // performs C=alpha op ( B ) op ( A ) + beta C

  int lda = (TransA == false) ? K : M;
  int ldb = (TransB == false) ? N : K;
  cblas_sgemm(CblasRowMajor, (TransA == false) ? CblasNoTrans : CblasTrans, (TransB == false) ? CblasNoTrans : CblasTrans, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
  //https://developer.apple.com/library/mac/documentation/Accelerate/Reference/BLAS_Ref/index.html#//apple_ref/doc/uid/TP30000414-SW27
}

template<>
void cpu_gemm<double>(const bool TransA,
                      const bool TransB, const int M, const int N, const int K,
                      const double alpha, const double* A, const double* B, const double beta,
                      double* C) 
{
  // M, N, K
  //M number of rows of matrix op(A) and C.
  //N is number of columns of matrix op(B) and C.]
  //K is number of rows of op(B) and columns of op(A).

  // op(A) is M by K
  // op(B) is K by N
  // C is M by N
  // cublasDgemm(handle, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) 
  // performs C=alpha op ( B ) op ( A ) + beta C

  int lda = (TransA == false) ? K : M;
  int ldb = (TransB == false) ? N : K;
  cblas_dgemm(CblasRowMajor, (TransA == false) ? CblasNoTrans : CblasTrans, (TransB == false) ? CblasNoTrans : CblasTrans, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void cpu_swap_ordering<float>(const long long int rows, const long long int cols, float *A, const bool row_major_ordering)
{
  bool Debug = false;
  if(Debug) {
    LOG("cpu_swap_ordering called");
    LOG("rows : "<<rows);
    LOG("cols : "<<cols);
    LOG("row_major_ordering : "<<row_major_ordering);
  }
  const long long int total = rows * cols;
  float* A_copy = NULL;
  A_copy  = (float *)malloc(total * sizeof(float));
  checkErrors(A_copy);
  
  host_copy<float>(total, A, A_copy);
  if(Debug) {
    LOG("Here");
  }
  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, total);
    if(Debug) {
      LOG("nthreads : "<<nthreads);
    }
    omp_set_num_threads(nthreads);
    omp_lock_t printlock;
    omp_init_lock(&printlock);
  #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int j = (long long int)th_id; j < total; j += (long long int)nthreads){
    //for(long long int j = (long long int)0; j < total; j += (long long int)1){
      if(!row_major_ordering){
          //starts in colum major ordering
        long long int row = j % rows;
        long long int col = j / rows;
          //i = row + rows * col;
        long long int new_i = cols * row + col;
        if(Debug) {
          omp_set_lock(&printlock);
          LOG("th_id : "<<th_id);
          LOG("j : "<<j);
          LOG("new_i : "<<new_i);

        }
        A[new_i] = A_copy[j];
        if(Debug){
          omp_unset_lock(&printlock);
        }
      }else{
          //starts in row major ordering
        long long int row = j / cols;
        long long int col = j % cols;
          //i = cols * row + col;
        long long int new_i = row + rows * col;
        if(Debug) {
          omp_set_lock(&printlock);
          LOG("th_id : "<<th_id);
          LOG("j : "<<j);
          LOG("new_i : "<<new_i);
        }
        A[new_i] = A_copy[j];
        if(Debug){
          omp_unset_lock(&printlock);
        }
      }
    }
  }
  free(A_copy);
  if(Debug) LOG("cpu_swap_ordering done");
}


template<>
float cpu_abs_max<float>(const long long int n,  const float* x) 
{
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1, "Long long long int too big");}

  thrust::pair<const float *, const float *> tuple = thrust::minmax_element(thrust::host, x, x + n);

  // if int data[6] = {1, 0, 2, 2, 1, 3};
  // thrust::pair<int *, int *> result = thrust::minmax_element(thrust::host, data, data + 6);
  // result.first is data + 1
  // result.second is data + 5
  // *result.first is 0
  // *result.second is 3

  float max;

  if(abs(*tuple.first) > abs(*tuple.second)){
    max =  abs(*tuple.first);
  }else{
    max =  abs(*tuple.second);
  };

  // save_device_array_to_file<float>(x, n , "gradient");
  // LOG(INFO) << "max : " <<max ;
  // LOG(INFO) << "Press Enter to continue." ;
  // std::cin.ignore();

  return max;

}

// square<T> computes the square of a number f(x) -> x*x 
template <typename Dtype> 
struct abss 
{ 
  __host__ __device__ Dtype operator()(const Dtype& x) const {
    if(x >= (Dtype)0.0){
      return x; 
    }else{
      return -x; 
    };
  } 
};

template <>
float cpu_expected_abs_value<float>(const long long int n,  const float* x) {
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  // setup arguments 
  abss<float> unary_op; 
  thrust::plus<float> binary_op; 
  float init = 0; 
  // compute norm 

  float s =  thrust::transform_reduce(thrust::host, x, x+n, unary_op, init, binary_op) ; 


  return (float)(s/(float)n);

}

//============================================================================================
// random utilities
//============================================================================================



// random seeding
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG("System entropy source not available, using fallback algorithm to generate seed instead.");
  if (f)
    fclose(f);

  pid = getpid();
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}


template <typename Dtype>
void fillupMatrix(Dtype *A , int lda , int rows, int cols)
{
  int seed = cluster_seedgen();
  int sign;
  for (int j = 0; j < cols; j++)
  {
    for (int i = 0; i < rows; i++)
    {
      sign = rand() % 2;
      A[i + lda*j ] = (Dtype) (((double)(((lda*i+j+seed) % 253)+1))/256.0, ((double)((((cols*i+j) + 123 + seed) % 253)+1))/256.0);
      if(sign==0){
        A[i + lda*j ] = A[i + lda*j ] * (Dtype)(-1);
      };
      /*
        if(i<5 && j==0){
            printf("A[%d + %d * %d] = %f \n", i, lda, j, A[i + lda*j ]);
        };
      */
    }
  }
}
/* Explicit instantiation */
template void  fillupMatrix<float>(float *A , int lda , int rows, int cols);
template void  fillupMatrix<double>(double *A , int lda , int rows, int cols);

template <typename Dtype>
Dtype nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
    b, std::numeric_limits<Dtype>::max());
}
template float nextafter(const float b);
template double nextafter(const double b);


template <typename Dtype>
void host_rng_uniform(const long long int n, const Dtype a, const Dtype b, Dtype* r) {
  //ABORT_IF_NEQ(0, 1, "host_rng_uniform not yet supported");
  bool Debug = false;
  if(Debug){
    LOG("a : "<<a);
    LOG("b : "<<b);
  }

  ABORT_IF_LESS(n, 1, "host_rng_uniform has n < 0");
  ABORT_IF_EQ(a, b, "host_rng_uniform has a = b");
  ABORT_IF_LESS(b, a, "host_rng_uniform has b < a");

  base_generator_type generator(static_cast<unsigned int>(cluster_seedgen()));

  // Define a uniform random number distribution which produces "double"
  // values between 0 and 1 (0 inclusive, 1 exclusive).
  boost::uniform_real<> uni_dist(static_cast<double>(a), static_cast<double>(b));
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);

  std::cout.setf(std::ios::fixed);
  // You can now retrieve random numbers from that distribution by means
  // of a STL Generator interface, i.e. calling the generator as a zero-
  // argument function.
  for(long long int i = (long long int)0; i < n; i+=(long long int)1)
    r[i] = static_cast<Dtype>(uni()) ;

  if(Debug){
    save_host_array_to_file<Dtype>(r, static_cast<int>(n), "random_array");
  }
  // boost::uniform_real<Dtype> random_distribution(a, nextafter<Dtype>(b));
  // boost::variate_generator<rng_t*, boost::uniform_real<Dtype> >
  //     variate_generator(_rng(), random_distribution);
  // for (int i = 0; i < n; ++i) {
  //   r[i] = variate_generator();
  // }
}

template void host_rng_uniform<float>(const long long int n, const float a, const float b, float* r);
template void host_rng_uniform<double>(const long long int n, const double a, const double b, double* r);

//============================================================================================
// Old
//============================================================================================

void getRandIntsBetween(int *A , int lower_bd , int upper_bd, int num)
{
  for (int i = 0; i < num; i++)
  {
    /* generate secret number between lower_bd and upper_bd: */
    A[i] = rand() % (upper_bd - lower_bd) + lower_bd;
    /*
      if(i<5){
          printf("A[%d] = %d \n", i, A[i]);
      };
    */
    }
  //printf("\n");

  }

  void secondarySort(int *rows , int *cols, int lower_bd , int upper_bd,  int num)
  {
    int *rows_bottom = rows;
    int *cols_bottom = cols;
    int current=rows_bottom[0];
    int count = 0;
    int next = 0;
    bool repeats = 1;

    while (count <= num)
    {
      while(rows_bottom[next ] == current){
      //if (count<100){printf("row[%d] =  %d, col[%d] =  %d, next = %d \n",  count + next, rows[count + next], count + next,  cols[count + next], next);};
        next+=1;
      };
      if (next>0){
      //if (count<100){printf("need to sort, count = %d , next = %d\n", count, next);};
        thrust::sort(cols_bottom, cols_bottom + next );
      //check for repeats
        while(repeats){
          repeats = 0;
          for (int j = 0; j < next-1; j++)
          {
            if(cols_bottom[j] == cols_bottom[j+1])
            {
              printf("repeat found: [ %d , %d ]\n", lower_bd, upper_bd);
              for (int k = 0; k < next; k++)
              {
                printf("[ %d , %d ]\n", rows_bottom[k], cols_bottom[k]);
              };

              repeats = 1;
              cols_bottom[j+1] = rand() % (upper_bd - lower_bd) + lower_bd;
              thrust::sort(cols_bottom, cols_bottom + next );
            };
          };
        }
      }
      count+=next ;
      rows_bottom = rows_bottom + next ;
      cols_bottom = cols_bottom + next ;
      current=rows_bottom[next ]; 
      next = 0;
    };
  }

  void gatherIndices(int *rows , int *cols, int num_rows , int num_cols)
  {
    int num = num_rows * num_cols;
    for (int i = 0; i < num_rows; i++)
    {
      for (int j = 0; j < num_cols; j++)
      {
        rows[i*num_cols +j]=i;
        cols[i*num_cols +j]=j; 
      };
    };
  }

  bool selectIndices(int *rows , int *cols, int *select_rows , int *select_cols,  int num_rows , int num_cols,  int nnz)
  {
    struct timeval time_start, time_end, while_start, while_end;
    double selectIndices_time, while_time;

    int* indicies = NULL;
    indicies  = (int *)malloc(nnz * sizeof(int));
    checkErrors(indicies);

    gettimeofday(&time_start, NULL);
  //generate random numbers:
    for (int i=0;i<nnz;i++)
    {   
      gettimeofday(&while_start, NULL);
    bool check; //variable to check or number is already used
    int n; //variable to store the number in
    do
    {
      n=rand()%(num_rows * num_cols);
        //check or number is already used:
      check=true;
      for (int j=0;j<i;j++)
      {
        if (n == indicies[j]) //if number is already used
        {
          check=false; //set check to false
          break; //no need to check the other elements of value[]
        }
      }
      gettimeofday(&while_end, NULL);
      while_time = (while_end.tv_sec * 1000 +(while_end.tv_usec/1000.0))-(while_start.tv_sec * 1000 +(while_start.tv_usec/1000.0));
    }while (!check && while_time < (double)30000.0); //loop until new, unique number is found
    gettimeofday(&time_end, NULL);
    selectIndices_time = (time_end.tv_sec * 1000 +(time_end.tv_usec/1000.0))-(time_start.tv_sec * 1000 +(time_start.tv_usec/1000.0));

    if(while_time < (double)30000.0 && selectIndices_time <(double)60000.0){
      indicies[i]=n; //store the generated number in the array
    }else{
      //printf("while_time: %d\n", while_time < (double)30.0);
      //printf("selectIndices_time: %d\n", selectIndices_time <(double)300.0);
      return false;
    };
  };


  thrust::sort(indicies, indicies + nnz );
  for (int j = 0; j < nnz; j++)
  {
    select_rows[j]=rows[indicies[j]];
    select_cols[j]=cols[indicies[j]];
  };
  free(indicies);

  return true;
}








void initializeVal(double * A, int rows, int cols , int ld, double val)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      A[j*ld +i] = val;
    }
  }
}






template <typename Dtype>
void fillupMatrixDebug(Dtype *A , int lda , int rows, int cols)
{
  for (int j = 0; j < cols; j++)
  {
    for (int i = 0; i < rows; i++)
    {
      A[i + lda*j ] = (Dtype) (i + j);
    }
  }
}




//============================================================================================
// Prints and Saves
//============================================================================================

template<typename Dtype>
std::string ToString(const Dtype val)
{
  std::stringstream stream;
  stream << val;
  return stream.str();
}
template std::string ToString<bool>(const bool val);
template std::string ToString<int>(const int val);
//template std::string ToString<long long int>(const long long int val);
template std::string ToString<float>(const float val);
template std::string ToString<double>(const double val);

extern "C" std::string readable_time(double ms){
  int ms_int = (int)ms;
  int hours = ms_int / 3600000;
  ms_int = ms_int % 3600000;
  int minutes = ms_int / 60000;
  ms_int = ms_int % 60000;
  int seconds = ms_int / 1000;
  ms_int = ms_int % 1000;

  return (ToString<int>(hours) + ":" +ToString<int>(minutes) + ":" + ToString<int>(seconds) + ":" + ToString<int>(ms_int));
}

template<typename Dtype>
void printPartialMatrices(Dtype * A, Dtype * B, int rows, int cols , int ld)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%f , ",  A[j*ld +i]);
    }
    printf("    ");
    for (int j = 0; j < cols; j++)
    {
      printf("%f , ",  B[j*ld +i]);
    }
    printf(";\n");
  }
}

template void printPartialMatrices<float>(float * A, float * B, int rows, int cols , int ld);
template void printPartialMatrices<int>(int * A, int * B, int rows, int cols , int ld);


template<typename Dtype>
void printPartialMtx(Dtype * A, int rows, int cols , int ld)
{
// This function assumes column major ordering, such that ld corresponds to the number of rows
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      if (j<cols-1){ printf("%d , ",  A[j*ld +i]); }
      else{ printf("%d ;\n ",  A[j*ld +i]); }
    }
  }
}
template void printPartialMtx<int>(int * A, int rows, int cols , int ld);
template void printPartialMtx<float>(float * A, int rows, int cols , int ld);
template void printPartialMtx<double>(double * A, int rows, int cols , int ld);


template<typename Dtype>
void print_host_array(const Dtype* host_pointer, int count, std::string title, std::string file_line)
{
  if(file_line != ""){
    LOG2(file_line, title<<" : ");
  }else{
    LOG(title<<" : ");
  }
  std::string line;
  line="[ ";
  for( int i = 0; i < count; i+= 1 ) {
    if (i==count-1){
      line = (line + ToString<Dtype>(host_pointer[i ])).c_str();
    }else{
      line = (line + ToString<Dtype>(host_pointer[i]) + " , ").c_str();
    };
  };     

  line = line+ " ];\n";
  if(file_line != ""){
    LOG2(file_line, line<<std::endl);
  }else{
    LOG(line<<std::endl);
  }
}

template void print_host_array<int>(const int* host_pointer, int count, std::string title, std::string file_line);
template void print_host_array<float>(const float* host_pointer, int count, std::string title, std::string file_line);

template<typename Dtype>
void print_host_mtx(const Dtype* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line)
{
  //assumes row major order
  std::string line = (title + " : \r\n").c_str();
  if(row_major_order){
    for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
        //line = (line + ToString<Dtype>(A_host[i + j * rows])).c_str();
        line = (line + ToString<Dtype>(A_host[i * cols + j])).c_str();
        if(j < cols - 1){
          line = (line + ", ").c_str();
        }
      }
      line = (line + "\r\n").c_str();
      //line = (line + "; ").c_str();
    }
  }else{
    for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
        line = (line + ToString<Dtype>(A_host[i + j * rows])).c_str();
        //line = (line + ToString<Dtype>(A_host[i * cols + j])).c_str();
        if(j < cols - 1){
          line = (line + ", ").c_str();
        }
      }
      line = (line + "\r\n").c_str();
      //line = (line + "; ").c_str();
    }    
  }
  if(file_line != ""){
    LOG2(file_line, line<<std::endl);
  }else{
    LOG(line<<std::endl);
  }
}

template void print_host_mtx<int>(const int* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);
//template void print_host_mtx<long long int>(const long long int* A_host, int rows, int cols, std::string title, bool row_major_order, std::string file_line);
template void print_host_mtx<float>(const float* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);
template void print_host_mtx<double>(const double* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);


template<typename Dtype>
void save_host_array_to_file(const Dtype* A_host, int count, std::string title, std::string file_line)
{
  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < count; i++){
    entries<<A_host[i ];
    if(i < count - 1){
      //entries<<", ";
      entries<<"\r\n";
    };
  };
  if(file_line != ""){
    LOG2(file_line, "save_host_array_to_file "<< title);
  }
}

template void save_host_array_to_file<int>(const int* A_host, int count, std::string title, std::string file_line);
template void save_host_array_to_file<float>(const float* A_host, int count, std::string title, std::string file_line);
template void save_host_array_to_file<double>(const double* A_host, int count, std::string title, std::string file_line);

template<typename Dtype>
void get_host_array_from_saved_txt(const Dtype* A_host, int count, std::string title)
{
  ABORT_IF_NEQ(0, 1, "Function not ready.");
  std::ifstream A_host_file ((ToString(title + ".txt")).c_str());

  for (int i = 0; i < count; i++){
    //A_host_file>>A_host[i];
  }

  //LOG(INFO)<<"It_count grabbed from file: "<<It_count;
  A_host_file.close();
}

template void get_host_array_from_saved_txt<int>(const int* A_host, int count, std::string title);
template void get_host_array_from_saved_txt<float>(const float* A_host, int count, std::string title);
template void get_host_array_from_saved_txt<double>(const double* A_host, int count, std::string title);

template<typename Dtype>
void append_host_array_to_file(const Dtype* A_host, int count, std::string title)
{


  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str(), std::ofstream::app);
  //entries<<"; ";
  entries<<"\r\n";
  for (int i = 0; i < count; i++){
    entries<<A_host[i ];
    if(i < count - 1){
      entries<<", ";
    };
  };
}

template void append_host_array_to_file<int>(const int* A_host, int count, std::string title);
template void append_host_array_to_file<float>(const float* A_host, int count, std::string title);
template void append_host_array_to_file<double>(const double* A_host, int count, std::string title);


void save_host_arrays_side_by_side_to_file(const int* A_host, const int* B_host, 
 const float* C_host, int count, std::string title)
{


  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < count; i++){
    entries<<"["<<i<<"] : "<<A_host[i ]<<", "<<B_host[i ]<<", "<<C_host[i ];
    if(i < count - 1){
    //entries<<", ";
      entries<<"\r\n";
    };

  };
  //LOG("file saved");

}



template<typename Dtype>
void save_host_mtx_to_file(const Dtype* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line)
{
  //assumes row major order

  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  if(row_major_order){
    for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
        //entries<<A_host[i + j * rows];
        entries<<A_host[i * cols + j];
        if(j < cols - 1){
          entries<<", ";
        }
      }
      entries<<"\r\n";
      //entries<<"; ";
    }
  }else{
    for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
        entries<<A_host[i + j * rows];
        //entries<<A_host[i * cols + j];
        if(j < cols - 1){
          entries<<", ";
        }
      }
      entries<<"\r\n";
    }    
  }
  if(file_line != ""){
    LOG2(file_line, "save_host_mtx_to_file "<< title);
  }
}

template void save_host_mtx_to_file<int>(const int* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);
//template void save_host_mtx_to_file<long long int>(const long long int* A_host, int rows, int cols, std::string title, bool row_major_order, std::string file_line);
template void save_host_mtx_to_file<float>(const float* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);
template void save_host_mtx_to_file<double>(const double* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);

void save_map(std::map<int, int>* items_dictionary, std::string title){

  std::map<int, int>::iterator it = items_dictionary -> begin();

  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());

  // Iterate over the map using Iterator till end.
  while (it != items_dictionary->end())
  {
    entries<<it->first;
    entries<<", ";
    entries<<it->second;
    entries<<"\r\n";
    it++;
  }
}

//============================================================================================
// Me Made
//============================================================================================

void cpu_fill_training_mtx(const long long int ratings_rows_training, const long long int ratings_cols_training,  
 const long long int num_entries_CU, const bool row_major_ordering,
 const int* csr_format_ratingsMtx_userID_host_training,
 const int* coo_format_ratingsMtx_itemID_host_training,
 const float* coo_format_ratingsMtx_rating_host_training,
 float* full_training_ratings_mtx)
{
  LOG("cpu_fill_training_mtx called...");
  bool Debug = false;
  LOG("ratings_rows_training : "<< ratings_rows_training);
  LOG("ratings_cols_training : "<< ratings_cols_training);
  //row major ordering
  for(long long int row = (long long int)0; row < ratings_rows_training; row +=(long long int)1){
    for(long long int i = (long long int)(csr_format_ratingsMtx_userID_host_training[row]); i < (long long int)(csr_format_ratingsMtx_userID_host_training[row + 1]); i+=(long long int)1){
      long long int col = (long long int)(coo_format_ratingsMtx_itemID_host_training[i]);
      float val = coo_format_ratingsMtx_rating_host_training[i]; 
      if(row_major_ordering){
        if(Debug) {
          LOG("num_entries_CU : "<< num_entries_CU);
          LOG("ratings_rows_training : "<< ratings_rows_training);
          LOG("ratings_cols_training : "<< ratings_cols_training);
          LOG("full_training_ratings_mtx["<<ratings_cols_training<<" * "<< row<<" + "<<col<<"] = "<<val) ;
          LOG("full_training_ratings_mtx["<<ratings_cols_training *  row + col<<"] = "<<val) ;
          LOG(val<<" = coo_format_ratingsMtx_rating_host_training["<<i<<"]") ;
          LOG(val<<" = "<<coo_format_ratingsMtx_rating_host_training[i]) ;
        }
        full_training_ratings_mtx[ratings_cols_training * row + col] = val;
      }else{
        full_training_ratings_mtx[row + ratings_rows_training * col] = val;
      };
    }
  }
  LOG("cpu_fill_training_mtx finished...");
}

template < typename Dtype>
void cpu_shuffle_array(const long long int n,  Dtype* x)
{
  bool Debug = false;
  if(too_big(n) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  double* order = (double *)malloc(n * sizeof(double));
  checkErrors(order);
  host_rng_uniform<double>(n, (double)0.0, (double)1.0, order);
  if(Debug){
    save_host_array_to_file<Dtype>(x, static_cast<int>(n), "before_shuffle");
  }
  thrust::sort_by_key(thrust::host, order, order + n, x);
  if(Debug){
    save_host_array_to_file<Dtype>(x, static_cast<int>(n), "after_shuffle");
  }
  free(order);

}


void cpu_shuffle_mtx_rows_or_cols(cublasHandle_t dn_handle, const long long int M, const long long int N, bool row_major_ordering, float* x, bool shuffle_rows)
{

  if(too_big(M) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  if(too_big(N) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}

  bool Debug = false;
  int * indicies_host = NULL;
  int * indicies_dev;

  /*
    A in row major ordering is equivalent to A^T in column major ordering
    A in column major ordering is equivalent to A^T in row major ordering
  */

  if(Debug) LOG("cpu_shuffle_mtx_rows_or_cols called") ;
  if(shuffle_rows){
    if(Debug) LOG("shuffle_rows is true") ;
    CUDA_CHECK(cudaMalloc((void**)&indicies_dev, M * sizeof(int)));
    indicies_host = (int *)malloc(M * sizeof(int));
    checkErrors(indicies_host);
    gpu_set_as_index(indicies_dev, M);
    gpu_shuffle_array<int>(dn_handle, M, indicies_dev);
    CUDA_CHECK(cudaMemcpy(indicies_host, indicies_dev, M * sizeof(int), cudaMemcpyDeviceToHost));
    if(row_major_ordering){
      if(Debug) LOG("row_major_ordering is true") ;
      cpu_permute<float>(x, indicies_host, N, M, false); 
    }else{
      cpu_permute<float>(x, indicies_host, M, N, true); 
    };
    checkCudaErrors(cudaFree(indicies_dev));
    free(indicies_host);
  }else{
    // shuffle columns
    CUDA_CHECK(cudaMalloc((void**)&indicies_dev, N * sizeof(int)));
    indicies_host = (int *)malloc(N * sizeof(int));
    checkErrors(indicies_host);
    gpu_set_as_index(indicies_dev, N);
    gpu_shuffle_array<int>(dn_handle, N, indicies_dev);
    CUDA_CHECK(cudaMemcpy(indicies_host, indicies_dev, N * sizeof(int), cudaMemcpyDeviceToHost));
    if(row_major_ordering){
      cpu_permute<float>(x, indicies_host, N, M, true); 
    }else{
      cpu_permute<float>(x, indicies_host, M, N, false); 
    };
    checkCudaErrors(cudaFree(indicies_dev));
    free(indicies_host);
  }
} 

void cpu_shuffle_map_second(const long long int M, std::map<int, int>* items_dictionary )
{

  if(too_big(M) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
  
  bool Debug = false;
  int * indicies_host = NULL;
  int * indicies_dev;

  CUDA_CHECK(cudaMalloc((void**)&indicies_dev, M * sizeof(int)));
  indicies_host = (int *)malloc(M * sizeof(int));
  checkErrors(indicies_host);

  gpu_set_as_index(indicies_dev, M);
  CUDA_CHECK(cudaMemcpy(indicies_host, indicies_dev, M * sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(indicies_dev));

  cpu_shuffle_array<int>(M, indicies_host);

  std::map<int, int>::iterator it = items_dictionary -> begin();
  int i = 0; 
  // Iterate over the map using Iterator till end.
  while (it != items_dictionary->end())
  {
    it->second = indicies_host[i];
    i++;
    it++;
  }

  free(indicies_host);

} 



template<typename Dtype>
void cpu_set_all(Dtype* x, const int N, Dtype alpha)
{
  for(int i=0; i <N; i++) {
    //origin+x1+rows*y1
    x[i]=alpha;
  };
}

template void cpu_set_all<int>(int* x, const int N, int alpha);
template void cpu_set_all<float>(float* x, const int N, float alpha);


template <>
void cpu_set_as_index<int>(int* x, const long long int rows, const long long int cols) 
{
  bool print = true;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, rows * cols);
    omp_set_num_threads(nthreads);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int i = (long long int)th_id; i < rows * cols; i += (long long int)nthreads){
    //for(long long int i = (long long int)0; i < rows * cols; i+=(long long int)1) {
      int row = (int)(i % rows);
      //int col = (int)(i / rows);
      x[i] = row;
    }
  }
  if(0) LOG("finished call to cpu_set_as_index") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("cpu_set_as_index run time : "<<readable_time(program_time));
}




void cpu_get_cosine_similarity(const long long int ratings_rows,
  const int* csr_format_ratingsMtx_userID_host,
  const int* coo_format_ratingsMtx_itemID_host,
  const float* coo_format_ratingsMtx_rating_host,
  float* cosine_similarity) 
{
  bool print = true;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  if(print) LOG("called cpu_get_cosine_similarity") ;

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, ratings_rows * ratings_rows);
    omp_set_num_threads(nthreads);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int entry = (long long int)th_id; entry < ratings_rows * ratings_rows; entry += (long long int)nthreads){
    //for( long long int entry = (long long int)0; entry < ratings_rows * ratings_rows; entry+=(long long int)1){
      long long int whole_index = from_below_diag_to_whole_faster(entry, ratings_rows);
      int user_i = (int)(whole_index % ratings_rows);
      int user_j = (int)(whole_index / ratings_rows);
      if( user_i == user_j){
        LOG("Thread "<<th_id<<" is bad : " <<user_i) ;
        cosine_similarity[entry] = (float)1.0;
      }else{
        int   count   = 0;
        float num     = (float)0.0;
        float denom_i = (float)0.0;
        float denom_j = (float)0.0;
        for(int i = csr_format_ratingsMtx_userID_host[user_i]; i < csr_format_ratingsMtx_userID_host[user_i + 1]; i++){
          int user_i_itemID = coo_format_ratingsMtx_itemID_host[i];
          int user_j_itemID = 0;
          int start_j = csr_format_ratingsMtx_userID_host[user_j];
          for(int j = start_j; j < csr_format_ratingsMtx_userID_host[user_j + 1]; j++){
            user_j_itemID = coo_format_ratingsMtx_itemID_host[j];
            if( user_i_itemID == user_j_itemID){
              count   += 1;
              num     += coo_format_ratingsMtx_rating_host[i] * coo_format_ratingsMtx_rating_host[j] ;
              denom_i += pow(coo_format_ratingsMtx_rating_host[i], (float)2.0) ;
              denom_j += pow(coo_format_ratingsMtx_rating_host[j], (float)2.0) ; 
              start_j = j + 1;
            }else if(user_i_itemID < user_j_itemID){
              start_j = j;
              break;
            }
          }
        }
        if(count > 0){
            //float temp = num / sqrt(denom_i * denom_j);
          float temp_i = (float)csr_format_ratingsMtx_userID_host[user_i + 1] - (float)csr_format_ratingsMtx_userID_host[user_i];
          float temp_j = (float)csr_format_ratingsMtx_userID_host[user_j + 1] - (float)csr_format_ratingsMtx_userID_host[user_j];
          float temp = ((float)count) / sqrtf(temp_i * temp_j);
          cosine_similarity[entry] = temp;
          if (::isinf(temp) || ::isnan(temp)){
            LOG("Thread "<<th_id<<" is bad : " <<temp) ;
          };
        }else{
         cosine_similarity[entry] = (float)0.0;
        }
      }
    }
  }

  if(0) LOG("finished call to cpu_get_cosine_similarity") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("cpu_get_cosine_similarity run time : "<<readable_time(program_time));
}


/* This function takes last element as pivot, places
   the pivot element at its correct position in sorted
    array, and places all smaller (smaller than pivot)
   to left of pivot and all greater elements to right
   of pivot */
template<typename Dtype>
int partition (Dtype* x, int low_index, int high_index, int* indicies)
{
    // pivot (Element to be placed at right position)
    Dtype pivot = x[high_index];  
    Dtype temp = 0.0;
    int temp_ = 0;
    int i = (low_index - 1);  // Index of smaller element

    for (int j = low_index; j < high_index; j++)
    {
        // If current element is smaller than the pivot
        if (x[j] < pivot)
        {
            i++;    // increment index of smaller element
            temp = x[i];
            x[i] = x[j];
            x[j] = temp;
            temp_ = indicies[i];
            indicies[i] = indicies[j];
            indicies[j] = temp_;
        }
    }
    temp = x[i + 1];
    x[i + 1] = x[high_index];
    x[high_index] = temp;
    temp_ = indicies[i + 1];
    indicies[i + 1] = indicies[high_index];
    indicies[high_index] = temp_;
    return (i + 1);
}

template int partition<int>(int* x, int low_index, int high_index, int* indicies);
template int partition<float>(float* x, int low_index, int high_index, int* indicies);


/* low  --> Starting index,  high  --> Ending index */
template<typename Dtype>
void quickSort_by_key(Dtype* x, int low_index, int high_index, int* indicies)
{
  ABORT_IF_NEQ(0, 1, "function not ready");
  if (x[low_index] < x[high_index])
  {
    /* pi is partitioning index, arr[pi] is now
       at right place */
    int pi = partition<Dtype>(x, low_index, high_index, indicies);

    quickSort_by_key<Dtype>(x, low_index, pi - 1, indicies);  // Before pi
    quickSort_by_key<Dtype>(x, pi + 1, high_index, indicies); // After pi
  }
}

template void quickSort_by_key<int>(int* x, int low_index, int high_index, int* indicies);
template void quickSort_by_key<float>(float* x, int low_index, int high_index, int* indicies);


template<typename Dtype>
void cpu_sort_index_by_max(const long long int rows, const long long int cols,  Dtype* x, int* indicies)
{
  bool print = true;
  struct timeval program_start, program_end;
  if(print) LOG("called cpu_sort_index_by_max") ;
  double program_time;
  gettimeofday(&program_start, NULL);

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, rows);
    omp_set_num_threads(nthreads);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int i = (long long int)th_id; i < rows; i += (long long int)nthreads){
    //for(long long int i = (long long int)0; i < rows; i+=(long long int)1){
      //thrust::sort_by_key sorts indicies by x smallest to x largest
      thrust::sort_by_key(thrust::host, x + i * cols, x + (i + 1) * cols , indicies + i * cols);
      //quickSort_by_key<Dtype>(x + i * cols, 0, cols, indicies + i * cols);
    }
  }

  if(0) LOG("finished call to cpu_sort_index_by_max") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("cpu_sort_index_by_max run time : "<<readable_time(program_time));
}


template void cpu_sort_index_by_max<int>(const long long int rows, const long long int cols,  int* x, int* indicies);
template void cpu_sort_index_by_max<float>(const long long int rows, const long long int cols,  float* x, int* indicies);

template<typename Dtype>
void cpu_sort_index_by_max(const long long int dimension,  Dtype* x, int* indicies, int top_N)
{
  bool print = true;
  bool debug = false;
  double avg_time = 0.0;
  struct timeval program_start, program_end;
  if(print) LOG("called cpu_sort_index_by_max") ;

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = std::min(nProcessors, (int)dimension);
    omp_set_num_threads(nthreads);
    if(debug){
      LOG("max threads: "<<nProcessors)
      LOG("number of threads: "<<nthreads);
    }
  #endif

  Dtype* temp_x  = (Dtype *)malloc((dimension - 1) * nthreads * sizeof(Dtype));
  int* temp_indicies  = (int *)malloc((dimension - 1) * nthreads * sizeof(int));
  checkErrors(temp_x);
  checkErrors(temp_indicies);
  //save_host_mtx_to_file(temp_x, (dimension - 1), nthreads, "temp_x");

  double program_time;
  gettimeofday(&program_start, NULL);
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int i = (long long int)th_id; i < dimension; i += (long long int)nthreads){
    //for(long long int i = (long long int)0; i < dimension; i+=(long long int)1){

      long long int num_below_diag = (long long int)0;
      long long int left_off = (long long int)0;
      long long int num_in_col = dimension - (long long int)1;

      for(long long int j = (long long int)0; j < i; j+=(long long int)1){
        left_off = num_below_diag + i - (dimension - num_in_col);

        temp_x[th_id * (dimension - 1) + j] = x[left_off];
        temp_indicies[th_id * (dimension - 1) + j] = j;

        num_below_diag += num_in_col;
        num_in_col -= (long long int)(1);
      }
      left_off = num_below_diag + (i + (long long int)(1)) - (dimension - num_in_col);
      for(long long int j = i + (long long int)1; j < dimension; j+=(long long int)1){
        temp_x[th_id * (dimension - 1) + j - 1] = x[left_off];
        temp_indicies[th_id * (dimension - 1) + j - 1] = j;
        left_off += (long long int)(1);
      }

      if(i == 0 && debug) {
        print_host_array((temp_x + th_id * (dimension - 1)), ((int)dimension - 1), ("temp_x"));
        print_host_array((temp_indicies + th_id * (dimension - 1)), ((int)dimension - 1), ("temp_indicies"));
      }
      //LOG("Hello from thread "<<th_id) ;
      //thrust::sort_by_key sorts temp_indicies by temp_x smallest to temp_x largest
      thrust::sort_by_key(thrust::host, temp_x + (long long int)th_id * (dimension - (long long int)1), temp_x + (long long int)(th_id + 1) * (dimension - (long long int)1) , temp_indicies + (long long int)th_id * (dimension - (long long int)1));
      //quickSort_by_key<Dtype>(temp_x + th_id * (dimension - 1), 0, dimension - 1, temp_indicies + th_id * (dimension - 1));
      host_copy(top_N, temp_indicies + (long long int)(th_id + 1) * (dimension - (long long int)1) - (long long int)top_N, indicies + i * (long long int)top_N);
      if(i == 0 && debug) {
        print_host_array((temp_x + (long long int)th_id * (dimension - (long long int)1)), ((int)dimension - 1), ("temp_x"));
        print_host_array((temp_indicies + (long long int)th_id * (dimension - (long long int)1)), ((int)dimension - 1), ("temp_indicies"));
        print_host_array((indicies + i * (long long int)top_N), (top_N), ("indicies"));
      }

      if(th_id == 0 && debug){
        //LOG("th_id * (dimension - 1) : "<<th_id * (dimension - 1));
        //save_host_array_to_file<Dtype>(temp_x + th_id * (dimension - 1), dimension - 1, "temp_x");
        //save_host_array_to_file<int>(temp_indicies + th_id * (dimension - 1), dimension - 1, "temp_indicies");
        LOG("for loop index : "<<i);
        gettimeofday(&program_end, NULL);
        program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
        LOG("cpu_sort_index_by_max average loop run time so far : "<<readable_time(program_time / (double)i));      
      }
    }
  }

  if(debug) LOG("Hello") ;
  free(temp_x);
  free(temp_indicies);

  if(0) LOG("finished call to cpu_sort_index_by_max") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("cpu_sort_index_by_max run time : "<<readable_time(program_time));
}


template void cpu_sort_index_by_max<int>(const long long int dimension,  int* x, int* indicies, int top_N);
template void cpu_sort_index_by_max<float>(const long long int dimension,  float* x, int* indicies, int top_N);



void cpu_count_appearances(const int top_N, const long long int dimension,
  int* count, const int* indicies )
{
  bool print = true;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, dimension);
    omp_set_num_threads(nthreads);
    omp_lock_t *locks = (omp_lock_t *)malloc(dimension * sizeof(omp_lock_t));
    checkErrors(locks);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int j = (long long int)th_id; j < dimension; j += (long long int)nthreads){
      omp_init_lock(locks + j);
      omp_unset_lock(locks + j);
    }
  }
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int j = (long long int)th_id; j < dimension; j += (long long int)nthreads){
      for(long long int i = (long long int)0; i < top_N; i+=(long long int)1){
        int temp = indicies[i + top_N * j];
        omp_set_lock(locks + temp);
        count[temp] += 1;
        omp_unset_lock(locks + temp);
      }
    }
  }

  #ifdef _OPENMP
    free(locks);
  #endif

  if(0) LOG("finished call to gpu_orthogonal_decomp") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("cpu_count_appearances run time : "<<readable_time(program_time));
}



void cpu_mark_CU_users(const int ratings_rows_CU, const int ratings_rows, const int* x, int* y )
{
  bool print = true;
  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min(nProcessors, ratings_rows_CU);
    omp_set_num_threads(nthreads);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int j = (long long int)th_id; j < ratings_rows_CU; j += (long long int)nthreads){
    //for(long long int j = (long long int)(ratings_rows - 1); j > (long long int)(ratings_rows - 1 - ratings_rows_CU); j-=(long long int)1){
      y[x[ratings_rows - 1 - j]] = 0;
    }
  }

  if(0) LOG("finished call to gpu_orthogonal_decomp") ;
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  if(print) LOG("cpu_mark_CU_users run time : "<<readable_time(program_time));

}

long long int from_below_diag_to_whole(long long int below_diag_index, long long int dimension){
  bool debug = false;
  const long long int num_below_diag = (dimension * (dimension - (long long int)1)) / (long long int)2;
  if( below_diag_index < (long long int)0 || below_diag_index > num_below_diag - (long long int)(1)) return (long long int)(-1);

  long long int num_so_far = (long long int)0;
  int col = 0;
  long long int num_in_col = (long long int)(dimension - 1);
  while(num_so_far < below_diag_index + (long long int)(1)){
    if(debug){
      LOG("num_so_far : "<<num_so_far);
      LOG("num_in_col : "<<num_in_col);
      LOG("col : "<<col);
      LOG("col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far = "<<col<<" * "<<dimension<<" + ("<<dimension<<" - "<<num_in_col<<") + "<<below_diag_index<<" - "<<num_so_far<<" = "<<(long long int)col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far);
    }
    
    if(num_so_far + num_in_col == below_diag_index + (long long int)(1)){
      return (long long int)(col + 1) * dimension - (long long int)(1);
    }
    if(num_so_far + num_in_col > below_diag_index + (long long int)(1)){ 
      break;
    }else{
      num_so_far += num_in_col;
      num_in_col -= (long long int)(1);
      col += 1;
    }
  }
  return (long long int)col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far;
}

long long int from_below_diag_to_whole_faster(long long int below_diag_index, long long int dimension)
{
  bool debug = true;
  const long long int num_below_diag = (dimension * (dimension - (long long int)1)) / (long long int)2;
  if( below_diag_index < (long long int)0 || below_diag_index > num_below_diag - (long long int)(1)){
    if(debug){
      LOG("num_below_diag : "<<num_below_diag);
      LOG("below_diag_index : "<<below_diag_index);
    }
    return (long long int)(-1);
  }

  long long int inverted_number = num_below_diag - below_diag_index;
  float n_float = ((float)1.0 + sqrt((float)1.0 + (float)8.0 * (float)inverted_number)) / (float)2.0;
  long long int n = (long long int)round(n_float);
  long long int row;
  long long int col;

  long long int one_less = ((n - (long long int)1) * (n - (long long int)2)) / (long long int)2;
  long long int on_it = (n * (n - (long long int)1)) / (long long int)2;
  long long int one_more = (n * (n + (long long int)1)) / (long long int)2;
  if(debug){
    LOG("below_diag_index : "<<below_diag_index);
    LOG("dimension : "<<dimension);
    LOG("inverted_number : "<<inverted_number);
    LOG("n_float : "<<n_float);
    LOG("n : "<<n);
    LOG("((n - 1) * (n - 2)) / 2 : "<<one_less);
    LOG("(n * (n - 1)) / 2 : "<<on_it);
    LOG("(n * (n + 1)) / 2 : "<<one_more);
  }


  if(one_more < inverted_number){
    if(debug) LOG("one_more < inverted_number");
    return (long long int)(-1);
    // col = dimension - n;
    // row = col + (inverted_number - one_more);
  }else if(one_less < inverted_number && inverted_number < on_it){
    if(debug) LOG("one_less < inverted_number && inverted_number < on_it");
    col = dimension - n;
    row = dimension -  (inverted_number - one_less);
  }else if(inverted_number == on_it){
    if(debug) LOG("inverted_number == on_it");
    col = dimension - n;
    row = col + (long long int)1;
  }else if(on_it < inverted_number && inverted_number < one_more){
    if(debug) LOG("on_it < inverted_number && inverted_number < one_more");
    col = dimension - n - (long long int)1;
    row = dimension - (inverted_number - on_it);
  }else if(inverted_number == one_more){
    if(debug) LOG("inverted_number == one_more");
    return (long long int)(-1);
    // col = dimension - n;
    // row = dimension - (long long int)1;
  } else {
    if(debug) LOG("inverted_number <= one_less");
    return (long long int)(-1);
    // col = dimension - n + (long long int)1;
    // row = dimension - (inverted_number - (n * (n - (long long int)1)) / (long long int)2);
  }
  
  if( row == col){
    if(debug){
      LOG("below_diag_index : "<<below_diag_index);
      LOG("dimension : "<<dimension);
      LOG("inverted_number : "<<inverted_number);
      LOG("n : "<<n);
      LOG("col : "<<col);
      LOG("row : "<<row);
    }
    return (long long int)(-1); 
  }
  if( col < (long long int)0 || col >= dimension){
    if(debug){
      LOG("below_diag_index : "<<below_diag_index);
      LOG("dimension : "<<dimension);
      LOG("inverted_number : "<<inverted_number);
      LOG("n : "<<n);
      LOG("col : "<<col);
      LOG("row : "<<row);
    }
    return (long long int)(-1);
  }
  if( row < (long long int)1 || row >= dimension){
    if(debug){
      LOG("below_diag_index : "<<below_diag_index);
      LOG("dimension : "<<dimension);
      LOG("inverted_number : "<<inverted_number);
      LOG("n : "<<n);
      LOG("col : "<<col);
      LOG("row : "<<row);
    }
    return (long long int)(-1);
  }
  return row + dimension * col;
}


/*
  long long int from_below_diag_to_whole_faster(long long int below_diag_index, int dimension){
    bool debug = false;
    const long long int num_below_diag = (dimension * (dimension - (long long int)1)) / (long long int)2;
    if( below_diag_index < (long long int)0 || below_diag_index > num_below_diag - (long long int)(1)) return (long long int)(-1);

    if (below_diag_index < num_below_diag / 2){
      long long int num_so_far = (long long int)0; // number to the left
      int col = 0;
      long long int num_in_col = (long long int)(dimension - 1);
      while(num_so_far < below_diag_index + (long long int)(1)){
        if(debug){
          LOG("num_so_far : "<<num_so_far);
          LOG("num_in_col : "<<num_in_col);
          LOG("col : "<<col);
          LOG("col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far = "<<col<<" * "<<dimension<<" + ("<<dimension<<" - "<<num_in_col<<") + "<<below_diag_index<<" - "<<num_so_far<<" = "<<(long long int)col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far);
        }
        
        if(num_so_far + num_in_col == below_diag_index + (long long int)(1)){
          return (long long int)(col + 1) * dimension - (long long int)(1);
        }
        if(num_so_far + num_in_col > below_diag_index + (long long int)(1)){ 
          break;
        }else{
          num_so_far += num_in_col;
          num_in_col -= (long long int)(1);
          col += 1;
        }
      }
      return (long long int)col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far;
    }else{
      long long int num_so_far = num_below_diag; // number to the left
      int col = dimension - 2;
      long long int num_in_col = (long long int)1;
      while(num_so_far >= below_diag_index + (long long int)(1)){
        if(false){
          LOG("num_so_far : "<<num_so_far - num_in_col);
          LOG("num_in_col : "<<num_in_col);
          LOG("col : "<<col);
          LOG("col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far = "<<col<<" * "<<dimension<<" + ("<<dimension<<" - "<<num_in_col<<") + "<<below_diag_index<<" - "<<num_so_far- num_in_col<<" = "<<(long long int)col * dimension + (dimension - num_in_col) + below_diag_index - (num_so_far-num_in_col));
        }
        
        if(num_so_far - num_in_col == below_diag_index + (long long int)(1)){
          return (long long int)(col) * dimension - (long long int)(1);
        }
        if(num_so_far - num_in_col < below_diag_index + (long long int)(1)){ 
          num_so_far -= num_in_col;
          break;
        }else{
          num_so_far -= num_in_col;
          num_in_col += (long long int)(1);
          col -= 1;
        }
      }
      LOG("col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far = "<<col<<" * "<<dimension<<" + ("<<dimension<<" - "<<num_in_col<<") + "<<below_diag_index<<" - "<<num_so_far<<" = "<<(long long int)col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far);
      return (long long int)col * dimension + (dimension - num_in_col) + below_diag_index - num_so_far;    
    }
  }
*/

long long int from_whole_to_below_diag(long long int whole_index, long long int dimension){
  bool debug = false;
  if(debug){
    LOG("whole_index : "<<whole_index);
    LOG("dimension : "<<dimension);
  }
  long long int row = (long long int)(whole_index % (long long int)dimension);
  long long int col = (long long int)(whole_index / (long long int)dimension);
  if(row == col) return (long long int)(-1);
  if(row < (long long int)0 || row > dimension - (long long int)1) return (long long int)(-1);
  if(col < (long long int)0 || col > dimension - (long long int)1) return (long long int)(-1);

  long long int temp = row;
  row = std::max(row, col);
  col = std::min(temp,col);
  // now row is larger than col
  if(debug){
    LOG("row : "<<row);
    LOG("col : "<<col);
  }

  long long int num_below_diag = (long long int)0;
  long long int count = (long long int)0;
  long long int num_in_col = (long long int)(dimension - 1);
  while(count < col){
    if(debug){
      LOG("count : "<<count);
      LOG("num_in_col : "<<num_in_col);
      LOG("num_below_diag : "<<num_below_diag);
      LOG("num_below_diag + row - (dimension - num_in_col) = "<<num_below_diag<<" + "<<row<<" - ("<<dimension<<" - "<<num_in_col<<") = "<<num_below_diag + row - (dimension - num_in_col));
    }
    num_below_diag += num_in_col;
    num_in_col -= (long long int)(1);
    count += (long long int)1;
  }
  return num_below_diag + row - (dimension - num_in_col);
}

template <>
void cpu_get_num_latent_factors<float>(const long long int m, float* S_host, 
  long long int* num_latent_factors, const float percent) 
{
  bool Debug = false;
  float sum = cpu_asum<float>( m, S_host);


  float sum_so_far;
  num_latent_factors[0] = m-1;
  for(int j = 0; j < m-1; j++){
    sum_so_far += S_host[j];
    if(sum_so_far / sum >= percent) {
      num_latent_factors[0] = j+1;
      if(Debug) LOG("num_latent_factors = "<< j+1);
      break;
    }
  }

}

template<typename Dtype>
void cpu_div_US_in_SVD(const long long int m, const long long int num_latent_factors, Dtype* U, const Dtype* S, 
  const bool right_divide_by_S) 
{
  //U is m by num_latent_factors in row major ordering
  for(long long int l = (long long int)0; l < m * num_latent_factors; l+=(long long int)1){
    long long int k;
    if(right_divide_by_S){
        k = l % num_latent_factors; //get column
      }else{
        k = l / num_latent_factors; //get row
      }

      //U[l] = (int)k;
      //U[l] = S[k];
      U[l] = U[l] / S[k];
  }

}
template void cpu_div_US_in_SVD<float>(const long long int m, const long long int num_latent_factors, float* U, const float* S, 
  const bool right_divide_by_S);
 

template<typename Dtype>
void cpu_mult_US_in_SVD(const long long int m, const long long int num_latent_factors, Dtype* U, const Dtype* S, 
  const bool right_multiply_by_S)  
{
  //U is m by num_latent_factors in row major ordering
  for(long long int l = (long long int)0; l < m * num_latent_factors; l+=(long long int)1){
    long long int k;
    if(right_multiply_by_S){
      k = l % num_latent_factors; //get column
    }else{
      k = l / num_latent_factors; //get row
    }

    U[l] = U[l] * S[k];
    //U[l] = S[k];
  }
}

template void cpu_mult_US_in_SVD<float>(const long long int m, const long long int num_latent_factors, float* U, const float* S, 
  const bool right_multiply_by_S);


template<>
void cpu_orthogonal_decomp<float>(const long long int m, const long long int n, const bool row_major_ordering,
                                  long long int* num_latent_factors, const float percent,
                                  float* A, float* U, float* V, bool SV_with_U, float* S)
{

  bool Debug = false;
  LOG("cpu_orthogonal_decomp called");
  std::string blank = "";

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);

  long long int lda, sda;
  float *d_U  = NULL;
  float *d_VT = NULL;

  /*
    A in row major ordering is equivalent to A^T in column major ordering
    A in column major ordering is equivalent to A^T in row major ordering
  */

  if(n > m){
    //we have to solve the transpose problem
    if(Debug) LOG("Solving the tranpose problem...");
    d_U  = V;
    d_VT = U;
    lda  = n;
    sda  = m;
    if(!row_major_ordering){
      if(Debug) LOG("A in row major ordering is equivalent to A^T in column major ordering...");
      MatrixInplaceTranspose<float>(A, m, n, row_major_ordering);
    }
  }else{
    d_U  = U;
    d_VT = V;
    lda  = m;
    sda  = n;
    if(row_major_ordering){
      if(Debug) LOG("swap ordering of A");
      cpu_swap_ordering<float>(m, n, A, row_major_ordering);
    }
  };

  long long int smaller_dim = (int)std::min(m, n);
  Eigen::MatrixXf eigen_A = Eigen::Map<Eigen::MatrixXf>( A, lda, sda);

  if(Debug) {
    LOG("eigen_A.rows() : "<<eigen_A.rows()) ;
    LOG("eigen_A.cols() : "<<eigen_A.cols()) ;
    LOG("eigen_A.size() : "<<eigen_A.size()) ;
    //std::cout << "Here is the matrix A:" << std::endl << eigen_A << std::endl;
  }

  // float* A_copy = NULL;
  // A_copy = (float *)malloc(m*n *  sizeof(float)); 
  // checkErrors(A_copy);  
  // cpu_set_all<float>(A_copy, m*n, (float)0.0);
  // Eigen::Map<Eigen::MatrixXf>( A_copy, sda, lda ) = eigen_A;
  // print_host_mtx<float>(A_copy, sda, lda, "A_copy", 1, strPreamble(blank));
  // free(A_copy);

  if(Debug) LOG("Compute svd...") ;
  if(smaller_dim <= (long long int)16){
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(eigen_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if(Debug){
      std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
      std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
      std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
    }
    eigen_A.resize(0,0);
    Eigen::Map<Eigen::MatrixXf>( d_VT, sda, smaller_dim) = svd.matrixV();
    Eigen::Map<Eigen::MatrixXf>( d_U, lda, smaller_dim ) =  svd.matrixU();
    if(S != NULL){
      if(Debug) LOG("saving singular values.");
      Eigen::Map<Eigen::VectorXf>( S, smaller_dim ) = svd.singularValues();
    }

  }else{
    //Eigen::JacobiSVD<Eigen::MatrixXf> svd(eigen_A, Eigen::ComputeFullV);
    Eigen::BDCSVD<Eigen::MatrixXf> svd(eigen_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if(Debug){
      std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
      std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
      std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
    }
    eigen_A.resize(0,0);
    Eigen::Map<Eigen::MatrixXf>( d_VT, sda, smaller_dim) = svd.matrixV();
    Eigen::Map<Eigen::MatrixXf>( d_U, lda, smaller_dim ) =  svd.matrixU();
    
    if(S != NULL){
      if(Debug) LOG("saving singular values.");
      Eigen::Map<Eigen::VectorXf>( S, smaller_dim ) = svd.singularValues();
    }
  }
  //MatrixXf eigen_V(smaller_dim, n) = svd.matrixV();
  if(Debug) LOG("svd computed...") ;
  

  // M, N, K
  //M number of rows of matrix op(A) and C.
  //N is number of columns of matrix op(B) and C.]
  //K is number of rows of op(B) and columns of op(A).

  // op(A) is M by K
  // op(B) is K by N
  // C is M by N
  // cublasDgemm(handle, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) 
  // performs C=alpha op ( B ) op ( A ) + beta C

  if(n > m){
    if(!row_major_ordering){
      if(Debug) LOG("tranpose back...");
      MatrixInplaceTranspose<float>(A, n, m);
    }

    // if(Debug) LOG("solving for V now...") ;
    // C←αAB + βC
    // cpu_gemm<float>(true, false, smaller_dim, n, m,
    //                 (float)1.0, U, A, (float)0.0, V);
    //cpu_mult_US_in_SVD<float>(m, smaller_dim, U, S, true);
    //cpu_div_US_in_SVD<float>(n, smaller_dim, V, S, true);
  }else{
    
    if(row_major_ordering){
      if(Debug) LOG("swap ordering of A back...");
      cpu_swap_ordering<float>(m, n, A, !row_major_ordering);
    }
    //if(Debug) LOG("solving for U now...") ;
    //C←αAB + βC
    // cpu_gemm<float>(false, false, m, smaller_dim, n,
    //                 (float)1.0, A, V, (float)0.0, U);
  };


  if(row_major_ordering){
    //if(Debug) LOG("A in row major ordering is equivalent to A^T in column major ordering...");
    //cpu_swap_ordering<float>(n, smaller_dim, V, !row_major_ordering);
    cpu_swap_ordering<float>(m, smaller_dim, U, !row_major_ordering);
    if(SV_with_U){
      cpu_mult_US_in_SVD<float>(m, smaller_dim, U, S, true);
    }else{
      cpu_mult_US_in_SVD<float>(smaller_dim, n, V, S, false);
    }
  }else{
    /*
      A in row major ordering is equivalent to A^T in column major ordering
      A in column major ordering is equivalent to A^T in row major ordering
    */
    if(SV_with_U){
      cpu_mult_US_in_SVD<float>(smaller_dim, m, U, S, false);
    }else{
      cpu_mult_US_in_SVD<float>(smaller_dim, n, V, S, false);
    }
  }
  cpu_get_num_latent_factors<float>(smaller_dim, S, num_latent_factors, percent);

  if(Debug && 0){

    LOG("num_latent_factors : "<<num_latent_factors[0]) ;
    save_host_mtx_to_file<float>(U, m, m, "U_3");
    save_host_mtx_to_file<float>(V, n, m, "V_3");
    save_host_mtx_to_file<float>(A, m, n, "A");


    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();
  }    
  save_host_array_to_file<float>(S, std::min(m,n), "singular_values");

  if(0){
    float* R  = NULL;
    R = (float *)malloc(m * n *  sizeof(float)); 
    checkErrors(R);

    /*
        A is m by n stored in row-maj ordering where m<<n
        V is n by m stored in row-maj ordering
        (V^T is m by n)
        U is m by m stored in row-maj ordering
    */
    // M, N, K
    //M number of rows of matrix op(A) and C.
    //N is number of columns of matrix op(B) and C.]
    //K is number of rows of op(B) and columns of op(A).

    // op(A) is M by K
    // op(B) is K by N
    // C is M by N
    // cublasDgemm(handle, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) 
    // performs C=alpha op ( B ) op ( A ) + beta C

    cpu_gemm<float>(false, true, m, m, smaller_dim /*num_latent_factors[0]*/,
     (float)1.0, U, U, (float)0.0, R);

    save_host_mtx_to_file<float>(R, m, m, "UUT");

    cpu_gemm<float>(true, false, smaller_dim, smaller_dim, n /*num_latent_factors[0]*/,
     (float)1.0, V, V, (float)0.0, R);

    save_host_mtx_to_file<float>(R, smaller_dim, smaller_dim, "VTV");



    cpu_gemm<float>(false, true, m, n, smaller_dim /*num_latent_factors[0]*/,
     (float)1.0, U, V, (float)0.0,
     R);

    cpu_axpby<float>(m * n, (float)1.0, A,
      (float)(-1.0), R);

    save_host_mtx_to_file<float>(R, m, n, "svd_error");

    //float range_A    = gpu_range<float>(m * n,  A);
        float error      = cpu_abs_max<float>(m * n, R); 
    //float error_expt = gpu_expected_abs_value<float>(m * n, R); 
        free(R);
    // LOG("A mtx range of values = "<<range_A) ;
        LOG("SVD max error = "<<error) ;
    // LOG("SVD max error over range of values = "<<error/range_A) ;
    // LOG("SVD expected absolute error = "<<error_expt) ;
    // LOG("SVD expected absolute error over range of values = "<<error_expt/range_A) ;
        LOG("Press Enter to continue.") ;
        std::cin.ignore();

  }
  gettimeofday(&program_end, NULL);
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(1) LOG("cpu_orthogonal_decomp run time : "<<readable_time(program_time)<<std::endl);

}


void cpu_orthogonal_decomp_test() {

  // MatrixXf M = MatrixXf::Random(3,2);
  // std::cout << "Here is the matrix M:" << std::endl << M << std::endl;
  // JacobiSVD<MatrixXf> svd(M, ComputeThinU | ComputeThinV);
  // std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
  // std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
  // std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;

  const int m = 19;
  const int n = 17;
  const int min_dim = std::min(m,n);
  const int max_dim = std::max(m,n);
  std::string blank = "";

  float* A = NULL;
  float* U = NULL;
  float* V = NULL;
  float* S = NULL;
  A = (float *)malloc(m*n *  sizeof(float)); 
  U = (float *)malloc(m*min_dim *  sizeof(float)); 
  V = (float *)malloc(n*min_dim *  sizeof(float)); 
  S = (float *)malloc(min_dim *  sizeof(float)); 
  checkErrors(A);
  checkErrors(U);
  checkErrors(V);
  checkErrors(S);

  cpu_set_all<float>(U, m*min_dim, (float)0.0);
  cpu_set_all<float>(V, min_dim*n, (float)0.0);
  cpu_set_all<float>(S, min_dim, (float)0.0);

  host_rng_uniform<float>(m*n, (float)(-10.0), (float)10.0, A);

  //save_host_mtx_to_file<float>(A, m, n, "A");

  long long int num_latent_factors;
  const float percent = (float)0.95;
  bool row_major_ordering = false;
  if(1){
    print_host_mtx<float>(A, m, n, "A", row_major_ordering, strPreamble(blank));
  }
  cpu_orthogonal_decomp<float>(m, n, row_major_ordering, &num_latent_factors, percent,
  A, U, V, false, S);


  if(1){
    print_host_mtx<float>(A, m, n, "A", row_major_ordering, strPreamble(blank));
    print_host_mtx<float>(U, m, min_dim, "U", row_major_ordering, strPreamble(blank));
    print_host_mtx<float>(V, n, min_dim, "V", false, strPreamble(blank));
    print_host_array<float>(S, min_dim, "S", strPreamble(blank));
  }

  if(row_major_ordering){
    float* R  = NULL;
    R = (float *)malloc(max_dim * max_dim *  sizeof(float)); 
    checkErrors(R);

    /*
        A is m by n stored in row-maj ordering where m<<n
        V is n by m stored in row-maj ordering
        (V^T is m by n)
        U is m by m stored in row-maj ordering
    */
    // M, N, K
    //M number of rows of matrix op(A) and C.
    //N is number of columns of matrix op(B) and C.]
    //K is number of rows of op(B) and columns of op(A).

    // op(A) is M by K
    // op(B) is K by N
    // C is M by N
    // cublasDgemm(handle, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) 
    // performs C=alpha op ( B ) op ( A ) + beta C


    cpu_gemm<float>(true, false, min_dim, min_dim, m /*num_latent_factors[0]*/,
     (float)1.0, U, U, (float)0.0, R);
    print_host_mtx<float>(R, min_dim, min_dim, "UTU", 1, strPreamble(blank));

    if(min_dim == m){
      cpu_gemm<float>(false, true, m, m, min_dim /*num_latent_factors[0]*/,
       (float)1.0, U, U, (float)0.0, R);
      print_host_mtx<float>(R, m, m, "UUT", 1, strPreamble(blank));
    }

    cpu_gemm<float>(true, false, min_dim, min_dim, n /*num_latent_factors[0]*/,
     (float)1.0, V, V, (float)0.0, R);
    print_host_mtx<float>(R, min_dim, min_dim, "VTV", 1, strPreamble(blank));

    if(min_dim == n){
      cpu_gemm<float>(false, true, n, n, min_dim /*num_latent_factors[0]*/,
       (float)1.0, V, V, (float)0.0, R);
      print_host_mtx<float>(R, n, n, "VVT", 1, strPreamble(blank));
    }



    cpu_gemm<float>(false, true, m, n, min_dim /*num_latent_factors[0]*/,
     (float)1.0, U, V, (float)0.0,
     R);

    cpu_axpby<float>(m * n, (float)1.0, A,
      (float)(-1.0), R);

    print_host_mtx<float>(R, m, n, "svd_error", 1, strPreamble(blank));

    //float range_A    = gpu_range<float>(m * n,  A);
    float error      = cpu_abs_max<float>(m * n, R); 
    //float error_expt = gpu_expected_abs_value<float>(m * n, R); 
    free(R);
    // LOG("A mtx range of values = "<<range_A) ;
    LOG("SVD max error = "<<error) ;
    // LOG("SVD max error over range of values = "<<error/range_A) ;
    // LOG("SVD expected absolute error = "<<error_expt) ;
    // LOG("SVD expected absolute error over range of values = "<<error_expt/range_A) ;
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();
  }

  free(A);
  free(U);
  free(V);
  free(S);  

}



void cpu_center_rows(const long long int rows, const long long int cols, 
                 float* X, const float val_when_var_is_zero, float* user_means,  float* user_var) 
{
  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, rows);
    omp_set_num_threads(nthreads);
  #endif

  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int row = (long long int)th_id; row < rows; row += (long long int)nthreads){
    //for(long long int row = (long long int)0; row < rows; row+=(long long int)1){
      float mean = (float)0.0;
      float std_dev = (float)0.0;
      for(long long int i = (long long int)0; i < cols; i+=(long long int)1){
        mean += X[row * cols + i];
        std_dev += pow(X[row * cols + i], (float)2.0);
      }
      mean /= (float)cols;
      std_dev /= (float)cols;
      std_dev = std_dev - pow(mean, (float)2.0);
      user_var[row] = std_dev;
      std_dev = sqrt(std_dev);
      user_means[row] = mean;

      if(std_dev == (float)0.0 ){
        for(long long int i = (long long int)0; i < cols; i+=(long long int)1){
          X[row * cols + i] = val_when_var_is_zero;
        } 
      }else{
        for(long long int i = (long long int)0; i < cols; i+=(long long int)1){
          X[row * cols + i] = (X[row * cols + i] - mean) / std_dev;
          if (::isinf(X[row * cols + i]) || ::isnan(X[row * cols + i])){
            ABORT_IF_NEQ(0, 1, "isBad");
          };
        } 
      }       
    }
  }
}

template <typename Dtype>
void cpu_sparse_nearest_row(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const int num_sparse_entries, const int* csr_rows_B, const int* coo_cols_B,
 const Dtype* coo_entries_B, int* selection, Dtype* error)
{
  bool Debug = false;
  LOG("cpu_sparse_nearest_row called");

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */
  const int row_skip = csr_rows_B[0];
  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min(nProcessors, rows_B);
    omp_set_num_threads(nthreads);
  #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int row_B = (long long int)th_id; row_B < (long long int)rows_B; row_B += (long long int)nthreads){
    //CUDA_KERNEL_LOOP(row_B,rows_B) {
      Dtype closest_A_row_dist = (Dtype)10000.0;
      int   closest_A_row      = 0;
      for(long long int row_A = (long long int)0; row_A < (long long int)rows_A; row_A+=(long long int)1){
        Dtype temp = (Dtype)0.0;
        for(long long int coo_index = (long long int)(csr_rows_B[row_B]); coo_index < (long long int)(csr_rows_B[row_B + 1]); coo_index+=(long long int)1){
          int col = coo_cols_B[coo_index];
          temp += pow(dense_mtx_A[row_A  * cols + col] - coo_entries_B[coo_index], (Dtype)2.0);
        }
        if(temp < closest_A_row_dist || row_A == 0){
          closest_A_row_dist = temp;
          closest_A_row      = row_A;
        }
      }
      selection[row_B] = closest_A_row;
      error[row_B] = closest_A_row_dist;

      if (::isinf(error[row_B]) || ::isnan(error[row_B])){
        ABORT_IF_NEQ(0, 1, "isBad");
      };
    }
  }
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(1) LOG("cpu_sparse_nearest_row run time : "<<readable_time(program_time)<<std::endl);
}

template void cpu_sparse_nearest_row<float>(const int rows_A, const int cols, const float* dense_mtx_A, 
 const int rows_B, const int num_sparse_entries, const int* csr_rows_B, const int* coo_cols_B,
 const float* coo_entries_B, int* selection, float* error);


template<typename Dtype>
void cpu_dense_nearest_row(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const Dtype* dense_mtx_B, int* selection, Dtype* error)
{
  bool Debug = false;
  LOG("cpu_dense_nearest_row called");

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min(nProcessors, rows_B);
    omp_set_num_threads(nthreads);
  #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int row_B = (long long int)th_id; row_B < (long long int)rows_B; row_B += (long long int)nthreads){
      //CUDA_KERNEL_LOOP(row_B,rows_B) {
        Dtype closest_A_row_dist = (Dtype)1000000.0;
        int   closest_A_row      = 0;
        for(long long int row_A = (long long int)0; row_A < (long long int)rows_A; row_A+=(long long int)1){
          Dtype temp = (Dtype)0.0;
          for(long long int col = (long long int)0; col < (long long int)cols; col+=(long long int)1){
            temp += pow(dense_mtx_A[row_A * cols + col] - dense_mtx_B[row_B * cols+ col],(Dtype)2.0);
          }
          if(temp < closest_A_row_dist || row_A == 0){
            closest_A_row_dist = temp;
            closest_A_row      = (int)row_A;
          }
        }
        selection[row_B] = closest_A_row;
        error[row_B] = closest_A_row_dist;

        if (::isinf(error[row_B]) || ::isnan(error[row_B])){
          ABORT_IF_EQ(0, 0, "isBad");
        };
      };
    }
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(1) LOG("cpu_dense_nearest_row run time : "<<readable_time(program_time)<<std::endl);
}


template void cpu_dense_nearest_row<float>(const int rows_A, const int cols, const float* dense_mtx_A, 
 const int rows_B, const float* dense_mtx_B, int* selection, float* error);




template<typename Dtype>
void cpu_calculate_KM_error_and_update(const int rows_A, const int cols, Dtype* dense_mtx_A, 
                                                    const int rows_B, const Dtype* dense_mtx_B, int* selection,  
                                                    Dtype alpha, Dtype lambda)
{
  bool Debug = false;
  LOG("cpu_calculate_KM_error_and_update called");

  struct timeval program_start, program_end;
  double program_time;
  gettimeofday(&program_start, NULL);
  /*
    subtract dense_mtx_A from sparse mtx B and put the sparse results in coo_errors
    dense_mtx_A must be in column major ordering
  */

  int nthreads = 1;
  #ifdef _OPENMP
    int nProcessors = omp_get_max_threads();
    nthreads = (int)std::min((long long int)nProcessors, (long long int)rows_A * (long long int)cols);
    omp_set_num_threads(nthreads);
  #endif
  #pragma omp parallel shared(nthreads)
  {
    int th_id = 0;
    #ifdef _OPENMP
      th_id = omp_get_thread_num();
    #endif
    for(long long int index = (long long int)th_id; index < (long long int)rows_A * (long long int)cols; index += (long long int)nthreads){
        //CUDA_KERNEL_LOOP(index, num) {
        long long int row_A = index / ((long long int) cols);
        long long int col = index % ((long long int) cols);
        Dtype temp = (Dtype)0.0;
        int count = 0;
        for(long long int row_B = (long long int)0; row_B < (long long int)rows_B; row_B +=(long long int)1){
          if(selection[row_B] == (int)row_A){
            temp += dense_mtx_B[row_B * (long long int)cols + col];
            count++;
          }
        }
        dense_mtx_A[row_A  * (long long int)cols + col] = ((float)1.0 - alpha * lambda) * dense_mtx_A[row_A  * (long long int)cols + col] + alpha * (temp / ((float)count));

        if (::isinf(dense_mtx_A[row_A  * (long long int)cols + col]) || ::isnan(dense_mtx_A[row_A  * (long long int)cols + col])){
          ABORT_IF_EQ(0, 0, "isBad");
        };
      };
  }
  program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
  //printf("program_time: %f\n", program_time);   
  if(1) LOG("cpu_calculate_KM_error_and_update run time : "<<readable_time(program_time)<<std::endl);
}

template void cpu_calculate_KM_error_and_update<float>(const int rows_A, const int cols, float* dense_mtx_A, 
    const int rows_B, const float* dense_mtx_B, int* selection, float alpha, float lambda);

