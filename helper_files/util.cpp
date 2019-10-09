#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
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
void host_copy(const int N, const Dtype* X, Dtype* Y) 
{
  if (X != Y) {
    memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
  }

}

template void host_copy<int>(const int N, const int* X, int* Y);
template void host_copy<unsigned int>(const int N, const unsigned int* X, unsigned int* Y);
template void host_copy<float>(const int N, const float* X, float* Y);
template void host_copy<double>(const int N, const double* X, double* Y);

//============================================================================================
// math functions
//============================================================================================


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
    for(long long int  j = 0; j <cols; j++) {
      long long int  ind=0;
      Dtype temp=0;

      for(long long int  i=0; i<(rows-1); i++){
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
  } else{
    if(Debug) LOG("permute_rows is false") ;
    //pvt is an array length cols
    for(long long int  i=0; i<rows; i++) {
      long long int  ind=0;
      Dtype temp=0;

      for(long long int  j=0; j<(cols-1); j++){
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
template void cpu_permute<float>(float* A, const int* P, const long long int rows, const long long int cols, bool permute_rows) ;
template void cpu_permute<double>(double* a, const int* pvt,const long long int  rows,const long long int  cols, bool direction);


// Non-square matrix transpose of matrix of size r x c and base address A 
template <>
void MatrixInplaceTranspose<float>(float *A, int r, int c) 
{ ABORT_IF_NEQ(0, 1, "Function Not Supported Yet");

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
} 

//============================================================================================
// random utilities
//============================================================================================


template <typename Dtype>
void fillupMatrix(Dtype *A , int lda , int rows, int cols, int seed)
{
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
template void  fillupMatrix<float>(float *A , int lda , int rows, int cols, int seed);
template void  fillupMatrix<double>(double *A , int lda , int rows, int cols, int seed);

template <typename Dtype>
Dtype nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
    b, std::numeric_limits<Dtype>::max());
}
template float nextafter(const float b);
template double nextafter(const double b);


template <typename Dtype>
void host_rng_uniform(const long long int n, const Dtype a, const Dtype b, Dtype* r) {
  ABORT_IF_NEQ(0, 1, "host_rng_uniform not yet supported");


  ABORT_IF_LESS(n, 1, "host_rng_uniform has n < 0");
  ABORT_IF_NEQ(a, b, "host_rng_uniform has a = b");
  ABORT_IF_LESS(b, a, "host_rng_uniform has b < a");
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
void print_host_array(const Dtype* host_pointer, int count, std::string title)
{
  LOG(title<<" : ");
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
  LOG(line<<std::endl);

  
}

template void print_host_array<int>(const int* host_pointer, int count, std::string title);
template void print_host_array<float>(const float* host_pointer, int count, std::string title);


template<typename Dtype>
void save_host_array_to_file(const Dtype* A_host, int count, std::string title)
{
  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < count; i++){
    entries<<A_host[i ];
    if(i < count - 1){
      entries<<", ";
    };
  };
}

template void save_host_array_to_file<int>(const int* A_host, int count, std::string title);
template void save_host_array_to_file<float>(const float* A_host, int count, std::string title);
template void save_host_array_to_file<double>(const double* A_host, int count, std::string title);

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
void save_host_mtx_to_file(const Dtype* A_host, const int lda, const int sda, std::string title)
{
  std::stringstream filename;
  filename << title<< ".txt";
  std::ofstream entries (filename.str().c_str());
  //entries<<"[ ";
  for (int i = 0; i < lda; i++){
    for (int j = 0; j < sda; j++){
      entries<<A_host[i + j * lda];
      entries<<", ";
    }
    entries<<"\r\n";
  }
}

template void save_host_mtx_to_file<int>(const int* A_host, const int lda, const int sda, std::string title);
//template void save_host_mtx_to_file<long long int>(const long long int* A_host, int lda, int sda, std::string title);
template void save_host_mtx_to_file<float>(const float* A_host, const int lda, const int sda, std::string title);
template void save_host_mtx_to_file<double>(const double* A_host, const int lda, const int sda, std::string title);


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
  bool Debug = false;
  LOG("ratings_rows_training : "<< ratings_rows_training);
  LOG("ratings_cols_training : "<< ratings_cols_training);
  //row major ordering
  for(long long int row = 0; row < ratings_rows_training; row ++){
    for(long long int i = csr_format_ratingsMtx_userID_host_training[row]; i < csr_format_ratingsMtx_userID_host_training[row + 1]; i++){
      long long int col = coo_format_ratingsMtx_itemID_host_training[i];
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
}


void cpu_shuffle_mtx_rows_or_cols(cublasHandle_t dn_handle, const long long int M, const long long int N, bool row_major_ordering, float* x, bool shuffle_rows)
{

  bool Debug = false;
  int * indicies_host = NULL;
  int * indicies_dev;

  if(Debug) LOG("cpu_shuffle_mtx_rows_or_cols called") ;
  if(shuffle_rows){
    if(Debug) LOG("shuffle_rows is true") ;
    CUDA_CHECK(cudaMalloc((void**)&indicies_dev, M * sizeof(int)));
    indicies_host = (int *)malloc(M * sizeof(int));
    gpu_set_as_index(indicies_dev, M);
    gpu_shuffle_array<int>(dn_handle, M, indicies_dev);
    CUDA_CHECK(cudaMemcpy(indicies_host, indicies_dev, M * sizeof(int), cudaMemcpyDeviceToHost));
    if(row_major_ordering){
      if(Debug) LOG("row_major_ordering is true") ;
      cpu_permute<float>(x, indicies_host, N, M, !shuffle_rows); 
    }else{
      cpu_permute<float>(x, indicies_host, M, N, shuffle_rows); 
    };
    cudaFree(indicies_dev);
    free(indicies_host);
  }else{
    // shuffle columns
    CUDA_CHECK(cudaMalloc((void**)&indicies_dev, N * sizeof(int)));
    indicies_host = (int *)malloc(N * sizeof(int));
    gpu_set_as_index(indicies_dev, N);
    gpu_shuffle_array<int>(dn_handle, N, indicies_dev);
    CUDA_CHECK(cudaMemcpy(indicies_host, indicies_dev, N * sizeof(int), cudaMemcpyDeviceToHost));
    if(row_major_ordering){
      cpu_permute<float>(x, indicies_host, N, M, shuffle_rows); 
    }else{
      cpu_permute<float>(x, indicies_host, M, N, !shuffle_rows); 
    };
    cudaFree(indicies_dev);
    free(indicies_host);
  }
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
    //for(long long int i = (long long int)0; i < rows * cols; i++) {
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
    //for( long long int entry = (long long int)0; entry < ratings_rows * ratings_rows; entry++){
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
  ABORT_IF_EQ(0, 1, "function not ready");
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
    //for(long long int i = (long long int)0; i < rows; i++){
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
    //for(long long int i = (long long int)0; i < dimension; i++){

      long long int num_below_diag = (long long int)0;
      long long int left_off = (long long int)0;
      long long int num_in_col = dimension - (long long int)1;

      for(long long int j = (long long int)0; j < i; j++){
        left_off = num_below_diag + i - (dimension - num_in_col);

        temp_x[th_id * (dimension - 1) + j] = x[left_off];
        temp_indicies[th_id * (dimension - 1) + j] = j;

        num_below_diag += num_in_col;
        num_in_col -= (long long int)(1);
      }
      left_off = num_below_diag + (i + (long long int)(1)) - (dimension - num_in_col);
      for(long long int j = i + 1; j < dimension; j++){
        temp_x[th_id * (dimension - 1) + j - 1] = x[left_off];
        temp_indicies[th_id * (dimension - 1) + j - 1] = j;
        left_off += (long long int)(1);
      }

      if(i == 0 && debug) {
        print_host_array<Dtype>(temp_x + th_id * (dimension - 1), (int)dimension - 1, "temp_x");
        print_host_array<int>(temp_indicies + th_id * (dimension - 1), (int)dimension - 1, "temp_indicies");
      }
      //LOG("Hello from thread "<<th_id) ;
      //thrust::sort_by_key sorts temp_indicies by temp_x smallest to temp_x largest
      thrust::sort_by_key(thrust::host, temp_x + th_id * (dimension - 1), temp_x + (th_id + 1) * (dimension - 1) , temp_indicies + th_id * (dimension - 1));
      //quickSort_by_key<Dtype>(temp_x + th_id * (dimension - 1), 0, dimension - 1, temp_indicies + th_id * (dimension - 1));
      host_copy(top_N, temp_indicies + (th_id + 1) * (dimension - 1) - top_N, indicies + i * top_N);
      if(i == 0 && debug) {
        print_host_array<Dtype>(temp_x + th_id * (dimension - 1), (int)dimension - 1, "temp_x");
        print_host_array<int>(temp_indicies + th_id * (dimension - 1), (int)dimension - 1, "temp_indicies");
        print_host_array<int>(indicies + i * top_N, top_N, "indicies");
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
      for(long long int i = (long long int)0; i < top_N; i++){
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
    //for(int j = ratings_rows - 1; j > ratings_rows - 1 - ratings_rows_CU; j--){
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





