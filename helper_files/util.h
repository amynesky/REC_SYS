#ifndef UTIL_H_
#define UTIL_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <iterator>
#include <string>
#include <algorithm>
#include <sys/time.h>
#include <boost/algorithm/string.hpp>
#include <boost/random.hpp>
//#include <Eigen/Dense>

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
 
#include "util_gpu.cuh"

#define ABORT_IF_EQ(a, b, output_statement) \
  if(a==b){ \
      std::cout<<__FILE__<<" line "<<__LINE__<<" : "<<output_statement<<std::endl;\
      if (abort) exit(EXIT_FAILURE);\
    }

#define ABORT_IF_NEQ(a, b, output_statement) \
	if(a!=b){ \
    	std::cout<<__FILE__<<" line "<<__LINE__<<" : "<<output_statement<<std::endl;\
    	if (abort) exit(EXIT_FAILURE);\
    }

#define ABORT_IF_LESS(a, b, output_statement) \
	if(a < b){ \
    	std::cout<<__FILE__<<" line "<<__LINE__<<" : "<<output_statement<<std::endl;\
    	if (abort) exit(EXIT_FAILURE);\
    }

#define ABORT_IF_LE(a, b, output_statement) \
  if(a <= b){ \
      std::cout<<__FILE__<<" line "<<__LINE__<<" : "<<output_statement<<std::endl;\
      if (abort) exit(EXIT_FAILURE);\
    }
#define ABORT_IF_GT(a, b, output_statement) \
  if(a > b){ \
      std::cout<<__FILE__<<" line "<<__LINE__<<" : "<<output_statement<<std::endl;\
      if (abort) exit(EXIT_FAILURE);\
    }

#define ABORT_IF_GE(a, b, output_statement) \
  if(a >= b){ \
      std::cout<<__FILE__<<" line "<<__LINE__<<" : "<<output_statement<<std::endl;\
      if (abort) exit(EXIT_FAILURE);\
    }

#define LOG(output_statement) \
    	std::cout<<__FILE__<<" line "<<__LINE__<<" at "<<currentDateTime()<<" : "<<output_statement<<std::endl;

#define LOG2(preamble, output_statement) \
      std::cout<<preamble<<" : "<<output_statement<<std::endl;

#define strPreamble(blank)\
    ((((blank + __FILE__) + " line ") + ToString(__LINE__)) + " at ") + currentDateTime()

#define SIZE_OF(val)\
    static_cast<long long int>(sizeof(val))


#define checkErrors(ans) { Hassert((ans), __FILE__, __LINE__); }
template <typename Dtype>
inline void Hassert(const Dtype *ptr, const char *file, int line, bool abort=true)
{
   if (ptr == NULL) 
   {
      fprintf(stderr,"assert: %s %d\n",  file, line);
      if (abort) {ABORT_IF_NEQ(0, 1, "Fatal: failed to allocate");};
   }
}

bool too_big(const long long int li);

const std::string currentDateTime();

template <typename Dtype>
bool cpu_isBad(const Dtype* A, long long int size, std::string var_name = "", std::string file_line = "");

//============================================================================================
// manipulate memory
//============================================================================================


template <typename Dtype>
void host_copy(const long long int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void copy_device_mtx_into_host_submtx(const long long int M, const long long int N, const Dtype* X, Dtype* Y, const long long int inc_Y);
//============================================================================================
// math functions
//============================================================================================

template <typename Dtype>
void cpu_incremental_average_array(const long long int increment_index, Dtype* old_avg, Dtype* new_val, int num);

template <typename Dtype>
void cpu_incremental_average(const long long int increment_index, Dtype* old_avg, Dtype new_val);

template<typename Dtype>
void cpu_mean_abs_nonzero(const long long int n, const Dtype* x, Dtype* y, bool Debug = false, std::string vec_name = "") ;

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype cpu_asum(const long long int n, const Dtype* x);

template <typename Dtype>
Dtype cpu_sum(const long long int n, const Dtype* x);

template<typename Dtype>
Dtype cpu_sum_of_squares(const long long int n, const Dtype* x);

template <typename Dtype>
void cpu_scal(const long long int N, const Dtype alpha, Dtype *X);

int gcd(int a, int b);

template < typename Dtype>
void cpu_permute(Dtype* a, const int* pvt, const long long int rows, const long long int cols, bool permute_rows);

template < typename Dtype>
void MatrixInplaceTranspose(Dtype *A, long long int r, long long int c, bool row_major_ordering = true);

template < typename Dtype>
void cpu_axpby(const long long int N, const Dtype alpha, const Dtype* X, const Dtype beta, Dtype* Y);

void cpu_axpby_test();

template <typename Dtype>
void cpu_gemm(const bool TransA, const bool TransB, 
              const long long int M, const long long int N, const long long int K,
             const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
             Dtype* C, long long int start_ = (long long int)0, long long int num_ = (long long int)0);

void cpu_gemm_test();

template <typename Dtype>
void cpu_swap_ordering(const long long int rows, const long long int cols, Dtype *A, const bool row_major_ordering);

template <typename Dtype>
Dtype cpu_min(const long long int n,  const Dtype* x);

template <typename Dtype>
Dtype cpu_abs_max(const long long int n,  const Dtype* x);

template <typename Dtype> 
Dtype cpu_expected_value(const long long int n,  const Dtype* x);

template <typename Dtype> 
Dtype cpu_variance(const long long int n,  const Dtype* x);

template <typename Dtype>
Dtype cpu_expected_abs_value(const long long int n,  const Dtype* x);

//============================================================================================
// random utilities
//============================================================================================

int64_t cluster_seedgen(void);

template < typename Dtype>
void  fillupMatrix(Dtype *A , int lda , int rows, int cols);

template <typename Dtype>
void host_rng_uniform(const long long int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void host_rng_gaussian(const long long int n, const Dtype a, const Dtype b, Dtype* r);

void getRandIntsBetween(int *A , int lower_bd , int upper_bd, int num);
//============================================================================================
// prints
//============================================================================================

template <typename Dtype>
std::string ToString(const Dtype val);

extern "C" std::string readable_time(double ms);

template < typename Dtype>
void printPartialMatrices(Dtype * A, Dtype * B, int rows, int cols , int ld);

template < typename Dtype>
void printPartialMtx(Dtype * A, int rows, int cols , int ld);

template<typename Dtype>
void print_host_array(const Dtype* host_pointer, int count, std::string title, std::string file_line = "");

template<typename Dtype>
void print_host_mtx(const Dtype* A_host, const int rows, const int cols, std::string title, 
                    bool row_major_order = true, std::string file_line = "");

template <typename Dtype>
void save_host_array_to_file(const Dtype* A_host, int count, std::string title, std::string file_line = "");

template <typename Dtype>
void get_host_array_from_saved_txt(const Dtype* A_host, int count, std::string title);

template <typename Dtype>
void append_host_array_to_file(const Dtype* A_host, int count, std::string title, std::string file_line = "");

template<typename Dtype>
void save_host_arrays_side_by_side_to_file(const Dtype* A_host, const Dtype* B_host, int count, std::string title, std::string file_line = "");

template<typename Dtype>
void save_host_array_side_by_side_with_device_array(const Dtype* A_host, const Dtype* B_dev, int count, std::string title, std::string file_line = "");

void save_host_arrays_side_by_side_to_file_(const int* A_host, const int* B_host, const float* C_host, int count, std::string title);

template <typename Dtype>
void save_host_mtx_to_file(const Dtype* A_host, const int rows, const int  cols, std::string title, bool row_major_order = true, std::string file_line = "");

template<typename Dtype>
void append_host_mtx_to_file(const Dtype* A_host, const int rows, const int cols, std::string title, bool row_major_order, std::string file_line);

void save_map(std::map<int, int>* items_dictionary, std::string title);

//============================================================================================
// Me Made
//============================================================================================

void cpu_fill_training_mtx(const long long int ratings_rows_training, const long long int ratings_cols_training,  
                           const long long int num_entries_CU, const bool row_major_ordering,
                            const int* csr_format_ratingsMtx_userID_host_training,
                            const int* coo_format_ratingsMtx_itemID_host_training,
                            const float* coo_format_ratingsMtx_rating_host_training,
                            float* full_training_ratings_mtx);

template < typename Dtype>
void cpu_shuffle_array(const long long int n,  Dtype* x);

void cpu_shuffle_map_second(const long long int M, std::map<int, int>* items_dictionary );

template <typename Dtype>
void cpu_set_all(Dtype* x, const long long int N, Dtype alpha);

template <typename Dtype>
void cpu_set_as_index(Dtype* x, const long long int rows, const long long int cols);

void cpu_shuffle_mtx_rows_or_cols(const long long int M, const long long int N, 
                                  bool row_major_ordering, float* x,
                                  bool shuffle_rows);

void cpu_get_cosine_similarity(const long long int ratings_rows, 
                              const int* csr_format_ratingsMtx_userID_host,
                              const int* coo_format_ratingsMtx_itemID_host,
                              const float* coo_format_ratingsMtx_rating_host,
                              float* cosine_similarity);

long long int cpu_compute_hidden_values (const long long int ratings_rows, 
  const long long int ratings_cols, const int Top_N, const long long int num_entries,
  const int* csr_format_ratingsMtx_userID_host,
  const int* coo_format_ratingsMtx_itemID_host,
  const float* coo_format_ratingsMtx_rating_host,
  const std::vector<std::vector<int> >* top_N_most_sim_itemIDs_host,
  const std::vector<std::vector<float> >* top_N_most_sim_item_similarity_host,
  int**   coo_format_ratingsMtx_userID_host_new,
  int**   coo_format_ratingsMtx_itemID_host_new,
  float** coo_format_ratingsMtx_rating_host_new);

template<typename Dtype>
void quickSort(Dtype* x, int low_index, int high_index);

template<typename Dtype, typename Itype>
void quickSort_by_key(Dtype* x, int low_index, int high_index, Itype* indicies);

template <typename Dtype>
void cpu_sort_csr_colums(const long long int ratings_rows, 
                                const int *csr_format_ratingsMtx_userID_host,
                                int* coo_format_ratingsMtx_itemID_host,
                                Dtype* coo_format_ratingsMtx_rating_host, 
                                long long int num_entries_ = (long long int)0,
                                std::string preprocessing_path = "");

void cpu_sort_csr_colums_test();

template<typename Dtype, typename Itype>
void naiveSort_by_key(Dtype* x, int total, int num_sorted, Itype* indicies, Itype* indicies_sorted = NULL, Dtype* x_sorted = NULL);

template <typename Dtype, typename Itype>
void cpu_sort_index_by_max(const long long int rows, const long long int cols,  Dtype* x, Itype* indicies);

template<typename Dtype>
void cpu_sort_index_by_max(const long long int dimension,  Dtype* x, int* indicies, int top_N, Dtype* x_sorted = NULL);


void cpu_count_appearances(const int top_N, const long long int dimension,  int* count, const int* indicies );

void cpu_rank_appearances(const int top_N, const long long int dimension,  float* rank, const int* indicies );

void cpu_mark_CU_users(const int ratings_rows_CU, const int ratings_rows, const int* x, int* y );



long long int from_below_diag_to_whole(long long int below_diag_index, long long int dimension);

long long int from_below_diag_to_whole_faster(long long int below_diag_index, long long int dimension);

void from_below_diag_to_whole_test();

long long int from_whole_to_below_diag(long long int whole_index, long long int dimension);


template <typename Dtype>
void cpu_get_num_latent_factors(const long long int m, Dtype* S_host, 
                                long long int* num_latent_factors, const Dtype percent);

template <typename Dtype>
void cpu_get_latent_factor_mass(const long long int m, Dtype* S_host, 
  const long long int num_latent_factors, Dtype *percent);

template <typename Dtype>
void cpu_orthogonal_decomp(const long long int m, const long long int n, const bool row_major_ordering,
                          long long int* num_latent_factors, const Dtype percent,
                          Dtype* A, Dtype* U, Dtype* V, bool SV_with_U = false, Dtype* S = NULL);

void cpu_orthogonal_decomp_test();

void cpu_center_rows(const long long int rows, const long long int cols, 
                 float* X, const float val_when_var_is_zero, float* user_means,  float* user_var);

template <typename Dtype>
void cpu_logarithmic_histogram_abs_val(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
                                       const int rows_B, const int num_sparse_entries, 
                                       const int* csr_rows_B, const int* coo_cols_B,
                                       const Dtype* coo_entries_B, int* selection, 
                                       int min_pow, int max_pow, Dtype* probability);

template <typename Dtype>
void cpu_sparse_nearest_row(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const int num_sparse_entries, const int* csr_rows_B, const int* coo_cols_B,
 const Dtype* coo_entries_B, int* selection,  
 Dtype* error);

template<typename Dtype>
void cpu_dense_nearest_row(const int rows_A, const int cols, const Dtype* dense_mtx_A, 
 const int rows_B, const Dtype* dense_mtx_B, int* selection,  
 Dtype* error);

template <typename Dtype>
void cpu_calculate_KM_error_and_update(const int rows_A, const int cols, Dtype* dense_mtx_A, 
    const int rows_B, const Dtype* dense_mtx_B, int* selection, Dtype alpha, Dtype lambda, Dtype* checking = NULL);


template <typename Dtype>
void cpu_supplement_training_mtx_with_item_sim(const long long int ratings_rows_training, 
  const long long int ratings_cols_training, 
  const bool row_major_ordering,
  const int* csr_format_ratingsMtx_userID_training,
  const int* coo_format_ratingsMtx_itemID_training,
  const Dtype* coo_format_ratingsMtx_rating_training,
  Dtype* full_training_ratings_mtx,
  const int* top_N_most_sim_itemIDs,
  const Dtype* top_N_most_sim_item_similarity, const int top_N);

#ifndef CPU_ONLY  // GPU

#endif  // !CPU_ONLY


#endif //UTIL_H_