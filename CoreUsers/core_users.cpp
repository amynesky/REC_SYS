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

#if defined(ENABLE_CUDA)
/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime.h>
#include "/home/nesky/REC_SYS/helper_files/helper_cuda.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include "cusolverSp.h"
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
//#include "helper_cusolver.h"
#include "cusolverDn.h"
#include "/home/nesky/REC_SYS/helper_files/util_gpu.cuh"
#endif

// Utilities and system includes
#include "/home/nesky/REC_SYS/helper_files/CSVReader.h"
#include "/home/nesky/REC_SYS/helper_files/util.h"

#include "core_users.h"

const char *sSDKname = "Core Users Recommender Systems";

const bool Debug = 1;

bool Content_Based = 0;
bool Conserve_GPU_Mem = 1;
bool load_from_preprocessing = 1;
std::string preprocessing_path = "";
bool consider_item_cosine_similarity = true;
    bool CU_ratingsMtx_has_hidden_values = false;
bool user_cosine_similarity_incorporates_ratings = true; // do you just count the intersection or do you us actual ratings?
bool frequency_based = false; // otherwise rank_based

#define update_Mem(new_mem) \
    allocatedMem += static_cast<long long int>(new_mem); \
    memLeft = static_cast<long long int>(devMem) - allocatedMem; 
    //ABORT_IF_LESS(memLeft, 0, "Out of Memory"); \
    //if(Debug) LOG(allocatedMem<<" allocated bytes on the device"); \
    //if(Debug) LOG(memLeft<<" available bytes left on the device");
    //ABORT_IF_LESS(allocatedMem, (long long int)((double)devMem * (double)0.75), "Out of Memory"); \
    // if(Debug) LOG((int)devMem <<" total bytes on the device");\
    // if(Debug) LOG(new_mem<<" change in memory on the device");\






// #define allocate_V() \
//     if (d_S    ) cudaFree(d_S);
//     if (d_S    ) cudaFree(d_S);
//     if (d_S    ) cudaFree(d_S);
//     checkCudaErrors(cudaMalloc((void**)&U_CU,       batch_size_CU       * batch_size_CU                       * SIZE_OF(float)));
//     checkCudaErrors(cudaMalloc((void**)&V,          ratings_cols        * ratings_cols                        * SIZE_OF(float)));
//     checkCudaErrors(cudaMalloc((void**)&U_testing, batch_size_testing * std::max(min_CU_dimension, batch_size_testing) * SIZE_OF(float)));
//     checkCudaErrors(cudaMalloc((void**)&R_testing, batch_size_testing * ratings_cols                        * SIZE_OF(float)));
//     checkCudaErrors(cudaMalloc((void**)&U_testing,  batch_size_testing  * std::max(min_CU_dimension, batch_size_testing)  * SIZE_OF(float)));
//     checkCudaErrors(cudaMalloc((void**)&R_testing,  batch_size_testing  * ratings_cols                        * SIZE_OF(float)));
//     update_Mem(Training_bytes);











int main(int argc, char *argv[])
{
    struct timeval program_start, program_end, testing_start, testing_end;
    double program_time;
    double testing_time;
    gettimeofday(&program_start, NULL);
    std::string blank = "";

    long long int allocatedMem = (long long int)0; 



    // gpu_sort_csr_colums_test();
    // return 0;
    


    /*

        from_below_diag_to_whole_test();
        return 0;


        const long long int dimension = (long long int)131262; //138493
        const long long int num_below_diag = (dimension * (dimension - (long long int)1)) / (long long int)2;

        for(long long int i = 0; i < num_below_diag; i +=(long long int)1){
            long long int whole_slow_way = from_below_diag_to_whole(i, dimension);
            long long int whole_faster = from_below_diag_to_whole_faster(i, dimension);
            if(whole_slow_way != whole_faster){
                LOG("below_diag_indicies "<<i<<" maps to -> whole_indicies "   << whole_slow_way);
                LOG("below_diag_indicies "<<i<<" maps to -> whole_indicies "   <<whole_faster );
            }
            long long int below_after = from_whole_to_below_diag(whole_faster, dimension);
            if(below_after != i){
                LOG("whole_indicies "<<whole_faster<<" maps to -> below_diag_indicies "   <<below_after );
            }
            below_after = from_whole_to_below_diag(whole_slow_way, dimension);
            if(below_after != i){
                LOG("whole_indicies "<<whole_slow_way<<" maps to -> below_diag_indicies "   <<below_after );
            }
        }

        return 0;
    */
    


    /*
        //long long int below_index = (long long int)37;
        long long int ratings_rows_ = (long long int)10;
        for(long long int below_index = 0; below_index < 45; below_index++){



            LOG("below_index : "<<below_index);

            long long int whole_index = from_below_diag_to_whole_faster(below_index, ratings_rows_);
            int user_i = (int)(whole_index % ratings_rows_);
            int user_j = (int)(whole_index / ratings_rows_);
            LOG("whole_index : "<<whole_index);
            LOG("max whole index =  : "<<ratings_rows_ * ratings_rows_ - (long long int)1);
            LOG("row : "<<user_i);
            LOG("col : "<<user_j);

            long long int whole_index_slow = from_below_diag_to_whole(below_index, ratings_rows_);
            int user_i_slow = (int)(whole_index_slow % ratings_rows_);
            int user_j_slow = (int)(whole_index_slow / ratings_rows_);
            LOG("slow whole_index : "<<whole_index_slow);
            LOG("slow row : "<<user_i_slow);
            LOG("slow col : "<<user_j_slow<<std::endl);
        }
        return 0;
        //}  
    */


    cublasStatus_t cublas_status     = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;


    printf("%s Starting...\n\n", sSDKname);
    std::cout << "Current Date and Time :" << currentDateTime() << std::endl<< std::endl;
    LOG("Debug = "<<Debug);

    /* initialize random seed: */
    srand (time(0));

    int dev = findCudaDevice(argc, (const char **) argv);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    //printf("Major revision number:         %d\n",  devProp.major);
    //printf("Minor revision number:         %d\n",  devProp.minor);
    int major = devProp.major;
    int minor = devProp.minor;
    float compute_capability = (float)major + (float)minor / (float)10;
    //printf("\nDevice Compute Capability: %f \n\n", compute_capability);

    long long int devMem = std::abs(static_cast<long long int>(getDeviceMemory()));
    LOG("devMem : "<<devMem);
    long long int memLeft = devMem;

    if (dev == -1)
    {
        return CUBLASTEST_FAILED;
    }

    //============================================================================================
    // Dense matrix handles and descriptors
    //============================================================================================
    cublasHandle_t dn_handle;
    if (cublasCreate(&dn_handle) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stdout, "CUBLAS initialization failed!\n");
        exit(EXIT_FAILURE);
    };

    cusolverDnHandle_t dn_solver_handle;
    if (cusolverDnCreate(&dn_solver_handle) != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stdout, "DN CUSOLVER initialization failed!\n");
        exit(EXIT_FAILURE);
    };


    //============================================================================================
    // Sparse matrix handles and descriptors
    //============================================================================================
    cusparseHandle_t sp_handle;

    if (cusparseCreate(&sp_handle) != CUSPARSE_STATUS_SUCCESS) { 
        fprintf(stdout, "CUSPARSE initialization failed!\n");
        exit(EXIT_FAILURE);
    };

    /* create and setup matrix descriptor */ 
    cusparseMatDescr_t sp_descr;
    if (cusparseCreateMatDescr(&sp_descr) != CUSPARSE_STATUS_SUCCESS) {  
        fprintf(stdout, "SP Matrix descriptor initialization failed\n");
        exit(EXIT_FAILURE);
    } 
    cusparseSetMatType(sp_descr,CUSPARSE_MATRIX_TYPE_GENERAL);



    cusolverSpHandle_t sp_solver_handle;
    if (cusolverSpCreate(&sp_solver_handle) != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stdout, "SP CUSOLVER initialization failed!\n");
        exit(EXIT_FAILURE);
    };

    //cpu_orthogonal_decomp_test();
    //gpu_block_orthogonal_decomp_from_host_test(dn_handle, dn_solver_handle);
    //return 0;

    //============================================================================================
    // Get the ratings data from CSV File
    //============================================================================================


    // Creating an object of CSVWriter

    std::string Dataset_Name;

    
    std::string csv_Ratings_Path;
    std::string csv_keyWords_path;

    long long int temp_num_entries;

    std::map<int, int> items_dictionary;

    int case_ = 1;

    std::string preamble = "";
    // if( current_path == "/bridges/REC_SYS/CoreUsers"){
    //     preamble = "/Users/amynesky";
    // }


    switch (case_)
    {
        case 1:{ // code to be executed if n = 1;

            Dataset_Name = "MovieLens 20 million";

            csv_Ratings_Path = (preamble + "/pylon5/ac560rp/nesky/REC_SYS/datasets/ml-20m/ratings.csv").c_str();
            csv_keyWords_path = (preamble + "/pylon5/ac560rp/nesky/REC_SYS/datasets/ml-20m/movies.csv").c_str();
            //temp_num_entries = csv_Ratings.num_rows() - 1; // the first row is a title row
            Content_Based = 0;
            preprocessing_path = "/pylon5/ac560rp/nesky/REC_SYS/CoreUsers/preprocessing/ml-20m/";
            temp_num_entries = 20000264 - 1;   // MovieLens 20 million
            break;
        }case 2:{ // code to be executed if n = 2;
            Dataset_Name = "Rent The Runaway";
            csv_Ratings_Path = (preamble + "/pylon5/ac560rp/nesky/REC_SYS/datasets/renttherunway_final_data.json").c_str();
            //csv_keyWords_path = (preamble + "/pylon5/ac560rp/nesky/REC_SYS/datasets/renttherunway_final_data.json").c_str();
            Content_Based = 0;
            temp_num_entries = 192544;           // use for Rent The Runaway dataset
            break;
        }default: 
            ABORT_IF_EQ(0, 1, "no valid dataset selected");
    }

    if(consider_item_cosine_similarity){
       preprocessing_path = (preprocessing_path + "consider_item_cosine_similarity/").c_str();
    }else{
        preprocessing_path = (preprocessing_path + "dont/").c_str();
    }
    if(user_cosine_similarity_incorporates_ratings){
       preprocessing_path = (preprocessing_path + "user_cosine_similarity_incorporates_ratings/").c_str();
    }else{
        preprocessing_path = (preprocessing_path + "dont/").c_str();
    }
    if(frequency_based){
       preprocessing_path = (preprocessing_path + "frequency_based/").c_str();
    }else{
        preprocessing_path = (preprocessing_path + "not/").c_str();
    } 

    LOG("Training using the "<< Dataset_Name <<" dataset");
    LOG("csv_Ratings_Path : "<< csv_Ratings_Path);
    LOG("csv_keyWords_path : "<< csv_keyWords_path <<" dataset");
    LOG("Content_Based : "<< Content_Based<<std::endl);
    LOG("consider_item_cosine_similarity : "<< consider_item_cosine_similarity);
    if(consider_item_cosine_similarity){
        LOG("CU_ratingsMtx_has_hidden_values : "<< CU_ratingsMtx_has_hidden_values<<std::endl);
    }else{
        CU_ratingsMtx_has_hidden_values = false;
        LOG("");
    }
    LOG("user_cosine_similarity_incorporates_ratings : "<< user_cosine_similarity_incorporates_ratings<<std::endl);
    LOG("frequency_based : "<< frequency_based<<std::endl);
    LOG("load_from_preprocessing : "<< load_from_preprocessing<<std::endl);
    LOG("preprocessing_path : "<< preprocessing_path<<std::endl);




    CSVReader csv_Ratings(csv_Ratings_Path);
    

    const long long int num_entries = temp_num_entries;
    //const long long int num_entries = temp_num_entries/10; //for debuging code

    LOG("The dataset has "<<num_entries<<" specified entries.");
    
    int*   coo_format_ratingsMtx_userID_host  = NULL;
    int*   coo_format_ratingsMtx_itemID_host  = NULL;
    float* coo_format_ratingsMtx_rating_host  = NULL;
    coo_format_ratingsMtx_userID_host = (int *)  malloc(num_entries *  SIZE_OF(int)); 
    coo_format_ratingsMtx_itemID_host = (int *)  malloc(num_entries *  SIZE_OF(int)); 
    coo_format_ratingsMtx_rating_host = (float *)malloc(num_entries *  SIZE_OF(float)); 
    checkErrors(coo_format_ratingsMtx_userID_host);
    checkErrors(coo_format_ratingsMtx_itemID_host);
    checkErrors(coo_format_ratingsMtx_rating_host);



    bool load_all = true;
    bool original_column_ordering = true;
    original_column_ordering = original_column_ordering || Content_Based;// || load_CU_from_preprocessing || load_full_GU_from_save
    if(!load_all){
        original_column_ordering = false;
    }


    switch (case_)
    {
        case 1:{ // code to be executed if n = 1;

            //Dataset_Name = "MovieLens 20 million";

            csv_Ratings.getData(coo_format_ratingsMtx_userID_host,
                                coo_format_ratingsMtx_itemID_host,
                                coo_format_ratingsMtx_rating_host, 
                                num_entries, 
                                load_all || original_column_ordering, 
                                &items_dictionary);
            break;
        }case 2:{ // code to be executed if n = 2;
            //Dataset_Name = "Rent The Runaway";
            int missing_ = 5;
            LOG("value used to fill in missing ratings : "<< missing_) ;
            csv_Ratings.getDataJSON(coo_format_ratingsMtx_userID_host,
                                    coo_format_ratingsMtx_itemID_host,
                                    coo_format_ratingsMtx_rating_host, 
                                    num_entries, missing_);
            break;
        }default: 
            ABORT_IF_EQ(0, 1, "no valid dataset selected");
    }

    int max_col = cpu_abs_max(num_entries, coo_format_ratingsMtx_itemID_host);
    int max_row = cpu_abs_max(num_entries, coo_format_ratingsMtx_userID_host);
    LOG("max_row : "<< max_row) ;
    LOG("max_col : "<< max_col) ;

    if(Debug && 0){
        // save_host_arrays_side_by_side_to_file(coo_format_ratingsMtx_userID_host, coo_format_ratingsMtx_itemID_host, 
        //                                       coo_format_ratingsMtx_rating_host, (int)num_entries, "rows_cols_rating");
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();          
    }
 



    int*   coo_format_ratingsMtx_userID_dev;
    int*   coo_format_ratingsMtx_itemID_dev;
    float* coo_format_ratingsMtx_rating_dev;
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_userID_dev,  num_entries * SIZE_OF(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev,  num_entries * SIZE_OF(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev,  num_entries * SIZE_OF(float)));
    update_Mem(2 * num_entries * SIZE_OF(int) + num_entries * SIZE_OF(float));

    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_userID_dev,  coo_format_ratingsMtx_userID_host,  num_entries * SIZE_OF(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_dev,  coo_format_ratingsMtx_itemID_host,  num_entries * SIZE_OF(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev,  coo_format_ratingsMtx_rating_host,  num_entries * SIZE_OF(float), cudaMemcpyHostToDevice));

    if(Debug){
        // Print the content of row by row on screen
        // for(int row = 0; row <2; row++)
        // {
        //     for(int col = 0; col <dims; col++)
        //     {
        //         std::cout<<ratings_host[row + col * num_entries] << " , ";
        //         if (col < dims-1 ) std::cout<<" , ";
        //     }
        //     std::cout<<std::endl;
        // }
    }

    const long long int ratings_rows = (long long int)(gpu_abs_max<int>(num_entries, coo_format_ratingsMtx_userID_dev) + 1); 
    const long long int ratings_cols = (long long int)(gpu_abs_max<int>(num_entries, coo_format_ratingsMtx_itemID_dev) + 1); 


    int*   csr_format_ratingsMtx_userID_dev;
    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev, (ratings_rows + 1) * SIZE_OF(int)));
    update_Mem( (ratings_rows + 1) * SIZE_OF(int) );

    cusparse_status = cusparseXcoo2csr(sp_handle, coo_format_ratingsMtx_userID_dev, num_entries, 
                                       ratings_rows, csr_format_ratingsMtx_userID_dev, CUSPARSE_INDEX_BASE_ZERO); 
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 
        fprintf(stdout, "Conversion from COO to CSR format failed\n");
        return 1; 
    } 

    LOG("The sparse data matrix has "<<ratings_rows<<" users and "<<ratings_cols<<" items with "<<num_entries<<" specified entries.");
    LOG("The sparse data matrix has "<<(float)(ratings_rows * ratings_cols - num_entries) / (float)(ratings_rows * ratings_cols)<<" empty entries.");
    if(Debug && 0){
        save_device_array_to_file<int>(csr_format_ratingsMtx_userID_dev, ratings_rows + 1, "csr_format_ratingsMtx_userID_dev");
        // LOG("csr_format_ratingsMtx_userID_dev : ");
        // print_gpu_array_entries<int>(csr_format_ratingsMtx_userID_dev, ratings_rows + 1, 1 , ratings_rows + 1);
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }




    long long int num_entries_keyWord_mtx_temp = (long long int)0;
    long long int num_keyWords_temp            = (long long int)0;

    int*   coo_format_keyWordMtx_itemID_host  = NULL;
    int*   coo_format_keyWordMtx_keyWord_host = NULL;
    int*   coo_format_keyWordMtx_itemID_dev   = NULL;
    int*   coo_format_keyWordMtx_keyWord_dev  = NULL;
    if(Content_Based){
        CSVReader csv_keyWords(csv_keyWords);
        //num_entries_keyWord_mtx_temp = csv_keyWords.num_entries();
        switch (case_)
        {
            case 1:{ // code to be executed if n = 1;
                //Dataset_Name = "MovieLens 20 million";
                num_entries_keyWord_mtx_temp = ratings_cols;
                break;
            }case 2:{ // code to be executed if n = 2;
                //Dataset_Name = "Rent The Runaway";
                //num_entries_keyWord_mtx_temp = 48087;
                break;
            }default: 
                ABORT_IF_EQ(0, 1, "no valid dataset selected");
        }
        //num_entries_keyWord_mtx = 48087;
        LOG("num_entries_keyWord_mtx : "<<num_entries_keyWord_mtx_temp);

        if(num_entries_keyWord_mtx_temp <= 0 ){
            return 0;
        }

        coo_format_keyWordMtx_itemID_host  = (int *)malloc(num_entries_keyWord_mtx_temp *  SIZE_OF(int)); 
        coo_format_keyWordMtx_keyWord_host = (int *)malloc(num_entries_keyWord_mtx_temp *  SIZE_OF(int)); 
        checkErrors(coo_format_keyWordMtx_itemID_host);
        checkErrors(coo_format_keyWordMtx_keyWord_host);

        num_keyWords_temp  = csv_keyWords.makeContentBasedcooKeyWordMtx(coo_format_keyWordMtx_itemID_host,
                                                                        coo_format_keyWordMtx_keyWord_host,
                                                                        num_entries_keyWord_mtx_temp);
        LOG("num_keyWords : "<<num_keyWords_temp);

        checkCudaErrors(cudaMalloc((void**)&coo_format_keyWordMtx_itemID_dev,   num_entries_keyWord_mtx_temp * SIZE_OF(int)));
        checkCudaErrors(cudaMalloc((void**)&coo_format_keyWordMtx_keyWord_dev,  num_entries_keyWord_mtx_temp * SIZE_OF(int)));
        update_Mem(2 * num_entries_keyWord_mtx_temp * SIZE_OF(int) );

        checkCudaErrors(cudaMemcpy(coo_format_keyWordMtx_itemID_dev,   coo_format_keyWordMtx_itemID_host,   num_entries_keyWord_mtx_temp * SIZE_OF(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(coo_format_keyWordMtx_keyWord_dev,  coo_format_keyWordMtx_keyWord_host,  num_entries_keyWord_mtx_temp * SIZE_OF(int), cudaMemcpyHostToDevice));
        free(coo_format_keyWordMtx_itemID_host);
        free(coo_format_keyWordMtx_keyWord_host);
    }
    const long long int num_entries_keyWord_mtx = num_entries_keyWord_mtx_temp;
    const long long int num_keyWords            = num_keyWords_temp;

    int*   csr_format_keyWordMtx_itemID_dev;
    if(Content_Based){
        checkCudaErrors(cudaMalloc((void**)&csr_format_keyWordMtx_itemID_dev, (ratings_cols + 1) * SIZE_OF(int)));
        update_Mem( (ratings_cols + 1) * SIZE_OF(int) );

        cusparse_status = cusparseXcoo2csr(sp_handle, coo_format_keyWordMtx_itemID_dev, num_entries_keyWord_mtx, 
                                           ratings_cols, csr_format_keyWordMtx_itemID_dev, CUSPARSE_INDEX_BASE_ZERO); 
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 
            fprintf(stdout, "Conversion from COO to CSR format failed\n");
            return 1; 
        } 
        cudaFree(coo_format_keyWordMtx_itemID_dev);           update_Mem(num_entries_keyWord_mtx * SIZE_OF(int) * (-1));
    }















    //============================================================================================
    // collect User Means and Variances
    //============================================================================================










    LOG("collect User Means and Variance... ");

    float* user_means;
    checkCudaErrors(cudaMalloc((void**)&user_means,  ratings_rows * SIZE_OF(float)));
    update_Mem(( ratings_rows )* SIZE_OF(float));

    float* user_var;
    checkCudaErrors(cudaMalloc((void**)&user_var,  ratings_rows * SIZE_OF(float)));
    update_Mem((ratings_rows)* SIZE_OF(float));

    collect_user_means(user_means, user_var,  (long long int)ratings_rows,
                       csr_format_ratingsMtx_userID_dev,
                       coo_format_ratingsMtx_rating_dev);

    if(Debug && 0){
        save_device_array_to_file<float>(user_means,  (int)ratings_rows,  "user_means");
        // LOG("user_means_testing : ");
        // print_gpu_array_entries(user_means_CU, ratings_rows_CU);
        // LOG("user_means_testing : ");
        // print_gpu_array_entries(user_means_CU, ratings_rows_CU);
        // LOG("user_means_CU : ");
        // print_gpu_array_entries(user_means_CU, ratings_rows_CU);
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }











    //============================================================================================
    // Center Data
    //============================================================================================










    LOG("Center Data... ");


    float * coo_format_ratingsMtx_row_centered_rating_dev;
    
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_row_centered_rating_dev, num_entries * SIZE_OF(float)));
    update_Mem( ( num_entries) * SIZE_OF(float) );




    //const float val_when_var_is_zero = (float)3.5774;        // use for MovieLens
    const float val_when_var_is_zero = (float)0.5;        // use for Rent The Runway
    LOG("rating used when the variance of the user's ratings is zero : "<< val_when_var_is_zero);

    center_ratings(user_means, user_var, 
                   ratings_rows, num_entries,
                   csr_format_ratingsMtx_userID_dev,
                   coo_format_ratingsMtx_rating_dev,
                   coo_format_ratingsMtx_row_centered_rating_dev, 
                   val_when_var_is_zero);

    //the stored means are useless now
    cudaFree(user_means);                        update_Mem(ratings_rows * SIZE_OF(float) * (-1));
    cudaFree(user_var);                          update_Mem(ratings_rows * SIZE_OF(float) * (-1));

    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev, coo_format_ratingsMtx_row_centered_rating_dev,  num_entries *  SIZE_OF(float), cudaMemcpyDeviceToDevice));
    cudaFree(coo_format_ratingsMtx_row_centered_rating_dev);             
    update_Mem((ratings_rows+ 1) * SIZE_OF(int) * (-1));
    
    float range = gpu_range<float>(num_entries,        coo_format_ratingsMtx_rating_dev);


    
    if( Debug && 0){

        //LOG("range_CU = "      <<range_CU) ;
        //LOG("range_testing = " <<range_testing) ;


        save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev,  ratings_rows + 1, "csr_format_ratingsMtx_userID_dev_post_centering");
        save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev, num_entries,      "coo_format_ratingsMtx_itemID_dev_post_centering");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev,  num_entries,      "coo_format_ratingsMtx_rating_dev_post_centering");
        

        // LOG("csr_format_ratingsMtx_userID_dev_testing : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_testing, ratings_rows_testing + 1, 1 );
        // LOG("csr_format_ratingsMtx_userID_dev_CU : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_CU, ratings_rows_CU + 1, 1 );
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }

    //checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_userID_host,  csr_format_ratingsMtx_userID_dev,  (ratings_rows + 1) * SIZE_OF(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host,  coo_format_ratingsMtx_itemID_dev,  num_entries * SIZE_OF(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host,  coo_format_ratingsMtx_rating_dev,  num_entries * SIZE_OF(float), cudaMemcpyDeviceToHost));
    
    free(coo_format_ratingsMtx_userID_host);
    //============================================================================================
    // split the data into core users and testing data
    //============================================================================================

    

    /*
        In the third method, we first compute the top-N (e.g. N 5 10, 20, 50) 
        most similar neighbors of each user based on the cosine similarities, 
        and then count how many times a user has appeared in other users’ top-N lists.
        Those users who appear most frequently are selected as the information core.
    */




    const float probability_CU       = (float)10.0/(float)100.0;
    const float probability_testing  = ((float)1.0 - probability_CU);
    LOG("percentage of users for testing: " <<probability_testing);
    LOG("percentage of users for CU: "      <<(float)1.0 - probability_testing<<std::endl);

    long long int ratings_rows_CU_temp       = (long long int)(probability_CU * (float)ratings_rows);
    if(1){
        ratings_rows_CU_temp  /= (long long int)1000;
        ratings_rows_CU_temp  *= (long long int)1000; // now ratings_rows_GU_temp is divisible by 100
    }
    const long long int ratings_rows_CU      = ratings_rows_CU_temp;
    const long long int ratings_rows_testing = ratings_rows - ratings_rows_CU;

    LOG("num testing users : "   <<ratings_rows_testing);
    LOG("num CU users : "        <<ratings_rows_CU);
    if(ratings_rows_CU == 0 || ratings_rows_testing == 0) {
        LOG("One group has no users in it.")
        return 0;
    }


    int top_N = std::min((int)((float)ratings_rows / (float)10.0), 50);
    LOG("Compute the top-"<<top_N<<" most similar neighbors of each user based on the cosine similarities.");
    int top_N_items = std::min((int)((float)ratings_cols / (float)10.0), 50);
    if(consider_item_cosine_similarity) LOG("Compute the top-"<<top_N_items<<" most similar neighbors of each item based on the cosine similarities.");




    int* top_users= (int *)malloc(ratings_rows * SIZE_OF(int));
    checkErrors(top_users);
    int* count = NULL;
    float* rank = NULL;
    count = (int *)malloc(ratings_rows * SIZE_OF(int));
    checkErrors(count);
    if(!frequency_based){
        rank = (float *)malloc(ratings_rows * SIZE_OF(float));
        checkErrors(rank);
    }
    int*   top_N_most_sim_itemIDs         = NULL;
    float* top_N_most_sim_item_similarity = NULL;
    int*   csr_format_ratingsMtx_userID_dev_with_hidden_values = NULL;
    int*   coo_format_ratingsMtx_userID_dev_with_hidden_values = NULL;
    int*   coo_format_ratingsMtx_itemID_dev_with_hidden_values = NULL;
    float* coo_format_ratingsMtx_rating_dev_with_hidden_values = NULL;
    long long int num_entries_with_hidden_values;

    
    float* item_cosine_similarity         = NULL;
    if(CU_ratingsMtx_has_hidden_values || !load_from_preprocessing){
        if(consider_item_cosine_similarity){
            LOG("consider_item_cosine_similarity is TRUE");
            int*   csc_format_ratingsMtx_itemID_dev;
            int*   csc_format_ratingsMtx_userID_dev;
            float*   csc_format_ratingsMtx_rating_dev;
            checkCudaErrors(cudaMalloc((void**)&csc_format_ratingsMtx_itemID_dev, (ratings_cols + (long long int)1) * SIZE_OF(int)));
            checkCudaErrors(cudaMalloc((void**)&csc_format_ratingsMtx_userID_dev, num_entries * SIZE_OF(int)));
            checkCudaErrors(cudaMalloc((void**)&csc_format_ratingsMtx_rating_dev, num_entries * SIZE_OF(float)));
            update_Mem( (ratings_cols + 1) * SIZE_OF(int) );
            update_Mem( num_entries * SIZE_OF(int) );
            update_Mem( num_entries * SIZE_OF(float) );

            /*
                int bufferSize;
                float* buffer;
                cusparse_status = cusparseCsr2cscEx2_bufferSize(sp_handle, ratings_rows, ratings_cols, num_entries,
                           coo_format_ratingsMtx_rating_dev,
                           csr_format_ratingsMtx_userID_dev,
                           coo_format_ratingsMtx_itemID_dev,
                           csc_format_ratingsMtx_rating_dev,
                           csc_format_ratingsMtx_itemID_dev,
                           csc_format_ratingsMtx_userID_dev,
                             CUDA_R_32F,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO,
                             CUSPARSE_CSR2CSC_ALG1,
                             &bufferSize);

                if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 
                    fprintf(stdout, "Conversion from CSR to CSC format failed\n");
                    return 1; 
                } 

                CUDA_CHECK(cudaMalloc((void**)&buffer , SIZE_OF(float) * bufferSize));
                
                cusparse_status = cusparseCsr2cscEx2(sp_handle, ratings_rows, ratings_cols, num_entries,
                           coo_format_ratingsMtx_rating_dev,
                           csr_format_ratingsMtx_userID_dev,
                           coo_format_ratingsMtx_itemID_dev,
                           csc_format_ratingsMtx_rating_dev,
                           csc_format_ratingsMtx_itemID_dev,
                           csc_format_ratingsMtx_userID_dev,
                           CUDA_R_32F,
                           CUSPARSE_ACTION_NUMERIC,
                           CUSPARSE_INDEX_BASE_ZERO,
                           CUSPARSE_CSR2CSC_ALG1,
                           buffer);
            */

            cusparse_status = cusparseScsr2csc(sp_handle, ratings_rows, ratings_cols, num_entries,
                       coo_format_ratingsMtx_rating_dev,
                       csr_format_ratingsMtx_userID_dev,
                       coo_format_ratingsMtx_itemID_dev,
                       csc_format_ratingsMtx_rating_dev,
                       csc_format_ratingsMtx_userID_dev,
                       csc_format_ratingsMtx_itemID_dev,
                       CUSPARSE_ACTION_NUMERIC,
                       CUSPARSE_INDEX_BASE_ZERO);

            if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 
                fprintf(stdout, "Conversion from CSR to CSC format failed\n");
                return 1; 
            } 
            if(Debug && 0){
                save_device_array_to_file<int>(csc_format_ratingsMtx_itemID_dev, ((int)ratings_cols + 1), "csc_format_ratingsMtx_itemID_dev", strPreamble(blank));
                save_device_array_to_file<int>(csc_format_ratingsMtx_userID_dev, (int)num_entries, "csc_format_ratingsMtx_userID_dev", strPreamble(blank));
                save_device_array_to_file<float>(csc_format_ratingsMtx_rating_dev, (int)num_entries, "csc_format_ratingsMtx_rating_dev", strPreamble(blank));
                return 0;
            }
            const long long int num_below_diag = (ratings_cols * (ratings_cols - (long long int)1)) / (long long int)2;

            item_cosine_similarity  = (float *)malloc(num_below_diag * SIZE_OF(float));
            checkErrors(item_cosine_similarity);
            LOG("num_below_diag : "<< num_below_diag);  

            
            get_cosine_similarity_host(ratings_cols, 
                                      csc_format_ratingsMtx_itemID_dev,
                                      csc_format_ratingsMtx_userID_dev,
                                      csc_format_ratingsMtx_rating_dev,
                                      item_cosine_similarity, true); 

            cudaFree(csc_format_ratingsMtx_itemID_dev);
            cudaFree(csc_format_ratingsMtx_userID_dev);
            cudaFree(csc_format_ratingsMtx_rating_dev);                                      
                                     

            //cpu_set_as_index(col_index, ratings_rows, ratings_rows);
            //gpu_set_as_index_host(col_index, ratings_rows, ratings_rows);

            top_N_most_sim_itemIDs          = (int *)  malloc((long long int)top_N_items * ratings_cols * SIZE_OF(int));
            top_N_most_sim_item_similarity  = (float *)malloc((long long int)top_N_items * ratings_cols * SIZE_OF(float));
            checkErrors(top_N_most_sim_itemIDs);
            checkErrors(top_N_most_sim_item_similarity);
            /*
                we want to know col indicies of the max N elements in each row
                excluding the row index itself
            */
            //cpu_sort_index_by_max(ratings_rows, ratings_rows,  cosine_similarity, col_index);

            cpu_sort_index_by_max<float>(ratings_cols, item_cosine_similarity, top_N_most_sim_itemIDs, top_N_items, top_N_most_sim_item_similarity);
            
            free(item_cosine_similarity);
            if(1){
                save_host_mtx_to_file<float>(top_N_most_sim_item_similarity, top_N_items, ratings_cols, preprocessing_path + "top_N_most_sim_item_similarity", false, strPreamble(blank));
                save_host_mtx_to_file<int>(top_N_most_sim_itemIDs, top_N_items, ratings_cols, preprocessing_path + "top_N_most_sim_itemIDs", false, strPreamble(blank));
            }

            std::vector<std::vector<int> > top_N_most_sim_itemIDs_vectors;
            std::vector<std::vector<float> > top_N_most_sim_item_similarity_vectors;
            for(long long int j = (long long int)0; j < ratings_cols; j+=(long long int)1){
                std::vector<int> int_vec;
                top_N_most_sim_itemIDs_vectors.push_back(int_vec);
                std::vector<float> float_vec;
                top_N_most_sim_item_similarity_vectors.push_back(float_vec);
            }
            // for each item, find which other items list it as a most simmilar item
            for(long long int j = (long long int)0; j < ratings_cols; j+=(long long int)1){
                for(long long int i = (long long int)0; i < (long long int)top_N_items; i+=(long long int)1){
                    int item_ID = top_N_most_sim_itemIDs[i + j * (long long int)top_N_items];
                    float rating_ = top_N_most_sim_item_similarity[i + j * (long long int)top_N_items];
                    if(std::abs(rating_) > 0.001 ){
                        top_N_most_sim_itemIDs_vectors[item_ID].push_back(j);
                        top_N_most_sim_item_similarity_vectors[item_ID].push_back(rating_);
                    }
                }
            }
            free(top_N_most_sim_itemIDs);
            free(top_N_most_sim_item_similarity);        
            //cpu_sort_index_by_max<int, float>(top_N_items, ratings_cols, top_N_most_sim_itemIDs, top_N_most_sim_item_similarity);
            

            
            int* csr_format_ratingsMtx_userID_host  = NULL;
            csr_format_ratingsMtx_userID_host = (int *)  malloc((ratings_rows + (long long int)1) *  SIZE_OF(int)); 
            checkErrors(csr_format_ratingsMtx_userID_host);
            checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host,  csr_format_ratingsMtx_userID_dev,  (ratings_rows + (long long int)1) *  SIZE_OF(int), cudaMemcpyDeviceToHost));
            
            int*   coo_format_ratingsMtx_userID_host_with_hidden_values;
            int*   coo_format_ratingsMtx_itemID_host_with_hidden_values;
            float* coo_format_ratingsMtx_rating_host_with_hidden_values;              
            

                       
            num_entries_with_hidden_values = cpu_compute_hidden_values(ratings_rows, ratings_cols, top_N_items, num_entries,
                                                                          csr_format_ratingsMtx_userID_host,
                                                                          coo_format_ratingsMtx_itemID_host,
                                                                          coo_format_ratingsMtx_rating_host,
                                                                          &top_N_most_sim_itemIDs_vectors,
                                                                          &top_N_most_sim_item_similarity_vectors,
                                                                          &coo_format_ratingsMtx_userID_host_with_hidden_values,
                                                                          &coo_format_ratingsMtx_itemID_host_with_hidden_values,
                                                                          &coo_format_ratingsMtx_rating_host_with_hidden_values); 
             
            LOG("num_entries_with_hidden_values : "<<num_entries_with_hidden_values); 

            free(coo_format_ratingsMtx_itemID_host);
            free(coo_format_ratingsMtx_rating_host);


            checkCudaErrors(cudaDeviceSynchronize());

            free(csr_format_ratingsMtx_userID_host);  

            checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_userID_dev_with_hidden_values,  num_entries_with_hidden_values * SIZE_OF(int)));
            checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_userID_dev_with_hidden_values,  coo_format_ratingsMtx_userID_host_with_hidden_values,  num_entries_with_hidden_values * SIZE_OF(int), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_with_hidden_values, (ratings_rows + (long long int)1) * SIZE_OF(int)));
            update_Mem( (ratings_rows + 1) * SIZE_OF(int) );            
            cusparse_status = cusparseXcoo2csr(sp_handle, coo_format_ratingsMtx_userID_dev_with_hidden_values, num_entries_with_hidden_values, 
                                               ratings_rows, csr_format_ratingsMtx_userID_dev_with_hidden_values, CUSPARSE_INDEX_BASE_ZERO); 

            if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 
                fprintf(stdout, "Conversion from COO to CSR format failed\n");
                return 1; 
            }
            checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_userID_host_with_hidden_values,  csr_format_ratingsMtx_userID_dev_with_hidden_values,  (ratings_rows + (long long int)1) * SIZE_OF(int), cudaMemcpyDeviceToHost));

            
            checkCudaErrors(cudaDeviceSynchronize());
            

            int ran_ind = 0;
            getRandIntsBetween(&ran_ind, 0, (int)ratings_rows - 2, 1);
            int first_place = (coo_format_ratingsMtx_userID_host_with_hidden_values[ran_ind]);
            int last_place = (coo_format_ratingsMtx_userID_host_with_hidden_values[ran_ind + 2]);

            if(Debug && 0){
                LOG("random row index : "<<ran_ind);
                LOG("first coo index : "<<first_place);
                LOG("last coo index : "<<last_place - 1);
                LOG("number of entries to print : "<<last_place - first_place);
                save_host_array_to_file<int>(coo_format_ratingsMtx_userID_host_with_hidden_values + ran_ind, 3, preprocessing_path + "csr_format_ratingsMtx_userID_host_with_hidden_values", strPreamble(blank));
                save_host_array_to_file<int>(coo_format_ratingsMtx_itemID_host_with_hidden_values + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_itemID_host_with_hidden_values_unsorted",strPreamble(blank));
                save_host_array_to_file<float>(coo_format_ratingsMtx_rating_host_with_hidden_values + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_rating_host_with_hidden_values_unsorted",strPreamble(blank));
            }
            //columns need to be sorted
            
            cpu_sort_csr_colums<float>(ratings_rows, 
                                       coo_format_ratingsMtx_userID_host_with_hidden_values,
                                       coo_format_ratingsMtx_itemID_host_with_hidden_values,
                                       coo_format_ratingsMtx_rating_host_with_hidden_values,
                                       num_entries_with_hidden_values, preprocessing_path);
            if(Debug && 0){
                save_host_array_to_file<int>(coo_format_ratingsMtx_userID_host_with_hidden_values + ran_ind, 3, preprocessing_path + "csr_format_ratingsMtx_userID_host_with_hidden_values",strPreamble(blank));
                save_host_array_to_file<int>(coo_format_ratingsMtx_itemID_host_with_hidden_values + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_itemID_host_with_hidden_values",strPreamble(blank));
                save_host_array_to_file<float>(coo_format_ratingsMtx_rating_host_with_hidden_values + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_rating_host_with_hidden_values",strPreamble(blank));
            } 

            checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_with_hidden_values,  num_entries_with_hidden_values * SIZE_OF(int)));
            checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_with_hidden_values,  num_entries_with_hidden_values * SIZE_OF(float)));
            checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_dev_with_hidden_values,  coo_format_ratingsMtx_itemID_host_with_hidden_values,  num_entries_with_hidden_values * SIZE_OF(int), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_with_hidden_values,  coo_format_ratingsMtx_rating_host_with_hidden_values,  num_entries_with_hidden_values * SIZE_OF(float), cudaMemcpyHostToDevice));


            update_Mem(2 * num_entries_with_hidden_values * SIZE_OF(int) + num_entries_with_hidden_values * SIZE_OF(float));



            free(coo_format_ratingsMtx_userID_host_with_hidden_values);
            free(coo_format_ratingsMtx_itemID_host_with_hidden_values);
            free(coo_format_ratingsMtx_rating_host_with_hidden_values);            

             

            //columns need to be sorted
            // gpu_sort_csr_colums<float>(ratings_rows, 
            //                            csr_format_ratingsMtx_userID_dev_with_hidden_values,
            //                            coo_format_ratingsMtx_itemID_dev_with_hidden_values,
            //                            coo_format_ratingsMtx_rating_dev_with_hidden_values,
            //                            num_entries_with_hidden_values, preprocessing_path);

            if(0){
                save_device_array_to_file<int>(coo_format_ratingsMtx_itemID_dev_with_hidden_values + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_itemID_dev_with_hidden_values");
                save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_with_hidden_values + first_place, last_place - first_place, preprocessing_path + "coo_format_ratingsMtx_rating_dev_with_hidden_values");
            }

            
        } // end if consider_item_cosine_similarity
    }
    if(!load_from_preprocessing){
        int*   top_N_most_sim_itemIDs_dev         = NULL;
        float* top_N_most_sim_item_similarity_dev = NULL;

        if(top_N_most_sim_itemIDs && 0){
            checkCudaErrors(cudaMalloc((void**)&top_N_most_sim_itemIDs_dev, (long long int)top_N_items * ratings_cols * SIZE_OF(int)));
            checkCudaErrors(cudaMalloc((void**)&top_N_most_sim_item_similarity_dev, (long long int)top_N_items * ratings_cols * SIZE_OF(float)));  
            checkCudaErrors(cudaMemcpy(top_N_most_sim_itemIDs_dev,  top_N_most_sim_itemIDs,  (long long int)top_N_items * ratings_cols * SIZE_OF(int), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(top_N_most_sim_item_similarity_dev,  top_N_most_sim_item_similarity,  (long long int)top_N_items * ratings_cols * SIZE_OF(float), cudaMemcpyHostToDevice));  
            //if(item_cosine_similarity) free(item_cosine_similarity);       
        }

        const long long int num_below_diag = (ratings_rows * (ratings_rows - (long long int)1)) / (long long int)2;

        float* cosine_similarity  = (float *)malloc(num_below_diag * SIZE_OF(float));
        int*   col_index          = (int *)malloc((long long int)top_N * ratings_rows * SIZE_OF(int));
        checkErrors(cosine_similarity);
        checkErrors(col_index);


        /*
        cpu_get_cosine_similarity(ratings_rows, 
                                  coo_format_ratingsMtx_userID_host,
                                  coo_format_ratingsMtx_itemID_host,
                                  coo_format_ratingsMtx_rating_host,
                                  cosine_similarity);
                                  */


        if(csr_format_ratingsMtx_userID_dev_with_hidden_values){
            LOG("Incorporating hidden values into user cosine similarity!");
            get_cosine_similarity_host(ratings_rows, 
                                      csr_format_ratingsMtx_userID_dev_with_hidden_values,
                                      coo_format_ratingsMtx_itemID_dev_with_hidden_values,
                                      coo_format_ratingsMtx_rating_dev_with_hidden_values,
                                      cosine_similarity, user_cosine_similarity_incorporates_ratings/*, top_N_most_sim_itemIDs_dev, 
                                      top_N_most_sim_item_similarity_dev, ratings_cols, top_N*/);

        }else{
            get_cosine_similarity_host(ratings_rows, 
                                      csr_format_ratingsMtx_userID_dev,
                                      coo_format_ratingsMtx_itemID_dev,
                                      coo_format_ratingsMtx_rating_dev,
                                      cosine_similarity, user_cosine_similarity_incorporates_ratings/*, top_N_most_sim_itemIDs_dev, 
                                      top_N_most_sim_item_similarity_dev, ratings_cols, top_N*/);
        }

                                  
        if(top_N_most_sim_itemIDs_dev) cudaFree(top_N_most_sim_itemIDs_dev); 
        if(top_N_most_sim_item_similarity_dev) cudaFree(top_N_most_sim_item_similarity_dev);                                   

        //cpu_set_as_index(col_index, ratings_rows, ratings_rows);
        //gpu_set_as_index_host(col_index, ratings_rows, ratings_rows);



        /*
            we want to know column indicies of the max N elements in each row
            excluding the row index itself
        */
        //cpu_sort_index_by_max(ratings_rows, ratings_rows,  cosine_similarity, col_index);
        cpu_sort_index_by_max<float>(ratings_rows, cosine_similarity, col_index, top_N);

        // NOTE: higher similarity has higher row index

        if(Debug && 0){
            save_host_array_to_file<float>(cosine_similarity, static_cast<int>(5*ratings_rows), "cosine_similarity_chunk");
            save_host_mtx_to_file<int>(col_index, static_cast<int>(top_N), ratings_rows, "col_index_sorted");
        }
        free(cosine_similarity);
        /*
            for each user index, count how many times that user appears in the top_N + 1 most similar users.
        */

        
        if(frequency_based){
            cpu_set_all<int>(count, ratings_rows, 0);
            cpu_count_appearances(top_N, ratings_rows, count, col_index);
        }else{
            cpu_set_all<float>(rank, ratings_rows, (float)0.0);
            cpu_rank_appearances(top_N, ratings_rows, rank, col_index);
        }
        free(col_index);
        if(Debug && 0){
            save_host_array_to_file<int>(count, (int)ratings_rows, "count");
        }

        /*
            sort the count to find the ratings_rows_CU core users
        */

        cpu_set_as_index<int>(top_users, ratings_rows, 1);

        if(frequency_based){
            cpu_sort_index_by_max<int>(ratings_rows, 1, count, top_users);
        }else{
            cpu_sort_index_by_max<float>(ratings_rows, 1, rank, top_users);
            free(rank);
        }
        
        
        save_host_array_to_file<int>(top_users, ratings_rows, preprocessing_path + "top_users");
        
    }else{
        LOG("Load top_users from saved file in "<<preprocessing_path);
        // Load top_users from saved file
        //get_host_array_from_saved_txt<int>(top_users, ratings_rows, preprocessing_path);

        CSVReader csv_Preprocessing(preprocessing_path + "top_users.txt");
        csv_Preprocessing.getData(top_users, ratings_rows, 1);
        LOG("Sanity Check : ");
        LOG("top_users[0] : "<<top_users[0]);
        LOG("top_users[ratings_rows - 1] : "<<top_users[ratings_rows - 1]);
    }
    LOG("here!");

    /*
        The ratings_rows_CU core users indicies are the indicies 
        in the last ratings_rows_CU entries of top_users 
    */
    
    cpu_set_all<int>(count, ratings_rows, 1);
    cpu_mark_CU_users(ratings_rows_CU, ratings_rows, top_users, count );
    free(top_users);
    if(Debug && 0){
        save_host_array_to_file<int>(count, ratings_rows, "top_user_bools");
        //LOG("Press Enter to continue.") ;
        //std::cin.ignore();
    }




    
    const int num_groups = 2;
    int *group_indicies;
    checkCudaErrors(cudaMalloc((void**)&group_indicies, ratings_rows * SIZE_OF(int)));
    update_Mem( ratings_rows * SIZE_OF(int) );
    
    checkCudaErrors(cudaMemcpy(group_indicies, count, ratings_rows * SIZE_OF(int), cudaMemcpyHostToDevice));
    free(count);

    int* group_sizes = NULL;
    group_sizes = (int *)malloc(num_groups * SIZE_OF(int)); 
    checkErrors(group_sizes);  

    count_each_group_from_coo(num_groups, group_indicies, num_entries, coo_format_ratingsMtx_userID_dev, group_sizes);
    long long int num_entries_CU_temp   = group_sizes[0];
    const long long int num_entries_testing  = group_sizes[1];

    if(coo_format_ratingsMtx_userID_dev_with_hidden_values && CU_ratingsMtx_has_hidden_values){
        LOG("CONSTRUCTING CU RATINGS MTX USING HIDDEN VALUES!");
        count_each_group_from_coo(num_groups, group_indicies, num_entries_with_hidden_values, coo_format_ratingsMtx_userID_dev_with_hidden_values, group_sizes);
        cudaFree(coo_format_ratingsMtx_userID_dev_with_hidden_values);  
        num_entries_CU_temp = group_sizes[0];
        LOG("num testing user ratings with hidden values : " <<group_sizes[1]);
        LOG("total ratings with hidden values : " <<group_sizes[0] + group_sizes[1]);
    }

    const long long int num_entries_CU = num_entries_CU_temp;

    // count_each_group(ratings_rows, group_indicies, group_sizes, num_groups);
    // const long long int ratings_rows_CU  = group_sizes[0];
    // const long long int ratings_rows_testing = group_sizes[1]; 


    LOG("num testing entries : " <<num_entries_testing);
    LOG("num CU entries : "      <<num_entries_CU<<std::endl);
    
    ABORT_IF_NEQ(ratings_rows_testing  + ratings_rows_CU, ratings_rows, "The number of rows does not add up correctly.");
    if(!CU_ratingsMtx_has_hidden_values)
        ABORT_IF_NEQ(num_entries_testing   + num_entries_CU,  num_entries, "The number of entries does not add up correctly.");

    if(Debug && 0){
        // LOG("coo_format_ratingsMtx_userID_dev : ");
        // print_gpu_array_entries<int>(coo_format_ratingsMtx_userID_dev, 100, 1 , num_entries);
        LOG("group_indicies :");
        print_gpu_mtx_entries<int>(group_indicies, (int)ratings_rows, 1 );
        //save_device_array_to_file<int>(group_indicies, ratings_rows, "testing_bools");
        //LOG("Press Enter to continue.") ;
        //std::cin.ignore();
    }

    cudaFree(coo_format_ratingsMtx_userID_dev);
    update_Mem( num_entries * SIZE_OF(int) * (-1) );

    int*   csr_format_ratingsMtx_userID_dev_testing;
    int*   coo_format_ratingsMtx_itemID_dev_testing;
    float* coo_format_ratingsMtx_rating_dev_testing;

    int*   csr_format_ratingsMtx_userID_dev_CU;
    int*   coo_format_ratingsMtx_itemID_dev_CU;
    float* coo_format_ratingsMtx_rating_dev_CU;


    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_testing,  (ratings_rows_testing + 1) * SIZE_OF(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_testing,  num_entries_testing        * SIZE_OF(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing        * SIZE_OF(float)));

    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_CU,  (ratings_rows_CU + 1) * SIZE_OF(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_CU,  num_entries_CU        * SIZE_OF(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_CU,  num_entries_CU        * SIZE_OF(float)));
    update_Mem(  (ratings_rows_testing + 1)  * SIZE_OF(int) + num_entries_testing  * SIZE_OF(int) + num_entries_testing  * SIZE_OF(float)
               + (ratings_rows_CU + 1)       * SIZE_OF(int) + num_entries_CU       * SIZE_OF(int) + num_entries_CU       * SIZE_OF(float)  );

    int*   csr_format_ratingsMtx_userID_dev_by_group_host  [num_groups] = { csr_format_ratingsMtx_userID_dev_CU,  csr_format_ratingsMtx_userID_dev_testing  };
    int*   coo_format_ratingsMtx_itemID_dev_by_group_host  [num_groups] = { coo_format_ratingsMtx_itemID_dev_CU,  coo_format_ratingsMtx_itemID_dev_testing  };
    float* coo_format_ratingsMtx_rating_dev_by_group_host  [num_groups] = { coo_format_ratingsMtx_rating_dev_CU,  coo_format_ratingsMtx_rating_dev_testing  };
    int    ratings_rows_by_group_host                      [num_groups] = { ratings_rows_CU                    ,  ratings_rows_testing                      };
    int**   csr_format_ratingsMtx_userID_dev_by_group;
    int**   coo_format_ratingsMtx_itemID_dev_by_group;
    float** coo_format_ratingsMtx_rating_dev_by_group;
    int*    ratings_rows_by_group;

    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_by_group,  num_groups*SIZE_OF(int*)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_by_group,  num_groups*SIZE_OF(int*)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_by_group,  num_groups*SIZE_OF(float*)));
    checkCudaErrors(cudaMalloc((void**)&ratings_rows_by_group,                      num_groups*SIZE_OF(int)));
    update_Mem( num_groups * SIZE_OF(int*) * 2 + num_groups * SIZE_OF(float*) + num_groups * SIZE_OF(int) );

    checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_dev_by_group,  csr_format_ratingsMtx_userID_dev_by_group_host,  num_groups * SIZE_OF(int*),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_dev_by_group,  coo_format_ratingsMtx_itemID_dev_by_group_host,  num_groups * SIZE_OF(int*),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_by_group,  coo_format_ratingsMtx_rating_dev_by_group_host,  num_groups * SIZE_OF(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ratings_rows_by_group,                      ratings_rows_by_group_host,                      num_groups * SIZE_OF(int),    cudaMemcpyHostToDevice));

    gpu_split_data(csr_format_ratingsMtx_userID_dev,
                   coo_format_ratingsMtx_itemID_dev,
                   coo_format_ratingsMtx_rating_dev, 
                   ratings_rows, group_indicies,
                   csr_format_ratingsMtx_userID_dev_by_group,
                   coo_format_ratingsMtx_itemID_dev_by_group,
                   coo_format_ratingsMtx_rating_dev_by_group,
                   ratings_rows_by_group); 
    if(csr_format_ratingsMtx_userID_dev_with_hidden_values && CU_ratingsMtx_has_hidden_values){
        LOG("CONSTRUCTING CU RATINGS MTX USING HIDDEN VALUES!");
        gpu_split_data(csr_format_ratingsMtx_userID_dev_with_hidden_values,
                       coo_format_ratingsMtx_itemID_dev_with_hidden_values,
                       coo_format_ratingsMtx_rating_dev_with_hidden_values, 
                       ratings_rows, group_indicies,
                       csr_format_ratingsMtx_userID_dev_by_group,
                       coo_format_ratingsMtx_itemID_dev_by_group,
                       coo_format_ratingsMtx_rating_dev_by_group,
                       ratings_rows_by_group, true);  
        cudaFree(csr_format_ratingsMtx_userID_dev_with_hidden_values);
        cudaFree(coo_format_ratingsMtx_itemID_dev_with_hidden_values);
        cudaFree(coo_format_ratingsMtx_rating_dev_with_hidden_values);       
    }




    
    cudaFree(csr_format_ratingsMtx_userID_dev);            update_Mem((ratings_rows + 1) * SIZE_OF(int) * (-1));
    cudaFree(coo_format_ratingsMtx_itemID_dev);            update_Mem(num_entries * SIZE_OF(int) * (-1));
    cudaFree(coo_format_ratingsMtx_rating_dev);            update_Mem(num_entries * SIZE_OF(float) * (-1));
    cudaFree(csr_format_ratingsMtx_userID_dev_by_group);   update_Mem(num_groups * SIZE_OF(int*) * (-1));
    cudaFree(coo_format_ratingsMtx_itemID_dev_by_group);   update_Mem(num_groups * SIZE_OF(int*) * (-1));
    cudaFree(coo_format_ratingsMtx_rating_dev_by_group);   update_Mem(num_groups * SIZE_OF(float*) * (-1));
    cudaFree(ratings_rows_by_group);                       update_Mem(num_groups * SIZE_OF(int) * (-1));
    cudaFree(group_indicies);                              update_Mem(num_groups * SIZE_OF(int)* (-1));
    
    free(group_sizes);


    if(Debug && 0){
        save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_CU,        (int)ratings_rows_CU + 1,       preprocessing_path + "csr_format_ratingsMtx_userID_dev_CU");
        save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_CU,        (int)num_entries_CU,            preprocessing_path + "coo_format_ratingsMtx_itemID_dev_CU");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_CU,        (int)num_entries_CU,            preprocessing_path + "coo_format_ratingsMtx_rating_dev_CU");
        save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_testing,   (int)ratings_rows_testing + 1,  preprocessing_path + "csr_format_ratingsMtx_userID_dev_testing");
        save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_testing,   (int)num_entries_testing,       preprocessing_path + "coo_format_ratingsMtx_itemID_dev_testing");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_testing,   (int)num_entries_testing,       preprocessing_path + "coo_format_ratingsMtx_rating_dev_testing");
        // LOG("csr_format_ratingsMtx_userID_dev_testing : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_testing, ratings_rows_testing + 1, 1 );
        // LOG("csr_format_ratingsMtx_userID_dev_CU : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_CU, ratings_rows_CU + 1, 1 );
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
        //return 0;
    }





    //============================================================================================
    // Fill CU Ratings Matrix
    //============================================================================================


    LOG("fill CU matrix... ");



    bool row_major_ordering = true;

    float * full_ratingsMtx_dev_CU;
    float * full_ratingsMtx_host_CU = NULL;

    const long long int CU_mtx_size = (long long int)ratings_rows_CU * (long long int)ratings_cols;
    const long long int CU_mtx_size_bytes = (long long int)ratings_rows_CU * (long long int)ratings_cols * (long long int)SIZE_OF(float);
    LOG("Will need "<<CU_mtx_size<< " floats for the CU mtx.") ;
    LOG("Will need "<<CU_mtx_size_bytes<< " bytes for the CU mtx.") ;
    bool Conserve_GPU_Mem_temp = Conserve_GPU_Mem;
    if(!Conserve_GPU_Mem_temp && allocatedMem + CU_mtx_size_bytes > (long long int)((double)devMem * (double)0.75)){
        LOG("Conserving Memory Now");
        Conserve_GPU_Mem = 1;
    }
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();


    int*   csr_format_ratingsMtx_userID_host_CU = NULL;
    int*   coo_format_ratingsMtx_itemID_host_CU = NULL;
    float* coo_format_ratingsMtx_rating_host_CU = NULL;

    if(Conserve_GPU_Mem){
        csr_format_ratingsMtx_userID_host_CU  = (int *)  malloc((ratings_rows_CU + 1) * SIZE_OF(int)  );
        coo_format_ratingsMtx_itemID_host_CU  = (int *)  malloc(num_entries_CU        * SIZE_OF(int)  );
        coo_format_ratingsMtx_rating_host_CU  = (float *)malloc(num_entries_CU        * SIZE_OF(float));
        checkErrors(csr_format_ratingsMtx_userID_host_CU);
        checkErrors(coo_format_ratingsMtx_itemID_host_CU);
        checkErrors(coo_format_ratingsMtx_rating_host_CU);
        checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host_CU,  csr_format_ratingsMtx_userID_dev_CU,  (ratings_rows_CU + 1) * SIZE_OF(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host_CU,  coo_format_ratingsMtx_itemID_dev_CU,  num_entries_CU        * SIZE_OF(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host_CU,  coo_format_ratingsMtx_rating_dev_CU,  num_entries_CU        * SIZE_OF(float), cudaMemcpyDeviceToHost));

        full_ratingsMtx_host_CU = (float *)malloc(CU_mtx_size_bytes);
        checkErrors(full_ratingsMtx_host_CU);

        cpu_set_all<float>(full_ratingsMtx_host_CU, CU_mtx_size, (float)0.0);
        
        cpu_fill_training_mtx((long long int)ratings_rows_CU, (long long int)ratings_cols, (long long int)num_entries_CU, 
                              row_major_ordering,  
                              csr_format_ratingsMtx_userID_host_CU,
                              coo_format_ratingsMtx_itemID_host_CU,
                              coo_format_ratingsMtx_rating_host_CU,
                              full_ratingsMtx_host_CU);

        if(0 && consider_item_cosine_similarity && load_from_preprocessing){
            top_N_most_sim_itemIDs  = (int *)malloc((long long int)top_N * ratings_cols * SIZE_OF(int));
            top_N_most_sim_item_similarity  = (float *)malloc((long long int)top_N * ratings_cols * SIZE_OF(float));
            checkErrors(top_N_most_sim_itemIDs);
            checkErrors(top_N_most_sim_item_similarity);

            CSVReader top_N_most_sim_itemIDs_Preprocessing(preprocessing_path + "top_N_most_sim_itemIDs.txt");
            top_N_most_sim_itemIDs_Preprocessing.getData(top_N_most_sim_itemIDs, top_N, ratings_cols);
            LOG("Sanity Check : ");
            LOG("top_N_most_sim_itemIDs[0] : "<<top_N_most_sim_itemIDs[0]);
            CSVReader top_N_most_sim_item_similarity_Preprocessing(preprocessing_path + "top_N_most_sim_item_similarity.txt");
            top_N_most_sim_item_similarity_Preprocessing.getData(top_N_most_sim_item_similarity, top_N, ratings_cols);
            LOG("Sanity Check : ");
            LOG("top_N_most_sim_item_similarity[0] : "<<top_N_most_sim_item_similarity[0]);
        }
        
        free(csr_format_ratingsMtx_userID_host_CU);
        free(coo_format_ratingsMtx_itemID_host_CU);
        free(coo_format_ratingsMtx_rating_host_CU);       

        
        cpu_shuffle_mtx_rows_or_cols(ratings_rows_CU, ratings_cols, 
                                     row_major_ordering, full_ratingsMtx_host_CU, 1);

        LOG("full_ratingsMtx_host_CU filled and shuffled") ;
    }else{
        checkCudaErrors(cudaMalloc((void**)&full_ratingsMtx_dev_CU, ratings_rows_CU * ratings_cols * SIZE_OF(float)));
        update_Mem(ratings_rows_CU * ratings_cols * SIZE_OF(float));


        gpu_set_all<float>(full_ratingsMtx_dev_CU, ratings_rows_CU * ratings_cols, (float)0.0);
        
        gpu_fill_training_mtx(ratings_rows_CU, ratings_cols, row_major_ordering,
                              csr_format_ratingsMtx_userID_dev_CU,
                              coo_format_ratingsMtx_itemID_dev_CU,
                              coo_format_ratingsMtx_rating_dev_CU,
                              full_ratingsMtx_dev_CU);

        if(csr_format_ratingsMtx_userID_dev_with_hidden_values)    cudaFree(csr_format_ratingsMtx_userID_dev_with_hidden_values);
        if(csr_format_ratingsMtx_userID_dev_with_hidden_values)    cudaFree(coo_format_ratingsMtx_itemID_dev_with_hidden_values);
        if(csr_format_ratingsMtx_userID_dev_with_hidden_values)    cudaFree(coo_format_ratingsMtx_rating_dev_with_hidden_values);

        if(Content_Based){
            gpu_supplement_training_mtx_with_content_based(ratings_rows_CU, 
                                                            ratings_cols, 
                                                            row_major_ordering,
                                                            csr_format_ratingsMtx_userID_dev_CU,
                                                            coo_format_ratingsMtx_itemID_dev_CU,
                                                            coo_format_ratingsMtx_rating_dev_CU,
                                                            full_ratingsMtx_dev_CU,
                                                            csr_format_keyWordMtx_itemID_dev,
                                                            coo_format_keyWordMtx_keyWord_dev);
        }

        if(Debug && 0){
            save_device_mtx_to_file(full_ratingsMtx_dev_CU, ratings_cols, ratings_rows_CU, "full_ratingsMtx_dev_CU_pre_shuffle", true);
        }

        cudaDeviceSynchronize();
        //shuffle CU rows
        gpu_shuffle_mtx_rows_or_cols(dn_handle, ratings_rows_CU, ratings_cols,  
                                     row_major_ordering, full_ratingsMtx_dev_CU, 1);
        if(Debug && 0){
            save_device_mtx_to_file(full_ratingsMtx_dev_CU, ratings_cols, ratings_rows_CU, "full_ratingsMtx_dev_CU", true);
            // LOG("Press Enter to continue.") ;
            // std::cin.ignore();
        }
        LOG("full_ratingsMtx_dev_CU filled and shuffled") ;

    }
    cudaFree(csr_format_ratingsMtx_userID_dev_CU);         update_Mem((ratings_rows_CU + 1) * SIZE_OF(int)   * (-1));
    cudaFree(coo_format_ratingsMtx_itemID_dev_CU);         update_Mem(num_entries_CU        * SIZE_OF(int)   * (-1));
    cudaFree(coo_format_ratingsMtx_rating_dev_CU);         update_Mem(num_entries_CU        * SIZE_OF(float) * (-1));

    if(Debug && 0){
        save_device_array_to_file<int>(  csr_format_ratingsMtx_userID_dev_testing,  (int)ratings_rows_testing + 1, "csr_format_ratingsMtx_userID_dev_testing" );
        save_device_array_to_file<int>(  coo_format_ratingsMtx_itemID_dev_testing,  (int)num_entries_testing,      "coo_format_ratingsMtx_itemID_dev_testing");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_testing,  (int)num_entries_testing,      "coo_format_ratingsMtx_rating_dev_testing" );
        // LOG("csr_format_ratingsMtx_userID_dev_testing : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_testing, ratings_rows_testing + 1, 1 );
        // LOG("csr_format_ratingsMtx_userID_dev_CU : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_CU, ratings_rows_CU + 1, 1 );
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }





    //============================================================================================
    // We want to find orthogonal matrices U, V such that R ~ U*V^T
    // 
    // R is batch_size_CU by ratings_cols
    // U is batch_size_CU by num_latent_factors
    // V is ratings_cols by num_latent_factors
    //============================================================================================

    bool        print_error    = true;

    // float       training_rate           = 0.01;      //use for movielens
    // const float regularization_constant = 5;         //use for movielens

    float       training_rate           = (float)0.0001;      //use for rent the runway
    const float regularization_constant = (float)0.01;         //use for rent the runway

    const float testing_fraction        = 0.2; //percent of known entries used for testing
    bool        compress                = true;
    bool        SV_with_U               = false;

    
    const int num_batches    = 83;     // number of training batches per iteration (batches index into training data)
    const int num_blocks     = ratings_rows_CU / 1000;    // number of blocks of generic users (a single block of generic users is updated in a batch)
    const int testing_rate   = 1;
    const long long int batch_size_CU       = std::max((long long int)1, ratings_rows_CU / (num_blocks));
    const long long int batch_size_testing  = std::min((long long int)200, std::min(ratings_rows_testing, batch_size_CU));//ratings_rows_testing / num_batches;
    long long int num_batches_testing  = (long long int)(std::ceil((float)ratings_rows_testing  / (float)batch_size_testing));

    LOG("training_rate : "          <<training_rate);
    LOG("regularization_constant : "<<regularization_constant);
    LOG("testing_fraction : "       <<testing_fraction);
    LOG("compress : "               <<compress);
    LOG("num_batches : "            <<num_batches);
    LOG("testing_rate : "           <<testing_rate);
    LOG("SV_with_U : "              <<SV_with_U);


    float * testing_error = NULL;
    testing_error = (float *)malloc(num_batches * SIZE_OF(float)); 
    checkErrors(testing_error);

    
    LOG(std::endl);
    LOG("batch_size_testing : " <<batch_size_testing);
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();


    long long int num_latent_factors = (long long int)((float)ratings_rows_CU * (float)0.95);
    long long int max_num_latent_factors = (long long int)12000;
    float percent_sv_mass              = (float)0.80;     // numbber of singular values to use
    
    const float percent              = (float)0.20;     // numbber of singular values to use


    float * U_CU;       // U_CU is ratings_rows_CU * ratings_rows_CU
    float * V_dev;          // V_CU is ratings_cols * ratings_cols
    float * V_host;          // V_CU is ratings_cols * ratings_cols

    if(Debug && 0){
        // const int batch_size_CU = 3;
        // const int ratings_cols = 2;
        // /*       | 1 2  |
        //  *   A = | 4 5  |
        //  *       | 2 1  |
        //  */
        // float A[batch_size_CU*ratings_cols] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
        // cudaMemcpy(full_ratingsMtx_dev_CU, A, SIZE_OF(float)*batch_size_CU*ratings_cols, cudaMemcpyHostToDevice);
    }

    long long int min_CU_dimension = std::min(ratings_rows_CU, ratings_cols);
    Conserve_GPU_Mem_temp = Conserve_GPU_Mem;
    const long long int Training_bytes = (batch_size_testing  * std::max(min_CU_dimension, batch_size_testing) + 
                                          ratings_cols        * min_CU_dimension +
                                          ratings_cols        * batch_size_testing)* (long long int)SIZE_OF(float) ;
    if(allocatedMem + Training_bytes > (long long int)((double)devMem * (double)0.75)) 
    {
        Conserve_GPU_Mem = 1;
    };


    if(!Conserve_GPU_Mem_temp && Conserve_GPU_Mem){
        LOG("Conserving Memory Now");
        //put the CU ratings mtx on the CPU;
        full_ratingsMtx_host_CU = (float *)malloc(CU_mtx_size_bytes);
        checkErrors(full_ratingsMtx_host_CU);
        checkCudaErrors(cudaMemcpy(full_ratingsMtx_host_CU, full_ratingsMtx_dev_CU, CU_mtx_size_bytes, cudaMemcpyDeviceToHost));
        cudaFree(full_ratingsMtx_dev_CU);
        update_Mem((float)(-1.0) * CU_mtx_size_bytes );
    };
    int *   csr_format_ratingsMtx_userID_host_testing  = NULL;
    int *   coo_format_ratingsMtx_itemID_host_testing = NULL;
    float * coo_format_ratingsMtx_rating_host_testing  = NULL;
    if(Conserve_GPU_Mem){
        csr_format_ratingsMtx_userID_host_testing  = (int *)  malloc((ratings_rows_testing + 1) *  SIZE_OF(int)); 
        coo_format_ratingsMtx_itemID_host_testing  = (int *)  malloc(num_entries_testing  *  SIZE_OF(int)); 
        coo_format_ratingsMtx_rating_host_testing  = (float *)malloc(num_entries_testing  *  SIZE_OF(float));
        checkErrors(csr_format_ratingsMtx_userID_host_testing);
        checkErrors(coo_format_ratingsMtx_itemID_host_testing);
        checkErrors(coo_format_ratingsMtx_rating_host_testing); 

        checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host_testing,  csr_format_ratingsMtx_userID_dev_testing,  (ratings_rows_testing + 1) *  SIZE_OF(int),   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host_testing, coo_format_ratingsMtx_itemID_dev_testing, num_entries_testing  *  SIZE_OF(int),   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host_testing,  coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing  *  SIZE_OF(float), cudaMemcpyDeviceToHost));

        cudaFree(csr_format_ratingsMtx_userID_dev_testing);  update_Mem((ratings_rows_testing + 1)*  SIZE_OF(int) * (-1));
        cudaFree(coo_format_ratingsMtx_itemID_dev_testing);  update_Mem(num_entries_testing  *  SIZE_OF(int) * (-1));
        cudaFree(coo_format_ratingsMtx_rating_dev_testing);  update_Mem(num_entries_testing  *  SIZE_OF(float) * (-1) );
        
        if(Debug && 0){
            save_host_array_to_file<int>  (csr_format_ratingsMtx_userID_host_testing,  ratings_rows_testing + 1, "csr_format_ratingsMtx_userID_host_testing_1");
            save_host_array_to_file<int>  (coo_format_ratingsMtx_itemID_host_testing,  num_entries_testing, "coo_format_ratingsMtx_itemID_host_testing");
            save_host_array_to_file<float>  (coo_format_ratingsMtx_rating_host_testing,  num_entries_testing, "coo_format_ratingsMtx_rating_host_testing");
        }

        U_CU = (float *)malloc(ratings_rows_CU * ratings_rows_CU * SIZE_OF(float));
        V_host = (float *)malloc(ratings_cols * ratings_rows_CU * SIZE_OF(float));
        checkErrors(U_CU);
        checkErrors(V_host);

        if(!row_major_ordering){
            ABORT_IF_EQ(0,1,"try again with row_major_ordering = true.")
        }
        CUDA_CHECK(cudaMalloc((void**)&V_dev, ratings_cols * (compress ? num_latent_factors : ratings_rows_CU) * SIZE_OF(float)));
        update_Mem(ratings_cols * (compress ? num_latent_factors : ratings_rows_CU) * SIZE_OF(float) );
        LOG("ratings_cols * (compress ? num_latent_factors : ratings_rows_CU) : " << ratings_cols * (compress ? num_latent_factors : ratings_rows_CU));
    }else{
        checkCudaErrors(cudaMalloc((void**)&U_CU,       ratings_rows_CU     * ratings_rows_CU            * SIZE_OF(float)));
        checkCudaErrors(cudaMalloc((void**)&V_dev,          ratings_cols        * min_CU_dimension           * SIZE_OF(float)));
        update_Mem((ratings_rows_CU * ratings_rows_CU + ratings_cols * min_CU_dimension) * SIZE_OF(float) );
    }
    
    
    // LOG(ratings_cols * ratings_cols * SIZE_OF(float)) ;

    // cudaDeviceSynchronize();
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();  





    float * testing_error_on_training_entries = NULL;
    float * testing_error_on_testing_entries = NULL;
    float * testing_iterations = NULL;
    float * meta_km_errors = NULL;
    testing_error_on_training_entries = (float *)malloc(num_batches_testing * SIZE_OF(float)); 
    checkErrors(testing_error_on_training_entries);
    cpu_set_all<float>(testing_error_on_training_entries, num_batches_testing, (float)0.0);
    testing_error_on_testing_entries = (float *)malloc(num_batches_testing * SIZE_OF(float)); 
    checkErrors(testing_error_on_testing_entries);
    cpu_set_all<float>(testing_error_on_testing_entries, num_batches_testing, (float)0.0);
    testing_iterations = (float *)malloc(num_batches_testing * SIZE_OF(float)); 
    checkErrors(testing_iterations);
    cpu_set_all<float>(testing_iterations, num_batches_testing, (float)0.0);
    meta_km_errors = (float *)malloc(num_batches_testing * SIZE_OF(float)); 
    checkErrors(meta_km_errors);
    cpu_set_all<float>(meta_km_errors, num_batches_testing, (float)0.0);

    float* logarithmic_histogram = (float *)malloc(7 * num_batches_testing  * SIZE_OF(float)); 
    checkErrors(logarithmic_histogram);
    cpu_set_all<float>(logarithmic_histogram, 7 * num_batches_testing , (float)0.0);

    float* logarithmic_histogram_km = (float *)malloc(7 * num_batches_testing  * SIZE_OF(float)); 
    checkErrors(logarithmic_histogram_km);
    cpu_set_all<float>(logarithmic_histogram_km, 7 * num_batches_testing , (float)0.0);

    float running_avg_testing_error_on_testing_entries = 0.0;
    float running_avg_testing_error_on_training_entries = 0.0;

    float min_training_error = 10000.0;
    float max_training_error = 0.0;

    gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("program_time: %f\n", program_time);   
    LOG("program_time so far: "<<readable_time(program_time));

    //============================================================================================
    // Begin Testing
    //============================================================================================  
    LOG(std::endl<<std::endl<<"                              Begin Testing..."<<std::endl); 
    gettimeofday(&testing_start, NULL);
    int num_tests = 0;
    float testing_error_on_testing_entries_temp = (float)0.0;
    float testing_error_on_training_entries_temp = (float)0.0;
    float total_iterations = (float)0.0;
    long long int total_testing_nnz = (long long int)0;
    int count_tests = 0;

    //============================================================================================
    // Find U_CU, V such that U_CU * V^T ~ R_CU 
    //============================================================================================  
    
    float* SV = NULL;
    SV = (float *)malloc(ratings_rows_CU *  SIZE_OF(float)); 
    checkErrors(SV);

    //save_device_mtx_to_file<float>(V_dev, ratings_cols, num_latent_factors, "V_compressed");
    
    if(compress){
        //ABORT_IF_NEQ(0, 1, "Not Yet Supported");
    }


    LOG(std::endl<<"      ~~~ TESTING ~~~ "); 
    if(Debug){
        LOG(memLeft<<" available bytes left on the device");
        LOG("batch_size_testing : "<<batch_size_testing);
    }
    
    float * SV_dev;
    checkCudaErrors(cudaMalloc((void**)&SV_dev, ratings_rows_CU * SIZE_OF(float)));
    update_Mem(ratings_rows_CU * SIZE_OF(float));
    if(Conserve_GPU_Mem){
        /*
            cpu_orthogonal_decomp<float>(ratings_rows_CU, ratings_cols, row_major_ordering,
                                    &num_latent_factors, percent,
                                    full_ratingsMtx_host_CU, U_CU, V_host, SV_with_U, SV);
        */
        LOG("num_blocks = "<< num_blocks);
        LOG("batch_size_CU = "<< batch_size_CU);
        
        gpu_block_orthogonal_decomp_from_host<float>(dn_handle, dn_solver_handle,
                                                     ratings_rows_CU, ratings_cols,
                                                     &num_latent_factors, percent_sv_mass,
                                                     full_ratingsMtx_host_CU, U_CU, 
                                                     V_host, batch_size_CU, SV_with_U, SV);
        LOG("largest singular value = "<< SV[0]<<std::endl);                                             
                                                    
        //LOG("num_latent_factors with percent mass = "<< num_latent_factors << " ( / "<< ratings_rows_CU<< " )");
        //LOG("percent singular vectors with mass = "<< (((float)num_latent_factors) / ((float)ratings_rows_CU)) <<std::endl);   

        num_latent_factors = static_cast<long long int>(percent * (float)ratings_rows_CU);
        num_latent_factors = std::min(num_latent_factors, std::min(ratings_rows_CU , max_num_latent_factors));

        LOG("num_latent_factors = "<< num_latent_factors << " ( / "<< ratings_rows_CU<< " )");
        LOG("percent singular vectors = "<< (((float)num_latent_factors) / ((float)ratings_rows_CU)) );
        
        cpu_get_latent_factor_mass<float>(ratings_rows_CU, SV, num_latent_factors, &percent_sv_mass);
        LOG("percent singular value mass = "<< percent_sv_mass<<std::endl);  

        checkCudaErrors(cudaMemcpy(V_dev, V_host, ratings_cols * (compress ? num_latent_factors : ratings_rows_CU) * SIZE_OF(float), cudaMemcpyHostToDevice));
        //if(Debug) {checkCudaErrors(cudaDeviceSynchronize()); LOG("Here");}
        
        if(Debug && 0){
            float mean_abs_nonzero_ = (float)0.0;
            cpu_mean_abs_nonzero(ratings_cols * (compress ? num_latent_factors : ratings_rows_CU), V_host, &mean_abs_nonzero_, true, "V_host");
            gpu_mean_abs_nonzero(ratings_cols * (compress ? num_latent_factors : ratings_rows_CU), V_dev, &mean_abs_nonzero_, true, "V_dev");
        }
        
        

        if(row_major_ordering){
            //gpu_swap_ordering<float>(ratings_cols, num_latent_factors, V_dev, true);
        }
        if(Debug) {
            //save_host_array_to_file<float>(SV, ratings_rows_CU, "sinCUlar_values", strPreamble(blank));
            //save_host_mtx_to_file<float>(U_CU_host, ratings_rows_CU, num_latent_factors, "U_CU_compressed");
            
            //save_host_array_side_by_side_with_device_array<float>(V_host, V_dev, static_cast<int>(ratings_cols * (compress ? num_latent_factors : ratings_rows_CU)), "V_host_dev", strPreamble(blank));
            //save_host_mtx_to_file<float>(V_host, ratings_cols, (compress ? num_latent_factors : ratings_rows_CU), "V_host", false, strPreamble(blank));
            //save_device_mtx_to_file<float>(V_dev, ratings_cols, (compress ? num_latent_factors : ratings_rows_CU), "V_dev", false, strPreamble(blank));
        }
        checkCudaErrors(cudaMemcpy(SV_dev, SV, ratings_rows_CU * SIZE_OF(float), cudaMemcpyHostToDevice));
        /*
            At this point U_CU is ratings_rows_CU by ratings_rows_CU in memory stored in row major
            ordering and V is ratings_cols by ratings_rows_CU stored in column major ordering

            There is no extra work to compress to compress V into ratings_cols by num_latent_factors, 
            just take the first num_latent_factors columns of each matrix.  
            The columns of U_CU are mixed in memory.
        */
        
    }else{
        if(row_major_ordering){
            //remember that R_CU is stored in row major ordering
            LOG("swap matrix indexing from row major to column major");
            gpu_swap_ordering<float>(ratings_rows_CU, ratings_cols, full_ratingsMtx_dev_CU, row_major_ordering);
            
        }

        gpu_orthogonal_decomp<float>(dn_handle, dn_solver_handle,
                                    ratings_rows_CU, ratings_cols, 
                                    &num_latent_factors, percent,
                                    full_ratingsMtx_dev_CU, U_CU, 
                                    V_dev, SV_with_U, SV_dev);

        //save_device_mtx_to_file<float>(U_CU_dev, ratings_rows_CU, num_latent_factors, "U_CU_compressed");
        /*
            At this point U_CU is ratings_rows_CU by ratings_rows_CU in memory stored in column major
            ordering and V is ratings_cols by ratings_rows_CU stored in column major ordering

            There is no extra work to compress U_CU into ratings_rows_CU by num_latent_factors, or
            to compress V into ratings_cols by num_latent_factors, just take the first num_latent_factors
            columns of each matrix
        */
    }  


    // float y;
    // LOG("V_dev : ");
    // gpu_msq_nonzero(ratings_cols * (compress ? num_latent_factors : ratings_rows_CU), V_dev, &y, true);
    // LOG("SV_dev : ");
    // gpu_msq_nonzero((compress ? num_latent_factors : ratings_rows_CU), SV_dev, &y, true);

    // return 0;
    
    int *   csr_format_ratingsMtx_userID_dev_testing_      = NULL;
    int *   coo_format_ratingsMtx_itemID_dev_testing_      = NULL;
    float * coo_format_ratingsMtx_rating_dev_testing_      = NULL;
    int*    csr_format_ratingsMtx_userID_dev_testing_batch = NULL;
    int*    coo_format_ratingsMtx_itemID_dev_testing_batch = NULL;
    float*  coo_format_ratingsMtx_rating_dev_testing_batch = NULL;
    if(Conserve_GPU_Mem){
        checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_testing_,  (batch_size_testing + 1) * SIZE_OF(int)));
        update_Mem((batch_size_testing + 1) * SIZE_OF(int) );
    }
    float * U_testing;
    float * R_testing;
    checkCudaErrors(cudaMalloc((void**)&U_testing,  batch_size_testing  * (compress ? num_latent_factors : ratings_rows_CU)  * SIZE_OF(float)));
    checkCudaErrors(cudaMalloc((void**)&R_testing,  batch_size_testing  * ratings_cols                        * SIZE_OF(float)));
    update_Mem(batch_size_testing  * (compress ? num_latent_factors : ratings_rows_CU)  * SIZE_OF(float));
    update_Mem(batch_size_testing  * ratings_cols                        * SIZE_OF(float));

    int real_num_batches = 75;
    for(int batch__ = 0; batch__ < std::min(real_num_batches, (int)num_batches_testing); batch__++){
        //for(int batch = 0; batch < num_batches_testing; batch++){

        LOG(std::endl<<"                                          ~~~ TESTING Batch "<<batch__<<" ( / "<<(std::min(real_num_batches, (int)num_batches_testing))<<" ) ~~~ "); 
        int batch = 0;
        getRandIntsBetween(&batch , 0 , (int)num_batches_testing - 1, 1);
        LOG("batch id : "<<batch);


        long long int batch_size_testing_temp = batch_size_testing;
        long long int first_row_index_in_batch_testing  = (batch_size_testing * (long long int)batch) /* % ratings_rows_testing*/;
        if(first_row_index_in_batch_testing + batch_size_testing >= ratings_rows_testing) {
            batch_size_testing_temp = ratings_rows_testing - first_row_index_in_batch_testing;
            if(Debug && 0){
                LOG("left over batch_size_testing : "<<batch_size_testing_temp);
            }
        } 
        LOG("batch_size_testing : "<<batch_size_testing_temp);
        if(Debug){
            //LOG(memLeft<<" available bytes left on the device");
            LOG("first_row_index_in_batch_testing : "<<first_row_index_in_batch_testing<< " ( / "<<ratings_rows_testing<<" )"); 
        };

        


        long long int nnz_testing;
        long long int first_coo_ind_testing;
        if(Conserve_GPU_Mem){
            if(Debug && 0){
                save_host_array_to_file<int>  (csr_format_ratingsMtx_userID_host_testing,  ratings_rows_testing + 1, "csr_format_ratingsMtx_userID_host_testing_2");
            }
            csr_format_ratingsMtx_userID_dev_testing_batch = csr_format_ratingsMtx_userID_host_testing +  first_row_index_in_batch_testing;
            first_coo_ind_testing = csr_format_ratingsMtx_userID_dev_testing_batch[0];
            int last_entry_index = (csr_format_ratingsMtx_userID_dev_testing_batch + batch_size_testing_temp)[0];

            nnz_testing = (long long int)last_entry_index - first_coo_ind_testing;
            if(Debug){
                LOG("first_coo_ind_testing : "<<first_coo_ind_testing  << " ( / "<< num_entries_testing<< " )");
                LOG("last_entry_index : "<<last_entry_index);
                LOG("nnz_testing : "<<nnz_testing);
            }

            if(nnz_testing <= 0){
                LOG("nnz_testing : "<<nnz_testing);
                ABORT_IF_EQ(0, 0, "nnz_testing <= 0");
            }
            
            checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_testing_,  nnz_testing        * SIZE_OF(int)));
            checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_testing_,  nnz_testing        * SIZE_OF(float)));
            update_Mem(nnz_testing * SIZE_OF(int) );
            update_Mem(nnz_testing * SIZE_OF(float) );

            checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_dev_testing_,  csr_format_ratingsMtx_userID_dev_testing_batch,  (batch_size_testing_temp + 1) *  SIZE_OF(int),   cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_dev_testing_,  coo_format_ratingsMtx_itemID_host_testing + first_coo_ind_testing, nnz_testing  *  SIZE_OF(int),   cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_testing_,  coo_format_ratingsMtx_rating_host_testing + first_coo_ind_testing,  nnz_testing  *  SIZE_OF(float), cudaMemcpyHostToDevice));
            
            csr_format_ratingsMtx_userID_dev_testing_batch = csr_format_ratingsMtx_userID_dev_testing_;
            coo_format_ratingsMtx_itemID_dev_testing_batch = coo_format_ratingsMtx_itemID_dev_testing_;
            coo_format_ratingsMtx_rating_dev_testing_batch = coo_format_ratingsMtx_rating_dev_testing_;
        }else{
            csr_format_ratingsMtx_userID_dev_testing_batch = csr_format_ratingsMtx_userID_dev_testing +  first_row_index_in_batch_testing;
            nnz_testing = gpu_get_num_entries_in_rows(0, batch_size_testing_temp - 1, csr_format_ratingsMtx_userID_dev_testing_batch);
            if(nnz_testing <=0){
                LOG("nnz_testing : "<<nnz_testing);
                ABORT_IF_EQ(0, 0, "nnz_testing <= 0");
            }
            first_coo_ind_testing = gpu_get_first_coo_index(0, csr_format_ratingsMtx_userID_dev_testing_batch);
            
            coo_format_ratingsMtx_itemID_dev_testing_batch = coo_format_ratingsMtx_itemID_dev_testing +  first_coo_ind_testing;
            coo_format_ratingsMtx_rating_dev_testing_batch = coo_format_ratingsMtx_rating_dev_testing +  first_coo_ind_testing;
        }

        ABORT_IF_LESS(nnz_testing, 1, "nnz < 1");
        
        float* coo_testing_errors;
        float* testing_entries;
        checkCudaErrors(cudaMalloc((void**)&coo_testing_errors, nnz_testing * SIZE_OF(float)));
        checkCudaErrors(cudaMalloc((void**)&testing_entries, nnz_testing * SIZE_OF(float)));
        update_Mem(2 * nnz_testing * SIZE_OF(float));

        long long int min_dim_ = std::min(ratings_rows_CU, ratings_cols);
        if(Debug && 0){
            LOG("testing requires " <<2 * nnz_testing * SIZE_OF(float) + batch_size_testing_temp  * std::max(min_dim_, batch_size_testing_temp) * SIZE_OF(float) 
                + batch_size_testing_temp  * ratings_cols * SIZE_OF(float) +
                (batch_size_testing_temp + 1) * SIZE_OF(int) + nnz_testing * SIZE_OF(int) + nnz_testing * SIZE_OF(float) << " bytes of memory");
            LOG("first_coo_ind in this TESTING batch : "<<first_coo_ind_testing<< " ( / "<<num_entries_testing<<" )");
            LOG("nnz in this TESTING batch : "<<nnz_testing);
            LOG("( next first_coo_ind in TESTING batch : "<<first_coo_ind_testing + nnz_testing<<" )");
            // save_device_mtx_to_file<float>(R, batch_size_training, ratings_cols, "error", false);
            // LOG("Press Enter to continue.") ;
            // std::cin.ignore();
        }
        //============================================================================================
        // Compute  R_testing * V = U_testing
        // Compute  Error = R_testing -  U_testing * V^T  <-- sparse
        //============================================================================================ 

        // gpu_R_error<float>(dn_handle, sp_handle, sp_descr,
        //                    batch_size_testing_temp, ratings_rows_CU, num_latent_factors, ratings_cols,
        //                    nnz_testing, first_coo_ind_testing, compress, 
        //                    testing_entries, coo_testing_errors, testing_fraction,
        //                    coo_format_ratingsMtx_rating_dev_testing + first_coo_ind_testing, 
        //                    csr_format_ratingsMtx_userID_dev_testing_batch, 
        //                    coo_format_ratingsMtx_itemID_dev_testing + first_coo_ind_testing,
        //                    V, U_testing, R_testing, "testing", (float)0.1, (float)0.01);


        gpu_R_error<float>(dn_handle, sp_handle, sp_descr, dn_solver_handle, 
                           batch_size_testing_temp, ratings_rows_CU, num_latent_factors, ratings_cols,
                           nnz_testing, first_coo_ind_testing, compress, 
                           testing_entries, coo_testing_errors, testing_fraction,
                           coo_format_ratingsMtx_rating_dev_testing_batch, 
                           csr_format_ratingsMtx_userID_dev_testing_batch, 
                           coo_format_ratingsMtx_itemID_dev_testing_batch,
                           V_dev, U_testing, R_testing, NULL, NULL,
                           training_rate, regularization_constant, 0, 0,
                           testing_error_on_training_entries + num_tests, testing_error_on_testing_entries + num_tests, 
                           testing_iterations + num_tests, SV_with_U, SV_dev, logarithmic_histogram + 7 * num_tests);

        //gpu_reverse_bools<float>(nnz_testing,  testing_entries);
        //gpu_hadamard<float>(nnz_testing, testing_entries, coo_testing_errors );
        //save_device_arrays_side_by_side_to_file<float>(coo_testing_errors, testing_entries, nnz_testing, "testing_entry_errors");

        //testing_error_temp += gpu_sum_of_squares<float>(nnz_testing, coo_testing_errors);
        
        

        checkCudaErrors(cudaFree(coo_testing_errors));
        checkCudaErrors(cudaFree(testing_entries));
        update_Mem(2 * nnz_testing * SIZE_OF(float) * (-1));  

        if(Conserve_GPU_Mem){
            checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_testing_));
            checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_testing_));
            update_Mem(nnz_testing * SIZE_OF(int) * (-1));
            update_Mem(nnz_testing * SIZE_OF(float) * (-1));
        }
        
        if(1){
            LOG("Batch "<<batch__<<" testing error per training entry: "<< testing_error_on_training_entries[num_tests]);
            LOG("Batch "<<batch__<<" expected number of training iterations when testing: "<< testing_iterations[num_tests]);
            LOG("Batch "<<batch__<<" testing error per testing entry: "<< testing_error_on_testing_entries[num_tests]);
            print_host_array(logarithmic_histogram + 7 * num_tests, 7, "Logarithmic Histogram of Errors From 10^(-3) to 10^3", strPreamble(blank));
            // LOG("Press Enter to continue.") ;
            // std::cin.ignore();
        }
        min_training_error = std::min(min_training_error, testing_error_on_training_entries[num_tests]);
        max_training_error = std::max(max_training_error, testing_error_on_training_entries[num_tests]);
        cpu_incremental_average((long long int)(num_tests + 1), &running_avg_testing_error_on_testing_entries, testing_error_on_testing_entries[num_tests]);
        cpu_incremental_average((long long int)(num_tests + 1), &running_avg_testing_error_on_training_entries, testing_error_on_training_entries[num_tests]);
        
        LOG("running_avg_testing_error_on_TESTING_entries :  "<<running_avg_testing_error_on_testing_entries<<std::endl);

        LOG("min_training_error :  "<<min_training_error);
        LOG("running_avg_testing_error_on_training_entries :  "<<running_avg_testing_error_on_training_entries);
        LOG("max_training_error :  "<<max_training_error<<std::endl);
        num_tests++;
    }//for loop on test batches

    checkCudaErrors(cudaFree(U_testing));
    checkCudaErrors(cudaFree(R_testing));
    update_Mem(batch_size_testing  * (compress ? num_latent_factors : ratings_rows_CU)  * SIZE_OF(float) * (-1));
    update_Mem(batch_size_testing  * ratings_cols                                                    * SIZE_OF(float) * (-1));


    //testing_error[num_tests] /= ((float)num_entries_testing * testing_fraction);
    //LOG("Iteration "<<it<<" MSQ testing error per testing entry: "<< testing_error[num_tests]);

    if(Conserve_GPU_Mem){
        checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_testing_));
        update_Mem((batch_size_testing + 1) * SIZE_OF(int) * (-1));
    }else{
        if(row_major_ordering){
            //remember that R_CU is stored in row major ordering
            LOG("swap matrix indexing from row major to column major");
            gpu_swap_ordering<float>(ratings_rows_CU, ratings_cols, full_ratingsMtx_dev_CU, !row_major_ordering);
            
        }
    }
    checkCudaErrors(cudaFree(SV_dev));
    update_Mem(ratings_rows_CU * SIZE_OF(float) * (-1));
    //LOG("HERE!"); checkCudaErrors(cudaDeviceSynchronize()); LOG("HERE!");



    //============================================================================================
    // Assume the rows of R_CU are the ratings_rows_CU-means for the training users - what is the error?
    //============================================================================================ 


    if(0){
        float* errors;
        int* selection;
        if(Conserve_GPU_Mem){
            errors  = (float *)malloc(ratings_rows_testing * SIZE_OF(float));
            selection  = (int *)malloc(ratings_rows_testing * SIZE_OF(int));
            checkErrors(errors);
            checkErrors(selection);
            cpu_sparse_nearest_row<float>(ratings_rows_CU, ratings_cols, full_ratingsMtx_host_CU, 
                                         ratings_rows_testing, num_entries_testing, 
                                         csr_format_ratingsMtx_userID_host_testing, 
                                         coo_format_ratingsMtx_itemID_host_testing,
                                         coo_format_ratingsMtx_rating_host_testing, 
                                         selection, errors);
            meta_km_errors[num_tests] = cpu_sum(ratings_rows_testing,  errors);
            //LOG("err norm when clustering : "<<std::sqrt(meta_km_errors[num_tests]));
            //meta_km_errors[num_tests] /= (float)num_entries_testing;
            LOG("mean err when clustering : "<<meta_km_errors[num_tests]);
            if(Debug){
                save_host_array_to_file<float>(errors, ratings_rows_testing, "meta_km_errors_single_it");
                save_host_array_to_file<int>(selection, ratings_rows_testing, "meta_km_selection_single_it");
            }
            cpu_logarithmic_histogram_abs_val<float>(ratings_rows_CU, ratings_cols, full_ratingsMtx_host_CU, 
                                                     ratings_rows_testing, num_entries_testing, 
                                                     csr_format_ratingsMtx_userID_host_testing, 
                                                     coo_format_ratingsMtx_itemID_host_testing,
                                                     coo_format_ratingsMtx_rating_host_testing, 
                                                     selection, (int)(0 - 3), 3, 
                                                     logarithmic_histogram_km + (7 * num_tests));
            free(errors);
            free(selection);
        }else{
            checkCudaErrors(cudaMalloc((void**)&errors, ratings_rows_testing * SIZE_OF(float)));
            checkCudaErrors(cudaMalloc((void**)&selection, ratings_rows_testing * SIZE_OF(int)));
            update_Mem(2 * ratings_rows_testing * SIZE_OF(int));
            gpu_sparse_nearest_row<float>(ratings_rows_CU, ratings_cols, full_ratingsMtx_dev_CU, 
                                     ratings_rows_testing, num_entries_testing, 
                                     csr_format_ratingsMtx_userID_dev_testing, 
                                     coo_format_ratingsMtx_itemID_dev_testing,
                                     coo_format_ratingsMtx_rating_dev_testing, 
                                     selection, errors, row_major_ordering);
            meta_km_errors[num_tests] = gpu_sum(ratings_rows_testing,  errors);
            LOG("err norm when clustering : "<<std::sqrt(meta_km_errors[num_tests]));
            meta_km_errors[num_tests] /= (float)ratings_rows_testing;
            LOG("mean sqed err when clustering : "<<meta_km_errors[num_tests]);
            save_device_array_to_file<float>(errors, ratings_rows_testing, "meta_km_errors");
            save_device_array_to_file<int>(selection, ratings_rows_testing, "meta_km_selection");
            checkCudaErrors(cudaFree(errors));
            checkCudaErrors(cudaFree(selection));
            update_Mem(2 * ratings_rows_testing * SIZE_OF(int) * (-1));
        }
    }
    
    
    LOG("      ~~~ DONE TESTING ~~~ "<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl); 

    // cpu_isBad<float>(testing_error_on_testing_entries + num_tests, (long long int)1, "testing_error_on_testing_entries", strPreamble(blank));
    // cpu_isBad<float>(testing_error_on_training_entries + num_tests, (long long int)1, "testing_error_on_training_entries", strPreamble(blank));
    // cpu_isBad<float>(testing_iterations + num_tests, (long long int)1, "testing_iterations", strPreamble(blank));
    // //cpu_isBad<float>(R_CU_abs_max + num_tests, (long long int)1, "R_CU_abs_max", strPreamble(blank));
    // cpu_isBad<float>(R_CU_max_sv + num_tests, (long long int)1, "R_CU_max_sv", strPreamble(blank));

    // if(load_full_CU_from_save){
    //     append_host_array_to_file(testing_error_on_testing_entries + num_tests, 1, "testing_error_on_testing_entries", strPreamble(blank));
    //     append_host_array_to_file(testing_error_on_training_entries + num_tests, 1, "testing_error_on_training_entries", strPreamble(blank));
    //     append_host_array_to_file(testing_iterations + num_tests, 1, "testing_iterations", strPreamble(blank));
    //     //append_host_array_to_file(R_CU_abs_max + num_tests, 1, "R_CU_abs_max", strPreamble(blank));
    //     append_host_array_to_file(R_CU_max_sv + num_tests, 1, "R_CU_max_sv", strPreamble(blank));
    //     append_host_array_to_file(meta_km_errors + num_tests, 1, "meta_km_errors", strPreamble(blank));
    //     append_host_mtx_to_file(logarithmic_histogram + num_tests * 7, 1, 7, "logarithmic_histogram", true, strPreamble(blank));
    //     append_host_mtx_to_file(logarithmic_histogram_km + num_tests * 7, 1, 7, "logarithmic_histogram_km", true, strPreamble(blank)); 
    //     num_tests += 1;
    // }else{
    //     num_tests += 1;
    //     save_host_array_to_file(testing_error_on_testing_entries, num_tests, "testing_error_on_testing_entries", strPreamble(blank));
    //     save_host_array_to_file(testing_error_on_training_entries, num_tests, "testing_error_on_training_entries", strPreamble(blank));
    //     save_host_array_to_file(testing_iterations, num_tests, "testing_iterations", strPreamble(blank));
    //     //save_host_array_to_file(R_CU_abs_max, num_tests, "R_CU_abs_max", strPreamble(blank));
    //     save_host_array_to_file(R_CU_max_sv, num_tests, "R_CU_max_sv", strPreamble(blank));
    //     save_host_array_to_file(meta_km_errors, num_tests, "meta_km_errors", strPreamble(blank));
    //     save_host_mtx_to_file(logarithmic_histogram, num_tests, 7, "logarithmic_histogram", true, strPreamble(blank)); 
    //     save_host_mtx_to_file(logarithmic_histogram_km, num_tests, 7, "logarithmic_histogram_km", true, strPreamble(blank));                
    // }





    //save_device_array_to_file<float>(testing_error, (num_iterations / testing_rate), "testing_error");

    
    cudaDeviceSynchronize();
    gettimeofday(&testing_end, NULL);
    testing_time = (testing_end.tv_sec * 1000 +(testing_end.tv_usec/1000.0))-(testing_start.tv_sec * 1000 +(testing_start.tv_usec/1000.0));  
    LOG("testing_time : "<<readable_time(testing_time));
    //============================================================================================
    // Destroy
    //============================================================================================
    LOG("Cleaning Up...");
    //free(user_means_testing_host);
    free(testing_error);
    if (full_ratingsMtx_host_CU  ) { free(full_ratingsMtx_host_CU); }


    cudaFree(U_CU);
    cudaFree(U_testing);
    cudaFree(V_dev);
    free(V_host);
    cudaFree(R_testing);
    update_Mem((ratings_rows_CU * std::min(ratings_rows_CU, ratings_cols)  + ratings_cols * std::min(ratings_rows_CU, ratings_cols))* static_cast<long long int>(SIZE_OF(float))* (-1));
    checkCudaErrors(cudaFree(SV_dev));

    if (testing_error_on_training_entries) free(testing_error_on_training_entries);
    if (testing_error_on_testing_entries) free(testing_error_on_testing_entries);
    if (testing_iterations) free(testing_iterations);
    if (meta_km_errors) free(meta_km_errors);
    if(logarithmic_histogram) free(logarithmic_histogram);
    if(logarithmic_histogram_km) free(logarithmic_histogram_km);
    
    
    if(!Conserve_GPU_Mem){
        cudaFree(full_ratingsMtx_dev_CU);
    }
    

    update_Mem((ratings_rows_CU * ratings_cols + /*ratings_rows_testing * ratings_cols +*/ num_entries_testing )* SIZE_OF(float));
    update_Mem(( (ratings_rows_testing + 1)  * static_cast<long long int>(SIZE_OF(int)) + num_entries_testing  * static_cast<long long int>(SIZE_OF(int)) + num_entries_testing  * static_cast<long long int>(SIZE_OF(float))
               + (ratings_rows_CU + 1)       * static_cast<long long int>(SIZE_OF(int)) + num_entries_CU       * static_cast<long long int>(SIZE_OF(int)) + num_entries_CU       * static_cast<long long int>(SIZE_OF(float))) * (-1) );
    

    if(Content_Based){
        cudaFree(csr_format_keyWordMtx_itemID_dev);          update_Mem((ratings_cols + 1)      * SIZE_OF(int) * (-1));
        cudaFree(coo_format_keyWordMtx_keyWord_dev);          update_Mem(num_entries_keyWord_mtx * SIZE_OF(int) * (-1));
    }

    cublasDestroy          (dn_handle);
    cusolverDnDestroy      (dn_solver_handle);

    cusparseDestroy        (sp_handle);
    cusparseDestroyMatDescr(sp_descr);
    cusolverSpDestroy      (sp_solver_handle);

    cudaDeviceSynchronize();
    gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("program_time: %f\n", program_time);   
    LOG("program_time : "<<readable_time(program_time));

    //if(Debug && memLeft!=devMem)LOG("WARNING POSSIBLE DEVICE MEMORY LEAK");
         
}
