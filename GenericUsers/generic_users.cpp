/**
    Author: Amy Nesky
**/
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

#include "generic_users.h"

const char *sSDKname = "Artificial Core Users Recommender Systems";

const bool Debug = 1;

bool Content_Based = 0;         // This means that we use extra knowledge about the user preferences or about the item relationships
bool Conserve_GPU_Mem = 1;      // This means that the full ACU rating mtx is stored on the host 
bool random_initialization = 0; // This means that we only split the data into two groups and the ACU mtx will be initialized randomly with perhaps other augmentation
bool load_CU_from_preprocessing = 1;
bool load_full_ACU_from_save = 1;
bool testing_only = 1;
std::string preprocessing_path = "";


#define update_Mem(new_mem) \
    allocatedMem += static_cast<long long int>(new_mem); \
    memLeft = static_cast<long long int>(devMem) - allocatedMem;
    // if(Debug) LOG(allocatedMem<<" allocated bytes on the device");\
    // if(Debug) LOG(memLeft<<" available bytes left on the device");
    // ABORT_IF_LESS(memLeft, 0, "Out of Memory"); 

    //
    //ABORT_IF_LESS(allocatedMem, (long long int)((double)devMem * (double)0.75), "Out of Memory"); \
    // if(Debug) LOG((int)devMem <<" total bytes on the device");\
    // if(Debug) LOG(new_mem<<" change in memory on the device");\






// #define allocate_V() \
//     if (d_S    ) checkCudaErrors(cudaFree(d_S));
//     if (d_S    ) checkCudaErrors(cudaFree(d_S));
//     if (d_S    ) checkCudaErrors(cudaFree(d_S));
//     checkCudaErrors(cudaMalloc((void**)&U_ACU,       batch_size_ACU       * batch_size_ACU                       * SIZE_OF(float)));
//     checkCudaErrors(cudaMalloc((void**)&V,          ratings_cols        * ratings_cols                        * SIZE_OF(float)));
//     checkCudaErrors(cudaMalloc((void**)&U_training, batch_size_training * std::max(min_, batch_size_training) * SIZE_OF(float)));
//     checkCudaErrors(cudaMalloc((void**)&R_training, batch_size_training * ratings_cols                        * SIZE_OF(float)));
//     checkCudaErrors(cudaMalloc((void**)&U_testing,  batch_size_testing  * std::max(min_, batch_size_testing)  * SIZE_OF(float)));
//     checkCudaErrors(cudaMalloc((void**)&R_testing,  batch_size_testing  * ratings_cols                        * SIZE_OF(float)));
//     update_Mem(Training_bytes);








int main(int argc, char *argv[])
{
    std::string blank = "";

    struct timeval program_start, program_end, training_start, training_end, testing_start, testing_end;
    double program_time;
    double training_time;
    gettimeofday(&program_start, NULL);

    long long int allocatedMem = (long long int)0; 


    cublasStatus_t cublas_status     = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;

    /* initialize random seed: */
    srand (cluster_seedgen());

    printf("%s Starting...\n\n", sSDKname);
    std::cout << "Current Date and Time :" << currentDateTime() << std::endl<< std::endl;
    LOG("Debug = "<<Debug);

    
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
    
        // long long int devMem;
        // long long int memLeft;

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


    // float* Beta = NULL;
    // checkCudaErrors(cudaMalloc((void**)&Beta, 200 * SIZE_OF(float)));
    // gpu_set_as_func_of_index<float>(Beta, 200, (float)1.0,  (float)1.0 - (float)0.01);
    // if(1){
    //   save_device_array_to_file<float>(Beta, 200, "V_regularizing_array");
    // }


    // gpu_block_orthogonal_decomp_from_host_test(dn_handle, dn_solver_handle);
    //return 0;

    //============================================================================================
    // Get the ratings data from CSV File
    //============================================================================================



    // Creating an object of CSVWriter

    std::string Dataset_Name;

    
    std::string csv_Ratings_Path;
    std::string csv_keyWords_path;
    std::string save_path = "/pylon5/ac560rp/nesky/REC_SYS/GenericUsers/observations/";



    long long int temp_num_entries;

    std::map<int, int> items_dictionary;

    int case_ = 1;

    switch (case_)
    {
        case 1:{ // code to be executed if n = 1;

            Dataset_Name = "MovieLens 20 million";

            csv_Ratings_Path = "/pylon5/ac560rp/nesky/REC_SYS/datasets/ml-20m/ratings_copy.csv";
            csv_keyWords_path = "/pylon5/ac560rp/nesky/REC_SYS/datasets/ml-20m/movies_copy.csv";
            //temp_num_entries = csv_Ratings.num_rows() - 1; // the first row is a title row
            preprocessing_path = "/pylon5/ac560rp/nesky/REC_SYS/CoreUsers/preprocessing/ml-20m/consider_item_cosine_similarity/user_cosine_similarity_incorporates_ratings/not/top_users.txt";
            temp_num_entries = 20000264 - 1;   // MovieLens 20 million
            save_path += "ml-20m/";
            break;
        }case 2:{ // code to be executed if n = 2;
            Dataset_Name = "Rent The Runaway";
            csv_Ratings_Path = "/pylon5/ac560rp/nesky/REC_SYS/datasets/renttherunway_final_data_copy.json";
            //csv_keyWords_path = "/pylon5/ac560rp/nesky/REC_SYS/datasets/renttherunway_final_data.json";
            preprocessing_path = "/pylon5/ac560rp/nesky/REC_SYS/CoreUsers/preprocessing/renttherunway/consider_item_cosine_similarity/user_cosine_similarity_incorporates_ratings/not/top_users.txt";
            Content_Based = 0;
            temp_num_entries = 192544;           // use for Rent The Runaway dataset
            save_path += "rentTheRunway/";
            break;
        }default: 
            ABORT_IF_EQ(0, 1, "no valid dataset selected");
    }
    if(Content_Based){
        save_path += "content_based_initialization/";
    }else if(load_CU_from_preprocessing){
        save_path += "cu/";
    }else{
       save_path += "not/"; 
    }

    if(load_CU_from_preprocessing) {
        random_initialization = 0;
    }
    // if(random_initialization) {
    //     load_CU_from_preprocessing = 0;
    // }

    LOG("Training using the "<< Dataset_Name <<" dataset");
    LOG("csv_Ratings_Path : "<< csv_Ratings_Path);
    LOG("csv_keyWords_path : "<< csv_keyWords_path <<" dataset");

    LOG("random_initialization : "<< random_initialization);// This means that we initialize the full ACU rating mtx randomly
    LOG("load_CU_from_preprocessing : "<< load_CU_from_preprocessing);// This means that we initialize the full ACU rating mtx randomly
    LOG("load_full_ACU_from_save : "<< load_full_ACU_from_save);
    LOG("Conserve_GPU_Mem : "<< Conserve_GPU_Mem);          // This means that the full ACU rating mtx is stored on the host
    LOG("Content_Based : "<< Content_Based<<std::endl);     // This means that we use extra knowledge about the user preferences or about the item relationships



    CSVReader csv_Ratings(csv_Ratings_Path);
    

    const long long int num_entries = temp_num_entries;
    //const int num_entries = 10000; //for debuging code
    bool load_all = true;
    bool original_column_ordering = true;
    original_column_ordering = original_column_ordering || Content_Based;// || load_CU_from_preprocessing || load_full_ACU_from_save
    if(!load_all){
        original_column_ordering = false;
    }
    
    int*   coo_format_ratingsMtx_userID_host  = NULL;
    int*   coo_format_ratingsMtx_itemID_host  = NULL;
    float* coo_format_ratingsMtx_rating_host  = NULL;
    coo_format_ratingsMtx_userID_host = (int *)  malloc(num_entries *  SIZE_OF(int)); 
    coo_format_ratingsMtx_itemID_host = (int *)  malloc(num_entries *  SIZE_OF(int)); 
    coo_format_ratingsMtx_rating_host = (float *)malloc(num_entries *  SIZE_OF(float)); 
    checkErrors(coo_format_ratingsMtx_userID_host);
    checkErrors(coo_format_ratingsMtx_itemID_host);
    checkErrors(coo_format_ratingsMtx_rating_host);

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
                                    num_entries, missing_, &items_dictionary);
            break;
        }default: 
            ABORT_IF_EQ(0, 1, "no valid dataset selected");
    }


    if(Debug && 0){
        save_host_arrays_side_by_side_to_file_(coo_format_ratingsMtx_userID_host, coo_format_ratingsMtx_itemID_host, 
                                              coo_format_ratingsMtx_rating_host, num_entries, "rows_cols_rating");
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
    

    free(coo_format_ratingsMtx_userID_host);
    free(coo_format_ratingsMtx_itemID_host);
    free(coo_format_ratingsMtx_rating_host);

    const long long int ratings_rows = (long long int)(gpu_abs_max<int>(num_entries, coo_format_ratingsMtx_userID_dev) + 1); 
    const long long int ratings_cols = (long long int)(gpu_abs_max<int>(num_entries, coo_format_ratingsMtx_itemID_dev) + 1); 

    LOG(std::endl<<"The sparse data matrix has "<<ratings_rows<<" users and "<<ratings_cols<<" items with "<<num_entries<<" specified entries.");
    LOG("The sparse data matrix has "<<(float)(ratings_rows * ratings_cols - num_entries) / (float)(ratings_rows * ratings_cols)<<" percent empty entries.");
    
    int*   csr_format_ratingsMtx_userID_dev;
    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev, (ratings_rows + 1) * SIZE_OF(int)));
    update_Mem( (ratings_rows + 1) * SIZE_OF(int) );

    cusparse_status = cusparseXcoo2csr(sp_handle, coo_format_ratingsMtx_userID_dev, num_entries, 
                                       ratings_rows, csr_format_ratingsMtx_userID_dev, CUSPARSE_INDEX_BASE_ZERO); 
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 
        fprintf(stdout, "Conversion from COO to CSR format failed\n");
        return 1; 
    } 

    

    if(Debug){
        // save_device_array_to_file<int>(csr_format_ratingsMtx_userID_dev, ratings_rows + 1, "csr_format_ratingsMtx_userID_dev");
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
        LOG("Content_Based --> This means that we use extra knowledge about the user preferences or about the item relationships") ;
        
        CSVReader csv_keyWords(csv_keyWords_path);
        LOG("Here!") ;
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

        LOG("Attempting to collect Content Based Information.");
        num_keyWords_temp  = csv_keyWords.makeContentBasedcooKeyWordMtx(coo_format_keyWordMtx_itemID_host,
                                                                        coo_format_keyWordMtx_keyWord_host,
                                                                        num_entries_keyWord_mtx_temp);
        LOG("num_keyWords : "<<num_keyWords_temp);
        if(num_keyWords_temp <= 0 ){
            return 0;
        }

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
    if(Content_Based){ // This means that we use extra knowledge about the user preferences or about the item relationships
        checkCudaErrors(cudaMalloc((void**)&csr_format_keyWordMtx_itemID_dev, (ratings_cols + 1) * SIZE_OF(int)));
        update_Mem( (ratings_cols + 1) * SIZE_OF(int) );

        cusparse_status = cusparseXcoo2csr(sp_handle, coo_format_keyWordMtx_itemID_dev, num_entries_keyWord_mtx, 
                                           ratings_cols, csr_format_keyWordMtx_itemID_dev, CUSPARSE_INDEX_BASE_ZERO); 
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 
            fprintf(stdout, "Conversion from COO to CSR format failed\n");
            return 1; 
        } 
        checkCudaErrors(cudaFree(coo_format_keyWordMtx_itemID_dev));           update_Mem(num_entries_keyWord_mtx * SIZE_OF(int) * (-1));
    }














    //============================================================================================
    // split the data into testing data and training data
    //============================================================================================

    


    /*
    random_initialization      -> fill ACU mtx with rand values from some rand distr.
    load_CU_from_preprocessing -> fill ACU mtx with Core Users

    */

    bool pull_rand_ACU_users = !random_initialization && !load_CU_from_preprocessing;





    const float probability_ACU       = (float)10.0/(float)100.0;
    const float p                    = (float)1.0/(float)10.0;
    const float probability_testing  = !pull_rand_ACU_users ?       p        :        p        * ((float)1.0 - probability_ACU);
    const float probability_training = !pull_rand_ACU_users ? (float)1.0 - p :((float)1.0 - p) * ((float)1.0 - probability_ACU);
    LOG("percentage of users for testing: " <<probability_testing);
    LOG("percentage of users for training: "<<probability_training);
    if(pull_rand_ACU_users) {
        LOG("percentage of users for ACU: "      <<(float)1.0 - probability_training - probability_testing<<std::endl);
    }

    long long int ratings_rows_ACU_temp       = (long long int)(probability_ACU * (float)ratings_rows);
    if(!pull_rand_ACU_users){
        ratings_rows_ACU_temp  /= (long long int)1000;
        ratings_rows_ACU_temp  *= (long long int)1000; // now ratings_rows_ACU_temp is divisible by 100
    }
    ABORT_IF_LE(probability_ACU, (float)0.0, "probability_ACU <= 0");
    ABORT_IF_LE(probability_testing, (float)0.0, "probability_testing <= 0");
    ABORT_IF_LE(probability_training, (float)0.0, "probability_training <= 0");



    int num_groups =  pull_rand_ACU_users ? 3 : 2;
    float  probability_of_groups_host [num_groups];
    probability_of_groups_host[0] = probability_testing;
    probability_of_groups_host[1] = probability_training;
    if(pull_rand_ACU_users){
        probability_of_groups_host[2] = (float)1.0 - probability_training - probability_testing;
    }

    float* probability_of_groups_dev;
    checkCudaErrors(cudaMalloc((void**)&probability_of_groups_dev, num_groups * SIZE_OF(float)));
    update_Mem( num_groups * SIZE_OF(float) );
    checkCudaErrors(cudaMemcpy(probability_of_groups_dev, probability_of_groups_host, num_groups * SIZE_OF(float), cudaMemcpyHostToDevice));

    int *group_indicies;
    checkCudaErrors(cudaMalloc((void**)&group_indicies, ratings_rows * SIZE_OF(int)));
    update_Mem( ratings_rows * SIZE_OF(int) );
    gpu_set_all<int>(group_indicies, ratings_rows, 1);

    gpu_get_rand_groups(ratings_rows,  group_indicies, probability_of_groups_dev, num_groups);

    int* top_users = NULL;
    if(load_CU_from_preprocessing && !load_full_ACU_from_save){
        top_users= (int *)malloc(ratings_rows * SIZE_OF(int));
        checkErrors(top_users);

        LOG("Load top_users from saved file in "<<preprocessing_path);
        // Load top_users from saved file
        //get_host_array_from_saved_txt<int>(top_users, ratings_rows, preprocessing_path);

        CSVReader csv_Preprocessing(preprocessing_path);
        csv_Preprocessing.getData(top_users, ratings_rows, 1);
        LOG("Sanity Check : ");
        LOG("top_users[0] : "<<top_users[0]);
        LOG("top_users[ratings_rows - 1] : "<<top_users[ratings_rows - 1]);

        gpu_mark_ACU_users(ratings_rows_ACU_temp, ratings_rows, top_users, group_indicies );
        free(top_users);
        LOG("here!");
    }
    if(load_full_ACU_from_save){
        LOG("LOADING PRESAVED TESTING AND TRAINING SETS");

        int* temp__ = NULL;
        temp__= (int *)malloc(ratings_rows * SIZE_OF(int));
        checkErrors(temp__);

        CSVReader csv_Preprocessing(save_path + "group_indicies.txt");
        csv_Preprocessing.getData(temp__, ratings_rows, 1);
        checkCudaErrors(cudaMemcpy(group_indicies, temp__, ratings_rows * SIZE_OF(int), cudaMemcpyHostToDevice));

        LOG("Sanity Check : ");
        LOG("group_indicies[0] : "<<temp__[0]);
        LOG("group_indicies[ratings_rows - 1] : "<<temp__[ratings_rows - 1]);
        free(temp__);
    }else{
        if(too_big(ratings_rows) ) {ABORT_IF_NEQ(0, 1,"Long long long int too big");}
        save_device_array_to_file<int>(group_indicies, (int)ratings_rows, save_path + "group_indicies", strPreamble(blank));
    }

    num_groups =  random_initialization ? 2 : 3;
    int* group_sizes = NULL;
    group_sizes = (int *)malloc(num_groups * SIZE_OF(int)); 
    checkErrors(group_sizes);

    count_each_group_from_coo(num_groups, group_indicies, num_entries, coo_format_ratingsMtx_userID_dev, group_sizes);
    const long long int num_entries_testing   = group_sizes[0];
    const long long int num_entries_training  = group_sizes[1];

    if(!random_initialization){
        ratings_rows_ACU_temp = group_sizes[2];
    }
    const long long int num_entries_ACU        = ratings_rows_ACU_temp;


    count_each_group(ratings_rows, group_indicies, group_sizes, num_groups);
    const long long int ratings_rows_testing  = group_sizes[0];
    const long long int ratings_rows_training = group_sizes[1];
    if(!random_initialization){
        ratings_rows_ACU_temp = group_sizes[2];
    }
    const long long int ratings_rows_ACU       = ratings_rows_ACU_temp;
    
    LOG("num testing users : "   <<ratings_rows_testing);
    LOG("num training users : "  <<ratings_rows_training);
    LOG("num ACU users : "        <<ratings_rows_ACU);
    LOG("num testing entries : " <<num_entries_testing);
    LOG("num training entries : "<<num_entries_training);

    LOG("MEMORY REDUCTION OF ACU USERS : "      <<static_cast<double>(ratings_rows_ACU * ratings_cols) / static_cast<double>(ratings_rows + (long long int)2 * num_entries)<<std::endl<<std::endl<<std::endl);

    
    
    if(!random_initialization){
        LOG("num ACU entries : "      <<num_entries_ACU<<std::endl);
        ABORT_IF_NEQ(ratings_rows_testing + ratings_rows_training + ratings_rows_ACU, ratings_rows, "The number of rows does not add up correctly.");
        ABORT_IF_NEQ(num_entries_testing  + num_entries_training  + num_entries_ACU,  num_entries, "The number of entries does not add up correctly.");
    }else{
        ABORT_IF_NEQ(ratings_rows_testing + ratings_rows_training, ratings_rows, "The number of rows does not add up correctly.");
        ABORT_IF_NEQ(num_entries_testing  + num_entries_training ,  num_entries, "The number of entries does not add up correctly.");        
    }
    ABORT_IF_LE(ratings_rows_ACU, (long long int)0, "ratings_rows_ACU <= 0");
    ABORT_IF_LE(ratings_rows_testing, (long long int)0, "ratings_rows_testing <= 0");
    ABORT_IF_LE(ratings_rows_training, (long long int)0, "ratings_rows_training <= 0");    

    if(Debug && 0){
        // LOG("coo_format_ratingsMtx_userID_dev : ");
        // print_gpu_array_entries<int>(coo_format_ratingsMtx_userID_dev, 100, 1 , num_entries);
        LOG("group_indicies :");
        print_gpu_mtx_entries<int>(group_indicies, ratings_rows, 1 );
        //save_device_array_to_file<int>(group_indicies, ratings_rows, "testing_bools");
        LOG("Press Enter to continue.") ;
        std::cin.ignore();
    }
    
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_userID_dev));
    update_Mem( num_entries * SIZE_OF(int) * (-1) );

    int*   csr_format_ratingsMtx_userID_dev_testing;
    int*   coo_format_ratingsMtx_itemID_dev_testing;
    float* coo_format_ratingsMtx_rating_dev_testing;

    int*   csr_format_ratingsMtx_userID_dev_training;
    int*   coo_format_ratingsMtx_itemID_dev_training;
    float* coo_format_ratingsMtx_rating_dev_training;

    int*   csr_format_ratingsMtx_userID_dev_ACU;
    int*   coo_format_ratingsMtx_itemID_dev_ACU;
    float* coo_format_ratingsMtx_rating_dev_ACU;

    
    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_testing,  (ratings_rows_testing + 1) * SIZE_OF(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_testing,  num_entries_testing        * SIZE_OF(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing        * SIZE_OF(float)));

    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_training,  (ratings_rows_training + 1) * SIZE_OF(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_training,  num_entries_training        * SIZE_OF(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_training,  num_entries_training        * SIZE_OF(float)));

    if(!random_initialization){
        checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_ACU,  (ratings_rows_ACU + 1) * SIZE_OF(int)));
        checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_ACU,  num_entries_ACU        * SIZE_OF(int)));
        checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_ACU,  num_entries_ACU        * SIZE_OF(float)));
    }
    update_Mem(  (ratings_rows_testing + 1)  * SIZE_OF(int) + num_entries_testing  * SIZE_OF(int) + num_entries_testing  * SIZE_OF(float)
               + (ratings_rows_training + 1) * SIZE_OF(int) + num_entries_training * SIZE_OF(int) + num_entries_training * SIZE_OF(float)
               + (ratings_rows_ACU + 1)       * SIZE_OF(int) + num_entries_ACU       * SIZE_OF(int) + num_entries_ACU       * SIZE_OF(float)  );
    
    int*   csr_format_ratingsMtx_userID_dev_by_group_host  [num_groups];
    csr_format_ratingsMtx_userID_dev_by_group_host[0] = csr_format_ratingsMtx_userID_dev_testing;
    csr_format_ratingsMtx_userID_dev_by_group_host[1] = csr_format_ratingsMtx_userID_dev_training;
    if(!random_initialization){
        csr_format_ratingsMtx_userID_dev_by_group_host[2] = csr_format_ratingsMtx_userID_dev_ACU;
    }
    int*   coo_format_ratingsMtx_itemID_dev_by_group_host  [num_groups];
    coo_format_ratingsMtx_itemID_dev_by_group_host[0] = coo_format_ratingsMtx_itemID_dev_testing;
    coo_format_ratingsMtx_itemID_dev_by_group_host[1] = coo_format_ratingsMtx_itemID_dev_training;
    if(!random_initialization){
        coo_format_ratingsMtx_itemID_dev_by_group_host[2] = coo_format_ratingsMtx_itemID_dev_ACU;
    }
    float*   coo_format_ratingsMtx_rating_dev_by_group_host  [num_groups];
    coo_format_ratingsMtx_rating_dev_by_group_host[0] = coo_format_ratingsMtx_rating_dev_testing;
    coo_format_ratingsMtx_rating_dev_by_group_host[1] = coo_format_ratingsMtx_rating_dev_training;
    if(!random_initialization){
        coo_format_ratingsMtx_rating_dev_by_group_host[2] = coo_format_ratingsMtx_rating_dev_ACU;
    }
    int ratings_rows_by_group_host[num_groups];
    ratings_rows_by_group_host[0] = ratings_rows_testing;
    ratings_rows_by_group_host[1] = ratings_rows_training;
    if(!random_initialization){
        ratings_rows_by_group_host[2] = ratings_rows_ACU;
    }
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
    if(Debug && 0){
        if(random_initialization){
            save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_ACU,        ratings_rows_ACU + 1,       "csr_format_ratingsMtx_userID_dev_ACU");
            save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_ACU,        num_entries_ACU,            "coo_format_ratingsMtx_itemID_dev_ACU");
            save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_ACU,        num_entries_ACU,            "coo_format_ratingsMtx_rating_dev_ACU");
        }
        save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_training,  ratings_rows_training + 1, "csr_format_ratingsMtx_userID_dev_training");
        save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_training,  num_entries_training,      "coo_format_ratingsMtx_itemID_dev_training");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_training,  num_entries_training,      "coo_format_ratingsMtx_rating_dev_training");
        save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_testing,   ratings_rows_testing + 1,  "csr_format_ratingsMtx_userID_dev_testing");
        save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_testing,   num_entries_testing,       "coo_format_ratingsMtx_itemID_dev_testing");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_testing,   num_entries_testing,       "coo_format_ratingsMtx_rating_dev_testing");
        // LOG("csr_format_ratingsMtx_userID_dev_training : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_training, ratings_rows_training + 1, 1 );
        // LOG("csr_format_ratingsMtx_userID_dev_ACU : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_ACU, ratings_rows_ACU + 1, 1 );
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }


    
    checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev));            update_Mem((ratings_rows + 1) * SIZE_OF(int) * (-1));
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev));            update_Mem(num_entries * SIZE_OF(int) * (-1));
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev));            update_Mem(num_entries * SIZE_OF(float) * (-1));
    checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_by_group));   update_Mem(num_groups * SIZE_OF(int*) * (-1));
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_by_group));   update_Mem(num_groups * SIZE_OF(int*) * (-1));
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_by_group));   update_Mem(num_groups * SIZE_OF(float*) * (-1));
    checkCudaErrors(cudaFree(ratings_rows_by_group));                       update_Mem(num_groups * SIZE_OF(int) * (-1));
    checkCudaErrors(cudaFree(probability_of_groups_dev));                   update_Mem(num_groups * SIZE_OF(float) * (-1));
    checkCudaErrors(cudaFree(group_indicies));                              update_Mem(num_groups * SIZE_OF(int)* (-1));
    
    free(group_sizes);






    






    //============================================================================================
    // collect User Means and Variances
    //============================================================================================










    LOG("collect User Means and Variance... ");

    float* user_means_training;
    float* user_means_testing;
    float* user_means_ACU;
    checkCudaErrors(cudaMalloc((void**)&user_means_training, ratings_rows_training * SIZE_OF(float)));
    checkCudaErrors(cudaMalloc((void**)&user_means_testing,  ratings_rows_testing  * SIZE_OF(float)));
    checkCudaErrors(cudaMalloc((void**)&user_means_ACU,       ratings_rows_ACU       * SIZE_OF(float)));
    update_Mem((ratings_rows_training + ratings_rows_testing + ratings_rows_ACU)* SIZE_OF(float));

    float* user_var_training;
    float* user_var_testing;
    float* user_var_ACU;
    checkCudaErrors(cudaMalloc((void**)&user_var_training, ratings_rows_training * SIZE_OF(float)));
    checkCudaErrors(cudaMalloc((void**)&user_var_testing,  ratings_rows_testing  * SIZE_OF(float)));
    checkCudaErrors(cudaMalloc((void**)&user_var_ACU,       ratings_rows_ACU       * SIZE_OF(float)));
    update_Mem((ratings_rows_training + ratings_rows_testing + ratings_rows_ACU)* SIZE_OF(float));


    collect_user_means(user_means_training, user_var_training,  (long long int)ratings_rows_training,
                       csr_format_ratingsMtx_userID_dev_training,
                       coo_format_ratingsMtx_rating_dev_training,
                       user_means_testing, user_var_testing,    (long long int)ratings_rows_testing,
                       csr_format_ratingsMtx_userID_dev_testing,
                       coo_format_ratingsMtx_rating_dev_testing,
                       user_means_ACU, user_var_ACU,              random_initialization ? (long long int)0 : (long long int)ratings_rows_ACU,
                       csr_format_ratingsMtx_userID_dev_ACU,
                       coo_format_ratingsMtx_rating_dev_ACU);

    if(Debug && 0){
        save_device_array_to_file<float>(user_means_testing,  ratings_rows_testing,  "user_means_testing");
        save_device_array_to_file<float>(user_var_testing,    ratings_rows_testing,  "user_var_testing");
        save_device_array_to_file<float>(user_means_training, ratings_rows_training, "user_means_training");
        save_device_array_to_file<float>(user_var_training,   ratings_rows_training, "user_var_training");
        if(!random_initialization){
            save_device_array_to_file<float>(user_means_ACU,       ratings_rows_ACU,       "user_means_ACU");
            save_device_array_to_file<float>(user_var_ACU,         ratings_rows_ACU,       "user_var_ACU");
        }
        // LOG("user_means_training : ");
        // print_gpu_array_entries(user_means_ACU, ratings_rows_ACU);
        // LOG("user_means_training : ");
        // print_gpu_array_entries(user_means_ACU, ratings_rows_ACU);
        // LOG("user_means_ACU : ");
        // print_gpu_array_entries(user_means_ACU, ratings_rows_ACU);
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }





    





    //============================================================================================
    // Center and Fill Training Data
    //============================================================================================










    LOG("Center Data and fill ACU matrix... ");


    float * coo_format_ratingsMtx_row_centered_rating_dev_ACU;
    float * coo_format_ratingsMtx_row_centered_rating_dev_testing;
    float * coo_format_ratingsMtx_row_centered_rating_dev_training;
    
    if(!random_initialization){
        checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_row_centered_rating_dev_ACU,       num_entries_ACU       * SIZE_OF(float)));
    }
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_row_centered_rating_dev_testing,  num_entries_testing  * SIZE_OF(float)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_row_centered_rating_dev_training, num_entries_training * SIZE_OF(float)));
    update_Mem( ( num_entries_ACU + num_entries_testing + num_entries_training) * SIZE_OF(float) );




    //const float val_when_var_is_zero = (float)3.5774;        // use for MovieLens
    const float val_when_var_is_zero = (float)0.5;        // use for Rent The Runway
    LOG("rating used when the variance of the user's ratings is zero : "<< val_when_var_is_zero);

    if(!random_initialization){
        center_ratings(user_means_ACU, user_var_ACU, 
                       ratings_rows_ACU, num_entries_ACU,
                        csr_format_ratingsMtx_userID_dev_ACU,
                        coo_format_ratingsMtx_rating_dev_ACU,
                        coo_format_ratingsMtx_row_centered_rating_dev_ACU, val_when_var_is_zero);

        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_ACU,       coo_format_ratingsMtx_row_centered_rating_dev_ACU,       num_entries_ACU       *  SIZE_OF(float), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_row_centered_rating_dev_ACU));             update_Mem((ratings_rows_ACU + 1) * SIZE_OF(int) * (-1));
        float range_ACU       = gpu_range<float>(num_entries_ACU,        coo_format_ratingsMtx_rating_dev_ACU);
    }

    center_ratings(user_means_testing, user_var_testing, 
                   ratings_rows_testing, num_entries_testing,
                    csr_format_ratingsMtx_userID_dev_testing,
                    coo_format_ratingsMtx_rating_dev_testing,
                    coo_format_ratingsMtx_row_centered_rating_dev_testing, val_when_var_is_zero);

    center_ratings(user_means_training, user_var_training, 
                   ratings_rows_training, num_entries_training,
                    csr_format_ratingsMtx_userID_dev_training,
                    coo_format_ratingsMtx_rating_dev_training,
                    coo_format_ratingsMtx_row_centered_rating_dev_training, val_when_var_is_zero);

    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_testing,  coo_format_ratingsMtx_row_centered_rating_dev_testing,  num_entries_testing  *  SIZE_OF(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_training, coo_format_ratingsMtx_row_centered_rating_dev_training, num_entries_training *  SIZE_OF(float), cudaMemcpyDeviceToDevice));
    
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_row_centered_rating_dev_testing));        update_Mem(num_entries_ACU        * SIZE_OF(int) * (-1));
    checkCudaErrors(cudaFree(coo_format_ratingsMtx_row_centered_rating_dev_training));       update_Mem(num_entries_ACU        * SIZE_OF(float) * (-1));

    
    
    float range_training = gpu_range<float>         (num_entries_training, coo_format_ratingsMtx_rating_dev_training);
    const float min_training   = gpu_min<float>           (num_entries_training, coo_format_ratingsMtx_rating_dev_training);
    const float max_training   = gpu_max<float>           (num_entries_training, coo_format_ratingsMtx_rating_dev_training);
    const float abs_max_training = std::max(max_training, std::abs(min_training));
    const float expt_training  = gpu_expected_value<float>(num_entries_training, coo_format_ratingsMtx_rating_dev_training);
    const float expt_abs_training  = gpu_expected_abs_value<float>(num_entries_training, coo_format_ratingsMtx_rating_dev_training);

    float range_testing  = gpu_range<float>(num_entries_testing,   coo_format_ratingsMtx_rating_dev_testing);
    const float min_testing   = gpu_min<float>           (num_entries_testing, coo_format_ratingsMtx_rating_dev_testing);
    const float max_testing   = gpu_max<float>           (num_entries_testing, coo_format_ratingsMtx_rating_dev_testing);
    const float abs_max_testing = std::max(std::abs(max_testing), std::abs(min_testing));
    const float expt_testing  = gpu_expected_value<float>(num_entries_testing, coo_format_ratingsMtx_rating_dev_testing);
    const float expt_abs_testing  = gpu_expected_abs_value<float>(num_entries_testing, coo_format_ratingsMtx_rating_dev_testing);

    range_training = (float)2.0 * expt_abs_training;

    //LOG(std::endl<<"range_ACU = "         <<range_ACU) ;
    //LOG("range_testing = "    <<range_testing) ;
    //LOG("range_training = "   <<range_training) ;
    LOG("max_training = "     <<max_training) ;
    LOG("min_training = "     <<min_training) ;
    LOG("abs_max_training = " <<abs_max_training) ;
    LOG("expt_training = "    <<expt_training) ;
    LOG("expt_abs_training = "<<expt_abs_training) ;

    LOG("max_testing = "     <<max_testing) ;
    LOG("min_testing = "     <<min_testing) ;
    LOG("abs_max_testing = " <<abs_max_testing) ;
    LOG("expt_testing = "    <<expt_testing) ;
    LOG("expt_abs_testing = "<<expt_abs_testing) ;

   
    if( Debug && 0){
        if(!random_initialization){
            save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_ACU,        ratings_rows_ACU + 1,       "csr_format_ratingsMtx_userID_dev_ACU");
            save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_ACU,        num_entries_ACU,            "coo_format_ratingsMtx_itemID_dev_ACU");
            save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_ACU,        num_entries_ACU,            "coo_format_ratingsMtx_rating_dev_ACU");
        }

    	// save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_training,  num_entries_training,      "coo_format_ratingsMtx_rating_dev_training_post_centering");
    	// save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_ACU,  num_entries_ACU,      "coo_format_ratingsMtx_rating_dev_ACU_post_centering");
    	// save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing,      "coo_format_ratingsMtx_rating_dev_testing_post_centering");

        // save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_testing,  ratings_rows_testing + 1, "csr_format_ratingsMtx_userID_dev_testing_post_centering");
        // save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_testing, num_entries_testing,      "coo_format_ratingsMtx_itemID_dev_testing_post_centering");
        // save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing,      "coo_format_ratingsMtx_rating_dev_testing_post_centering");
        
        // save_device_arrays_side_by_side_to_file(coo_format_ratingsMtx_itemID_dev_training, coo_format_ratingsMtx_rating_dev_training, 
        //                                         num_entries_training, "training_users_cols_rating_centered");
        // save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_training,  ratings_rows_training + 1, "csr_format_ratingsMtx_userID_dev_training_post_centering" );
        // save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_training, num_entries_training,      "coo_format_ratingsMtx_itemID_dev_training_post_centering");
        // save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_training,  num_entries_training,      "coo_format_ratingsMtx_rating_dev_training_post_centering" );

        // LOG("csr_format_ratingsMtx_userID_dev_training : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_training, ratings_rows_training + 1, 1 );
        // LOG("csr_format_ratingsMtx_userID_dev_ACU : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_ACU, ratings_rows_ACU + 1, 1 );
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }





    bool row_major_ordering = true;
    LOG("ACU matrix row_major_ordering = " <<row_major_ordering);

    float * full_ratingsMtx_dev_ACU = NULL;
    float * full_ratingsMtx_host_ACU = NULL;

    const long long int ACU_mtx_size = (long long int)ratings_rows_ACU * (long long int)ratings_cols;
    const long long int ACU_mtx_size_bytes = (long long int)ratings_rows_ACU * (long long int)ratings_cols * (long long int)SIZE_OF(float);
    LOG(std::endl);
    LOG("Will need "<<ACU_mtx_size<< " floats for the ACU mtx.") ;
    LOG("Will need "<<ACU_mtx_size_bytes<< " bytes for the ACU mtx.") ;
    if(allocatedMem + ACU_mtx_size_bytes > (long long int)((double)devMem * (double)0.75)){
        LOG("Conserving Memory Now --> This means that the full ACU rating mtx is stored on the host ");
        Conserve_GPU_Mem = 1;
    }
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();

    
    int*   csr_format_ratingsMtx_userID_host_ACU = NULL;
    int*   coo_format_ratingsMtx_itemID_host_ACU = NULL;
    float* coo_format_ratingsMtx_rating_host_ACU = NULL;
    
    if(Conserve_GPU_Mem){
        // This means that the full ACU rating mtx is stored on the host
        
        full_ratingsMtx_host_ACU = (float *)malloc(ACU_mtx_size_bytes);
        checkErrors(full_ratingsMtx_host_ACU);
        
        if(load_full_ACU_from_save){
            //load later

            // std::string full_ratingsMtx_ACU_filepath = save_path + "full_ratingsMtx_ACU.txt";

            // CSVReader ACU_mtx_reader(full_ratingsMtx_ACU_filepath);
            // ACU_mtx_reader.getData(full_ratingsMtx_host_ACU, ratings_rows_ACU, ratings_cols);
            // if(!row_major_ordering){
            //     ABORT_IF_EQ(0, 0, "Option not ready");
            // }

        }else{
            //cpu_set_all<float>(full_ratingsMtx_host_ACU, ACU_mtx_size, (float)0.0);
            //host_rng_uniform(ratings_rows_ACU * ratings_cols, min_training, max_training, full_ratingsMtx_host_ACU);
            if(random_initialization){
                //host_rng_uniform(ratings_rows_ACU * ratings_cols, (float)((-1.0)* std::sqrt(3.0)), (float)(std::sqrt(3.0)), full_ratingsMtx_host_ACU);
                host_rng_gaussian(ratings_rows_ACU * ratings_cols, (float)0.0, (float)1.0, full_ratingsMtx_host_ACU);
            }else{
                //host_rng_uniform(ratings_rows_ACU * ratings_cols, (float)((-1.0)* 0.1), (float)0.1, full_ratingsMtx_host_ACU);
                //host_rng_gaussian(ratings_rows_ACU * ratings_cols, (float)0.0, (float)0.1, full_ratingsMtx_host_ACU);
                cpu_set_all<float>(full_ratingsMtx_host_ACU, ACU_mtx_size, (float)0.0);

                csr_format_ratingsMtx_userID_host_ACU  = (int *)  malloc((ratings_rows_ACU + 1) * SIZE_OF(int)  );
                coo_format_ratingsMtx_itemID_host_ACU  = (int *)  malloc(num_entries_ACU        * SIZE_OF(int)  );
                coo_format_ratingsMtx_rating_host_ACU  = (float *)malloc(num_entries_ACU        * SIZE_OF(float));


                checkErrors(csr_format_ratingsMtx_userID_host_ACU);
                checkErrors(coo_format_ratingsMtx_itemID_host_ACU);
                checkErrors(coo_format_ratingsMtx_rating_host_ACU);
                checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host_ACU,  csr_format_ratingsMtx_userID_dev_ACU,  (ratings_rows_ACU + 1) * SIZE_OF(int), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host_ACU,  coo_format_ratingsMtx_itemID_dev_ACU,  num_entries_ACU        * SIZE_OF(int), cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host_ACU,  coo_format_ratingsMtx_rating_dev_ACU,  num_entries_ACU        * SIZE_OF(float), cudaMemcpyDeviceToHost));
                
                cpu_fill_training_mtx((long long int)ratings_rows_ACU, (long long int)ratings_cols, (long long int)num_entries_ACU, 
                                      row_major_ordering,  
                                      csr_format_ratingsMtx_userID_host_ACU,
                                      coo_format_ratingsMtx_itemID_host_ACU,
                                      coo_format_ratingsMtx_rating_host_ACU,
                                      full_ratingsMtx_host_ACU);

                if(Debug & 0){
                    save_host_array_to_file(full_ratingsMtx_host_ACU, 1000/*ratings_cols * ratings_rows_ACU*/, "full_ratingsMtx_ACU", strPreamble(blank));
                    return 0;
                }

                cpu_shuffle_mtx_rows_or_cols(ratings_rows_ACU, ratings_cols, 
                                         row_major_ordering, full_ratingsMtx_host_ACU, 1);

                free(csr_format_ratingsMtx_userID_host_ACU);
                free(coo_format_ratingsMtx_itemID_host_ACU);
                free(coo_format_ratingsMtx_rating_host_ACU); 
            }

        }

        float R_ACU_mean_abs_nonzero = (float)0.0;
        //cpu_mean_abs_nonzero<float>(ratings_rows_ACU * ratings_cols, full_ratingsMtx_host_ACU, &R_ACU_mean_abs_nonzero, Debug);

        LOG("full_ratingsMtx_host_ACU filled and shuffled") ;
    }else{
        checkCudaErrors(cudaMalloc((void**)&full_ratingsMtx_dev_ACU, ratings_rows_ACU * ratings_cols * SIZE_OF(float)));
        update_Mem(ratings_rows_ACU * ratings_cols * SIZE_OF(float));

        if(load_full_ACU_from_save){
            ABORT_IF_EQ(0, 0, "Option not ready");
        }else{
            //gpu_set_all<float>(full_ratingsMtx_dev_ACU, ratings_rows_ACU * ratings_cols, (float)0.0);
            //gpu_rng_uniform<float>(dn_handle, ratings_rows_ACU * ratings_cols, min_training, max_training, full_ratingsMtx_dev_ACU);
            gpu_rng_gaussian<float>(ratings_rows_ACU * ratings_cols, (float)0.0, (float)0.00007, full_ratingsMtx_dev_ACU);

            if(!random_initialization){
                gpu_fill_training_mtx(ratings_rows_ACU, ratings_cols, row_major_ordering,
                                      csr_format_ratingsMtx_userID_dev_ACU,
                                      coo_format_ratingsMtx_itemID_dev_ACU,
                                      coo_format_ratingsMtx_rating_dev_ACU,
                                      full_ratingsMtx_dev_ACU);

                if(Content_Based){ // This means that we use extra knowledge about the user preferences or about the item relationships
                    gpu_supplement_training_mtx_with_content_based(ratings_rows_ACU, 
                                                                    ratings_cols, 
                                                                    row_major_ordering,
                                                                    csr_format_ratingsMtx_userID_dev_ACU,
                                                                    coo_format_ratingsMtx_itemID_dev_ACU,
                                                                    coo_format_ratingsMtx_rating_dev_ACU,
                                                                    full_ratingsMtx_dev_ACU,
                                                                    csr_format_keyWordMtx_itemID_dev,
                                                                    coo_format_keyWordMtx_keyWord_dev);

                    const float max_training_supplement   = gpu_max<float>(ratings_rows_ACU * ratings_cols, full_ratingsMtx_dev_ACU);
                    if (max_training_supplement > max_training) {
                        LOG("max_training_supplement : "<<max_training_supplement);
                    }
                    const float min_training_supplement   = gpu_min<float>(ratings_rows_ACU * ratings_cols, full_ratingsMtx_dev_ACU);
                    if (min_training_supplement < min_training) {
                        LOG("min_training_supplement : "<<min_training_supplement);
                    }
                }

                if(Debug && 0){
                    save_device_mtx_to_file(full_ratingsMtx_dev_ACU, ratings_cols, ratings_rows_ACU, "full_ratingsMtx_dev_ACU_pre_shuffle", true);
                }

                checkCudaErrors(cudaDeviceSynchronize());
                //shuffle ACU rows
                gpu_shuffle_mtx_rows_or_cols(dn_handle, ratings_rows_ACU, ratings_cols,  
                                             row_major_ordering, full_ratingsMtx_dev_ACU, 1);
            }
            if(Debug && 0){
                save_device_mtx_to_file(full_ratingsMtx_dev_ACU, ratings_cols, ratings_rows_ACU, "full_ratingsMtx_dev_ACU", true);
                // LOG("Press Enter to continue.") ;
                // std::cin.ignore();
            }
        }
        LOG("full_ratingsMtx_dev_ACU filled and shuffled"<<std::endl) ;

        const float min_ACU       = gpu_min<float>               (ratings_rows_ACU * ratings_cols, full_ratingsMtx_dev_ACU);
        const float max_ACU       = gpu_max<float>               (ratings_rows_ACU * ratings_cols, full_ratingsMtx_dev_ACU);
        const float abs_max_ACU   = std::max(max_ACU, std::abs(min_ACU));
        const float expt_ACU      = gpu_expected_value<float>    (ratings_rows_ACU * ratings_cols, full_ratingsMtx_dev_ACU);
        const float expt_abs_ACU  = gpu_expected_abs_value<float>(ratings_rows_ACU * ratings_cols, full_ratingsMtx_dev_ACU);


        LOG("max_ACU = "     <<max_ACU) ;
        LOG("min_ACU = "     <<min_ACU) ;
        LOG("abs_max_ACU = " <<abs_max_ACU) ;
        LOG("expt_ACU = "    <<expt_ACU) ;
        LOG("expt_abs_ACU = "<<expt_abs_ACU) ;

    }
    if(!random_initialization){
        checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_ACU));         update_Mem((ratings_rows_ACU + 1) * SIZE_OF(int)   * (-1));
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_ACU));         update_Mem(num_entries_ACU        * SIZE_OF(int)   * (-1));
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_ACU));         update_Mem(num_entries_ACU        * SIZE_OF(float) * (-1));
    }

    if(Debug && 0){
        save_device_array_to_file<int>(  csr_format_ratingsMtx_userID_dev_testing,  ratings_rows_testing + 1, "csr_format_ratingsMtx_userID_dev_testing" );
        save_device_array_to_file<int>(  coo_format_ratingsMtx_itemID_dev_testing,  num_entries_testing,      "coo_format_ratingsMtx_itemID_dev_testing");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing,      "coo_format_ratingsMtx_rating_dev_testing" );
        // LOG("csr_format_ratingsMtx_userID_dev_training : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_training, ratings_rows_training + 1, 1 );
        // LOG("csr_format_ratingsMtx_userID_dev_ACU : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_ACU, ratings_rows_ACU + 1, 1 );
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }



    float* user_means_testing_host = NULL;
    float* user_var_testing_host   = NULL;
    user_means_testing_host        = (float *)malloc(ratings_rows_testing *  SIZE_OF(float)); 
    user_var_testing_host          = (float *)malloc(ratings_rows_testing *  SIZE_OF(float)); 
    checkErrors(user_means_testing_host);
    checkErrors(user_var_testing_host);
    checkCudaErrors(cudaMemcpy(user_means_testing_host, user_means_testing, ratings_rows_testing *  SIZE_OF(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(user_var_testing_host,   user_var_testing,   ratings_rows_testing *  SIZE_OF(float), cudaMemcpyDeviceToHost));
    
    checkCudaErrors(cudaFree(user_means_testing));                        update_Mem(ratings_rows_testing * SIZE_OF(float) * (-1));
    checkCudaErrors(cudaFree(user_var_testing));                          update_Mem(ratings_rows_testing * SIZE_OF(float) * (-1));

    //the stored means are useless now
    checkCudaErrors(cudaFree(user_means_ACU));                             update_Mem(ratings_rows_ACU * SIZE_OF(float) * (-1));
    checkCudaErrors(cudaFree(user_var_ACU));                               update_Mem(ratings_rows_ACU * SIZE_OF(float) * (-1));
    checkCudaErrors(cudaFree(user_means_training));                       update_Mem(ratings_rows_training * SIZE_OF(float) * (-1));
    checkCudaErrors(cudaFree(user_var_training));                         update_Mem(ratings_rows_training * SIZE_OF(float) * (-1));


    if (user_means_testing_host) free(user_means_testing_host);
    if (user_var_testing_host) free(user_var_testing_host);

    //============================================================================================
    // We want to find orthogonal matrices U, V such that R ~ U*V^T
    // 
    // R is batch_size_ACU by ratings_cols
    // U is batch_size_ACU by num_latent_factors
    // V is ratings_cols by num_latent_factors
    //============================================================================================
    LOG(std::endl);

    bool        print_training_error    = true;


    float meta_training_rate;
    float micro_training_rate; 
    float testing_training_rate;     
    float regularization_constant;         

    const float testing_fraction            = 0.2; //percent of known entries used for testing
    bool        compress_when_training      = true;
    bool        compress_when_testing       = true;
    bool        update_U                    = true;
    bool        regularize_U                = true;
    bool        regularize_V                = true;
    bool        regularize_R                = true;
    bool        regularize_R_distribution   = false;
    bool        normalize_V_rows            = false;
    bool        SV_with_U                   = false;

    switch (case_)
    {
        case 1:{ // code to be executed if n = 1;

            //Dataset_Name = "MovieLens 20 million";
            meta_training_rate          = (float)0.01; 
            micro_training_rate         = (float)0.0005;      //use for movielens
            testing_training_rate       = (float)0.00001;// (float)0.00001; //*Submitted
            regularization_constant     = (float)0.1;         //use for movielens
            update_U                    = true;
            regularize_U                = false;
            regularize_V                = true;
            regularize_R                = false;
            regularize_R_distribution   = false;
            normalize_V_rows            = false;
            compress_when_testing       = true;
            break;
        }case 2:{ // code to be executed if n = 2;
            //Dataset_Name = "Rent The Runaway";
            meta_training_rate          = (float)0.01; 
            micro_training_rate         = (float)0.01;      //use for movielens
            testing_training_rate       = (float)0.00001;      //use for movielens
            regularization_constant     = (float)0.01;         //use for movielens
            update_U                    = true;
            regularize_U                = false;
            regularize_V                = true;
            regularize_R                = false;
            regularize_R_distribution   = false;
            normalize_V_rows            = false;
            compress_when_testing       = true;
            break;
        }default: 
            ABORT_IF_EQ(0, 1, "no valid dataset selected");
    }
    // if(Conserve_GPU_Mem){
    //     compress_when_training = true;
    // }
    if(compress_when_training){
        compress_when_testing = true;
    }

    long long int batch_size_training = 1500;

    int num_iterations = 10000;  // number of training iterations
    if(testing_only){
        num_iterations = 181;
    }
    const int num_batches    = std::max((long long int)1, ratings_rows_training / (batch_size_training));     // number of training batches per iteration (batches index into training data)     // submitted version uses 10
    const int num_blocks     = static_cast<int>(ratings_rows_ACU / (long long int)1000);    // number of blocks of AC users (a single block of AC users is updated in a batch)
    const int testing_rate   = 20;      // 

    LOG("meta_training_rate : "        <<meta_training_rate);
    LOG("micro_training_rate : "       <<micro_training_rate);
    LOG("testing_training_rate : "     <<testing_training_rate);
    LOG("regularization_constant : "   <<regularization_constant);
    LOG("testing_fraction : "          <<testing_fraction);
    LOG("update_U : "                  <<update_U);
    LOG("regularize_U : "              <<regularize_U);
    LOG("regularize_V : "              <<regularize_V);
    LOG("regularize_R : "              <<regularize_R);
    LOG("regularize_R_distribution : " <<regularize_R_distribution);
    LOG("compress_when_training : "    <<compress_when_training);
    LOG("compress_when_testing : "     <<compress_when_testing);
    LOG("num_iterations : "            <<num_iterations);
    LOG("num_batches : "               <<num_batches);
    LOG("num_blocks: "                 <<num_blocks);
    LOG("testing_rate : "              <<testing_rate);
    LOG("SV_with_U : "                 <<SV_with_U);



    batch_size_training = std::max((long long int)1, ratings_rows_training / (num_batches));
    const long long int batch_size_ACU       = std::max((long long int)1, ratings_rows_ACU / (num_blocks));
    const long long int batch_size_testing  = std::min((long long int)200, std::min(ratings_rows_testing, batch_size_ACU));
    LOG(std::endl);
    LOG("batch_size_testing : " <<batch_size_testing);
    LOG("batch_size_training : "<<batch_size_training);
    LOG("batch_size_ACU : "      <<batch_size_ACU);
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();

    ABORT_IF_LE(batch_size_training, (long long int)1, "batch size training is too small");
    ABORT_IF_LE(batch_size_testing, (long long int)1, "batch size testing is too small");
    ABORT_IF_LE(batch_size_ACU, (long long int)1, "batch size ACU is too small");
    // ABORT_IF_LE((long long int)3000, batch_size_ACU, "batch size ACU is too large");
    // ABORT_IF_LE((long long int)3000, batch_size_training, "batch size training is too large");
    // ABORT_IF_LE((long long int)3000, batch_size_testing, "batch size testing is too large");

    ABORT_IF_NEQ(ratings_rows_ACU % batch_size_ACU, (long long int)0, "ratings_rows_ACU % batch_size_ACU != 0"<<std::endl);

    long long int num_batches_training = (long long int)(std::ceil((float)ratings_rows_training  / (float)batch_size_training)); // = num_batches
    //long long int num_batches_ACU       = std::ceil(ratings_rows_ACU       / batch_size_ACU);
    long long int num_batches_testing  = (long long int)(std::ceil((float)ratings_rows_testing  / (float)batch_size_testing));

    LOG("ratings_rows_ACU : " <<ratings_rows_ACU);
    LOG("num_entries_ACU : "<<num_entries_ACU);
    LOG("batch_size_ACU : " <<batch_size_ACU);
    LOG("num_blocks : " <<num_blocks<<std::endl);

    LOG("ratings_rows_training : " <<ratings_rows_training);
    LOG("num_entries_training : "<<num_entries_training);
    LOG("batch_size_training : " <<batch_size_training);
    LOG("num_batches_training : " <<num_batches_training<<std::endl);

    LOG("ratings_rows_testing : " <<ratings_rows_testing);
    LOG("num_entries_testing : "<<num_entries_testing);
    LOG("batch_size_testing : " <<batch_size_testing);
    LOG("num_batches_testing : " <<num_batches_testing<<std::endl);


    LOG("ratings_cols : " <<ratings_cols<<std::endl);
    
    const float percent              = (float)0.05;  // How much of the full SV spectrum do you want to keep as latent factors?
    long long int num_latent_factors = ratings_rows_ACU;
    long long int max_num_latent_factors = (long long int)((float)ratings_rows_ACU * percent);


    /*
        There are a couple different factorization possibilities:
            - You could factor the whole R_ACU (When R_ACU is large this is done by factoring blocks and agregating the factored results)
            - You could factor only a horizontal strip of R_ACU containing a subset of the ACU users 
    */

    long long int min_dim_block = std::min(batch_size_ACU, ratings_cols); // minimum dimension when factoring only a block of R_ACU
    long long int min_dim_ = std::min(ratings_rows_ACU, ratings_cols);    // minimum dimension when factoring the entire R_ACU mtx


    float * old_R_ACU = NULL;         // R_ACU is ratings_rows_ACU * ratings_cols
    float * U_ACU_host = NULL;        // U_ACU is ratings_rows_ACU * ratings_rows_ACU
    float * U_ACU_dev = NULL;         // U_ACU is ratings_rows_ACU * ratings_rows_ACU
    float * V_host = NULL;           // V_ACU is ratings_cols * ratings_cols
    float * V_dev = NULL;            // V_ACU is ratings_cols * ratings_cols


    bool temp = Conserve_GPU_Mem;
    const long long int Training_bytes = (ratings_rows_ACU     * min_dim_ +
                                          batch_size_training * std::max(min_dim_block, batch_size_training) + 
                                          batch_size_testing  * std::max(min_dim_, batch_size_testing) + 
                                          ratings_cols        * min_dim_ +
                                          ratings_cols        * batch_size_training +
                                          ratings_cols        * batch_size_testing)* (long long int)SIZE_OF(float) ;
    if(allocatedMem + Training_bytes > (long long int)((double)devMem * (double)0.75)) 
    {
        Conserve_GPU_Mem = 1;
    };


    if(!temp && Conserve_GPU_Mem){
        LOG("Conserving Memory Now");
        //put the ACU ratings mtx on the CPU;
        full_ratingsMtx_host_ACU = (float *)malloc(ACU_mtx_size_bytes);
        checkErrors(full_ratingsMtx_host_ACU);
        
        checkCudaErrors(cudaMemcpy(full_ratingsMtx_host_ACU, full_ratingsMtx_dev_ACU, ACU_mtx_size_bytes, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(full_ratingsMtx_dev_ACU));
        update_Mem((float)(-1.0) * ACU_mtx_size_bytes );
    };

    int *   csr_format_ratingsMtx_userID_host_testing = NULL;
    int *   coo_format_ratingsMtx_itemID_host_testing = NULL;
    float * coo_format_ratingsMtx_rating_host_testing = NULL;
    int *   csr_format_ratingsMtx_userID_host_training = NULL;
    int *   coo_format_ratingsMtx_itemID_host_training = NULL;
    float * coo_format_ratingsMtx_rating_host_training = NULL;

    float* SV = NULL;

    if(Conserve_GPU_Mem){
        //============================================================================================
        // Conserve Memory
        //============================================================================================
        csr_format_ratingsMtx_userID_host_testing  = (int *)  malloc((ratings_rows_testing + 1) *  SIZE_OF(int)); 
        coo_format_ratingsMtx_itemID_host_testing  = (int *)  malloc(num_entries_testing  *  SIZE_OF(int)); 
        coo_format_ratingsMtx_rating_host_testing  = (float *)malloc(num_entries_testing  *  SIZE_OF(float));
        csr_format_ratingsMtx_userID_host_training = (int *)  malloc((ratings_rows_training + 1) *  SIZE_OF(int)); 
        coo_format_ratingsMtx_itemID_host_training = (int *)  malloc(num_entries_training  *  SIZE_OF(int)); 
        coo_format_ratingsMtx_rating_host_training = (float *)malloc(num_entries_training  *  SIZE_OF(float));

        old_R_ACU  = (float *)malloc(ratings_rows_ACU * ratings_cols  *  SIZE_OF(float));

        checkErrors(csr_format_ratingsMtx_userID_host_testing);
        checkErrors(coo_format_ratingsMtx_itemID_host_testing);
        checkErrors(coo_format_ratingsMtx_rating_host_testing); 
        checkErrors(csr_format_ratingsMtx_userID_host_training);
        checkErrors(coo_format_ratingsMtx_itemID_host_training);
        checkErrors(coo_format_ratingsMtx_rating_host_training); 
        checkErrors(old_R_ACU); 

        checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host_testing,  csr_format_ratingsMtx_userID_dev_testing,  (ratings_rows_testing + 1) *  SIZE_OF(int),   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host_testing, coo_format_ratingsMtx_itemID_dev_testing, num_entries_testing  *  SIZE_OF(int),   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host_testing,  coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing  *  SIZE_OF(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host_training,  csr_format_ratingsMtx_userID_dev_training,  (ratings_rows_training + 1) *  SIZE_OF(int),   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host_training, coo_format_ratingsMtx_itemID_dev_training, num_entries_training  *  SIZE_OF(int),   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host_training,  coo_format_ratingsMtx_rating_dev_training,  num_entries_training  *  SIZE_OF(float), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_testing));  update_Mem((ratings_rows_testing + 1)*  SIZE_OF(int) * (-1));
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_testing));  update_Mem(num_entries_testing  *  SIZE_OF(int) * (-1));
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_testing));  update_Mem(num_entries_testing  *  SIZE_OF(float) * (-1) );
        checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_training));  update_Mem((ratings_rows_training + 1)*  SIZE_OF(int) * (-1));
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_training));  update_Mem(num_entries_training  *  SIZE_OF(int) * (-1));
        checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_training));  update_Mem(num_entries_training  *  SIZE_OF(float) * (-1) );
        
        if(Debug && 0){
            save_host_array_to_file<int>  (csr_format_ratingsMtx_userID_host_testing,  ratings_rows_testing + 1, "csr_format_ratingsMtx_userID_host_testing_1");
            save_host_array_to_file<int>  (coo_format_ratingsMtx_itemID_host_testing,  num_entries_testing, "coo_format_ratingsMtx_itemID_host_testing");
            save_host_array_to_file<float>  (coo_format_ratingsMtx_rating_host_testing,  num_entries_testing, "coo_format_ratingsMtx_rating_host_testing");
        }

        U_ACU_host = (float *)malloc(ratings_rows_ACU * ratings_rows_ACU * SIZE_OF(float));
        V_host = (float *)malloc(ratings_cols * ratings_rows_ACU * SIZE_OF(float));
        checkErrors(U_ACU_host);
        checkErrors(V_host);

        
        num_latent_factors = std::min(ratings_rows_ACU, max_num_latent_factors);
        if(Debug && 0) {
            checkCudaErrors(cudaDeviceSynchronize()); 
            LOG("num_latent_factors = "<< num_latent_factors);
            LOG("min_dim_ = "<< min_dim_);
        }

        CUDA_CHECK(cudaMalloc((void**)&V_dev, ratings_cols * std::max(min_dim_block, (compress_when_testing ? num_latent_factors : ratings_rows_ACU)) * SIZE_OF(float)));
        update_Mem(ratings_cols * (compress_when_testing ? num_latent_factors : ratings_rows_ACU) * SIZE_OF(float) );
        LOG("ratings_cols * (compress_when_testing ? num_latent_factors : ratings_rows_ACU) : " << ratings_cols * (compress_when_testing ? num_latent_factors : ratings_rows_ACU));
        
        if(!row_major_ordering){
            ABORT_IF_EQ(0,1,"try again with row_major_ordering = true.")
        }

        // checkCudaErrors(cudaMalloc((void**)&full_ratingsMtx_dev_ACU, batch_size_ACU * ratings_cols * SIZE_OF(float)));
        // update_Mem(batch_size_ACU * ratings_cols* SIZE_OF(float) );

        SV = (float *)malloc(ratings_rows_ACU *  SIZE_OF(float)); 
        checkErrors(SV);
    }else{
        if(num_blocks > 1){
            ABORT_IF_EQ(0, 1, "Does not train blocks when Conserve_GPU_Mem is false.")
        }
        checkCudaErrors(cudaMalloc((void**)&U_ACU_dev,       ratings_rows_ACU     * min_dim_                     * SIZE_OF(float)));
        checkCudaErrors(cudaMalloc((void**)&V_dev,          ratings_cols        * min_dim_                     * SIZE_OF(float)));
        update_Mem((ratings_rows_ACU * min_dim_ + ratings_cols * min_dim_) * SIZE_OF(float) );
        checkCudaErrors(cudaMalloc((void**)&old_R_ACU, ratings_rows_ACU * ratings_cols * SIZE_OF(float)));
        update_Mem(ratings_rows_ACU * ratings_cols * SIZE_OF(float));
    }



    // LOG(ratings_cols * ratings_cols * SIZE_OF(float)) ;

    // checkCudaErrors(cudaDeviceSynchronize());
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();  




    if(Conserve_GPU_Mem){
        if(row_major_ordering && !load_full_ACU_from_save){
            //remember that R_ACU is stored in row major ordering
            LOG("swap matrix indexing from row major to column major");
            for(int block = 0; block < num_blocks; block++){
                long long int first_row_in_batch_ACU = (batch_size_ACU * (long long int)block) ;
                cpu_swap_ordering<float>(batch_size_ACU, ratings_cols, full_ratingsMtx_host_ACU + ratings_cols * first_row_in_batch_ACU, row_major_ordering);
            }
        }
        row_major_ordering = false;
    }
    




    
    long long int total_testing_iterations = (long long int) 10000; //deprecated
    


    int count_batches_so_far = 0;
    



    if(Debug) LOG(memLeft<<" available bytes left on the device");









    float epsilon = (float)0.001;

    const float min_training_rate = (float)0.000001;

    float min_error_so_far = (float)100000.0;







    float * testing_error_on_training_entries = NULL;
    float * testing_error_on_testing_entries = NULL;
    float * testing_iterations = NULL;
    float * meta_km_errors = NULL;
    int * num_latent_factors_ = NULL;

    float * training_error_v = NULL;
    float * training_error_u = NULL;
    float * training_iterations = NULL;

    float * delta_R_ACU_exp = NULL;
    float * R_ACU_abs_max = NULL;
    float * R_ACU_max_sv = NULL;
    float * R_ACU_exp = NULL;
    float * R_ACU_var = NULL;
    
    testing_error_on_training_entries = (float *)malloc((num_iterations / testing_rate) * SIZE_OF(float)); 
    checkErrors(testing_error_on_training_entries);
    cpu_set_all<float>(testing_error_on_training_entries, (num_iterations / testing_rate), (float)0.0);
    testing_error_on_testing_entries = (float *)malloc((num_iterations / testing_rate) * SIZE_OF(float)); 
    checkErrors(testing_error_on_testing_entries);
    cpu_set_all<float>(testing_error_on_testing_entries, (num_iterations / testing_rate), (float)0.0);
    testing_iterations = (float *)malloc((num_iterations / testing_rate) * SIZE_OF(float)); 
    checkErrors(testing_iterations);
    cpu_set_all<float>(testing_iterations, (num_iterations / testing_rate), (float)0.0);
    meta_km_errors = (float *)malloc((num_iterations / testing_rate) * SIZE_OF(float)); 
    checkErrors(meta_km_errors);
    cpu_set_all<float>(meta_km_errors, (num_iterations / testing_rate), (float)0.0);
    num_latent_factors_ = (int *)malloc((num_iterations / testing_rate) * SIZE_OF(int)); 
    checkErrors(num_latent_factors_);
    cpu_set_all<int>(num_latent_factors_, (num_iterations / testing_rate), 0);

    training_error_v = (float *)malloc(num_iterations * SIZE_OF(float)); 
    checkErrors(training_error_v);
    cpu_set_all<float>(training_error_v, num_iterations, (float)0.0);
    training_error_u = (float *)malloc(num_iterations * SIZE_OF(float)); 
    checkErrors(training_error_u);
    cpu_set_all<float>(training_error_u, num_iterations, (float)0.0);
    training_iterations = (float *)malloc(num_iterations * SIZE_OF(float)); 
    checkErrors(training_iterations);
    cpu_set_all<float>(training_iterations, num_iterations, (float)0.0);

    delta_R_ACU_exp = (float *)malloc(num_iterations * SIZE_OF(float)); 
    checkErrors(delta_R_ACU_exp);
    cpu_set_all<float>(delta_R_ACU_exp, num_iterations, (float)0.0);
    
    R_ACU_abs_max = (float *)malloc((num_iterations / testing_rate)  * SIZE_OF(float)); 
    checkErrors(R_ACU_abs_max);
    cpu_set_all<float>(R_ACU_abs_max, (num_iterations / testing_rate) , (float)0.0);
    R_ACU_max_sv = (float *)malloc((num_iterations / testing_rate)  * SIZE_OF(float)); 
    checkErrors(R_ACU_max_sv);
    cpu_set_all<float>(R_ACU_max_sv, (num_iterations / testing_rate) , (float)0.0);
    R_ACU_exp = (float *)malloc((num_iterations / testing_rate)  * SIZE_OF(float)); 
    checkErrors(R_ACU_exp);
    cpu_set_all<float>(R_ACU_exp, (num_iterations / testing_rate) , (float)0.0);
    R_ACU_var = (float *)malloc((num_iterations / testing_rate)  * SIZE_OF(float)); 
    checkErrors(R_ACU_var);
    cpu_set_all<float>(R_ACU_var, (num_iterations / testing_rate) , (float)0.0);


    float* logarithmic_histogram = (float *)malloc(7 * (num_iterations / testing_rate)  * SIZE_OF(float)); 
    checkErrors(logarithmic_histogram);
    cpu_set_all<float>(logarithmic_histogram, 7 * (num_iterations / testing_rate) , (float)0.0);

    float* logarithmic_histogram_km = (float *)malloc(7 * (num_iterations / testing_rate)  * SIZE_OF(float)); 
    checkErrors(logarithmic_histogram_km);
    cpu_set_all<float>(logarithmic_histogram_km, 7 * (num_iterations / testing_rate) , (float)0.0);



    //============================================================================================
    // Assume the rows of R_ACU are the ratings_rows_ACU-means for the training users - what is the error?
    //============================================================================================ 

    if(0){
        float k_means_er = (float)0.0;
        float* errors;
        int* selection;
        if(Conserve_GPU_Mem){
            errors  = (float *)malloc(ratings_rows_training * SIZE_OF(float));
            selection  = (int *)malloc(ratings_rows_training * SIZE_OF(int));
            checkErrors(errors);
            checkErrors(selection);
            cpu_sparse_nearest_row<float>(ratings_rows_ACU, ratings_cols, full_ratingsMtx_host_ACU, 
                                         ratings_rows_training, num_entries_training, 
                                         csr_format_ratingsMtx_userID_host_training, 
                                         coo_format_ratingsMtx_itemID_host_training,
                                         coo_format_ratingsMtx_rating_host_training, 
                                         selection, errors);
            //k_means_er = cpu_sum(ratings_rows_training,  errors);
            //LOG("err norm when clustering : "<<std::sqrt(k_means_er));
            LOG("expected MSQER when clustering : "<<cpu_expected_value(ratings_rows_training, errors));
            save_host_array_to_file<float>(errors, ratings_rows_training, "meta_km_errors");
            save_host_array_to_file<int>(selection, ratings_rows_training, "meta_km_selection");
            free(errors);
            free(selection);
        }else{
            checkCudaErrors(cudaMalloc((void**)&errors, ratings_rows_training * SIZE_OF(float)));
            checkCudaErrors(cudaMalloc((void**)&selection, ratings_rows_training * SIZE_OF(int)));
            update_Mem(2 * ratings_rows_training * SIZE_OF(int));
            gpu_sparse_nearest_row<float>(ratings_rows_ACU, ratings_cols, full_ratingsMtx_dev_ACU, 
                                     ratings_rows_training, num_entries_training, 
                                     csr_format_ratingsMtx_userID_dev_training, 
                                     coo_format_ratingsMtx_itemID_dev_training,
                                     coo_format_ratingsMtx_rating_dev_training, 
                                     selection, errors, row_major_ordering);
            k_means_er = gpu_sum(ratings_rows_training,  errors);
            LOG("err norm when clustering : "<<std::sqrt(k_means_er));
            LOG("mean sqed err when clustering : "<<k_means_er / (float)num_entries_training);
            save_device_array_to_file<float>(errors, ratings_rows_training, "meta_km_errors");
            save_device_array_to_file<int>(selection, ratings_rows_training, "meta_km_selection");
            checkCudaErrors(cudaFree(errors));
            checkCudaErrors(cudaFree(selection));
            update_Mem(2 * ratings_rows_training * SIZE_OF(int) * (-1));
        }
    }



    //============================================================================================
    // Begin Training
    //============================================================================================  
    LOG(std::endl<<std::endl<<"                              Begin Training..."<<std::endl); 
    int num_tests = 0;
    
        bool not_done = true;

    int first_it = 0;
    int first_training_batch__ = 0;

    if(load_full_ACU_from_save && !testing_only){
        first_it = 60;
        first_training_batch__ = 0;
    }
    int it = first_it;

    double avg_testing_time = (double)0.0;
    double avg_training_time = (double)0.0;



    if(Conserve_GPU_Mem && load_full_ACU_from_save && !testing_only){
        LOG("Loading full_ratingsMtx_host_ACU from saved batches.");
        for(int block = 0; block < num_blocks; block++){
            std::string full_ratingsMtx_ACU_filepath = save_path + "full_ratingsMtx_ACU_block_" + ToString<int>(block)+ "_it_"+ToString<int>(it - 1)+".txt";

            long long int first_row_in_batch_ACU = (batch_size_ACU * (long long int)block) ;


            CSVReader ACU_mtx_reader(full_ratingsMtx_ACU_filepath);
            ACU_mtx_reader.getData(full_ratingsMtx_host_ACU + ratings_cols * first_row_in_batch_ACU, batch_size_ACU, ratings_cols, row_major_ordering);
            //ACU_mtx_reader.getData(full_ratingsMtx_host_ACU + ratings_cols * first_row_in_batch_ACU, (long long int)1, batch_size_ACU * ratings_cols, row_major_ordering);
            if(Debug){
                print_host_array(full_ratingsMtx_host_ACU + ratings_cols * first_row_in_batch_ACU, 5, full_ratingsMtx_ACU_filepath, 
                    strPreamble(blank));
            }
            
        }
        LOG("Done loading full_ratingsMtx_host_ACU from saved batches.");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("program_time: %f\n", program_time);   
    LOG("RUN TIME SO FAR: "<<readable_time(program_time));

    while(not_done && it < num_iterations && meta_training_rate >= min_training_rate){
        //for(int it = 0; it < num_iterations; it ++){
        LOG("it : "<<it);

        bool mid_it = false;
        if(load_full_ACU_from_save && it == first_it /*&& first_training_batch__ > 0*/){
            mid_it = true;
        }
        
        //============================================================================================
        // TESTING
        //============================================================================================ 
        bool do_ = true;
        
        if( it % testing_rate == 0 && !mid_it /*&& it != 0*/){
            gettimeofday(&testing_start, NULL);
            LOG(std::endl<<"                                                                                        ~~~ TESTING ~~~ "); 
            if(Debug){
                LOG(memLeft<<" available bytes left on the device");
                LOG("batch_size_testing : "<<batch_size_testing);
            }
            
            float * SV_dev;
            checkCudaErrors(cudaMalloc((void**)&SV_dev, ratings_rows_ACU * SIZE_OF(float)));
            update_Mem(ratings_rows_ACU * SIZE_OF(float));
            if(load_full_ACU_from_save && testing_only){
                if(Conserve_GPU_Mem){
                    LOG("Loading full_ratingsMtx_host_ACU from saved batches.");
                    for(int block = 0; block < num_blocks; block++){
                        std::string full_ratingsMtx_ACU_filepath = save_path + "6_13/full_ratingsMtx_GU_block_" + ToString<int>(block)+ "_it_"+ToString<int>(it - 1)+".txt";

                        long long int first_row_in_batch_ACU = (batch_size_ACU * (long long int)block) ;


                        CSVReader ACU_mtx_reader(full_ratingsMtx_ACU_filepath);
                        ACU_mtx_reader.getData(full_ratingsMtx_host_ACU + ratings_cols * first_row_in_batch_ACU, batch_size_ACU, ratings_cols, row_major_ordering);
                        //ACU_mtx_reader.getData(full_ratingsMtx_host_ACU + ratings_cols * first_row_in_batch_ACU, (long long int)1, batch_size_ACU * ratings_cols, row_major_ordering);
                        if(Debug){
                            print_host_array(full_ratingsMtx_host_ACU + ratings_cols * first_row_in_batch_ACU, 5, full_ratingsMtx_ACU_filepath, 
                                strPreamble(blank));
                        }
                        
                    }
                    LOG("Done loading full_ratingsMtx_host_ACU from saved batches.");
                }
            }
            if(Conserve_GPU_Mem){
                /*
                    cpu_orthogonal_decomp<float>(ratings_rows_ACU, ratings_cols, row_major_ordering,
                                            &num_latent_factors, percent,
                                            full_ratingsMtx_host_ACU, U_ACU, V_host, SV_with_U, SV);
                */
                LOG("num_blocks = "<< num_blocks);
                LOG("batch_size_ACU = "<< batch_size_ACU);
                
                gpu_block_orthogonal_decomp_from_host<float>(dn_handle, dn_solver_handle,
                                                             ratings_rows_ACU, ratings_cols,
                                                             &num_latent_factors, (float)1.0,
                                                             full_ratingsMtx_host_ACU, U_ACU_host, 
                                                             V_host, row_major_ordering,
                                                             batch_size_ACU, SV_with_U, SV);
                R_ACU_max_sv[num_tests] = SV[0];                                             
                                                             
                LOG("num_latent_factors = "<< num_latent_factors << " ( / "<< ratings_rows_ACU<< " )");
                num_latent_factors_[num_tests] = static_cast<int>(std::min(num_latent_factors, std::min(ratings_rows_ACU , max_num_latent_factors)));

                num_latent_factors = max_num_latent_factors;//std::min(num_latent_factors, std::min(ratings_rows_ACU , max_num_latent_factors));

                LOG("num_latent_factors = "<< num_latent_factors << " ( / "<< ratings_rows_ACU<< " )");

                LOG("MEMORY REDUCTION OF ACU USERS : "      <<static_cast<double>((ratings_rows_ACU + ratings_cols) * (compress_when_testing ? num_latent_factors : ratings_rows_ACU)) / static_cast<double>(ratings_rows + (long long int)2 * num_entries)<<std::endl<<std::endl<<std::endl);
                LOG("MEMORY REDUCTION OF ACU Item Vectors : "      <<static_cast<double>(ratings_cols * (compress_when_testing ? num_latent_factors : ratings_rows_ACU)) / static_cast<double>(ratings_rows + (long long int)2 * num_entries)<<std::endl<<std::endl<<std::endl);

                checkCudaErrors(cudaMemcpy(V_dev, V_host, ratings_cols * (compress_when_testing ? num_latent_factors : ratings_rows_ACU) * SIZE_OF(float), cudaMemcpyHostToDevice));
                //if(Debug) {checkCudaErrors(cudaDeviceSynchronize()); LOG("Here");}
                
                if(Debug && 0){
                    float mean_abs_nonzero_ = (float)0.0;
                    cpu_mean_abs_nonzero(ratings_cols * (compress_when_testing ? num_latent_factors : ratings_rows_ACU), V_host, &mean_abs_nonzero_, true, "V_host");
                    gpu_mean_abs_nonzero(ratings_cols * (compress_when_testing ? num_latent_factors : ratings_rows_ACU), V_dev, &mean_abs_nonzero_, true, "V_dev");
                }
                
                

                if(row_major_ordering){
                    //gpu_swap_ordering<float>(ratings_cols, num_latent_factors, V_dev, true);
                }
                if(Debug) {
                    //save_host_array_to_file<float>(SV, ratings_rows_ACU, "singular_values", strPreamble(blank));
                    //save_host_mtx_to_file<float>(U_ACU_host, ratings_rows_ACU, num_latent_factors, "U_ACU_compressed");
                    
                    //save_host_array_side_by_side_with_device_array<float>(V_host, V_dev, static_cast<int>(ratings_cols * (compress_when_testing ? num_latent_factors : ratings_rows_ACU)), "V_host_dev", strPreamble(blank));
                    //save_host_mtx_to_file<float>(V_host, ratings_cols, (compress_when_testing ? num_latent_factors : ratings_rows_ACU), "V_host", false, strPreamble(blank));
                    //save_device_mtx_to_file<float>(V_dev, ratings_cols, (compress_when_testing ? num_latent_factors : ratings_rows_ACU), "V_dev", false, strPreamble(blank));
                }
                save_host_array_to_file<float>(SV, ratings_rows_ACU, save_path + "6_13/singular_values_it_"+ ToString<int>(it), strPreamble(blank));
                checkCudaErrors(cudaMemcpy(SV_dev, SV, ratings_rows_ACU * SIZE_OF(float), cudaMemcpyHostToDevice));
                /*
                    At this point U_ACU is ratings_rows_ACU by ratings_rows_ACU in memory stored in row major
                    ordering and V is ratings_cols by ratings_rows_ACU stored in column major ordering

                    There is no extra work to compress to compress V into ratings_cols by num_latent_factors, 
                    just take the first num_latent_factors columns of each matrix.  
                    The columns of U_ACU are mixed in memory.
                */

                if(1){
                    R_ACU_exp[num_tests] = cpu_expected_value(ratings_rows_ACU * ratings_cols,  full_ratingsMtx_host_ACU);
                    R_ACU_var[num_tests] = cpu_variance(ratings_rows_ACU * ratings_cols,  full_ratingsMtx_host_ACU);
                }
                
            }else{
                if(row_major_ordering){
                    //remember that R_ACU is stored in row major ordering
                    LOG("swap matrix indexing from row major to column major");
                    gpu_swap_ordering<float>(ratings_rows_ACU, ratings_cols, full_ratingsMtx_dev_ACU, row_major_ordering);
                    
                }
                if(Debug){
                    //save_device_mtx_to_file(full_ratingsMtx_dev_ACU, ratings_rows_ACU, ratings_cols, "full_ratingsMtx_dev_ACU", false);

                    float R_ACU_abs_exp = gpu_expected_abs_value<float>(ratings_rows_ACU * ratings_cols, full_ratingsMtx_dev_ACU);
                    R_ACU_abs_max[it] = gpu_abs_max<float>           (ratings_rows_ACU * ratings_cols, full_ratingsMtx_dev_ACU); 
                    LOG("full_ratingsMtx_dev_ACU_abs_max = "<<R_ACU_abs_max[it]) ;
                    LOG("full_ratingsMtx_dev_ACU_abs_exp = "<<R_ACU_abs_exp) ;
                    ABORT_IF_EQ(R_ACU_abs_exp, R_ACU_abs_max[it], "R_ACU is constant");
                    ABORT_IF_LESS((float)10.0 * abs_max_training, std::abs(R_ACU_abs_max[it]), "unstable growth");
                    ABORT_IF_LESS( std::abs(R_ACU_abs_max[it]), abs_max_training / (float)10.0 , "unstable shrinking");
                    // LOG("Press Enter to continue.") ;
                    // std::cin.ignore();
                }

                gpu_orthogonal_decomp<float>(dn_handle, dn_solver_handle,
                                            ratings_rows_ACU, ratings_cols, 
                                            &num_latent_factors, percent,
                                            full_ratingsMtx_dev_ACU, U_ACU_dev, 
                                            V_dev, SV_with_U, SV_dev);

                //save_device_mtx_to_file<float>(U_ACU_dev, ratings_rows_ACU, num_latent_factors, "U_ACU_compressed");
                /*
                    At this point U_ACU is ratings_rows_ACU by ratings_rows_ACU in memory stored in column major
                    ordering and V is ratings_cols by ratings_rows_ACU stored in column major ordering

                    There is no extra work to compress U_ACU into ratings_rows_ACU by num_latent_factors, or
                    to compress V into ratings_cols by num_latent_factors, just take the first num_latent_factors
                    columns of each matrix
                */
            }  
    

            // float y;
            // LOG("V_dev : ");
            // gpu_msq_nonzero(ratings_cols * (compress_when_testing ? num_latent_factors : ratings_rows_ACU), V_dev, &y, true);
            // LOG("SV_dev : ");
            // gpu_msq_nonzero((compress_when_testing ? num_latent_factors : ratings_rows_ACU), SV_dev, &y, true);

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
            checkCudaErrors(cudaMalloc((void**)&U_testing,  batch_size_testing  * (compress_when_testing ? num_latent_factors : ratings_rows_ACU)  * SIZE_OF(float)));
            checkCudaErrors(cudaMalloc((void**)&R_testing,  batch_size_testing  * ratings_cols                        * SIZE_OF(float)));
            update_Mem(batch_size_testing  * (compress_when_testing ? num_latent_factors : ratings_rows_ACU)  * SIZE_OF(float));
            update_Mem(batch_size_testing  * ratings_cols                        * SIZE_OF(float));

            int val_ = std::min(20, (int)num_batches_testing);
            val_ = 0;
            for(int batch__ = 0; batch__ < val_; batch__++){
                //for(int batch = 0; batch < num_batches_testing; batch++){

                LOG(std::endl<<"                                                                              ~~~ TESTING Batch "<<batch__<<" ( / "<<val_<<" ) ~~~ "); 
                int batch = batch__;
                getRandIntsBetween(&batch , 0 , (int)num_batches_testing, 1);
                LOG("batch id "<<batch << " out of "<<num_batches_testing);


                long long int batch_size_testing_temp = batch_size_testing;
                long long int first_row_index_in_batch_testing  = (batch_size_testing * (long long int)batch) /* % ratings_rows_testing*/;
                if(first_row_index_in_batch_testing + batch_size_testing - (long long int)1 >= ratings_rows_testing) {
                    //batch_size_testing_temp = ratings_rows_testing - first_row_index_in_batch_testing;
                    first_row_index_in_batch_testing = ratings_rows_testing - batch_size_testing_temp;
                    ABORT_IF_LE(first_row_index_in_batch_testing, (long long int)0, "first_row_index_in_batch_testing too small");
                    if(Debug && 0){
                        LOG("left over batch_size_testing : "<<batch_size_testing_temp);
                    }
                } 
                LOG("first_row_index_in_batch_testing : "<<first_row_index_in_batch_testing<< " ( / "<<ratings_rows_testing<<" )"); 
                if(Debug){
                    //LOG(memLeft<<" available bytes left on the device");
                    LOG("batch_size_testing : "<<batch_size_testing_temp);
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
                //                    batch_size_testing_temp, ratings_rows_ACU, num_latent_factors, ratings_cols,
                //                    nnz_testing, first_coo_ind_testing, compress, 
                //                    testing_entries, coo_testing_errors, testing_fraction,
                //                    coo_format_ratingsMtx_rating_dev_testing + first_coo_ind_testing, 
                //                    csr_format_ratingsMtx_userID_dev_testing_batch, 
                //                    coo_format_ratingsMtx_itemID_dev_testing + first_coo_ind_testing,
                //                    V, U_testing, R_testing, "testing", (float)0.1, (float)0.01);


                gpu_R_error<float>(dn_handle, sp_handle, sp_descr, dn_solver_handle, 
                                   batch_size_testing_temp, ratings_rows_ACU, num_latent_factors, ratings_cols,
                                   nnz_testing, first_coo_ind_testing, compress_when_testing, 
                                   testing_entries, coo_testing_errors, testing_fraction,
                                   coo_format_ratingsMtx_rating_dev_testing_batch, 
                                   csr_format_ratingsMtx_userID_dev_testing_batch, 
                                   coo_format_ratingsMtx_itemID_dev_testing_batch,
                                   V_dev, U_testing, R_testing, NULL, NULL,
                                   testing_training_rate, regularization_constant, batch__, it,
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
                    LOG("Iteration "<<it<<" testing error per testing entry: "<< testing_error_on_testing_entries[num_tests]);
                    LOG("Iteration "<<it<<" testing error per training entry: "<< testing_error_on_training_entries[num_tests]<<std::endl);

                    LOG("Iteration "<<it<<" expected number of training iterations when testing: "<< testing_iterations[num_tests]);
                    print_host_array(logarithmic_histogram + 7 * num_tests, 7, "Logarithmic Histogram of Errors From 10^(-3) to 10^3", strPreamble(blank));
                    // LOG("Press Enter to continue.") ;
                    // std::cin.ignore();
                }
            }//for loop on test batches

            checkCudaErrors(cudaFree(U_testing));
            checkCudaErrors(cudaFree(R_testing));
            update_Mem(batch_size_testing  * (compress_when_testing ? num_latent_factors : ratings_rows_ACU)  * SIZE_OF(float) * (-1));
            update_Mem(batch_size_testing  * ratings_cols                                                    * SIZE_OF(float) * (-1));


            //testing_error[num_tests] /= ((float)num_entries_testing * testing_fraction);
            //LOG("Iteration "<<it<<" MSQ testing error per testing entry: "<< testing_error[num_tests]);

            if(Conserve_GPU_Mem){
                checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_testing_));
                update_Mem((batch_size_testing + 1) * SIZE_OF(int) * (-1));
            }else{
                if(row_major_ordering){
                    //remember that R_ACU is stored in row major ordering
                    LOG("swap matrix indexing from row major to column major");
                    gpu_swap_ordering<float>(ratings_rows_ACU, ratings_cols, full_ratingsMtx_dev_ACU, !row_major_ordering);
                    
                }
            }
            checkCudaErrors(cudaFree(SV_dev));
            update_Mem(ratings_rows_ACU * SIZE_OF(float) * (-1));
            //LOG("HERE!"); checkCudaErrors(cudaDeviceSynchronize()); LOG("HERE!");

        

            //============================================================================================
            // Assume the rows of R_ACU are the ratings_rows_ACU-means for the training users - what is the error?
            //============================================================================================ 

            if(testing_only && Conserve_GPU_Mem){
                // gpu_gemm<float>(dn_handle, true, false, 
                // ratings_cols, batch_size_ACU, 
                // compress ? num_latent_factors : batch_size_ACU,
                // (float)1.0,
                // V, U_ACU, 
                // (float)0.0,
                // R_ACU);

                  /*
                    M, N, K
                    M number of rows of matrix op(A) and C.
                    N is number of columns of matrix op(B) and C.]
                    K is number of rows of op(B) and columns of op(A).

                    op(A) is M by K
                    op(B) is K by N
                    C is M by N
                    cblas_sgemm(CblasRowMajor, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) 
                    performs C=alpha op ( A )op ( B ) + beta C
                  */

                //V is in column major ordering
                //U_ACU_host
                cpu_gemm<float>(0, 0, 
                         ratings_rows_ACU, ratings_cols, num_latent_factors,
                         (float)1.0, U_ACU_host, V_host, (float)0.0,
                         full_ratingsMtx_host_ACU);
            }

            if(1){
                float* errors;
                int* selection;
                if(Conserve_GPU_Mem){
                    errors  = (float *)malloc(ratings_rows_testing * SIZE_OF(float));
                    selection  = (int *)malloc(ratings_rows_testing * SIZE_OF(int));
                    checkErrors(errors);
                    checkErrors(selection);
                    cpu_sparse_nearest_row<float>(ratings_rows_ACU, ratings_cols, full_ratingsMtx_host_ACU, 
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
                    cpu_logarithmic_histogram_abs_val<float>(ratings_rows_ACU, ratings_cols, full_ratingsMtx_host_ACU, 
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
                    gpu_sparse_nearest_row<float>(ratings_rows_ACU, ratings_cols, full_ratingsMtx_dev_ACU, 
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
            gettimeofday(&program_end, NULL);
            program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
            //printf("program_time: %f\n", program_time);   
            LOG("program_time so far: "<<readable_time(program_time));
            LOG("      ~~~ DONE TESTING ~~~ "<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl); 
            if(it > 0 && 0){
              if(testing_error_on_testing_entries[num_tests] < epsilon || ::isinf(testing_error_on_testing_entries[num_tests]) || ::isnan(testing_error_on_testing_entries[num_tests])){
                LOG("Finished at iteration : "<<it);
                do_ = false;
                not_done = false;
                break;
              }
              if(testing_error_on_testing_entries[num_tests] > min_error_so_far * (float)1.3){
                // we jumped over the minimum
                // meta_training_rate = meta_training_rate / (float)10.0;
                // micro_training_rate = micro_training_rate / (float)10.0;
                if(1) {
                  LOG("Jumped over minimum meta iteration : "<<it);
                  LOG("min_error_so_far : "<<min_error_so_far);
                  LOG("new error : "<<testing_error_on_testing_entries[num_tests]);
                  LOG("meta_training_rate : "<<meta_training_rate);
                }
                if(Debug && 0){
                    if(Conserve_GPU_Mem){
                        host_copy(ratings_rows_ACU * ratings_cols, old_R_ACU, full_ratingsMtx_host_ACU);
                    }else{
                        checkCudaErrors(cudaMemcpy(full_ratingsMtx_dev_ACU, old_R_ACU,
                                                ratings_rows_ACU * ratings_cols * SIZE_OF(float), cudaMemcpyDeviceToDevice));
                    }
                    //do_ = false;
                }

              }else{
                  //have we stopped improving?
                  // if ((testing_error_on_testing_entries[num_tests - 1] - testing_error_on_testing_entries[num_tests]) < -(float)2.0 * min_error_so_far){
                  //   meta_training_rate = meta_training_rate / (float)10.0;
                  //   micro_training_rate = micro_training_rate / (float)10.0;
                  //   LOG("Reducing meta_training_rate : "<<it);
                  //   LOG("previous error : "<<testing_error_on_testing_entries[num_tests - 1]);
                  //   LOG("new error : "<<testing_error_on_testing_entries[num_tests]);
                  //   LOG("diff : "<<std::abs(testing_error_on_testing_entries[num_tests - 1] - testing_error_on_testing_entries[num_tests]));
                  //   LOG("meta_training_rate : "<<meta_training_rate);
                  // }
              }

            }
            min_error_so_far = std::min(min_error_so_far, testing_error_on_testing_entries[num_tests]);

            cpu_isBad<float>(testing_error_on_testing_entries + num_tests, (long long int)1, "testing_error_on_testing_entries", strPreamble(blank));
            cpu_isBad<float>(testing_error_on_training_entries + num_tests, (long long int)1, "testing_error_on_training_entries", strPreamble(blank));
            cpu_isBad<float>(testing_iterations + num_tests, (long long int)1, "testing_iterations", strPreamble(blank));
            //cpu_isBad<float>(R_ACU_abs_max + num_tests, (long long int)1, "R_ACU_abs_max", strPreamble(blank));
            cpu_isBad<float>(R_ACU_max_sv + num_tests, (long long int)1, "R_ACU_max_sv", strPreamble(blank));

            if(load_full_ACU_from_save && ! testing_only){
                append_host_array_to_file(testing_error_on_testing_entries + num_tests, 1, "testing_error_on_testing_entries", strPreamble(blank));
                append_host_array_to_file(testing_error_on_training_entries + num_tests, 1, "testing_error_on_training_entries", strPreamble(blank));
                append_host_array_to_file(testing_iterations + num_tests, 1, "testing_iterations", strPreamble(blank));
                //append_host_array_to_file(R_ACU_abs_max + num_tests, 1, "R_ACU_abs_max", strPreamble(blank));
                append_host_array_to_file(R_ACU_max_sv + num_tests, 1, "R_ACU_max_sv", strPreamble(blank));
                append_host_array_to_file(R_ACU_exp + num_tests, 1, "R_ACU_exp", strPreamble(blank));
                append_host_array_to_file(R_ACU_var + num_tests, 1, "R_ACU_var", strPreamble(blank));
                append_host_array_to_file(meta_km_errors + num_tests, 1, "meta_km_errors", strPreamble(blank));
                append_host_array_to_file(num_latent_factors_ + num_tests, 1, "num_latent_factors", strPreamble(blank));
                append_host_mtx_to_file(logarithmic_histogram + num_tests * 7, 1, 7, "logarithmic_histogram", true, strPreamble(blank));
                append_host_mtx_to_file(logarithmic_histogram_km + num_tests * 7, 1, 7, "logarithmic_histogram_km", true, strPreamble(blank)); 
                num_tests += 1;
            }else{
                num_tests += 1;
                save_host_array_to_file(testing_error_on_testing_entries, num_tests, "testing_error_on_testing_entries", strPreamble(blank));
                save_host_array_to_file(testing_error_on_training_entries, num_tests, "testing_error_on_training_entries", strPreamble(blank));
                save_host_array_to_file(testing_iterations, num_tests, "testing_iterations", strPreamble(blank));
                //save_host_array_to_file(R_ACU_abs_max, num_tests, "R_ACU_abs_max", strPreamble(blank));
                save_host_array_to_file(R_ACU_max_sv, num_tests, "R_ACU_max_sv", strPreamble(blank));
                save_host_array_to_file(R_ACU_exp, num_tests, "R_ACU_exp", strPreamble(blank));
                save_host_array_to_file(R_ACU_var, num_tests, "R_ACU_var", strPreamble(blank));
                save_host_array_to_file(meta_km_errors, num_tests, "meta_km_errors", strPreamble(blank));
                save_host_array_to_file(num_latent_factors_, num_tests, "num_latent_factors", strPreamble(blank));
                save_host_mtx_to_file(logarithmic_histogram, num_tests, 7, "logarithmic_histogram", true, strPreamble(blank)); 
                save_host_mtx_to_file(logarithmic_histogram_km, num_tests, 7, "logarithmic_histogram_km", true, strPreamble(blank));                
            }
            gettimeofday(&testing_end, NULL);
            program_time = (testing_end.tv_sec * 1000 +(testing_end.tv_usec/1000.0))-(testing_start.tv_sec * 1000 +(testing_start.tv_usec/1000.0));   
            cpu_incremental_average((long long int)num_tests, &avg_testing_time, program_time);
            LOG("average testing time: "<<readable_time(avg_testing_time));
            LOG("average testing time per test: "<<readable_time(avg_testing_time /static_cast<double>(val_)));

            if(num_tests > 2){
                if(meta_km_errors[num_tests - 1] > meta_km_errors[num_tests - 2]){
                    LOG("WARNING: meta_km_errors is growing");
                    //ABORT_IF_EQ(0, 0, "meta_km_errors is growing"); 
                }
            }
        }//end is testing?
        

        if(!testing_only){
            //============================================================================================
            // TRAINING
            //============================================================================================ 
            gettimeofday(&training_start, NULL);

            if(meta_training_rate < min_training_rate || do_ == false) break;
            if(micro_training_rate < min_training_rate || do_ == false) break;

            int count_ACU_rounds = 0;

            long long int total_training_nnz = (long long int)0;

            float max_abs_delta_R_ACU = (float)0.0;
            float exp_abs_delta_R_ACU = (float)0.0;

            float max_abs_R_ACU = (float)0.0;
            float exp_abs_R_ACU = (float)0.0;

            float largest_sv = (float)0.0;

            int *   csr_format_ratingsMtx_userID_dev_training_      = NULL;
            int *   coo_format_ratingsMtx_itemID_dev_training_      = NULL;
            float * coo_format_ratingsMtx_rating_dev_training_      = NULL;
            int*    csr_format_ratingsMtx_userID_dev_training_batch = NULL;
            int*    coo_format_ratingsMtx_itemID_dev_training_batch = NULL;
            float*  coo_format_ratingsMtx_rating_dev_training_batch = NULL;
            float*  full_ratingsMtx_dev_ACU_current_batch            = NULL;

            float * U_training = NULL;
            checkCudaErrors(cudaMalloc((void**)&U_training, batch_size_training * min_dim_block * SIZE_OF(float)));
            update_Mem(batch_size_training * min_dim_block * SIZE_OF(float));
            float * R_training = NULL;
            checkCudaErrors(cudaMalloc((void**)&R_training, batch_size_training * ratings_cols * SIZE_OF(float)));
            update_Mem(batch_size_training * ratings_cols * SIZE_OF(float));
            float * SV_dev = NULL;
            checkCudaErrors(cudaMalloc((void**)&SV_dev, batch_size_ACU * SIZE_OF(float)));  

            if(Conserve_GPU_Mem){
                checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_training_,  (batch_size_training + 1) * SIZE_OF(int)));
                update_Mem((batch_size_training + 1) * SIZE_OF(int) );

                checkCudaErrors(cudaMalloc((void**)&full_ratingsMtx_dev_ACU_current_batch, batch_size_ACU * ratings_cols * SIZE_OF(float)));
                update_Mem(batch_size_ACU * ratings_cols* SIZE_OF(float) );

                checkCudaErrors(cudaMalloc((void**)&U_ACU_dev,  batch_size_ACU  * min_dim_block  * SIZE_OF(float)));
                update_Mem(batch_size_ACU  * min_dim_block  * SIZE_OF(float));
            }

            bool calc_delta_R_ACU = true;
            for(int batch__ = 0; batch__ < num_blocks; batch__++){
                if(load_full_ACU_from_save && it == first_it && batch__ == 0){
                    batch__ = first_training_batch__;
                }
                //for(int batch = 0; batch < num_batches; batch++){

                if(meta_training_rate < min_training_rate || do_ == false) break;
                if( print_training_error){
                    //LOG(std::endl<<"                                       ~ITERATION "<<it<<", BATCH "<<batch__<<"~"<<std::endl);
                    LOG(std::endl<<std::endl<<"                                                           ITERATION "<<it<<" ( / "<<num_iterations<<" ), BATCH "<<batch__<<" ( / "<<num_blocks<<" )"/*<<", ACU Round "<<count_ACU_rounds<<" ( / "<<num_batches / num_blocks<<" )"*/);
                }
                if(Debug && 0){
                    //LOG(std::endl<<"                              ITERATION "<<it<<", BATCH "<<batch__);
                    LOG(std::endl<<std::endl<<"                              ITERATION "<<it<<" ( / "<<num_iterations<<" ), BATCH "<<batch__<<" ( / "<<num_blocks<<" ), ACU Round "<<count_ACU_rounds<<" ( / "<<num_batches / num_blocks<<" )");
                }
     
                int batch = 0;
                getRandIntsBetween(&batch , 0 , (int)num_batches, 1);
                if(Debug && 0)LOG("batch id : "<<batch);
                
                long long int batch_size_training_temp = batch_size_training;
                long long int first_row_in_batch_training = batch_size_training * (long long int)batch; /* % ratings_rows_testing*/;
                if(first_row_in_batch_training + batch_size_training - (long long int)1 >= ratings_rows_training) {
                    //batch_size_training_temp = ratings_rows_training - first_row_in_batch_training;
                    first_row_in_batch_training = ratings_rows_training - batch_size_training_temp;
                    ABORT_IF_LE(first_row_in_batch_training, (long long int)0, "first_row_in_batch_training too small");
                    if(Debug && 0){
                        LOG("left over batch_size_training : "<<batch_size_training_temp);
                        LOG("first_row_in_batch_training : "<<first_row_in_batch_training<< " ( / "<<ratings_rows_training<<" )");
                    }
                    
                }
                // int temp_first_row_in_batch_training = 0;
                // getRandIntsBetween(&temp_first_row_in_batch_training , 0 , (int)(ratings_rows_training - batch_size_training_temp) + 1, 1); 
                // first_row_in_batch_training = (long long int)temp_first_row_in_batch_training;  
                // LOG("first_row_in_batch_training : "<<first_row_in_batch_training<< " ( / "<<ratings_rows_training<<" )");      
                long long int ACU_batch = (long long int)batch__ % num_blocks;
                long long int first_row_in_batch_ACU = (batch_size_ACU * (long long int)ACU_batch) ;            
                //============================================================================================
                // Find U_ACU, V such that U_ACU * V^T ~ R_ACU 
                //============================================================================================  
                
                //LOG("batch_size_training : "<<batch_size_training_temp);




                if(Debug && 0){
                    LOG("ACU_batch : "<< ACU_batch <<" ( / "<<num_blocks<<" )");
                    LOG(memLeft<<" available bytes left on the device");
                    //LOG("num_latent_factors = "<< num_latent_factors);

                    LOG("first_row_in_batch_training : "<<first_row_in_batch_training<< " ( / "<<ratings_rows_training<<" )");
                    LOG("batch_size_training : "<<batch_size_training_temp);
                    //LOG("( next first_row_in_batch_training : "<<first_row_in_batch_training + batch_size_training_temp<<" )");
                    LOG("first_row_in_batch_ACU : "<<first_row_in_batch_ACU<<  " ( / "<<ratings_rows_ACU<<" )");
                    LOG("batch_size_ACU : "<<batch_size_ACU);
                    //LOG("( next first_row_in_batch_ACU : "<<first_row_in_batch_ACU + batch_size_ACU<<" )");
                };


                // if(!row_major_ordering && num_blocks > (long long int)1){
                //     ABORT_IF_EQ(0, 0, "cannot store in column major ordering with more than one block");
                // }
                if(Conserve_GPU_Mem){
                    // old way
                    checkCudaErrors(cudaMemcpy(full_ratingsMtx_dev_ACU_current_batch, full_ratingsMtx_host_ACU + ratings_cols * first_row_in_batch_ACU, 
                                                batch_size_ACU * ratings_cols * SIZE_OF(float), cudaMemcpyHostToDevice));
                }else{
                    full_ratingsMtx_dev_ACU_current_batch = full_ratingsMtx_dev_ACU + ratings_cols * first_row_in_batch_ACU;
                }

                if(row_major_ordering){
                    //remember that ratings_ACU is stored in row major ordering
                    LOG("swap matrix indexing from row major to column major");
                    gpu_swap_ordering<float>(batch_size_ACU, ratings_cols, full_ratingsMtx_dev_ACU_current_batch, row_major_ordering);
                }

                long long int nnz_training;
                long long int first_coo_ind_training;
                if(Conserve_GPU_Mem){
                    if(Debug && 0){
                        save_host_array_to_file<int>  (csr_format_ratingsMtx_userID_host_training,  ratings_rows_training + 1, "csr_format_ratingsMtx_userID_host_training_2");
                    }
                    csr_format_ratingsMtx_userID_dev_training_batch = csr_format_ratingsMtx_userID_host_training +  first_row_in_batch_training;
                    first_coo_ind_training = csr_format_ratingsMtx_userID_dev_training_batch[0];
                    int last_entry_index = (csr_format_ratingsMtx_userID_dev_training_batch + batch_size_training_temp)[0];

                    nnz_training = (long long int)last_entry_index - first_coo_ind_training;
                    if(Debug  && 0){
                        LOG("first_coo_ind_training : "<<first_coo_ind_training);
                        LOG("last_entry_index : "<<last_entry_index);
                        LOG("nnz_training : "<<nnz_training);
                    }

                    if(nnz_training <= 0){
                        LOG("nnz_training : "<<nnz_training);
                        ABORT_IF_EQ(0, 0, "nnz_training <= 0");
                    }
                    
                    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_training_,  nnz_training        * SIZE_OF(int)));
                    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_training_,  nnz_training        * SIZE_OF(float)));
                    update_Mem(nnz_training * SIZE_OF(int) );
                    update_Mem(nnz_training * SIZE_OF(float) );

                    checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_dev_training_,  csr_format_ratingsMtx_userID_dev_training_batch,  (batch_size_training_temp + 1) *  SIZE_OF(int),   cudaMemcpyHostToDevice));
                    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_dev_training_,  coo_format_ratingsMtx_itemID_host_training + first_coo_ind_training, nnz_training  *  SIZE_OF(int),   cudaMemcpyHostToDevice));
                    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_training_,  coo_format_ratingsMtx_rating_host_training + first_coo_ind_training,  nnz_training  *  SIZE_OF(float), cudaMemcpyHostToDevice));
                    
                    csr_format_ratingsMtx_userID_dev_training_batch = csr_format_ratingsMtx_userID_dev_training_;
                    coo_format_ratingsMtx_itemID_dev_training_batch = coo_format_ratingsMtx_itemID_dev_training_;
                    coo_format_ratingsMtx_rating_dev_training_batch = coo_format_ratingsMtx_rating_dev_training_;
                }else{
                    csr_format_ratingsMtx_userID_dev_training_batch = csr_format_ratingsMtx_userID_dev_training +  first_row_in_batch_training;
                    nnz_training = gpu_get_num_entries_in_rows(0, batch_size_training_temp - 1, csr_format_ratingsMtx_userID_dev_training_batch);
                    if(nnz_training <=0){
                        LOG("nnz_training : "<<nnz_training);
                        ABORT_IF_EQ(0, 0, "nnz_training <= 0");
                    }
                    first_coo_ind_training = gpu_get_first_coo_index(0, csr_format_ratingsMtx_userID_dev_training_batch);
                    
                    coo_format_ratingsMtx_itemID_dev_training_batch = coo_format_ratingsMtx_itemID_dev_training + first_coo_ind_training;
                    coo_format_ratingsMtx_rating_dev_training_batch = coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training;
                }
                ABORT_IF_LESS(nnz_training, 1, "nnz < 1");
                total_training_nnz += nnz_training;


                float* coo_training_errors;
                //float* testing_entries;
                checkCudaErrors(cudaMalloc((void**)&coo_training_errors, nnz_training * SIZE_OF(float)));
                //checkCudaErrors(cudaMalloc((void**)&testing_entries, nnz_training * SIZE_OF(float)));
                update_Mem(nnz_training * SIZE_OF(float));

                if(Debug && 0){
                    LOG("training requires " <<nnz_training * SIZE_OF(float) + batch_size_training  * std::max(min_dim_block, batch_size_training) * SIZE_OF(float) 
                                + batch_size_training  * ratings_cols * SIZE_OF(float) +
                                (batch_size_training + 1) * SIZE_OF(int) + nnz_training * SIZE_OF(int) + nnz_training * SIZE_OF(float) << " bytes of memory");

                    LOG("first_coo_ind in this training batch : "<<first_coo_ind_training<< " ( / "<<num_entries_training<<" )");
                    LOG("nnz in this training batch : "<<nnz_training);
                    LOG("( next first_coo_ind in training batch : "<<first_coo_ind_training + nnz_training <<" )");
                    // save_device_array_to_file<int>(csr_format_ratingsMtx_userID_dev_training_batch, batch_size_training + 1, "csr_format_ratingsMtx_userID_dev_training_batch");
                    // LOG("first entry of csr_format_ratingsMtx_userID_dev_training_batch: ");
                    // print_gpu_array_entries<int>(csr_format_ratingsMtx_userID_dev_training_batch, 1);
                    // save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_training + (int)first_coo_ind_training, nnz_training, "coo_format_ratingsMtx_rating_dev_training_batch");
                    // save_device_array_to_file<int>(coo_format_ratingsMtx_itemID_dev_training + (int)first_coo_ind_training, nnz_training, "coo_format_ratingsMtx_itemID_dev_training");
                    // LOG("Press Enter to continue.") ;
                    // std::cin.ignore();
                }




                for(int round_ = 0; round_ < 1; round_++){
                    //LOG(std::endl<<"                                                      ROUND : "<<round_);

                    gpu_orthogonal_decomp<float>(dn_handle, dn_solver_handle,
                                                batch_size_ACU, ratings_cols, 
                                                &num_latent_factors, (float)0.999,
                                                full_ratingsMtx_dev_ACU_current_batch, 
                                                U_ACU_dev, V_dev, SV_with_U, SV_dev);

                    //LOG("num_latent_factors = "<< num_latent_factors << " ( / "<< batch_size_ACU<< " )");

                    num_latent_factors = std::min(num_latent_factors, (long long int)((percent + (float)0.1) * (float)batch_size_ACU));

                    LOG("num_latent_factors = "<< num_latent_factors << " ( / "<< batch_size_ACU<< " )");

                    //checkCudaErrors(cudaMemcpy(U_ACU_host, U_ACU_dev, batch_size_ACU * min_dim_block * SIZE_OF(float), cudaMemcpyDeviceToHost));

                    //save_device_mtx_to_file<float>(U_ACU, ratings_rows_ACU, num_latent_factors, "U_ACU_compressed");

                    /*
                        At this point U_ACU is batch_size_ACU by batch_size_ACU in memory stored in column major
                        ordering and V is ratings_cols by batch_size_ACU stored in column major ordering

                        There is no extra work to compress U_ACU into batch_size_ACU by num_latent_factors, or
                        to compress V into ratings_cols by num_latent_factors, just take the first num_latent_factors
                        columns of each matrix
                    */
                     

                    //============================================================================================
                    // Compute  R_training * V = U_training
                    // Compute  Error = R_training -  U_training * V^T  <-- sparse
                    //============================================================================================ 
                    //if(Debug) LOG("iteration "<<it<<" made it to check point");


                    
                    bool calc_delta_V = false;
                    float* delta_V;

                    if(calc_delta_V){
                        checkCudaErrors(cudaMalloc((void**)&delta_V, ratings_cols * min_dim_block * SIZE_OF(float)));
                        checkCudaErrors(cudaMemcpy(delta_V, V_dev, ratings_cols * min_dim_block * SIZE_OF(float), cudaMemcpyDeviceToDevice));
                        update_Mem(ratings_cols * min_dim_block * SIZE_OF(float));
                    } 

                    bool calc_delta_U = false;
                    float* delta_U_ACU;
                    if(calc_delta_U){
                        checkCudaErrors(cudaMalloc((void**)&delta_U_ACU, batch_size_ACU * min_dim_block * SIZE_OF(int)));
                        checkCudaErrors(cudaMemcpy(delta_U_ACU, U_ACU_dev, batch_size_ACU * min_dim_block * SIZE_OF(float), cudaMemcpyDeviceToDevice));
                    }

                    calc_delta_R_ACU = false;
                    float* delta_R_ACU;
                    if(calc_delta_R_ACU){
                        checkCudaErrors(cudaMalloc((void**)&delta_R_ACU, batch_size_ACU * ratings_cols * SIZE_OF(float)));
                        update_Mem(batch_size_ACU * ratings_cols * SIZE_OF(float));
                        checkCudaErrors(cudaMemcpy(delta_R_ACU, full_ratingsMtx_dev_ACU_current_batch, 
                                                    batch_size_ACU * ratings_cols * SIZE_OF(float), cudaMemcpyDeviceToDevice));
                        
                        if(Debug  && 0){
                            //save_device_mtx_to_file<float>(delta_R_ACU, batch_size_ACU, 1, "delta_R_ACU_0", true, strPreamble(blank));
                            //save_device_mtx_to_file<float>(full_ratingsMtx_dev_ACU, batch_size_ACU, 1, "R_ACU_0", true, strPreamble(blank));
                        }
                        
                    }
                    gpu_R_error<float>(dn_handle, sp_handle, sp_descr,dn_solver_handle, 
                                       batch_size_training_temp, batch_size_ACU, num_latent_factors, ratings_cols,
                                       nnz_training, first_coo_ind_training, compress_when_training, 
                                       NULL, coo_training_errors, (float)0.0,
                                       coo_format_ratingsMtx_rating_dev_training_batch, 
                                       csr_format_ratingsMtx_userID_dev_training_batch, 
                                       coo_format_ratingsMtx_itemID_dev_training_batch,
                                       V_dev, U_training, R_training, U_ACU_dev, full_ratingsMtx_dev_ACU_current_batch,
                                       micro_training_rate, regularization_constant, batch__, it,
                                       training_error_v + it, training_error_u + it, 
                                       training_iterations + it, SV_with_U, SV_dev);

                    

                    // gpu_R_error_training<float>(dn_handle, sp_handle, sp_descr,
                    //                            batch_size_training_temp, batch_size_ACU, num_latent_factors, ratings_cols,
                    //                            nnz_training, first_coo_ind_training, false /*compress_when_training*/, coo_training_errors,
                    //                            coo_format_ratingsMtx_rating_dev_training_batch, 
                    //                            csr_format_ratingsMtx_userID_dev_training_batch,         // <-- already has shifted to correct start
                    //                            coo_format_ratingsMtx_itemID_dev_training_batch,
                    //                            V_dev, U_training, R_training, training_rate, regularization_constant, SV_with_U, SV_dev);


                    

                    
                    if(num_blocks - 1 == ACU_batch){
                        if( print_training_error ){
                            //LOG("           ~Finished round "<<count_ACU_rounds<<" of ACU training~"<<std::endl); 
                            LOG("TRAINING AVERAGE SQUARED ERROR : "<< ToString<float>(training_error_v[it])); 

                            // float temp = gpu_sum_of_squares<float>(nnz_training, testing_entries);
                            // float temp = gpu_sum_of_squares_of_diff(dn_handle, nnz_training, 
                            //                                         coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training, 
                            //                                         testing_entries);
                            
                            // LOG("training error over range of ratings: "<< training_error_temp / ((float)(nnz_) * range_training));
                            // LOG("training error normalized: "<< training_error_temp / temp<<std::endl); 
                            // LOG("training error : "<< training_error_temp / (float)(nnz_* 2.0)); 

                        } 
                        count_ACU_rounds += 1;  
                        total_training_nnz = (long long int)0;
                        
                    }

                    if(Debug && 0){

                        //save_device_arrays_side_by_side_to_file<float>(coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training, testing_entries, coo_training_errors, nnz_training, "ratings_testing_errors");
                        
                        float coo_training_errors_abs_exp = gpu_expected_abs_value<float>(nnz_training, coo_training_errors);
                        float coo_training_errors_abs_max = gpu_abs_max<float>           (nnz_training, coo_training_errors);
                        LOG("coo_training_errors_abs_max = "<<ToString<float>(coo_training_errors_abs_max)) ;
                        //LOG("coo_training_errors_abs_max over range of ratings = "<<coo_training_errors_abs_max / range_training) ;
                        LOG("coo_training_errors_abs_exp = "<<ToString<float>(coo_training_errors_abs_exp)) ;
                        //LOG("coo_training_errors_abs_exp over range of ratings = "<<coo_training_errors_abs_exp / range_training) ;

                        // coo_training_errors_abs_exp = gpu_expected_abs_value<float>(nnz_training, coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training);
                        // coo_training_errors_abs_max = gpu_abs_max<float>           (nnz_training, coo_format_ratingsMtx_rating_dev_training + first_coo_ind_training);
                        // LOG("coo_training_abs_max = "<<coo_training_errors_abs_max) ;
                        // LOG("coo_training_abs_exp = "<<coo_training_errors_abs_exp) ;

                        // LOG("Press Enter to continue.") ;
                        // std::cin.ignore();
                    }

                    //checkCudaErrors(cudaFree(testing_entries));

                    bool update_U_temp = update_U;

                    if(training_error_v[it] < (float)0.01 && update_U_temp){
                        update_U = true;
                    }else{
                        //update_U = false;
                    }

                    float alpha =  meta_training_rate;
                    float beta = (float)1.0 - meta_training_rate * regularization_constant;
                    //if( round_ % 2 == 0 || !update_U ) 
                    {

                        //============================================================================================
                        // Update  V = V * (1 - alpha * lambda) + alpha * Error^T * U_training 
                        // (Update  U = U * (1 - alpha * lambda) + alpha * Error * V ) <- random error?????
                        //============================================================================================ 
                        //if(Debug) LOG("iteration "<<it<<" made it to check point");
                        /*
                            m,n,k
                            This function performs one of the following matrix-matrix operations:

                            C = α ∗ op ( A ) ∗ op ( B ) + β ∗ C

                            A is an m×k sparse matrix that is defined in CSR storage format by the three arrays csrValA, csrRowPtrA, and csrColIndA); B and C are dense matrices; α  and  β are scalars; and

                            op ( A ) = A if transA == CUSPARSE_OPERATION_NON_TRANSPOSE, A^T if transA == CUSPARSE_OPERATION_TRANSPOSE, A^H if transA == CUSPARSE_OPERATION_CONGACUTE_TRANSPOSE
                            and

                            op ( B ) = B if transB == CUSPARSE_OPERATION_NON_TRANSPOSE, B^T if transB == CUSPARSE_OPERATION_TRANSPOSE, B^H not supported
                            array of dimensions (ldb, n) if op(B)=B and (ldb, k) otherwise.

                            n is the number of columns of dense matrix op(B) and C.
                        */


                        if(regularize_V){
                            float beta = (float)1.0 - meta_training_rate * regularization_constant;
                        }else{
                            beta = (float)1.0;
                        }

                        // gpu_spXdense_MMM<float>(sp_handle, true, false, batch_size_training_temp, min_dim_block, 
                        //                         ratings_cols, nnz_training, first_coo_ind_training, &meta_training_rate, sp_descr, 
                        //                         coo_training_errors, 
                        //                         csr_format_ratingsMtx_userID_dev_training_batch, 
                        //                         coo_format_ratingsMtx_itemID_dev_training_batch,
                        //                         U_training, batch_size_training_temp, &beta, V_dev, ratings_cols, false);

                        if(normalize_V_rows && SV_with_U){
                            LOG("Normalizing the rows of V...");
                            gpu_normalize_mtx_rows_or_cols(ratings_cols, min_dim_block, false, V_dev, false);
                        }

                        /*
                            N is number of rows of matrix op(B) and C.
                            M number of columns of matrix op(A) and C.
                            K is number of columns of op(B) and rows of op(A).
                            
                            op(B) is N by K
                            op(A) is K by M
                            C is N by M

                            performs C=alpha op ( B ) op ( A ) + beta C
                        */


                        /*
                            gpu_noisey_gemm<float>(dn_handle, false, false, 
                                            batch_size_ACU, (compress_when_training == false) ? batch_size_ACU : num_latent_factors, 
                                            ratings_cols,
                                            beta, //(float)1.0
                                            V, training_rate, beta,
                                            U_ACU);
                        */


                        

                        if(calc_delta_V){
                            // float* copy;
                            // checkCudaErrors(cudaMalloc((void**)&copy, ratings_cols * min_dim_block * SIZE_OF(float)));
                            // update_Mem(ratings_cols * min_dim_block * SIZE_OF(float) );
                            // checkCudaErrors(cudaMemcpy(copy, delta_V, ratings_cols * min_dim_block * SIZE_OF(float), cudaMemcpyDeviceToDevice));
                            gpu_axpby<float>(dn_handle, ratings_cols * min_dim_block, 
                                             (float)(-1.0), V_dev,
                                             (float)(1.0), delta_V);
                            float delta_abs_exp = gpu_expected_abs_value<float>(ratings_cols * min_dim_block, delta_V);
                            float delta_abs_max = gpu_abs_max<float>(ratings_cols * min_dim_block, delta_V); 
                            LOG("delta V maximum absolute value = "<<ToString<float>(delta_abs_max)) ;
                            LOG("delta V expected absolute value = "<<ToString<float>(delta_abs_exp)) ;
                            ABORT_IF_EQ(delta_abs_max, delta_abs_exp, "delta V is constant");

                            gpu_mean_abs_nonzero((min_dim_block * ratings_cols), delta_V, &delta_abs_exp, 1, "delta_V");
                            LOG("delta V expected absolute value over only non zero = "<<ToString<float>(delta_abs_exp)) ;
                            // save_device_arrays_side_by_side_to_file(copy, V, delta_V, ratings_cols * min_dim_block, "old_new_delta_V");
                            // LOG("Press Enter to continue.") ;
                            // std::cin.ignore();
                            if(delta_V) {
                                checkCudaErrors(cudaFree(delta_V));
                                update_Mem(ratings_cols * min_dim_block * SIZE_OF(float) * (-1));
                            }
                            // checkCudaErrors(cudaFree(copy));
                            // update_Mem(ratings_cols * min_dim_block * SIZE_OF(float) * (-1));
                        }

                    
                    }
                    //else
                    {

                        if(update_U){
                            /*
                                LOG("Update U_ACU...");

                                // if(compress_when_training){
                                //     gpu_scale<float>(dn_handle, batch_size_ACU * num_latent_factors, beta, U_ACU_dev);
                                // }else{
                                //     gpu_scale<float>(dn_handle, batch_size_ACU * batch_size_ACU, beta, U_ACU_dev);
                                // }   

                            
                                float* errors;
                                int* selection;
                                checkCudaErrors(cudaMalloc((void**)&errors, batch_size_training_temp * SIZE_OF(float)));
                                checkCudaErrors(cudaMalloc((void**)&selection, batch_size_training_temp * SIZE_OF(int)));
                                
                                
                                bool dense_update = true;
                                if(dense_update){
                                    gpu_dense_nearest_row<float>(batch_size_ACU, min_dim_block, U_ACU_dev, 
                                                                 batch_size_training_temp, U_training, 
                                                                 selection, errors, false);
                                }else{

                                    gpu_sparse_nearest_row<float>(batch_size_ACU, ratings_cols, full_ratingsMtx_dev_ACU_current_batch, 
                                                                 batch_size_training_temp, nnz_training, 
                                                                 csr_format_ratingsMtx_userID_dev_training_batch, 
                                                                 coo_format_ratingsMtx_itemID_dev_training_batch,
                                                                 coo_format_ratingsMtx_rating_dev_training_batch, 
                                                                 selection, errors, false);
                                }


                                //int min_selection = gpu_min<int>(batch_size_training_temp, selection);

                                float k_means_er_training = gpu_sum(batch_size_training_temp,  errors);
                                //LOG("err norm when clustering U rows : "<<std::sqrt(k_means_er));
                                k_means_er_training /= (float)(batch_size_training_temp);
                                LOG("mean sqed err when clustering U rows : "<<ToString<float>(k_means_er_training));

                                cpu_incremental_average((long long int)(it + 1), training_error_u + it, k_means_er_training);

                                if(Debug){
                                    save_device_array_to_file<float>(errors, (int)batch_size_training_temp, "km_errors", strPreamble(blank));
                                    save_device_array_to_file<int>(selection, (int)batch_size_training_temp, "km_selection", strPreamble(blank));                        
                                }
                                checkCudaErrors(cudaFree(errors));


                                //long long int skip = min_selection * batch_size_ACU;
                                if(Debug){
                                    //LOG("min_selection : "<<min_selection);
                                    // save_device_mtx_to_file<float>(delta_U_ACU + skip, 3, (compress_when_training == false) ? batch_size_ACU : num_latent_factors, "U_ACU_old_0", true, strPreamble(blank));
                                    // save_device_mtx_to_file<float>(U_ACU + skip, 3, (compress_when_training == false) ? batch_size_ACU : num_latent_factors, "U_ACU_0", true, strPreamble(blank));
                                }
                                gpu_calculate_KM_error_and_update(batch_size_ACU, min_dim_block, U_ACU_dev, 
                                                                 batch_size_training_temp, U_training, csr_format_ratingsMtx_userID_dev_training_batch,
                                                                 selection, meta_training_rate, regularization_constant, it);
                                checkCudaErrors(cudaFree(selection));

                            */

                            if(calc_delta_U){
                                // save_device_mtx_to_file<float>(delta_U_ACU + skip, 3, (compress_when_training == false) ? batch_size_ACU : num_latent_factors, "U_ACU_old_1", true, strPreamble(blank));
                                // save_device_mtx_to_file<float>(U_ACU + skip, 3, (compress_when_training == false) ? batch_size_ACU : num_latent_factors, "U_ACU_1", true, strPreamble(blank));
                            
                                gpu_axpby<float>(dn_handle, (batch_size_ACU * min_dim_block), (float)(-1.0), U_ACU_dev, (float)(1.0), delta_U_ACU);
                            
                                // save_device_mtx_to_file<float>(delta_U_ACU + skip, 3, (compress_when_training == false) ? batch_size_ACU : num_latent_factors, "U_ACU_old_2", true, strPreamble(blank));
                                // save_device_mtx_to_file<float>(U_ACU + skip, 3, (compress_when_training == false) ? batch_size_ACU : num_latent_factors, "U_ACU_2", true, strPreamble(blank));
                            
                                // LOG("first 5 entries of delta_U_ACU :") ;
                                // print_gpu_array_entries(delta_U_ACU, 5, strPreamble(blank));

                                float delta_abs_exp = gpu_expected_abs_value<float>((batch_size_ACU * min_dim_block), delta_U_ACU);
                                float delta_abs_max = gpu_abs_max<float>((batch_size_ACU * min_dim_block), delta_U_ACU); 
                                LOG("delta U maximum absolute value = "<<ToString<float>(delta_abs_max)) ;
                                LOG("delta U expected absolute value over all = "<<ToString<float>(delta_abs_exp)) ;

                                gpu_mean_abs_nonzero((batch_size_ACU * min_dim_block), delta_U_ACU, &delta_abs_exp, 1, "delta_U_ACU");
                                LOG("delta U expected absolute value over only non zero = "<<ToString<float>(delta_abs_exp)) ;
                                ABORT_IF_EQ(delta_abs_max, delta_abs_exp, "delta U is constant");                    
                                checkCudaErrors(cudaFree(delta_U_ACU));
                            }

                        }
                    }

                    update_U = update_U_temp;

                    if(Debug){
                        //save_device_mtx_to_file<float>(V_dev, ratings_cols, num_latent_factors, "V_compressed");
                        //save_device_mtx_to_file<float>(U_ACU_dev, batch_size_ACU, num_latent_factors, "U_ACU_compressed");
                    }
                    //============================================================================================
                    // Update  R_ACU = U_ACU * V^T
                    //============================================================================================ 
                    //if(Debug) LOG("iteration "<<it<<" made it to check point");
                    if(ACU_batch == 0 && 0){
                        // store backup
                        if(Conserve_GPU_Mem){
                            host_copy(ratings_rows_ACU * ratings_cols, full_ratingsMtx_host_ACU, old_R_ACU);
                        }else{
                            checkCudaErrors(cudaMemcpy(old_R_ACU, full_ratingsMtx_dev_ACU, 
                                                    ratings_rows_ACU * ratings_cols * SIZE_OF(float), cudaMemcpyDeviceToDevice));
                            if(row_major_ordering) 
                                gpu_swap_ordering<float>(batch_size_ACU, ratings_cols, old_R_ACU + ratings_cols * first_row_in_batch_ACU, !row_major_ordering);
                        }
                    }

                    if(regularize_R){
                        float beta = (float)1.0 - meta_training_rate * regularization_constant;
                    }else{
                        beta = (float)1.0;
                    }

                    // gpu_gemm<float>(dn_handle, true, false, 
                    //                 ratings_cols, batch_size_ACU, 
                    //                 compress_when_training ? num_latent_factors : min_dim_block,
                    //                 (float)1.0,
                    //                 V_dev, U_ACU_dev, 
                    //                 (float)0.0,
                    //                 full_ratingsMtx_dev_ACU_current_batch);

                    if (regularize_R_distribution){
                        LOG("regularize_R_distribution...") ;
                        float* user_means_ACU;
                        float* user_var_ACU;
                        checkCudaErrors(cudaMalloc((void**)&user_means_ACU, batch_size_ACU * SIZE_OF(float)));
                        checkCudaErrors(cudaMalloc((void**)&user_var_ACU,   batch_size_ACU * SIZE_OF(float)));
                        update_Mem(2 * batch_size_ACU * SIZE_OF(float));
                        center_rows(batch_size_ACU, ratings_cols, full_ratingsMtx_dev_ACU_current_batch, 
                                    val_when_var_is_zero, user_means_ACU,  user_var_ACU);
                        checkCudaErrors(cudaFree(user_means_ACU));
                        checkCudaErrors(cudaFree(user_var_ACU));
                        update_Mem(2 * batch_size_ACU * SIZE_OF(float) * (-1));
                        if(Debug  && 0){
                            //save_device_mtx_to_file<float>(delta_R_ACU, batch_size_ACU, 1, "delta_R_ACU_1", true, strPreamble(blank));
                            save_device_mtx_to_file<float>(full_ratingsMtx_dev_ACU_current_batch, batch_size_ACU, 1, "R_ACU_1", true, strPreamble(blank));
                        }
                    }

                    if(calc_delta_R_ACU){
                        //float* copy;
                        //checkCudaErrors(cudaMalloc((void**)&copy, batch_size_ACU * ratings_cols * SIZE_OF(float)));
                        //update_Mem(batch_size_ACU * ratings_cols * SIZE_OF(float));
                        //checkCudaErrors(cudaMemcpy(copy, delta_R_ACU, batch_size_ACU * ratings_cols * SIZE_OF(float), cudaMemcpyDeviceToDevice));
                        gpu_axpby<float>(dn_handle, batch_size_ACU * ratings_cols, 
                                         (float)(-1.0), full_ratingsMtx_dev_ACU_current_batch,
                                         (float)(1.0), delta_R_ACU);

                        float delta_abs_exp = gpu_expected_abs_value<float>(batch_size_ACU * ratings_cols, delta_R_ACU);
                        float delta_abs_max = gpu_abs_max<float>           (batch_size_ACU * ratings_cols, delta_R_ACU); 
                        LOG("delta R_ACU maximum absolute value = "<<ToString<float>(delta_abs_max)) ;
                        LOG("delta R_ACU expected absolute value over all = "<<ToString<float>(delta_abs_exp)) ;

                        cpu_incremental_average((long long int)(batch__ + 1), delta_R_ACU_exp + it, delta_abs_exp);

                        gpu_mean_abs_nonzero((batch_size_ACU * ratings_cols), delta_R_ACU, &delta_abs_exp, 1, "delta_R_ACU");
                        LOG("delta R_ACU expected absolute value over only non zero = "<<ToString<float>(delta_abs_exp)) ;
                        max_abs_delta_R_ACU = std::max(max_abs_delta_R_ACU, delta_abs_max);
                        if(Debug && 0){
                            LOG("first 5 entries of delta_R_ACU :") ;
                            print_gpu_array_entries(delta_R_ACU, 5, strPreamble(blank));
                            // save_device_arrays_side_by_side_to_file(copy, full_ratingsMtx_dev_ACU_current_batch, delta_R_ACU, batch_size_ACU * ratings_cols, "old_new_delta");
                            // LOG("Press Enter to continue.") ;
                            // std::cin.ignore();
                        }
                        checkCudaErrors(cudaFree(delta_R_ACU));
                        //checkCudaErrors(cudaFree(copy));
                        update_Mem(/*2 * */batch_size_ACU * ratings_cols * SIZE_OF(float) * (-1));
                    }
                    
                    if(Debug && 0){
                        LOG("first 5 entries of full_ratingsMtx_dev_ACU_current_batch :") ;
                        print_gpu_array_entries(full_ratingsMtx_dev_ACU_current_batch, 5, strPreamble(blank));
                        //save_device_mtx_to_file(full_ratingsMtx_dev_ACU_current_batch, batch_size_ACU, ratings_cols, "full_ratingsMtx_dev_ACU_current_batch", false);
                    }

                    // if(batch > 0 || it > 0){
                    //     float R_ACU_abs_exp = gpu_expected_abs_value<float>(batch_size_ACU * ratings_cols, full_ratingsMtx_dev_ACU_current_batch);
                    //     float R_ACU_abs_max = gpu_abs_max<float>           (batch_size_ACU * ratings_cols, full_ratingsMtx_dev_ACU_current_batch); 
                    //     LOG("full_ratingsMtx_dev_ACU_current_batch_abs_max = "<<ToString<float>(R_ACU_abs_max)) ;
                    //     LOG("full_ratingsMtx_dev_ACU_current_batch_abs_exp = "<<ToString<float>(R_ACU_abs_exp)) ;
                    //     ABORT_IF_EQ(R_ACU_abs_max, R_ACU_abs_exp, "R_ACU is constant");
                    //     ABORT_IF_LESS((float)10.0 * abs_max_training, std::abs(R_ACU_abs_max), "unstable growth");
                    //     ABORT_IF_LESS( std::abs(R_ACU_abs_max), abs_max_training / (float)10.0 , "unstable shrinking");
                    //     // LOG("Press Enter to continue.") ;
                    //     // std::cin.ignore();
                    // }

                } // end loop on rounds




                if(row_major_ordering){
                    //remember that ratings_ACU is stored in row major ordering
                    LOG("swap matrix indexing from column major to row major");
                    gpu_swap_ordering<float>(batch_size_ACU, ratings_cols, full_ratingsMtx_dev_ACU_current_batch, !row_major_ordering);
                }
                
                if(Conserve_GPU_Mem){
                    LOG("copy R_ACU to host storage");
                    checkCudaErrors(cudaMemcpy(full_ratingsMtx_host_ACU + ratings_cols * first_row_in_batch_ACU, full_ratingsMtx_dev_ACU_current_batch, 
                                                batch_size_ACU * ratings_cols * SIZE_OF(float), cudaMemcpyDeviceToHost));
                    checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_training_));
                    checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_training_));
                    
                    update_Mem(nnz_training * SIZE_OF(int)   * (-1) );
                    update_Mem(nnz_training * SIZE_OF(float) * (-1) ); 
                }
                checkCudaErrors(cudaFree(coo_training_errors));
                update_Mem(nnz_training * SIZE_OF(float) * (-1));

                if(it % testing_rate == (testing_rate - 1)){
                
                    std::string full_ratingsMtx_ACU_filepath = save_path + "full_ratingsMtx_ACU_block_" + ToString<int>(batch__) + "_it_"+ ToString<int>(it);
                    if(Conserve_GPU_Mem){
                        save_host_mtx_to_file(full_ratingsMtx_host_ACU + ratings_cols * first_row_in_batch_ACU, batch_size_ACU, ratings_cols, full_ratingsMtx_ACU_filepath, row_major_ordering, strPreamble(blank));
                    }else{
                        ABORT_IF_EQ(0, 0, "Option not yet available.");
                        if(row_major_ordering){
                            save_device_mtx_to_file(full_ratingsMtx_dev_ACU_current_batch, ratings_cols, batch_size_ACU, full_ratingsMtx_ACU_filepath, row_major_ordering, strPreamble(blank));
                        }else{
                            save_device_mtx_to_file(full_ratingsMtx_dev_ACU_current_batch, batch_size_ACU, ratings_cols, full_ratingsMtx_ACU_filepath, !row_major_ordering, strPreamble(blank));
                        }

                    }
                }

                


                /*
                    //============================================================================================
                    // Assume the rows of R_ACU are the ratings_rows_ACU-means for the training users - what is the error?
                    //============================================================================================ 

                    float* errors;
                    int* selection;
                    if(Conserve_GPU_Mem){
                        errors  = (float *)malloc(ratings_rows_training * SIZE_OF(float));
                        selection  = (int *)malloc(ratings_rows_training * SIZE_OF(int));
                        checkErrors(errors);
                        checkErrors(selection);
                        cpu_sparse_nearest_row<float>(ratings_rows_ACU, ratings_cols, full_ratingsMtx_host_ACU, 
                                                     ratings_rows_training, num_entries_training, 
                                                     csr_format_ratingsMtx_userID_host_training, 
                                                     coo_format_ratingsMtx_itemID_host_training,
                                                     coo_format_ratingsMtx_rating_host_training, 
                                                     selection, errors);
                        float mean_ACUess_error = cpu_sum_of_squares<float>(num_entries_training, coo_format_ratingsMtx_rating_host_training);
                        float k_means_er = cpu_sum(ratings_rows_training,  errors);
                        LOG("err norm when clustering : "<<std::sqrt(k_means_er));
                        LOG("err norm when clustering over err when guessing mean one cluster: "<<std::sqrt(k_means_er) / std::sqrt(mean_ACUess_error));
                        LOG("mean sqed err when clustering : "<<k_means_er / (float)num_entries_training);
                        if(Debug && 0){
                            save_host_array_to_file<float>(errors, ratings_rows_training, "errors");
                            save_host_array_to_file<int>(selection, ratings_rows_training, "selection");
                        }
                        free(errors);
                        free(selection);
                    }else{
                        checkCudaErrors(cudaMalloc((void**)&errors, ratings_rows_training * SIZE_OF(float)));
                        checkCudaErrors(cudaMalloc((void**)&selection, ratings_rows_training * SIZE_OF(int)));
                        update_Mem(2 * ratings_rows_training * SIZE_OF(int));
                        gpu_sparse_nearest_row<float>(ratings_rows_ACU, ratings_cols, full_ratingsMtx_dev_ACU, 
                                                 ratings_rows_training, num_entries_training, 
                                                 csr_format_ratingsMtx_userID_dev_training, 
                                                 coo_format_ratingsMtx_itemID_dev_training,
                                                 coo_format_ratingsMtx_rating_dev_training, 
                                                 selection, errors, row_major_ordering);
                        float k_means_er = gpu_sum(ratings_rows_training,  errors);
                        LOG("err norm when clustering : "<<std::sqrt(k_means_er));
                        LOG("mean sqed err when clustering : "<<k_means_er / (float)num_entries_training);
                        save_device_array_to_file<float>(errors, ratings_rows_training, "errors");
                        save_device_array_to_file<int>(selection, ratings_rows_training, "selection");
                        checkCudaErrors(cudaFree(errors));
                        checkCudaErrors(cudaFree(selection));
                        update_Mem(2 * ratings_rows_training * SIZE_OF(int) * (-1));
                    }
                */
                
                //sanity check
                // if(gpu_abs_max(nnz_training, coo_format_ratingsMtx_rating_dev_training_batch) > abs_max_training){
                //     ABORT_IF_EQ(0, 0, "abs_max_training should not grow");
                // }

                checkCudaErrors(cudaDeviceSynchronize());
                gettimeofday(&program_end, NULL);
                program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
                //printf("program_time: %f\n", program_time);   
                LOG("RUN TIME SO FAR: "<<readable_time(program_time));
                
                // count_batches_so_far++;
                // gettimeofday(&training_end, NULL);
                // training_time = (training_end.tv_sec * 1000 +(training_end.tv_usec/1000.0))-(training_start.tv_sec * 1000 +(training_start.tv_usec/1000.0));  
                // //LOG("average training iteration time : "<<readable_time(training_time / (double)(count_batches_so_far)));
                // //LOG("predicted run time left: "<<readable_time((training_time / (double)(count_batches_so_far)) * (double)(num_iterations * num_batches - count_batches_so_far))<<std::endl);
                // LOG("training_time : "<<readable_time(training_time));


            }//end for loop on batches

            checkCudaErrors(cudaFree(U_training));
            update_Mem(batch_size_training  * std::max(min_dim_block, batch_size_training)  * SIZE_OF(float) * (-1));
            checkCudaErrors(cudaFree(R_training));
            update_Mem(batch_size_training * ratings_cols * SIZE_OF(float) * (-1));

            if (SV_dev) checkCudaErrors(cudaFree(SV_dev));
            update_Mem(min_dim_ * SIZE_OF(float) * (-1));

            if(Conserve_GPU_Mem){
                checkCudaErrors(cudaFree(full_ratingsMtx_dev_ACU_current_batch));
                checkCudaErrors(cudaFree(U_ACU_dev));
                update_Mem(batch_size_ACU  * min_dim_block * SIZE_OF(float) * (-1) );
                checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_training_));
                update_Mem((batch_size_training + 1) * SIZE_OF(int) * (-1));

            }

            if(num_blocks != (long long int)1){
                if(Conserve_GPU_Mem){
                    LOG("shuffle ACU matrix rows");
                    // cpu_shuffle_mtx_rows_or_cols(dn_handle, ratings_rows_ACU, ratings_cols, 
                    //                              row_major_ordering, full_ratingsMtx_host_ACU, 1);
                }else{
                    LOG("shuffle ACU matrix rows");
                    //shuffle ACU rows
                    gpu_shuffle_mtx_rows_or_cols(dn_handle, ratings_rows_ACU, ratings_cols,  
                                                row_major_ordering, full_ratingsMtx_dev_ACU, 1);


                    //shuffle training rows??
                }
            }
            if(it % (num_iterations / 4) == 0){
               meta_training_rate =  meta_training_rate / (float)10.0;
            }



            gettimeofday(&training_end, NULL);
            program_time = (training_end.tv_sec * 1000 +(training_end.tv_usec/1000.0))-(training_start.tv_sec * 1000 +(training_start.tv_usec/1000.0));   
            cpu_incremental_average((long long int)(it + 1), &avg_training_time, program_time);
            LOG("average training time: "<<readable_time(avg_training_time));
            LOG("average training time per block: "<<readable_time(avg_training_time / static_cast<double>(num_blocks)));
            if(it % testing_rate == 0){
                LOG(std::endl<<std::endl<<std::endl);
                LOG("Iteration "<<it<<" MSQ training error (used to update V): "<< ToString<float>(training_error_v[it]));
                LOG("Iteration "<<it<<" expected number of training iterations: "<< ToString<float>(training_iterations[it]));
                LOG("Iteration "<<it<<" MSQ when clustering (used to update U): "<<ToString<float>(training_error_u[it]));
                //LOG("Iteration "<<it<<" err norm when clustering : "<<std::sqrt(k_means_er));
                
                LOG("Iteration "<<it<<" MSQ testing error per testing entry: "<< ToString<float>(testing_error_on_testing_entries[num_tests - 1]));
                LOG("Iteration "<<it<<" expected number of testing iterations: "<< ToString<float>(testing_iterations[num_tests - 1]));
                
                LOG("Iteration "<<it<<" MSQ testing error per testing entry: "<< ToString<float>(testing_error_on_testing_entries[num_tests - 1]));
                LOG("Iteration "<<it<<" expected number of testing iterations: "<< ToString<float>(testing_iterations[num_tests - 1]));
                
                LOG("Finished Iteration "<<it<<"."<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl<<std::endl);
                // LOG("Press Enter to continue.") ;
                // std::cin.ignore();
            }
            

            cpu_isBad<float>(training_error_v + it, (long long int)1, "training_error_v", strPreamble(blank));
            cpu_isBad<float>(training_iterations + it, (long long int)1, "training_iterations", strPreamble(blank));
            cpu_isBad<float>(training_error_u + it, (long long int)1, "training_error_u", strPreamble(blank));
            cpu_isBad<float>(delta_R_ACU_exp + it, (long long int)1, "delta_R_ACU_exp", strPreamble(blank));  

            if(load_full_ACU_from_save){
                append_host_array_to_file(training_error_v + it, 1, "training_error_v", strPreamble(blank));
                append_host_array_to_file(training_iterations + it, 1, "training_iterations", strPreamble(blank));
                if(update_U) append_host_array_to_file(training_error_u + it, 1, "training_error_u", strPreamble(blank));
                if(calc_delta_R_ACU)append_host_array_to_file(delta_R_ACU_exp + it, 1, "delta_R_ACU_exp", strPreamble(blank));
                it++;  
            }else{
                it++;
                save_host_array_to_file(training_error_v, it, "training_error_v", strPreamble(blank));
                save_host_array_to_file(training_iterations, it, "training_iterations", strPreamble(blank));
                if(update_U) save_host_array_to_file(training_error_u, it, "training_error_u", strPreamble(blank));
                if(calc_delta_R_ACU)save_host_array_to_file(delta_R_ACU_exp, it, "delta_R_ACU_exp", strPreamble(blank));            
            }
        }else{
            it++;
        }

    }//end for loop on iterations





    //============================================================================================
    // Assume the rows of R_ACU are the ratings_rows_ACU-means for the training users - what is the error?
    //============================================================================================ 


    if(0){
        float k_means_er = (float)0.0;
        float* errors;
        int* selection;
        if(Conserve_GPU_Mem){
            errors  = (float *)malloc(ratings_rows_training * SIZE_OF(float));
            selection  = (int *)malloc(ratings_rows_training * SIZE_OF(int));
            checkErrors(errors);
            checkErrors(selection);
            cpu_sparse_nearest_row<float>(ratings_rows_ACU, ratings_cols, full_ratingsMtx_host_ACU, 
                                         ratings_rows_training, num_entries_training, 
                                         csr_format_ratingsMtx_userID_host_training, 
                                         coo_format_ratingsMtx_itemID_host_training,
                                         coo_format_ratingsMtx_rating_host_training, 
                                         selection, errors);
            //k_means_er = cpu_sum(ratings_rows_training,  errors);
            //LOG("err norm when clustering : "<<std::sqrt(k_means_er));
            LOG("expected MSQER when clustering : "<<ToString<float>(cpu_expected_value(ratings_rows_training, errors)));
            save_host_array_to_file<float>(errors, ratings_rows_training, "meta_km_errors");
            save_host_array_to_file<int>(selection, ratings_rows_training, "meta_km_selection");
            free(errors);
            free(selection);
        }else{
            checkCudaErrors(cudaMalloc((void**)&errors, ratings_rows_training * SIZE_OF(float)));
            checkCudaErrors(cudaMalloc((void**)&selection, ratings_rows_training * SIZE_OF(int)));
            update_Mem(2 * ratings_rows_training * SIZE_OF(int));
            gpu_sparse_nearest_row<float>(ratings_rows_ACU, ratings_cols, full_ratingsMtx_dev_ACU, 
                                     ratings_rows_training, num_entries_training, 
                                     csr_format_ratingsMtx_userID_dev_training, 
                                     coo_format_ratingsMtx_itemID_dev_training,
                                     coo_format_ratingsMtx_rating_dev_training, 
                                     selection, errors, row_major_ordering);
            k_means_er = gpu_sum(ratings_rows_training,  errors);
            LOG("err norm when clustering : "<<ToString<float>(std::sqrt(k_means_er)));
            LOG("mean sqed err when clustering : "<<ToString<float>(k_means_er / (float)num_entries_training));
            save_device_array_to_file<float>(errors, ratings_rows_training, "meta_km_errors");
            save_device_array_to_file<int>(selection, ratings_rows_training, "meta_km_selection");
            checkCudaErrors(cudaFree(errors));
            checkCudaErrors(cudaFree(selection));
            update_Mem(2 * ratings_rows_training * SIZE_OF(int) * (-1));
        }
    }





    //============================================================================================
    // Destroy
    //============================================================================================
    LOG("Cleaning Up...");
    //free(user_means_training_host);

    if (testing_error_on_training_entries) free(testing_error_on_training_entries);
    if (testing_error_on_testing_entries) free(testing_error_on_testing_entries);
    if (testing_iterations) free(testing_iterations);
    if (training_error_v) free(training_error_v);
    if (training_error_u) free(training_error_u);
    if (training_iterations) free(training_iterations);
    if (delta_R_ACU_exp) free(delta_R_ACU_exp);
    if (R_ACU_abs_max) free(R_ACU_abs_max);
    if (R_ACU_max_sv) free(R_ACU_max_sv);
    if(logarithmic_histogram) free(logarithmic_histogram);
    if(logarithmic_histogram_km) free(logarithmic_histogram_km);

    if (user_means_testing_host) free(user_means_testing_host);
    if (user_var_testing_host) free(user_var_testing_host);

    if (full_ratingsMtx_host_ACU) { free(full_ratingsMtx_host_ACU); }
    if (full_ratingsMtx_dev_ACU) { checkCudaErrors(cudaFree(full_ratingsMtx_dev_ACU)); }

    if (U_ACU_host) free(U_ACU_host);
    if (U_ACU_dev) checkCudaErrors(cudaFree(U_ACU_dev));
    if (V_dev) checkCudaErrors(cudaFree(V_dev));
    

    if(Conserve_GPU_Mem){
        if (V_host) free(V_host);
        if (csr_format_ratingsMtx_userID_host_testing) free(csr_format_ratingsMtx_userID_host_testing);
        if (coo_format_ratingsMtx_itemID_host_testing) free(coo_format_ratingsMtx_itemID_host_testing);
        if (coo_format_ratingsMtx_rating_host_testing) free(coo_format_ratingsMtx_rating_host_testing); 
        if (csr_format_ratingsMtx_userID_host_training) free(csr_format_ratingsMtx_userID_host_training);
        if (coo_format_ratingsMtx_itemID_host_training) free(coo_format_ratingsMtx_itemID_host_training);
        if (coo_format_ratingsMtx_rating_host_training) free(coo_format_ratingsMtx_rating_host_training); 
        if (old_R_ACU) free(old_R_ACU);
    }else{
        if (old_R_ACU) checkCudaErrors(cudaFree(old_R_ACU));
    }
    
    


    update_Mem((batch_size_ACU * std::min(batch_size_ACU, ratings_cols) +batch_size_training * batch_size_training + ratings_cols * std::min(batch_size_ACU, ratings_cols))* static_cast<long long int>(SIZE_OF(float))* (-1));
    

    if (user_means_ACU) { checkCudaErrors(cudaFree(user_means_ACU)); update_Mem(batch_size_ACU * SIZE_OF(float) * (-1)); }
    if (user_var_ACU) { checkCudaErrors(cudaFree(user_var_ACU));   update_Mem(batch_size_ACU * SIZE_OF(float) * (-1)); }


    if (csr_format_ratingsMtx_userID_dev_training) checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_training));
    if (coo_format_ratingsMtx_itemID_dev_training) checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_training));
    if (coo_format_ratingsMtx_rating_dev_training) checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_training));
    
    if (csr_format_ratingsMtx_userID_dev_testing) checkCudaErrors(cudaFree(csr_format_ratingsMtx_userID_dev_testing));
    if (coo_format_ratingsMtx_itemID_dev_testing) checkCudaErrors(cudaFree(coo_format_ratingsMtx_itemID_dev_testing));
    if (coo_format_ratingsMtx_rating_dev_testing) checkCudaErrors(cudaFree(coo_format_ratingsMtx_rating_dev_testing));

 
    if (SV) free(SV);

    update_Mem((ratings_rows_ACU * ratings_cols + /*ratings_rows_training * ratings_cols +*/ num_entries_testing + num_entries_training)* SIZE_OF(float));
    update_Mem(( (ratings_rows_testing + 1)  * static_cast<long long int>(SIZE_OF(int)) + num_entries_testing  * static_cast<long long int>(SIZE_OF(int)) + num_entries_testing  * static_cast<long long int>(SIZE_OF(float))
               + (ratings_rows_training + 1) * static_cast<long long int>(SIZE_OF(int)) + num_entries_training * static_cast<long long int>(SIZE_OF(int)) + num_entries_training * static_cast<long long int>(SIZE_OF(float))
               + (ratings_rows_ACU + 1)       * static_cast<long long int>(SIZE_OF(int)) + num_entries_ACU       * static_cast<long long int>(SIZE_OF(int)) + num_entries_ACU       * static_cast<long long int>(SIZE_OF(float))) * (-1) );
    

    if(Content_Based){ // This means that we use extra knowledge about the user preferences or about the item relationships
        if (csr_format_keyWordMtx_itemID_dev) checkCudaErrors(cudaFree(csr_format_keyWordMtx_itemID_dev));          update_Mem((ratings_cols + 1)      * SIZE_OF(int) * (-1));
        if (coo_format_keyWordMtx_keyWord_dev) checkCudaErrors(cudaFree(coo_format_keyWordMtx_keyWord_dev));          update_Mem(num_entries_keyWord_mtx * SIZE_OF(int) * (-1));
    }

    cublasDestroy          (dn_handle);
    cusolverDnDestroy      (dn_solver_handle);
    cusparseDestroy        (sp_handle);
    cusparseDestroyMatDescr(sp_descr);
    cusolverSpDestroy      (sp_solver_handle);

    checkCudaErrors(cudaDeviceSynchronize());
    gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("program_time: %f\n", program_time);   
    LOG("program_time : "<<readable_time(program_time));

    //if(Debug && memLeft!=devMem)LOG("WARNING POSSIBLE DEVICE MEMORY LEAK");
         
}
