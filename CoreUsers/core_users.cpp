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
#include <cuda_runtime.h>
#include "/home/nesky/REC_SYS/helper_files/helper_cuda.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include "cusolverSp.h"
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
//#include "helper_cusolver.h"
#include "cusolverDn.h"

// Utilities and system includes
#include "/home/nesky/REC_SYS/helper_files/CSVReader.h"
#include "/home/nesky/REC_SYS/helper_files/util.h"
#include "/home/nesky/REC_SYS/helper_files/util_gpu.cuh"
#include "core_users.h"

const char *sSDKname = "Core Users Recommender Systems";

const bool Debug = 1;

bool Content_Based = 1;
bool Conserve_GPU_Mem = 0;

#define update_Mem(new_mem) \
    allocatedMem += static_cast<long long int>(new_mem); \
    memLeft = static_cast<long long int>(devMem) - allocatedMem; \
    ABORT_IF_LESS(memLeft, 0, "Out of Memory"); 
    // if(Debug) LOG(allocatedMem<<" allocated bytes on the device");
    //if(Debug) LOG(memLeft<<" available bytes left on the device");
    //
    //ABORT_IF_LESS(allocatedMem, (long long int)((double)devMem * (double)0.75), "Out of Memory"); \
    // if(Debug) LOG((int)devMem <<" total bytes on the device");\
    // if(Debug) LOG(new_mem<<" change in memory on the device");\






// #define allocate_V() \
//     if (d_S    ) cudaFree(d_S);
//     if (d_S    ) cudaFree(d_S);
//     if (d_S    ) cudaFree(d_S);
//     checkCudaErrors(cudaMalloc((void**)&U_CU,       batch_size_CU       * batch_size_CU                       * sizeof(float)));
//     checkCudaErrors(cudaMalloc((void**)&V,          ratings_cols        * ratings_cols                        * sizeof(float)));
//     checkCudaErrors(cudaMalloc((void**)&U_testing, batch_size_testing * std::max(min_CU_dimension, batch_size_testing) * sizeof(float)));
//     checkCudaErrors(cudaMalloc((void**)&R_testing, batch_size_testing * ratings_cols                        * sizeof(float)));
//     checkCudaErrors(cudaMalloc((void**)&U_testing,  batch_size_testing  * std::max(min_CU_dimension, batch_size_testing)  * sizeof(float)));
//     checkCudaErrors(cudaMalloc((void**)&R_testing,  batch_size_testing  * ratings_cols                        * sizeof(float)));
//     update_Mem(Training_bytes);











int main(int argc, char *argv[])
{
    struct timeval program_start, program_end, training_start, training_end;
    double program_time;
    double training_time;
    gettimeofday(&program_start, NULL);

    long long int allocatedMem = (long long int)0; 
    /*
        int dimension = 4;

        int below_diag_indicies[6] = {0,1,2,3,4,5};
        int whole_indicies[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

        LOG("below_diag_indicies 0 maps to -> whole_indicies "   <<from_below_diag_to_whole(below_diag_indicies[0], dimension) );
        LOG("below_diag_indicies 1 maps to -> whole_indicies "   <<from_below_diag_to_whole(below_diag_indicies[1], dimension) );
        LOG("below_diag_indicies 2 maps to -> whole_indicies "   <<from_below_diag_to_whole(below_diag_indicies[2], dimension) );

        LOG("whole_indicies 0 maps to -> below_diag_indicies "   <<from_whole_to_below_diag(whole_indicies[0], dimension) );
        LOG("whole_indicies 1 maps to -> below_diag_indicies "   <<from_whole_to_below_diag(whole_indicies[1], dimension) );
        LOG("whole_indicies 2 maps to -> below_diag_indicies "   <<from_whole_to_below_diag(whole_indicies[2], dimension) );
        LOG("whole_indicies 3 maps to -> below_diag_indicies "   <<from_whole_to_below_diag(whole_indicies[3], dimension) );
        LOG("whole_indicies 4 maps to -> below_diag_indicies "   <<from_whole_to_below_diag(whole_indicies[4], dimension) );
        LOG("whole_indicies 5 maps to -> below_diag_indicies "   <<from_whole_to_below_diag(whole_indicies[5], dimension) );
        LOG("whole_indicies 6 maps to -> below_diag_indicies "   <<from_whole_to_below_diag(whole_indicies[6], dimension) );
        LOG("whole_indicies 7 maps to -> below_diag_indicies "   <<from_whole_to_below_diag(whole_indicies[7], dimension) );
        LOG("whole_indicies 8 maps to -> below_diag_indicies "   <<from_whole_to_below_diag(whole_indicies[8], dimension) );


        LOG("Press Enter to continue.") ;
        std::cin.ignore();
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





    //============================================================================================
    // Get the ratings data from CSV File
    //============================================================================================


    // Creating an object of CSVWriter

    std::string Dataset_Name;

    
    std::string csv_Ratings_Path;
    std::string csv_keyWords_path;

    long long int temp_num_entries;

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

    LOG("Training using the "<< Dataset_Name <<" dataset");
    LOG("csv_Ratings_Path : "<< csv_Ratings_Path);
    LOG("csv_keyWords_path : "<< csv_keyWords_path <<" dataset");
    LOG("Content_Based : "<< Content_Based<<std::endl);



    CSVReader csv_Ratings(csv_Ratings_Path);
    

    const long long int num_entries = temp_num_entries;
    //const int num_entries = 100000; //for debuging code

    LOG("The dataset has "<<num_entries<<" specified entries.");
    
    int*   coo_format_ratingsMtx_userID_host  = NULL;
    int*   coo_format_ratingsMtx_itemID_host  = NULL;
    float* coo_format_ratingsMtx_rating_host  = NULL;
    coo_format_ratingsMtx_userID_host = (int *)  malloc(num_entries *  sizeof(int)); 
    coo_format_ratingsMtx_itemID_host = (int *)  malloc(num_entries *  sizeof(int)); 
    coo_format_ratingsMtx_rating_host = (float *)malloc(num_entries *  sizeof(float)); 
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
                                num_entries);
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



    if(Debug && 0){
        save_host_arrays_side_by_side_to_file(coo_format_ratingsMtx_userID_host, coo_format_ratingsMtx_itemID_host, 
                                              coo_format_ratingsMtx_rating_host, num_entries, "rows_cols_rating");
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();          
    }
 



    int*   coo_format_ratingsMtx_userID_dev;
    int*   coo_format_ratingsMtx_itemID_dev;
    float* coo_format_ratingsMtx_rating_dev;
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_userID_dev,  num_entries * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev,  num_entries * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev,  num_entries * sizeof(float)));
    update_Mem(2 * num_entries * sizeof(int) + num_entries * sizeof(float));

    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_userID_dev,  coo_format_ratingsMtx_userID_host,  num_entries * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_dev,  coo_format_ratingsMtx_itemID_host,  num_entries * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev,  coo_format_ratingsMtx_rating_host,  num_entries * sizeof(float), cudaMemcpyHostToDevice));

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
    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev, (ratings_rows + 1) * sizeof(int)));
    update_Mem( (ratings_rows + 1) * sizeof(int) );

    cusparse_status = cusparseXcoo2csr(sp_handle, coo_format_ratingsMtx_userID_dev, num_entries, 
                                       ratings_rows, csr_format_ratingsMtx_userID_dev, CUSPARSE_INDEX_BASE_ZERO); 
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 
        fprintf(stdout, "Conversion from COO to CSR format failed\n");
        return 1; 
    } 

    LOG("The sparse data matrix has "<<ratings_rows<<" users and "<<ratings_cols<<" items with "<<num_entries<<" specified entries.");
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

        coo_format_keyWordMtx_itemID_host  = (int *)malloc(num_entries_keyWord_mtx_temp *  sizeof(int)); 
        coo_format_keyWordMtx_keyWord_host = (int *)malloc(num_entries_keyWord_mtx_temp *  sizeof(int)); 
        checkErrors(coo_format_keyWordMtx_itemID_host);
        checkErrors(coo_format_keyWordMtx_keyWord_host);

        num_keyWords_temp  = csv_keyWords.makeContentBasedcooKeyWordMtx(coo_format_keyWordMtx_itemID_host,
                                                                        coo_format_keyWordMtx_keyWord_host,
                                                                        num_entries_keyWord_mtx_temp);
        LOG("num_keyWords : "<<num_keyWords_temp);

        checkCudaErrors(cudaMalloc((void**)&coo_format_keyWordMtx_itemID_dev,   num_entries_keyWord_mtx_temp * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&coo_format_keyWordMtx_keyWord_dev,  num_entries_keyWord_mtx_temp * sizeof(int)));
        update_Mem(2 * num_entries_keyWord_mtx_temp * sizeof(int) );

        checkCudaErrors(cudaMemcpy(coo_format_keyWordMtx_itemID_dev,   coo_format_keyWordMtx_itemID_host,   num_entries_keyWord_mtx_temp * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(coo_format_keyWordMtx_keyWord_dev,  coo_format_keyWordMtx_keyWord_host,  num_entries_keyWord_mtx_temp * sizeof(int), cudaMemcpyHostToDevice));
        free(coo_format_keyWordMtx_itemID_host);
        free(coo_format_keyWordMtx_keyWord_host);
    }
    const long long int num_entries_keyWord_mtx = num_entries_keyWord_mtx_temp;
    const long long int num_keyWords            = num_keyWords_temp;

    int*   csr_format_keyWordMtx_itemID_dev;
    if(Content_Based){
        checkCudaErrors(cudaMalloc((void**)&csr_format_keyWordMtx_itemID_dev, (ratings_cols + 1) * sizeof(int)));
        update_Mem( (ratings_cols + 1) * sizeof(int) );

        cusparse_status = cusparseXcoo2csr(sp_handle, coo_format_keyWordMtx_itemID_dev, num_entries_keyWord_mtx, 
                                           ratings_cols, csr_format_keyWordMtx_itemID_dev, CUSPARSE_INDEX_BASE_ZERO); 
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 
            fprintf(stdout, "Conversion from COO to CSR format failed\n");
            return 1; 
        } 
        cudaFree(coo_format_keyWordMtx_itemID_dev);           update_Mem(num_entries_keyWord_mtx * sizeof(int) * (-1));
    }















    //============================================================================================
    // collect User Means and Variances
    //============================================================================================










    LOG("collect User Means and Variance... ");

    float* user_means;
    checkCudaErrors(cudaMalloc((void**)&user_means,  ratings_rows * sizeof(float)));
    update_Mem(( ratings_rows )* sizeof(float));

    float* user_var;
    checkCudaErrors(cudaMalloc((void**)&user_var,  ratings_rows * sizeof(float)));
    update_Mem((ratings_rows)* sizeof(float));

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
    
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_row_centered_rating_dev, num_entries * sizeof(float)));
    update_Mem( ( num_entries) * sizeof(float) );




    //const float val_when_var_is_zero = (float)3.5774;        // use for MovieLens
    const float val_when_var_is_zero = (float)0.5;        // use for Rent The Runway
    LOG("rating used when the variance of the user's ratings is zero : "<< val_when_var_is_zero);

    center_ratings(user_means, user_var, 
                   ratings_rows, num_entries,
                   csr_format_ratingsMtx_userID_dev,
                   coo_format_ratingsMtx_rating_dev,
                   coo_format_ratingsMtx_row_centered_rating_dev, 
                   val_when_var_is_zero);



    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev, coo_format_ratingsMtx_row_centered_rating_dev,  num_entries *  sizeof(float), cudaMemcpyDeviceToDevice));
    cudaFree(coo_format_ratingsMtx_row_centered_rating_dev);             
    update_Mem((ratings_rows+ 1) * sizeof(int) * (-1));
    
    float range = gpu_range<float>(num_entries,        coo_format_ratingsMtx_rating_dev);


    
    if( Debug && 0){

        //LOG("range_CU = "      <<range_CU) ;
        //LOG("range_testing = " <<range_testing) ;


        // save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev,  ratings_rows + 1, "csr_format_ratingsMtx_userID_dev_post_centering");
        // save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev, num_entries,      "coo_format_ratingsMtx_itemID_dev_post_centering");
        // save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev,  num_entries,      "coo_format_ratingsMtx_rating_dev_post_centering");
        

        // LOG("csr_format_ratingsMtx_userID_dev_testing : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_testing, ratings_rows_testing + 1, 1 );
        // LOG("csr_format_ratingsMtx_userID_dev_CU : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_CU, ratings_rows_CU + 1, 1 );
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }

    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_userID_host,  csr_format_ratingsMtx_userID_dev,  (ratings_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host,  coo_format_ratingsMtx_itemID_dev,  num_entries * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host,  coo_format_ratingsMtx_rating_dev,  num_entries * sizeof(float), cudaMemcpyDeviceToHost));
    


    //============================================================================================
    // split the data into core users and testing data
    //============================================================================================

    

    /*
        In the third method, we first compute the top-N (e.g. N 5 10, 20, 50) 
        most similar neighbors of each user based on the cosine similarities, 
        and then count how many times a user has appeared in other users’ top-N lists.
        Those users who appear most frequently are selected as the information core.
    */







    int top_N = std::min((int)ratings_rows / 10, 50);
    LOG("Compute the top-"<<top_N<<" most similar neighbors of each user based on the cosine similarities.");
    const float probability_CU       = (float)1.0/(float)100.0;
    const float probability_testing  = ((float)1.0 - probability_CU);
    LOG("percentage of users for testing: " <<probability_testing);
    LOG("percentage of users for CU: "      <<(float)1.0 - probability_testing<<std::endl);

    const long long int ratings_rows_CU      = (long long int)(probability_CU * ratings_rows);
    const long long int ratings_rows_testing = ratings_rows - ratings_rows_CU;
    LOG("num testing users : "   <<ratings_rows_testing);
    LOG("num CU users : "        <<ratings_rows_CU);

    if(ratings_rows_CU == 0 || ratings_rows_testing == 0) {
        LOG("One group has no users in it.")
        return 0;
    }

    const long long int num_below_diag = (ratings_rows * (ratings_rows - (long long int)1)) / (long long int)2;

    float* cosine_similarity  = (float *)malloc(num_below_diag * sizeof(float));
    int*   col_index          = (int *)malloc(top_N * ratings_rows * sizeof(int));
    checkErrors(cosine_similarity);
    checkErrors(col_index);



    // cpu_get_cosine_similarity(ratings_rows, num_entries,
    //                           coo_format_ratingsMtx_userID_host,
    //                           coo_format_ratingsMtx_itemID_host,
    //                           coo_format_ratingsMtx_rating_host,
    //                           cosine_similarity);

    get_cosine_similarity_host(ratings_rows, 
                              csr_format_ratingsMtx_userID_dev,
                              coo_format_ratingsMtx_itemID_dev,
                              coo_format_ratingsMtx_rating_dev,
                              cosine_similarity);

    //cpu_set_as_index(col_index, ratings_rows, ratings_rows);
    //gpu_set_as_index_host(col_index, ratings_rows, ratings_rows);



    /*
        we want to know column indicies of the max N elements in each row
        excluding the row index itself
    */
    //cpu_sort_index_by_max(ratings_rows, ratings_rows,  cosine_similarity, col_index);
    cpu_sort_index_by_max<float>(ratings_rows,  cosine_similarity, col_index, top_N);
    free(cosine_similarity);
    if(Debug){
        save_host_mtx_to_file<int>(col_index, static_cast<int>(top_N), static_cast<int>(ratings_rows), "col_index_sorted");
    }

    /*
        for each user index, count how many times that user appears in the top_N + 1 most similar users.
    */
    int* count= (int *)malloc(ratings_rows * sizeof(int));
    checkErrors(count);
    cpu_set_all<int>(count, ratings_rows, 0);

    cpu_count_appearances(top_N, ratings_rows, count, col_index);
    free(col_index);
    if(Debug ){
        save_host_array_to_file<int>(count, (int)ratings_rows, "count");
    }

    /*
        sort the count to find the top_N core users
    */
    int* top_users= (int *)malloc(ratings_rows * sizeof(int));
    checkErrors(top_users);
    cpu_set_as_index<int>(top_users, ratings_rows, 1);


    cpu_sort_index_by_max(1, ratings_rows,  count, top_users);
    if(Debug){
        save_host_array_to_file<int>(top_users, ratings_rows, "top_users");

    }
    /*
        The ratings_rows_CU core users indicies are the indicies 
        in the last ratings_rows_CU entries of top_users 
    */
    cpu_set_all<int>(count, ratings_rows, 1);
    cpu_mark_CU_users(ratings_rows_CU, ratings_rows, top_users, count );
    free(top_users);
    if(Debug ){
        save_host_array_to_file<int>(count, ratings_rows, "top_user_bools");
        //LOG("Press Enter to continue.") ;
        //std::cin.ignore();
    }

    free(coo_format_ratingsMtx_userID_host);
    free(coo_format_ratingsMtx_itemID_host);
    free(coo_format_ratingsMtx_rating_host);




    
    const int num_groups = 2;
    int *group_indicies;
    checkCudaErrors(cudaMalloc((void**)&group_indicies, ratings_rows * sizeof(int)));
    update_Mem( ratings_rows * sizeof(int) );
    
    checkCudaErrors(cudaMemcpy(group_indicies,  count,  ratings_rows * sizeof(int), cudaMemcpyHostToDevice));
    free(count);

    int* group_sizes = NULL;
    group_sizes = (int *)malloc(num_groups * sizeof(int)); 
    checkErrors(group_sizes);  

    count_each_group_from_coo(num_groups, group_indicies, num_entries, coo_format_ratingsMtx_userID_dev, group_sizes);
    const long long int num_entries_CU   = group_sizes[0];
    const long long int num_entries_testing  = group_sizes[1];

    // count_each_group(ratings_rows, group_indicies, group_sizes, num_groups);
    // const long long int ratings_rows_CU  = group_sizes[0];
    // const long long int ratings_rows_testing = group_sizes[1]; 


    LOG("num testing entries : " <<num_entries_testing);
    LOG("num CU entries : "      <<num_entries_CU<<std::endl);
    
    ABORT_IF_NEQ(ratings_rows_testing  + ratings_rows_CU, ratings_rows, "The number of rows does not add up correctly.");
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
    update_Mem( num_entries * sizeof(int) * (-1) );

    int*   csr_format_ratingsMtx_userID_dev_testing;
    int*   coo_format_ratingsMtx_itemID_dev_testing;
    float* coo_format_ratingsMtx_rating_dev_testing;

    int*   csr_format_ratingsMtx_userID_dev_CU;
    int*   coo_format_ratingsMtx_itemID_dev_CU;
    float* coo_format_ratingsMtx_rating_dev_CU;


    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_testing,  (ratings_rows_testing + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_testing,  num_entries_testing        * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing        * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_CU,  (ratings_rows_CU + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_CU,  num_entries_CU        * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_CU,  num_entries_CU        * sizeof(float)));
    update_Mem(  (ratings_rows_testing + 1)  * sizeof(int) + num_entries_testing  * sizeof(int) + num_entries_testing  * sizeof(float)
               + (ratings_rows_CU + 1)       * sizeof(int) + num_entries_CU       * sizeof(int) + num_entries_CU       * sizeof(float)  );
    
    int*   csr_format_ratingsMtx_userID_dev_by_group_host  [num_groups] = { csr_format_ratingsMtx_userID_dev_CU,  csr_format_ratingsMtx_userID_dev_testing  };
    int*   coo_format_ratingsMtx_itemID_dev_by_group_host  [num_groups] = { coo_format_ratingsMtx_itemID_dev_CU,  coo_format_ratingsMtx_itemID_dev_testing  };
    float* coo_format_ratingsMtx_rating_dev_by_group_host  [num_groups] = { coo_format_ratingsMtx_rating_dev_CU,  coo_format_ratingsMtx_rating_dev_testing  };
    int    ratings_rows_by_group_host                      [num_groups] = { ratings_rows_CU                    ,  ratings_rows_testing                      };
    int**   csr_format_ratingsMtx_userID_dev_by_group;
    int**   coo_format_ratingsMtx_itemID_dev_by_group;
    float** coo_format_ratingsMtx_rating_dev_by_group;
    int*    ratings_rows_by_group;

    checkCudaErrors(cudaMalloc((void**)&csr_format_ratingsMtx_userID_dev_by_group,  num_groups*sizeof(int*)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_itemID_dev_by_group,  num_groups*sizeof(int*)));
    checkCudaErrors(cudaMalloc((void**)&coo_format_ratingsMtx_rating_dev_by_group,  num_groups*sizeof(float*)));
    checkCudaErrors(cudaMalloc((void**)&ratings_rows_by_group,                      num_groups*sizeof(int)));
    update_Mem( num_groups * sizeof(int*) * 2 + num_groups * sizeof(float*) + num_groups * sizeof(int) );

    checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_dev_by_group,  csr_format_ratingsMtx_userID_dev_by_group_host,  num_groups * sizeof(int*),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_dev_by_group,  coo_format_ratingsMtx_itemID_dev_by_group_host,  num_groups * sizeof(int*),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_dev_by_group,  coo_format_ratingsMtx_rating_dev_by_group_host,  num_groups * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ratings_rows_by_group,                      ratings_rows_by_group_host,                      num_groups * sizeof(int),    cudaMemcpyHostToDevice));

    gpu_split_data(csr_format_ratingsMtx_userID_dev,
                   coo_format_ratingsMtx_itemID_dev,
                   coo_format_ratingsMtx_rating_dev, 
                   ratings_rows, group_indicies,
                   csr_format_ratingsMtx_userID_dev_by_group,
                   coo_format_ratingsMtx_itemID_dev_by_group,
                   coo_format_ratingsMtx_rating_dev_by_group,
                   ratings_rows_by_group); 
    if(Debug && 0){
        save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_CU,        (int)ratings_rows_CU + 1,       "csr_format_ratingsMtx_userID_dev_CU");
        save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_CU,        (int)num_entries_CU,            "coo_format_ratingsMtx_itemID_dev_CU");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_CU,        (int)num_entries_CU,            "coo_format_ratingsMtx_rating_dev_CU");
        save_device_array_to_file<int>  (csr_format_ratingsMtx_userID_dev_testing,   (int)ratings_rows_testing + 1,  "csr_format_ratingsMtx_userID_dev_testing");
        save_device_array_to_file<int>  (coo_format_ratingsMtx_itemID_dev_testing,   (int)num_entries_testing,       "coo_format_ratingsMtx_itemID_dev_testing");
        save_device_array_to_file<float>(coo_format_ratingsMtx_rating_dev_testing,   (int)num_entries_testing,       "coo_format_ratingsMtx_rating_dev_testing");
        // LOG("csr_format_ratingsMtx_userID_dev_testing : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_testing, ratings_rows_testing + 1, 1 );
        // LOG("csr_format_ratingsMtx_userID_dev_CU : ");
        // print_gpu_mtx_entries<int>(csr_format_ratingsMtx_userID_dev_CU, ratings_rows_CU + 1, 1 );
        // LOG("Press Enter to continue.") ;
        // std::cin.ignore();
    }


    
    cudaFree(csr_format_ratingsMtx_userID_dev);            update_Mem((ratings_rows + 1) * sizeof(int) * (-1));
    cudaFree(coo_format_ratingsMtx_itemID_dev);            update_Mem(num_entries * sizeof(int) * (-1));
    cudaFree(coo_format_ratingsMtx_rating_dev);            update_Mem(num_entries * sizeof(float) * (-1));
    cudaFree(csr_format_ratingsMtx_userID_dev_by_group);   update_Mem(num_groups * sizeof(int*) * (-1));
    cudaFree(coo_format_ratingsMtx_itemID_dev_by_group);   update_Mem(num_groups * sizeof(int*) * (-1));
    cudaFree(coo_format_ratingsMtx_rating_dev_by_group);   update_Mem(num_groups * sizeof(float*) * (-1));
    cudaFree(ratings_rows_by_group);                       update_Mem(num_groups * sizeof(int) * (-1));
    cudaFree(group_indicies);                              update_Mem(num_groups * sizeof(int)* (-1));
    
    free(group_sizes);








    //============================================================================================
    // Fill CU Ratings Matrix
    //============================================================================================


    LOG("fill CU matrix... ");



    bool row_major_ordering = true;

    float * full_ratingsMtx_dev_CU;
    float * full_ratingsMtx_host_CU = NULL;

    const long long int CU_mtx_size = (long long int)ratings_rows_CU * (long long int)ratings_cols;
    const long long int CU_mtx_size_bytes = (long long int)ratings_rows_CU * (long long int)ratings_cols * (long long int)sizeof(float);
    LOG("Will need "<<CU_mtx_size<< " floats for the CU mtx.") ;
    LOG("Will need "<<CU_mtx_size_bytes<< " bytes for the CU mtx.") ;
    if(allocatedMem + CU_mtx_size_bytes > (long long int)((double)devMem * (double)0.75)){
        LOG("Conserving Memory Now");
        Conserve_GPU_Mem = 1;
    }
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();


    int*   csr_format_ratingsMtx_userID_host_CU = NULL;
    int*   coo_format_ratingsMtx_itemID_host_CU = NULL;
    float* coo_format_ratingsMtx_rating_host_CU = NULL;

    if(Conserve_GPU_Mem){
        csr_format_ratingsMtx_userID_host_CU  = (int *)  malloc((ratings_rows_CU + 1) * sizeof(int)  );
        coo_format_ratingsMtx_itemID_host_CU  = (int *)  malloc(num_entries_CU        * sizeof(int)  );
        coo_format_ratingsMtx_rating_host_CU  = (float *)malloc(num_entries_CU        * sizeof(float));


        checkErrors(csr_format_ratingsMtx_userID_host_CU);
        checkErrors(coo_format_ratingsMtx_itemID_host_CU);
        checkErrors(coo_format_ratingsMtx_rating_host_CU);
        checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host_CU,  csr_format_ratingsMtx_userID_dev_CU,  (ratings_rows_CU + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host_CU,  coo_format_ratingsMtx_itemID_dev_CU,  num_entries_CU        * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host_CU,  coo_format_ratingsMtx_rating_dev_CU,  num_entries_CU        * sizeof(float), cudaMemcpyDeviceToHost));

        full_ratingsMtx_host_CU = (float *)malloc(CU_mtx_size_bytes);
        checkErrors(full_ratingsMtx_host_CU);

        cpu_set_all<float>(full_ratingsMtx_host_CU, CU_mtx_size, (float)0.0);
        
        cpu_fill_training_mtx((long long int)ratings_rows_CU, (long long int)ratings_cols, (long long int)num_entries_CU, 
                              row_major_ordering,  
                              csr_format_ratingsMtx_userID_host_CU,
                              coo_format_ratingsMtx_itemID_host_CU,
                              coo_format_ratingsMtx_rating_host_CU,
                              full_ratingsMtx_host_CU);

        free(csr_format_ratingsMtx_userID_host_CU);
        free(coo_format_ratingsMtx_itemID_host_CU);
        free(coo_format_ratingsMtx_rating_host_CU);       

        
        cpu_shuffle_mtx_rows_or_cols(dn_handle, ratings_rows_CU, ratings_cols, 
                                     row_major_ordering, full_ratingsMtx_host_CU, 1);

        LOG("full_ratingsMtx_host_CU filled and shuffled") ;
    }else{
        checkCudaErrors(cudaMalloc((void**)&full_ratingsMtx_dev_CU, ratings_rows_CU * ratings_cols * sizeof(float)));
        update_Mem(ratings_rows_CU * ratings_cols * sizeof(float));


        gpu_set_all<float>(full_ratingsMtx_dev_CU, ratings_rows_CU * ratings_cols, (float)0.0);
        
        gpu_fill_training_mtx(ratings_rows_CU, ratings_cols, row_major_ordering,
                              csr_format_ratingsMtx_userID_dev_CU,
                              coo_format_ratingsMtx_itemID_dev_CU,
                              coo_format_ratingsMtx_rating_dev_CU,
                              full_ratingsMtx_dev_CU);
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
    cudaFree(csr_format_ratingsMtx_userID_dev_CU);         update_Mem((ratings_rows_CU + 1) * sizeof(int)   * (-1));
    cudaFree(coo_format_ratingsMtx_itemID_dev_CU);         update_Mem(num_entries_CU        * sizeof(int)   * (-1));
    cudaFree(coo_format_ratingsMtx_rating_dev_CU);         update_Mem(num_entries_CU        * sizeof(float) * (-1));

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



    //the stored means are useless now
    cudaFree(user_means);                        update_Mem(ratings_rows * sizeof(float) * (-1));
    cudaFree(user_var);                          update_Mem(ratings_rows * sizeof(float) * (-1));

    //============================================================================================
    // Conserve Memory
    //============================================================================================

    // int *   csr_format_ratingsMtx_userID_host_testing  = NULL;
    // int *   coo_format_ratingsMtx_itemID_host_testing = NULL;
    // float * coo_format_ratingsMtx_rating_host_testing  = NULL;

    // csr_format_ratingsMtx_userID_host_testing  = (int *)  malloc(ratings_rows_testing *  sizeof(int)); 
    // coo_format_ratingsMtx_itemID_host_testing = (int *)  malloc(num_entries_testing  *  sizeof(int)); 
    // coo_format_ratingsMtx_rating_host_testing  = (float *)malloc(num_entries_testing  *  sizeof(float));
    // checkErrors(csr_format_ratingsMtx_userID_host_testing);
    // checkErrors(coo_format_ratingsMtx_itemID_host_testing);
    // checkErrors(coo_format_ratingsMtx_rating_host_testing); 

    // checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host_testing,  csr_format_ratingsMtx_userID_dev_testing,  ratings_rows_testing *  sizeof(int),   cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host_testing, coo_format_ratingsMtx_itemID_dev_testing, num_entries_testing  *  sizeof(int),   cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host_testing,  coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing  *  sizeof(float), cudaMemcpyDeviceToHost));

    // cudaFree(csr_format_ratingsMtx_userID_dev_testing);
    // cudaFree(coo_format_ratingsMtx_itemID_dev_testing);
    // cudaFree(coo_format_ratingsMtx_rating_dev_testing);
    
    // int *   csr_format_ratingsMtx_userID_host_testing  = NULL;
    // int *   coo_format_ratingsMtx_itemID_host_testing = NULL;
    // float * coo_format_ratingsMtx_rating_host_testing  = NULL;

    // csr_format_ratingsMtx_userID_host_testing  = (int *)  malloc(ratings_rows_testing *  sizeof(int)); 
    // coo_format_ratingsMtx_itemID_host_testing = (int *)  malloc(num_entries_testing  *  sizeof(int)); 
    // coo_format_ratingsMtx_rating_host_testing  = (float *)malloc(num_entries_testing  *  sizeof(float)); 
    // checkErrors(csr_format_ratingsMtx_userID_host_testing);
    // checkErrors(coo_format_ratingsMtx_itemID_host_testing);
    // checkErrors(coo_format_ratingsMtx_rating_host_testing); 

    // checkCudaErrors(cudaMemcpy(csr_format_ratingsMtx_userID_host_testing,  csr_format_ratingsMtx_userID_dev_testing,  ratings_rows_testing *  sizeof(int),   cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_itemID_host_testing, coo_format_ratingsMtx_itemID_dev_testing, num_entries_testing  *  sizeof(int),   cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(coo_format_ratingsMtx_rating_host_testing,  coo_format_ratingsMtx_rating_dev_testing,  num_entries_testing  *  sizeof(float), cudaMemcpyDeviceToHost));

    // cudaFree(csr_format_ratingsMtx_userID_dev_testing);
    // cudaFree(coo_format_ratingsMtx_itemID_dev_testing);
    // cudaFree(coo_format_ratingsMtx_rating_dev_testing);


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

    float       training_rate           = (float)0.1;      //use for rent the runway
    const float regularization_constant = (float)0.01;         //use for rent the runway

    const float testing_fraction        = 0.25; //percent of known entries used for testing
    bool        compress                = false;

    const int num_batches    = 100;
    const int testing_rate   = 1;

    LOG("training_rate : "          <<training_rate);
    LOG("regularization_constant : "<<regularization_constant);
    LOG("testing_fraction : "       <<testing_fraction);
    LOG("compress : "               <<compress);
    LOG("num_batches : "            <<num_batches);
    LOG("testing_rate : "           <<testing_rate);

    float * testing_error = NULL;
    testing_error = (float *)malloc(num_batches * sizeof(float)); 
    checkErrors(testing_error);

    const long long int batch_size_testing  = ratings_rows_testing / num_batches;
    LOG(std::endl);
    LOG("batch_size_testing : " <<batch_size_testing);
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();


    long long int num_latent_factors = (long long int)((float)ratings_rows_CU * (float)0.95);
    const float percent              = (float)0.95;


    float * U_CU;       // U_CU is ratings_rows_CU * ratings_rows_CU
    float * U_testing;
    float * V;          // V_CU is ratings_cols * ratings_cols
    float * R_testing;
    if(Debug && 0){
        // const int batch_size_CU = 3;
        // const int ratings_cols = 2;
        // /*       | 1 2  |
        //  *   A = | 4 5  |
        //  *       | 2 1  |
        //  */
        // float A[batch_size_CU*ratings_cols] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
        // cudaMemcpy(full_ratingsMtx_dev_CU, A, sizeof(float)*batch_size_CU*ratings_cols, cudaMemcpyHostToDevice);
    }

    long long int min_CU_dimension = std::min(ratings_rows_CU, ratings_cols);
    bool temp = Conserve_GPU_Mem;
    const long long int Training_bytes = (batch_size_testing  * std::max(min_CU_dimension, batch_size_testing) + 
                                          ratings_cols        * min_CU_dimension +
                                          ratings_cols        * batch_size_testing)* (long long int)sizeof(float) ;
    if(allocatedMem + Training_bytes > (long long int)((double)devMem * (double)0.75)) 
    {
        Conserve_GPU_Mem = 1;
    };


    if(!temp && Conserve_GPU_Mem){
        LOG("Conserving Memory Now");
        //put the CU ratings mtx on the CPU;
        full_ratingsMtx_host_CU = (float *)malloc(CU_mtx_size_bytes);
        checkErrors(full_ratingsMtx_host_CU);
        checkCudaErrors(cudaMemcpy(full_ratingsMtx_host_CU, full_ratingsMtx_dev_CU, CU_mtx_size_bytes, cudaMemcpyDeviceToHost));
        cudaFree(full_ratingsMtx_dev_CU);
        update_Mem((float)(-1.0) * CU_mtx_size_bytes );
    };
    if(Conserve_GPU_Mem){
        // full_ratingsMtx_dev_CU hasn't been allocated yet
        checkCudaErrors(cudaMalloc((void**)&full_ratingsMtx_dev_CU, ratings_rows_CU * ratings_cols * sizeof(float)));
        checkCudaErrors(cudaMemcpy(full_ratingsMtx_dev_CU, full_ratingsMtx_host_CU, ratings_rows_CU * ratings_cols * sizeof(float), cudaMemcpyHostToDevice));
        update_Mem(ratings_rows_CU * ratings_cols* sizeof(float) );
    }
    
    
    // LOG(ratings_cols * ratings_cols * sizeof(float)) ;
    checkCudaErrors(cudaMalloc((void**)&U_CU,       ratings_rows_CU     * ratings_rows_CU            * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&V,          ratings_cols        * min_CU_dimension           * sizeof(float)));
    // cudaDeviceSynchronize();
    // LOG("Press Enter to continue.") ;
    // std::cin.ignore();  


    checkCudaErrors(cudaMalloc((void**)&U_testing,  batch_size_testing  * std::max(min_CU_dimension, batch_size_testing)  * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&R_testing,  batch_size_testing  * ratings_cols                                    * sizeof(float)));
    update_Mem(Training_bytes);


    //============================================================================================
    // Begin Testing
    //============================================================================================  
    LOG(std::endl<<std::endl<<"                              Begin Testing..."<<std::endl); 
    gettimeofday(&training_start, NULL);
    int num_tests = 0;
    float testing_error_temp = (float)0.0;
    long long int total_testing_nnz = (long long int)0;
    int count_tests = 0;
    if(row_major_ordering){
        //rember that ratings_CU is stored in row major ordering
        swap_ordering<float>(ratings_rows_CU, ratings_cols, full_ratingsMtx_dev_CU, row_major_ordering);
    }
    //============================================================================================
    // Find U_CU, V such that U_CU * V^T ~ R_CU 
    //============================================================================================  
    
    gpu_orthogonal_decomp<float>(dn_handle, dn_solver_handle,
                                ratings_rows_CU, ratings_cols, 
                                &num_latent_factors, percent,
                                full_ratingsMtx_dev_CU, U_CU, V);

    /*
        At this point U_CU is batch_size_CU by batch_size_CU in memory stored in column major
        ordering and V is ratings_cols by batch_size_CU stored in column major ordering

        There is no extra work to compress U_CU into batch_size_CU by num_latent_factors, or
        to compress V into ratings_cols by num_latent_factors, just take the first num_latent_factors
        columns of each matrix
    */
    LOG("num_latent_factors = "<< num_latent_factors);
    if(compress){
        //ABORT_IF_NEQ(0, 1, "Not Yet Supported");
    }

    for(int batch = 0; batch < num_batches; batch ++){
        if( print_error){
            LOG(std::endl<<"                                       ~BATCH "<<batch<<"~"<<std::endl);
        }

        if(Debug){

        };

        //============================================================================================
        // Compute  R_testing * V = U_testing
        // Compute  Error = R_testing -  U_testing * V^T  <-- sparse
        //============================================================================================ 

        long long int first_row_in_batch_testing  = (batch_size_testing * (long long int)batch) /* % ratings_rows_testing*/;
        if(first_row_in_batch_testing + batch_size_testing > ratings_rows_testing) {
            //LOG("SKIPPING test iteration "<<it<<", batch "<<batch);
            //break;
        }else{
            LOG("      ~~~ TESTING ~~~ "); 
            if(Debug){
                
                LOG("first_row_in_batch_testing : "<<first_row_in_batch_testing<< " ( / "<<ratings_rows_testing<<" )");
                LOG("batch_size_testing : "<<batch_size_testing);
                LOG("( next first_row_in_batch_testing : "<<first_row_in_batch_testing + batch_size_testing<<" )");
            };
            int* csr_format_ratingsMtx_userID_dev_testing_batch = csr_format_ratingsMtx_userID_dev_testing +  first_row_in_batch_testing;
            
            long long int nnz_testing = gpu_get_num_entries_in_rows(0, batch_size_testing - 1, csr_format_ratingsMtx_userID_dev_testing_batch);
            ABORT_IF_LESS(nnz_testing, 1, "nnz < 1");
            total_testing_nnz += nnz_testing;
            long long int first_coo_ind_testing = gpu_get_first_coo_index(0, csr_format_ratingsMtx_userID_dev_testing_batch);
            
            float* coo_testing_errors;
            float* testing_entries;
            checkCudaErrors(cudaMalloc((void**)&coo_testing_errors, nnz_testing * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&testing_entries, nnz_testing * sizeof(float)));

            if(Debug){
                
                LOG("first_coo_ind in this TESTING batch : "<<first_coo_ind_testing<< " ( / "<<num_entries_testing<<" )");
                LOG("nnz in this TESTING batch : "<<nnz_testing);
                LOG("( nest first_coo_ind in TESTING batch : "<<first_coo_ind_testing+ nnz_testing<<" )");
                // save_device_mtx_to_file<float>(R, batch_size_testing, ratings_cols, "error", false);
                // LOG("Press Enter to continue.") ;
                // std::cin.ignore();
            }
            //============================================================================================
            // Compute  R_testing * V = U_testing
            // Compute  Error = R_testing -  U_testing * V^T  <-- sparse
            //============================================================================================ 
            // gpu_R_error<float>(dn_handle, sp_handle, sp_descr,
            //                    batch_size_testing, batch_size_CU, num_latent_factors, ratings_cols,
            //                    nnz_testing, first_coo_ind_testing, compress, 
            //                    testing_entries, coo_testing_errors, testing_fraction,
            //                    coo_format_ratingsMtx_rating_dev_testing + first_coo_ind_testing, 
            //                    csr_format_ratingsMtx_userID_dev_testing_batch, 
            //                    coo_format_ratingsMtx_itemID_dev_testing + first_coo_ind_testing,
            //                    V, U_testing, R_testing, time(0), "testing", (float)0.1, (float)0.01);

            gpu_R_error_testing<float>(dn_handle, sp_handle, sp_descr,
                               batch_size_testing, ratings_rows_CU, num_latent_factors, ratings_cols,
                               nnz_testing, first_coo_ind_testing, compress, 
                               testing_entries, coo_testing_errors, testing_fraction,
                               coo_format_ratingsMtx_rating_dev_testing + first_coo_ind_testing, 
                               csr_format_ratingsMtx_userID_dev_testing_batch, 
                               coo_format_ratingsMtx_itemID_dev_testing + first_coo_ind_testing,
                               V, U_testing, R_testing, time(0), training_rate, regularization_constant);

            testing_error_temp += gpu_sum_of_squares<float>(nnz_testing, coo_testing_errors);
            
            cudaFree(coo_testing_errors);
            cudaFree(testing_entries);  

            count_tests +=1;


            float nnz_ = (float)total_testing_nnz * testing_fraction;
            long long int num_batches_ = (long long int)((float)(batch_size_testing * num_batches) * testing_fraction);
            LOG("testing error : "<< testing_error_temp / nnz_ ); 
            testing_error[num_tests] = testing_error_temp / nnz_ ;
            //testing_error[num_tests] = testing_error_temp / (float)(nnz_ * 2.0);
            save_device_array_to_file<float>(testing_error, (int)num_tests, "testing_error");
            num_tests += 1;
            total_testing_nnz = (long long int)0;

            
        }


    }//end for loop on batches


    LOG("      ~~~ DONE TESTING ~~~ "<<std::endl); 


    //save_device_array_to_file<float>(testing_error, (num_iterations / testing_rate), "testing_error");

    
    cudaDeviceSynchronize();
    gettimeofday(&training_end, NULL);
    training_time = (training_end.tv_sec * 1000 +(training_end.tv_usec/1000.0))-(training_start.tv_sec * 1000 +(training_start.tv_usec/1000.0));  
    LOG("training_time : "<<training_time<<"ms");
    //============================================================================================
    // Destroy
    //============================================================================================
    LOG("Cleaning Up...");
    //free(user_means_testing_host);
    free(testing_error);
    if (full_ratingsMtx_host_CU  ) { free(full_ratingsMtx_host_CU); }


    cudaFree(U_CU);
    cudaFree(U_testing);
    cudaFree(V);
    cudaFree(R_testing);
    update_Mem((ratings_rows_CU * std::min(ratings_rows_CU, ratings_cols)  + ratings_cols * std::min(ratings_rows_CU, ratings_cols))* static_cast<long long int>(sizeof(float))* (-1));
    
    
    
    
    cudaFree(full_ratingsMtx_dev_CU);
    
    cudaFree(csr_format_ratingsMtx_userID_dev_testing);
    cudaFree(coo_format_ratingsMtx_itemID_dev_testing);
    cudaFree(coo_format_ratingsMtx_rating_dev_testing);
    

    update_Mem((ratings_rows_CU * ratings_cols + /*ratings_rows_testing * ratings_cols +*/ num_entries_testing )* sizeof(float));
    update_Mem(( (ratings_rows_testing + 1)  * static_cast<long long int>(sizeof(int)) + num_entries_testing  * static_cast<long long int>(sizeof(int)) + num_entries_testing  * static_cast<long long int>(sizeof(float))
               + (ratings_rows_CU + 1)       * static_cast<long long int>(sizeof(int)) + num_entries_CU       * static_cast<long long int>(sizeof(int)) + num_entries_CU       * static_cast<long long int>(sizeof(float))) * (-1) );
    

    if(Content_Based){
        cudaFree(csr_format_keyWordMtx_itemID_dev);          update_Mem((ratings_cols + 1)      * sizeof(int) * (-1));
        cudaFree(coo_format_keyWordMtx_keyWord_dev);          update_Mem(num_entries_keyWord_mtx * sizeof(int) * (-1));
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
    LOG("program_time : "<<program_time<<"ms");

    //if(Debug && memLeft!=devMem)LOG("WARNING POSSIBLE DEVICE MEMORY LEAK");
         
}
