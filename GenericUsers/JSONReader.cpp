#include <iostream>
#include <map>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <sys/time.h>
#include <boost/algorithm/string.hpp>
#include "JSONReader.h"
#include "util.h"




int JSONReader::num_rows()
{
	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
	std::cout<<"calling JSONReader::num_rows()"<<std::endl;


	std::ifstream file(fileName.c_str());

	int mycount = std::count(std::istreambuf_iterator<char>(file), 
 	std::istreambuf_iterator<char>(), '\n');
	std::cout<<"There are "<<mycount<<" lines in "<<fileName.c_str()<<std::endl;
	  

	// Close the File
	file.close();

 	std::cout<<"finished call to JSONReader::num_rows()"<<std::endl;
 	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    printf("getData runtime: %f\n", program_time); 
	
	return mycount;
	
}

int JSONReader::num_cols()
{
	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
	std::cout<<"calling JSONReader::num_cols()"<<std::endl;


	std::ifstream file(fileName.c_str());
	std::string line = "";
	getline(file, line);

	std::vector<std::string> vec;
	boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
	int cols = vec.size();
	
	// Close the File
	file.close();

 	std::cout<<"finished call to JSONReader::num_cols()"<<std::endl;
 	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    printf("getData runtime: %f\n", program_time); 
	
	return cols;
	
}

long long int JSONReader::num_entries()
{  
	// return the number of coo entries in a JSON File with multiple mtx entries per JSON row
  	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
  	LOG("JSONReader::num_entries() on "<<fileName.c_str());


	std::ifstream file(fileName.c_str());

	bool first = 1;

	std::string line = "";
	// Iterate through each line and split the content using delimeter
	long long int num_entries = 0;
	while (getline(file, line))
	{
	    //std::cout<<line<<std::endl;
	    std::vector<std::string> temp;
	    boost::algorithm::split(temp, line, boost::is_any_of(delimeter));
	    std::vector<std::string> vec;
	    boost::algorithm::split(vec, temp[2], boost::is_any_of("|"));
	    long long int num_keyWords = (long long int)vec.size();

	    if(first){
	        first = 0;
	        // userID, movieID, rating, timestamp
	    }else{
	    	// for(int i = 0; i<num_keyWords; i++){
	    	// 	LOG(vec[i].c_str());
	    	// }


	        num_entries += num_keyWords;
	        // LOG("num_entries : "<<num_entries);
	        // LOG("Press Enter to continue.") ;
	        // std::cin.ignore();
	    };

	    
	}

    

	// Close the File
	file.close();

  
	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("getData runtime: %f\n", program_time); 
  	LOG("finished call to JSONReader::num_entries() in "<<program_time<< "ms");
  	LOG("num_entries : "<<num_entries);
  	//std::cout<<std::endl;

  	return num_entries;
}


 
/*
* Parses through csv file line by line and returns the data
* in vector of vector of strings.
*/
std::vector<std::vector<std::string> > JSONReader::getData()
{
	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
	std::cout<<"calling JSONReader::getData()"<<std::endl;


	std::ifstream file(fileName.c_str());

	std::vector<std::vector<std::string> > dataList;
 
	std::string line = "";
	// Iterate through each line and split the content using delimeter
	while (getline(file, line))
	{
		// std::cout<<line<<std::endl;
		// std::cout << "Press Enter to continue." ;
  		// std::cin.ignore();
		std::vector<std::string> vec;
		boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
		dataList.push_back(vec);
	}
	// Close the File
	file.close();

 	std::cout<<"finished call to JSONReader::getData()"<<std::endl;
 	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    printf("getData runtime: %f\n", program_time); 
	
	return dataList;
}







/*
* Parses through csv file line by line and returns the data
* in array.
*/
void JSONReader::getData(float* data, const int rows, const int cols)
{
	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
	std::cout<<"JSONReader::getData() on "<<fileName.c_str()<<std::endl;


	std::ifstream file(fileName.c_str());

	bool first = 1;

	std::string line = "";
	// Iterate through each line and split the content using delimeter
	int row = 0;
	while (getline(file, line))
	{
		//std::cout<<line<<std::endl;
		std::vector<std::string> vec;
		boost::algorithm::split(vec, line, boost::is_any_of(delimeter));

		if(first){
			first = 0;
			// userID, movieID, rating, timestamp
		}else{
			for(int col = 0; col <cols; col++)
	        {
	            data[row + col * rows] = ::atof(vec[col].c_str());
	            // std::cout<<data[row + col * rows]<< " , ";

	            //std::cout << "column ["<<col<<"] :" <<vec[col]<<std::endl;
	            //std::cout<<vec[col]<< " , ";
	        }
	        // std::cout<<std::endl;

	        row++;
	    	// std::cout << "Press Enter to continue." ;
 	 		// std::cin.ignore();
		};

		
	}

	  

	// Close the File
	file.close();

 	
 	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("getData runtime: %f\n", program_time); 
	std::cout<<"finished call to JSONReader::getData() in "<<program_time<< "ms"<<std::endl<<std::endl;
	//std::cout<<std::endl;
}


/*
* Parses through csv file line by line and returns the data
* in array.
*/
void JSONReader::getData(int* coo_format_ratingsMtx_userID_host,
					    int* coo_format_ratingsMtx_movieID_host,
					    float* coo_format_ratingsMtx_rating_host,
					    int num_entries)
{
	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
	LOG("JSONReader::getData() on "<<fileName.c_str());


	std::ifstream file(fileName.c_str());

	bool first = 1;

	std::string line = "";
	// Iterate through each line and split the content using delimeter
	int row = 0;
	while (getline(file, line) && row < num_entries)
	{
		//std::cout<<line<<std::endl;
		std::vector<std::string> vec;
		boost::algorithm::split(vec, line, boost::is_any_of(delimeter));

		if(first){
			first = 0;
			// userID, movieID, rating, timestamp
		}else{
            coo_format_ratingsMtx_userID_host[row] = (int)(::atof(vec[0].c_str())) - 1; /*make the user index start at zero*/
            coo_format_ratingsMtx_movieID_host[row] = (int)(::atof(vec[1].c_str())) - 1;/*make the movie index start at zero*/
            coo_format_ratingsMtx_rating_host[row] = ::atof(vec[2].c_str());
            // std::cout<<data[row + col * rows]<< " , ";

            //std::cout << "column ["<<col<<"] :" <<vec[col]<<std::endl;
            //std::cout<<vec[col]<< " , ";
	        // std::cout<<std::endl;

	        row++;
	    	// std::cout << "Press Enter to continue." ;
 	 		// std::cin.ignore();
		};

		
	}

	  

	// Close the File
	file.close();

 	
 	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("getData runtime: %f\n", program_time); 
	LOG("finished call to JSONReader::getData() in "<<program_time<< "ms");
	//std::cout<<std::endl;
}



/*
* Parses through csv file line by line and returns the data
* in array.
*/
long long int JSONReader::makeContentBasedcooKeyWordMtx(int* coo_format_keyWordMtx_movieID_host,
													  int* coo_format_keyWordMtx_keyWord_host,
													  const long long int num_entries)
{
	bool Debug = false;
	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
	LOG("JSONReader::makeContentBasedcrsKeyWordMtx() on "<<fileName.c_str());


	std::map<std::string, long long int> keyWords;

	long long int total_keyWords = (long long int)0;


	std::ifstream file(fileName.c_str());
	std::string line = "";
	bool first = 1;
	int row = 0;
	int coo_entry = 0;
	// Iterate through each line and split the content using delimeter
	while (getline(file, line))
	{
	    std::vector<std::string> temp;
	    boost::algorithm::split(temp, line, boost::is_any_of(delimeter));
	    std::vector<std::string> vec;
	    boost::algorithm::split(vec, temp[2], boost::is_any_of("|"));
		long long int num_keyWords_in_line = vec.size();
		if(Debug) LOG("num_keyWords_in_line : "<<num_keyWords_in_line);

		if(first){
			first = 0;
			// movieID, movie name, keyword list
		}else{
			if(Debug) LOG("movieiD : "<<row);
			for(int i = 0; i< num_keyWords_in_line; i++){
				if(Debug) LOG("coo_entry : "<<coo_entry);
				coo_format_keyWordMtx_movieID_host[coo_entry] = row;

				if( keyWords.find(vec[i].c_str()) == keyWords.end() ){
					if(Debug) LOG(total_keyWords<<"th new word : "<<vec[i].c_str());
					coo_format_keyWordMtx_keyWord_host[coo_entry] = total_keyWords;
					keyWords.insert ( std::pair<std::string,long long int>(vec[i].c_str() ,total_keyWords) );
					total_keyWords++;
				}else{
					int col = keyWords.find(vec[i].c_str()) -> second;
					if(Debug) LOG(col<<"th word, not new : "<<vec[i].c_str());
					coo_format_keyWordMtx_keyWord_host[coo_entry] = col;
				}
				coo_entry++;

			}
			if(Debug){
				LOG("Press Enter to continue.") ;
 	 			std::cin.ignore();					
			}

	        row++;

		};

		
	}

	  

	// Close the File
	file.close();

 	
 	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("getData runtime: %f\n", program_time); 
	LOG("finished call to JSONReader::makeContentBasedcrsKeyWordMtx() in "<<program_time<< "ms");
	//std::cout<<std::endl;

	return total_keyWords;
}














 