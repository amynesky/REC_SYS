#include <iostream>
#include <map>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <sys/time.h>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "CSVReader.h"
#include "util.h"




int CSVReader::num_rows()
{
	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
	std::cout<<"calling CSVReader::num_rows()"<<std::endl;

	int mycount;
	try{
		std::ifstream file(fileName.c_str());

		mycount = std::count(std::istreambuf_iterator<char>(file), 
	 	std::istreambuf_iterator<char>(), '\n');
		std::cout<<"There are "<<mycount<<" lines in "<<fileName.c_str()<<std::endl;
		  

		// Close the File
		file.close();
	}catch(const std::ifstream::failure& e){
		std::cout<<"CSVReader::num_rows() failure."<<std::endl;
	}

 	std::cout<<"finished call to CSVReader::num_rows()"<<std::endl;
 	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    printf("getData runtime: %f\n", program_time); 
	
	return mycount;
	
}

int CSVReader::num_cols()
{
	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
	std::cout<<"calling CSVReader::num_cols()"<<std::endl;

	int cols;
	try{
		std::ifstream file(fileName.c_str());
		std::string line = "";
		getline(file, line);

		std::vector<std::string> vec;
		boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
		cols = vec.size();
		
		// Close the File
		file.close();
	}catch(const std::ifstream::failure& e){
		std::cout<<"CSVReader::num_cols() failure."<<std::endl;
	}
 	std::cout<<"finished call to CSVReader::num_cols()"<<std::endl;
 	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    printf("getData runtime: %f\n", program_time); 
	
	return cols;
	
}

long long int CSVReader::num_entries()
{  
	// return the number of coo entries in a CSV File with multiple mtx entries per CSV row
  	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
  	LOG("CSVReader::num_entries() on "<<fileName.c_str());

  	long long int num_entries = (long long int)0;
  	try{
		std::ifstream file(fileName.c_str());

		bool first = 1;

		std::string line = "";
		// Iterate through each line and split the content using delimeter
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
		        // userID, itemID, rating, timestamp
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
	}catch(const std::ifstream::failure& e){
		std::cout<<"CSVReader::num_entries() failure."<<std::endl;
	}
  
	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("getData runtime: %f\n", program_time); 
  	LOG("finished call to CSVReader::num_entries() in "<<program_time<< "ms");
  	LOG("num_entries : "<<num_entries);
  	//std::cout<<std::endl;

  	return num_entries;
}


 
/*
* Parses through csv file line by line and returns the data
* in vector of vector of strings.
*/
std::vector<std::vector<std::string> > CSVReader::getData()
{
	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
	std::cout<<"calling CSVReader::getData()"<<std::endl;

	std::vector<std::vector<std::string> > dataList;
	try{
		std::ifstream file(fileName.c_str());

	 
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
	}catch(const std::ifstream::failure& e){
		std::cout<<"CSVReader::getData() failure."<<std::endl;
	}

 	std::cout<<"finished call to CSVReader::getData()"<<std::endl;
 	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    printf("getData runtime: %f\n", program_time); 
	
	return dataList;
}







/*
* Parses through csv file line by line and returns the data
* in array.
*/
void CSVReader::getData(float* data, const int rows, const int cols)
{
	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
	std::cout<<"CSVReader::getData() on "<<fileName.c_str()<<std::endl;

	try{
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
				// userID, itemID, rating, timestamp
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
	}catch(const std::ifstream::failure& e){
		std::cout<<"CSVReader::getData() failure."<<std::endl;
	}
 	
 	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("getData runtime: %f\n", program_time); 
	std::cout<<"finished call to CSVReader::getData() in "<<program_time<< "ms"<<std::endl<<std::endl;
	//std::cout<<std::endl;
}


/*
* Parses through csv file line by line and returns the data
* in array.
*/
void CSVReader::getData(int* coo_format_ratingsMtx_userID_host,
					    int* coo_format_ratingsMtx_itemID_host,
					    float* coo_format_ratingsMtx_rating_host,
					    int num_entries)
{
	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
	LOG("CSVReader::getData() on "<<fileName.c_str());

	try{
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
				// userID, itemID, rating, timestamp
			}else{
	            coo_format_ratingsMtx_userID_host[row] = (int)(::atof(vec[0].c_str())) - 1; /*make the user index start at zero*/
	            coo_format_ratingsMtx_itemID_host[row] = (int)(::atof(vec[1].c_str())) - 1;/*make the movie index start at zero*/
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
	}catch(const std::ifstream::failure& e){
		std::cout<<"CSVReader::getData() failure."<<std::endl;
	}
 	
 	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("getData runtime: %f\n", program_time); 
	LOG("finished call to CSVReader::getData() in "<<program_time<< "ms");
	//std::cout<<std::endl;
}



/*
* Parses through csv file line by line and returns the data
* in array.
*/
void CSVReader::getDataJSON(int* coo_format_ratingsMtx_userID_host,
						    int* coo_format_ratingsMtx_itemID_host,
						    float* coo_format_ratingsMtx_rating_host,
						    int num_entries, const int rating_if_missing)
{
	bool Debug = false;

	namespace pt = boost::property_tree;

	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
	LOG("CSVReader::getDataJSON() on "<<fileName.c_str());
	try{
		// The property tree for holding the JSON contents
	    pt::ptree tree;

	    // Load the JSON file in this property tree
	    pt::read_json(fileName.c_str(), tree);

	    // Print the JSON content to stdout
	    // pt::write_json(std::cout, tree);

		std::map<int, std::pair<int ,int > > Users;
		int total_users = 0;

		std::map<int, int> Items;
		int total_items = 0;

		BOOST_FOREACH(pt::ptree::value_type &entry, tree)
		{
			int user_id = entry.second.get<int>("user_id");
			int item_id = entry.second.get<int>("item_id");

			// BOOST_FOREACH(pt::ptree::value_type &property, entry.second)
			// {
			// 	LOG("Entry property key: " << property.first);
			// 	LOG("Entry property data: " << property.second.data());
			// 	LOG("Press Enter to continue.") ;
			// 	std::cin.ignore();  
			// }
			std::map<int, std::pair<int ,int > >::iterator it = Users.find(user_id);
			if( it == Users.end() ){
				//add the new user
				std::pair<int ,int > temp = std::pair<int ,int >(total_users, 1);
				Users.insert ( std::pair<int, std::pair<int ,int > >(user_id,temp) );
				total_users++;
			}else{
				//increase the number of ratings that user has made
				std::pair<int ,int > temp = it -> second;
				temp.second += 1;
				it -> second = temp;
			}	

		    std::map<int, int >::iterator it2 = Items.find(item_id);
		    if( it2 == Items.end() ){
		      //add the new user
		      Items.insert ( std::pair<int, int >(item_id,total_items) );
		      total_items++;
		    }	
		}
		if(Debug) {
			LOG("total_users : " << total_users);
			LOG("total_items : "<< total_items);
		}
		std::vector< std::pair<int ,int> > item_n_rating[total_users];
		if(Debug) LOG("here");

		BOOST_FOREACH(pt::ptree::value_type &entry, tree)
		{

			int user_id = entry.second.get<int>("user_id");
			int item_id = entry.second.get<int>("item_id");
			int rating;
			try {
				rating  = entry.second.get<int>("rating");
			} catch(...) {
				if(Debug && 0) {
					LOG("no rating available ");
					LOG("Press Enter to continue.") ;
			 		std::cin.ignore();	
				}
				rating  = rating_if_missing;

			}


			std::map<int, std::pair<int ,int > >::iterator it = Users.find(user_id);
			std::map<int, int >::iterator it2 = Items.find(item_id);
			if(Debug && 0){
				LOG("row : "<<it -> second.first);
				LOG("col : "<<it2 -> second);
				LOG("rating : "<< rating);
			}
			std::pair<int ,int > temp = std::pair<int ,int >(it2 -> second, rating);
			item_n_rating[it -> second.first].push_back(temp);
		}

		for(int us = 0; us < total_users; us++){
			sort(item_n_rating[us].begin(), item_n_rating[us].end());
		}

		if(Debug) LOG("here");
		int coo = 0;
		for(int us = 0; us < total_users; us++){
			int vec_size = item_n_rating[us].size();
			for(int v = 0; v < vec_size; v++){
	        	coo_format_ratingsMtx_userID_host[coo] = us; /*make the user index start at zero*/
	        	coo_format_ratingsMtx_itemID_host[coo] = item_n_rating[us][v].first;/*make the movie index start at zero*/
	        	coo_format_ratingsMtx_rating_host[coo] = (float)(item_n_rating[us][v].second);
				coo+=1;
			}
		}
	}catch(const pt::json_parser_error& e){
		std::cout<<"CSVReader::getDataJSON() failure."<<std::endl;
	}

	if(Debug) LOG("here");


 	
 	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("getData runtime: %f\n", program_time); 
	LOG("finished call to CSVReader::getDataJSON() in "<<program_time<< "ms"<<std::endl);
	//std::cout<<std::endl;
}


/*
* Parses through csv file line by line and returns the data
* in array.
*/
long long int CSVReader::makeContentBasedcooKeyWordMtx(int* coo_format_keyWordMtx_itemID_host,
													  int* coo_format_keyWordMtx_keyWord_host,
													  const long long int num_entries)
{
	bool Debug = false;
	struct timeval program_start, program_end;
    double program_time;
    gettimeofday(&program_start, NULL);
	LOG("CSVReader::makeContentBasedcrsKeyWordMtx() on "<<fileName.c_str());


	std::map<std::string, long long int> keyWords;

	long long int total_keyWords = (long long int)0;

	try{
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
				// itemID, movie name, keyword list
			}else{
				if(Debug) LOG("movieiD : "<<row);
				for(int i = 0; i< num_keyWords_in_line; i++){
					if(Debug) LOG("coo_entry : "<<coo_entry);
					coo_format_keyWordMtx_itemID_host[coo_entry] = row;

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
	}catch(const std::ifstream::failure& e){
		std::cout<<"CSVReader::makeContentBasedcooKeyWordMtx() failure."<<std::endl;
	}
 	
 	gettimeofday(&program_end, NULL);
    program_time = (program_end.tv_sec * 1000 +(program_end.tv_usec/1000.0))-(program_start.tv_sec * 1000 +(program_start.tv_usec/1000.0));
    //printf("getData runtime: %f\n", program_time); 
	LOG("finished call to CSVReader::makeContentBasedcrsKeyWordMtx() in "<<program_time<< "ms");
	//std::cout<<std::endl;

	return total_keyWords;
}














 