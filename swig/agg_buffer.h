/* agg_buffer.h	-- John D. Hunter
 */

#ifndef _AGG_BUFFER_H
#define _AGG_BUFFER_H

#include <iostream>

namespace agg {
  

  typedef struct binary_data {
    int size;
    unsigned char* data;
  } binary_data;

  struct buffer {
  public: 
    buffer(unsigned width, unsigned height, unsigned stride) : 
      width(width), height(height), stride(stride) {
      
      data = new int8u[height*stride];
      
    }
    ~buffer() {delete [] data;}
    
    void speak() {
      for (size_t i=0; i < 20; i++) {
	std::cout << "RGBA: " 
		  << int(data[4*i+0]) << " " 
		  << int(data[4*i+1]) << " " 
		  << int(data[4*i+2]) << " " 
		  << int(data[4*i+1]) << std::endl;
      }

    }

    binary_data to_string() { 
      binary_data result;
      result.size = height*stride;
      result.data = data;
      return result;
    }

    const unsigned width, height, stride;
    
    int8u *data;
  };
}  

#endif
