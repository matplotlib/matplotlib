/* agg_buffer.h	-- John D. Hunter
 */

#ifndef _AGG_BUFFER_H
#define _AGG_BUFFER_H

#include <iostream>
#include "agg_basics.h"

namespace agg {
  

  typedef struct binary_data {
    int size;
    unsigned char* data;
  } binary_data;

  struct buffer {
  public: 
    buffer(unsigned width, unsigned height, unsigned stride, bool freemem=true) : 
      width(width), height(height), stride(stride), freemem(freemem) {
      
      data = new int8u[height*stride];
      
    }
    ~buffer() {
      //std::cout << "bye bye " << freemem << std::endl;
      if (freemem) {
	delete [] data;
	data = NULL;      
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
    bool freemem;
  };
}  

#endif
