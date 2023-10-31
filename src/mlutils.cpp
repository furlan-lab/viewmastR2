// we only include RcppArrayFire.h which pulls Rcpp.h in for us
#include "RcppArrayFire.h"
#include <arrayfire.h>
#include <af/util.h>


// [[Rcpp::depends(RcppArrayFire)]]
// [[Rcpp::plugins(cpp11)]]

using namespace af;

//' @export
 // [[Rcpp::export]]
 bool set_device(std::string device = "GPU") {
   if (device == "GPU") {
     try {
       fprintf(stderr, "Trying OpenCL Backend\n");
       setBackend(AF_BACKEND_OPENCL);
       std::cerr << "AF_BACKEND_OPENCL: " << AF_BACKEND_OPENCL << std::endl;
       // testBackend();
     } catch (exception& e) {
       fprintf(stderr,"Caught exception when trying OpenCL backend\n");
       fprintf(stderr, "%s\n", e.what());
     }
   } else if (device == "GPU_CUDA") {
     try {
       fprintf(stderr,"Trying CUDA Backend\n");
       af::setBackend(AF_BACKEND_CUDA);
       std::cerr << "AF_BACKEND_CUDA: " << AF_BACKEND_CUDA << std::endl;
       // testBackend();
     } catch (af::exception& e) {
       fprintf(stderr,"Caught exception when trying CUDA backend\n");
       fprintf(stderr, "%s\n", e.what());
     }
   } else if (device == "GPU_OpenCL") {
     try {
       fprintf(stderr,"Trying OPENCL Backend\n");
       setBackend(AF_BACKEND_OPENCL);
       std::cerr << "AF_BACKEND_OPENCL: " << AF_BACKEND_OPENCL << std::endl;
       // testBackend();
     } catch (exception& e) {
       fprintf(stderr,"Caught exception when trying OpenCL backend\n");
       fprintf(stderr, "%s\n", e.what());
     }
   } else if (device == "CPU") {
     try {
       fprintf(stderr,"Trying CPU Backend\n");
       setBackend(AF_BACKEND_CPU);
       std::cerr << "AF_BACKEND_CPU: " << AF_BACKEND_CPU << std::endl;
       // testBackend();
     } catch (exception& e) {
       fprintf(stderr,"Caught exception when trying CPU backend\n");
       fprintf(stderr, "%s\n", e.what());
     }
   } else {
     std::cerr << "DEVICE:" << device << " not found!!" << std::endl;
     return true;
   }
   return false;
 }
