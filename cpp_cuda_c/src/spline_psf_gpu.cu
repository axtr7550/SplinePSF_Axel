//
//  Created by Lucas Müller on 12.02.2020
//  Copyright © 2020 Lucas-Raphael Müller. All rights reserved.
//
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include "spline_psf_gpu.cuh"
using namespace spline_psf_gpu;


// internal declarations
void check_host_coeff(const float *h_coeff);

auto forward_rois(spline *d_sp, float *d_rois, const int n, const int roi_size_x, const int roi_size_y,
    const float *d_x, const float *d_y, const float *d_z, const float *d_phot, const bool normalize) -> void;

auto forward_drv_rois(spline *d_sp, float *d_rois, float *d_drv_rois, const int n, const int roi_size_x, const int roi_size_y,
    const float *d_x, const float *d_y, const float *d_z, const float *d_phot, const float *d_bg, const bool add_bg) -> void;

/*
template <unsigned int blockSize>
__global__
void reduce_dot_product_matrix(float* g_idata_a, float* g_idata_b, float* g_odata,
unsigned int stride, unsigned int start, unsigned int n);
*/
template <int blockSize>

__device__ 
void warpReduce(volatile float* sdata, int tid) ;

__global__ 
void reduce_small(float *rois, float *g_odata, int r, const int np, int n_blocks, const int startPos);

__global__ 
void normalize(float *rois, float factor, int r, const int n_pixels);

__global__ 
void  get_factor(const float* phot_, float *total_sum, float *factors);

__device__
auto kernel_computeDelta3D(spline *sp,
    float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf,
    float x_delta, float y_delta, float z_delta) -> void;

__global__
auto kernel_derivative(spline *sp, float *rois, float *drv_rois, const int roi_ix, const int npx,
    const int npy, int xc, int yc, int zc, const float phot, const float bg,
    const float x_delta, const float y_delta, const float z_delta, const bool add_bg) -> void;

__global__
auto fAt3Dj(spline *sp, float* rois, int roi_ix, int npx, int npy,
    int xc, int yc, int zc, float phot, float x_delta, float y_delta, float z_delta) -> void;

__global__
auto kernel_roi(spline *sp, float *rois, const int npx, const int npy,
    const float* xc_, const float* yc_, const float* zc_, const float* phot_) -> void;

__global__
void kernel_sum_up(spline *sp, float *rois, float *sum_array, const int npx, const int npy);

__global__ 
void kernel_normalize(float *rois, float *factors, int npx, int npy);

__global__
auto kernel_derivative_roi(spline *sp, float *rois, float *drv_rois, const int npx, const int npy,
    const float *xc_, const float *yc_, const float *zc_,
    const float *phot_, const float *bg_, const bool add_bg) -> void;

__global__
auto roi_accumulate(float *frames, const int frame_size_x, const int frame_size_y, const int n_frames,
    const float *rois, const int n_rois,
    const int *frame_ix, const int *x0, const int *y0,
    const int roi_size_x, const int roi_size_y) -> void;



namespace spline_psf_gpu {

    // check cuda availability by device count
    auto cuda_is_available(void) -> bool {

        int d_count = 0;
        cudaError_t err = cudaGetDeviceCount(&d_count);

        if (err != cudaSuccess) {
            return false;
        }

        float min_compute_cap = 3.7;

        bool at_least_one_device = false;
        for (int i = 0; i < d_count; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);

            float compute_cap = prop.major + prop.minor / 10;

            if (compute_cap >= min_compute_cap) {
                at_least_one_device = true;
                break;
            }

        }

        if (at_least_one_device) {
            return true;
        }

        return false;
    }

    // Create struct and ship it to device
    auto d_spline_init(const float *h_coeff, int xsize, int ysize, int zsize, int device_ix) -> spline* {

        // allocate struct on host and ship it to device later
        // ToDo: C++11ify this
        spline* sp;
        sp = (spline *)malloc(sizeof(spline));

        sp->xsize = xsize;
        sp->ysize = ysize;
        sp->zsize = zsize;

        sp->roi_out_eps = 1e-10;
        sp->roi_out_deriv_eps = 0.0;

        sp->n_par = 5;
        sp->n_coeff = 64;

        int tsize = xsize * ysize * zsize * 64;

        cudaSetDevice(device_ix);

        float *d_coeff;
        cudaMalloc(&d_coeff, tsize * sizeof(float));
        cudaMemcpy(d_coeff, h_coeff, tsize * sizeof(float), cudaMemcpyHostToDevice);

        sp->coeff = d_coeff;  // for some reason this should happen here and not d_sp->coeff = d_coeff ...

        // ship to device
        spline* d_sp;
        cudaMalloc(&d_sp, sizeof(spline));
        cudaMemcpy(d_sp, sp, sizeof(spline), cudaMemcpyHostToDevice);

        // delete on host
        free(sp);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::stringstream rt_err;
            rt_err << "Error during allocation of spline struct on device.\nCode: "<< err << "\nInformation: \n" << cudaGetErrorString(err);
            throw std::runtime_error(rt_err.str());
        }

        return d_sp;
    }

    auto destructor(spline *d_sp) -> void {

        // first create host helper to be able to access the pointer to coeff, dereferencing d_sp is illegal
        spline *sp;
        sp = (spline*)malloc(sizeof(spline));
        cudaMemcpy(sp, d_sp, sizeof(spline), cudaMemcpyDeviceToHost);

        cudaFree(sp->coeff);
        free(sp);
        cudaFree(d_sp);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::stringstream rt_err;
            rt_err << "Error during destruxtor.\nCode: "<< err << "\nInformation: \n" << cudaGetErrorString(err);
            throw std::runtime_error(rt_err.str());
        }
    }


    // Wrapper function to compute the ROIs on the device.
    // Takes in all the host arguments and returns leaves the ROIs on the device
    //
    auto forward_rois_host2device(spline *d_sp, const int n, const int roi_size_x, const int roi_size_y,
    const float *h_x, const float *h_y, const float *h_z, const float *h_phot, const bool normalize) -> float* {

        cudaError_t err;

        // allocate and copy coordinates and photons
        float *d_x, *d_y, *d_z, *d_phot;
        cudaMalloc(&d_x, n * sizeof(float));
        cudaMalloc(&d_y, n * sizeof(float));
        cudaMalloc(&d_z, n * sizeof(float));
        cudaMalloc(&d_phot, n * sizeof(float));
        cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, h_z, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phot, h_phot, n * sizeof(float), cudaMemcpyHostToDevice);

        // allocate space for rois on device
        float* d_rois;
        cudaMalloc(&d_rois, n * roi_size_x * roi_size_y * sizeof(float));

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::stringstream rt_err;
            rt_err << "Error during ROI memory allocation.\nCode: "<< err << "\nInformation: \n" << cudaGetErrorString(err);
            throw std::runtime_error(rt_err.str());
        }

        cudaMemset(d_rois, 0.0, n * roi_size_x * roi_size_y * sizeof(float));

        #if DEBUG
            check_spline<<<1,1>>>(d_sp);
            cudaDeviceSynchronize();
        #endif

        // call to actual implementation
        forward_rois(d_sp, d_rois, n, roi_size_x, roi_size_y, d_x, d_y, d_z, d_phot, normalize);

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        cudaFree(d_phot);

        return d_rois;
    }

    // Wrapper function to ocmpute the ROIs on the device and ships it back to the host
    // Takes in all the host arguments and returns the ROIs to the host
    // Allocation for rois must have happened outside
    //
    auto forward_rois_host2host(spline *d_sp, float *h_rois, const int n, const int roi_size_x, const int roi_size_y,
        const float *h_x, const float *h_y, const float *h_z, const float *h_phot, const bool normalize) -> void {

        auto d_rois = forward_rois_host2device(d_sp, n, roi_size_x, roi_size_y, h_x, h_y, h_z, h_phot, normalize);

        cudaMemcpy(h_rois, d_rois, n * roi_size_x * roi_size_y * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_rois);
        return;
    }

    auto forward_drv_rois_host2device(spline *d_sp, float *d_rois, float *d_drv_rois, const int n, const int roi_size_x, const int roi_size_y,
        const float *h_x, const float *h_y, const float *h_z, const float *h_phot, const float *h_bg, const bool add_bg) -> void {

        // allocate and copy coordinates and photons
        float *d_x, *d_y, *d_z, *d_phot, *d_bg;
        cudaMalloc(&d_x, n * sizeof(float));
        cudaMalloc(&d_y, n * sizeof(float));
        cudaMalloc(&d_z, n * sizeof(float));
        cudaMalloc(&d_phot, n * sizeof(float));
        cudaMalloc(&d_bg, n * sizeof(float));

        cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, h_z, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phot, h_phot, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bg, h_bg, n * sizeof(float), cudaMemcpyHostToDevice);

        const int n_par = 5;
        cudaMemset(d_rois, 0.0, n * roi_size_x * roi_size_y * sizeof(float));
        cudaMemset(d_drv_rois, 0.0, n_par * n * roi_size_x * roi_size_y * sizeof(float));

        // call to actual implementation
        forward_drv_rois(d_sp, d_rois, d_drv_rois, n, roi_size_x, roi_size_y, d_x, d_y, d_z, d_phot, d_bg, add_bg);

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        cudaFree(d_phot);
        cudaFree(d_bg);

        return;
    }

    auto forward_drv_rois_host2host(spline *d_sp, float *h_rois, float *h_drv_rois, const int n, const int roi_size_x, const int roi_size_y,
        const float *h_x, const float *h_y, const float *h_z, const float *h_phot, const float *h_bg, const bool add_bg) -> void {

        cudaError_t err;

        // allocate space for rois and derivatives on device
        const int n_par = 5;
        float *d_rois, *d_drv_rois;

        cudaMalloc(&d_rois, n * roi_size_x * roi_size_y * sizeof(float));
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::stringstream rt_err;
            rt_err << "Error during ROI memory allocation.\nCode: "<< err << "\nInformation: \n" << cudaGetErrorString(err);
            throw std::runtime_error(rt_err.str());
        }

        cudaMalloc(&d_drv_rois, n_par * n * roi_size_x * roi_size_y * sizeof(float));
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::stringstream rt_err;
            rt_err << "Error during derivative ROI memory allocation.\nCode: "<< err << "\nInformation: \n" << cudaGetErrorString(err);
            throw std::runtime_error(rt_err.str());
        }

        // forward
        forward_drv_rois_host2device(d_sp, d_rois, d_drv_rois, n, roi_size_x, roi_size_y, h_x, h_y, h_z, h_phot, h_bg, add_bg);

        cudaMemcpy(h_rois, d_rois, n * roi_size_x * roi_size_y * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_drv_rois, d_drv_rois, n * n_par * roi_size_x * roi_size_y * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_rois);
        cudaFree(d_drv_rois);

        return;
    }

    auto forward_frames_host2host(spline *d_sp, float *h_frames, const int frame_size_x, const int frame_size_y, const int n_frames,
        const int n_rois, const int roi_size_x, const int roi_size_y,
        const int *h_frame_ix, const float *h_xr0, const float *h_yr0, const float *h_z0,
        const int *h_x_ix, const int *h_y_ix, const float *h_phot, const bool normalize) -> void {

        auto d_frames = forward_frames_host2device(d_sp, frame_size_x, frame_size_y, n_frames,
            n_rois, roi_size_x, roi_size_y, h_frame_ix, h_xr0, h_yr0, h_z0, h_x_ix, h_y_ix, h_phot, normalize);

        cudaMemcpy(h_frames, d_frames, n_frames * frame_size_x * frame_size_y * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_frames);
        return;
    }

    auto forward_frames_host2device(spline *d_sp, const int frame_size_x, const int frame_size_y, const int n_frames,
        const int n_rois, const int roi_size_x, const int roi_size_y,
        const int *h_frame_ix, const float *h_xr0, const float *h_yr0, const float *h_z0,
        const int *h_x_ix, const int *h_y_ix, const float *h_phot, const bool normalize) -> float* {

        cudaError_t err;

        // ToDo: maybe convert to stream
        float* d_frames;

        // Creates d_frames on GPU to store all the frames

        cudaMalloc(&d_frames, n_frames * frame_size_x * frame_size_y * sizeof(float));
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::stringstream rt_err;
            rt_err << "Error during Frame memory allocation.\nCode: "<< err << "\nInformation: \n" << cudaGetErrorString(err);
            throw std::runtime_error(rt_err.str());
        }
        cudaMemset(d_frames, 0.0, n_frames * frame_size_x * frame_size_y * sizeof(float));

        // allocate indices
        int *d_xix, *d_yix, *d_fix;
        cudaMalloc(&d_xix, n_rois * sizeof(int));
        cudaMalloc(&d_yix, n_rois * sizeof(int));
        cudaMalloc(&d_fix, n_rois * sizeof(int));
        cudaMemcpy(d_xix, h_x_ix, n_rois * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_yix, h_y_ix, n_rois * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fix, h_frame_ix, n_rois * sizeof(int), cudaMemcpyHostToDevice);

        auto d_rois = forward_rois_host2device(d_sp, n_rois, roi_size_x, roi_size_y, h_xr0, h_yr0, h_z0, h_phot, normalize);

        // accumulate rois into frames
        const int blocks = (n_rois * roi_size_x * roi_size_y) / 256 + 1;
        const int thread_p_block = 256;
        // One thread per roi pixel
        // Still quite quick
        roi_accumulate<<<blocks, thread_p_block>>>(d_frames, frame_size_x, frame_size_y, n_frames,
        d_rois, n_rois, d_fix, d_xix, d_yix, roi_size_x, roi_size_y);

        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::stringstream rt_err;
            rt_err << "Error during frame computation computation.\nCode: "<< err << "\nInformation: \n" << cudaGetErrorString(err);
            throw std::runtime_error(rt_err.str());
        }

        cudaFree(d_xix);
        cudaFree(d_yix);
        cudaFree(d_fix);
        cudaFree(d_rois);

        return d_frames;
    }
} // namespace spline_psf_gpu


auto forward_rois(spline *d_sp, float *d_rois, const int n, const int roi_size_x, const int roi_size_y,
    const float *d_x, const float *d_y, const float *d_z, const float *d_phot, const bool normalize) -> void {
    
    // printf("IN CUDA! \n");

    // init cuda_err
    cudaError_t err = cudaSuccess;


    // start n blocks which itself start threads corresponding to the number of px childs (dynamic parallelism)
    kernel_roi<<<n, 1>>>(d_sp, d_rois, roi_size_x, roi_size_y, d_x, d_y, d_z, d_phot);
    cudaDeviceSynchronize();


    
    if (normalize){

    float *sum_array;

    //int stride = ((roi_size_x*roi_size_y)/1024);

    cudaMalloc(&sum_array, n*sizeof(float));
    cudaMemset(sum_array, 0.0, n*sizeof(float));



    kernel_sum_up<<<n, 1>>>(d_sp, d_rois, sum_array, roi_size_x, roi_size_y);
    cudaDeviceSynchronize();


    // A bit of cheating, will work when n-elements is below 2048

    float *factors_array;

    cudaMalloc(&factors_array, n*sizeof(float));
    cudaMemset(factors_array, 0.0, n*sizeof(float));

    get_factor<<<n, 1>>>(d_phot, sum_array, factors_array);
    cudaDeviceSynchronize();

    kernel_normalize<<<n, 1>>>(d_rois, factors_array, roi_size_x, roi_size_y);
    
    cudaFree(sum_array);

    cudaFree(factors_array);
    }
    // Sum array should contain sum from each block. Sum each roi and get factor. Then in parallel multiply each pixel by factor.
    

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::stringstream rt_err;
        rt_err << "Error during ROI computation computation.\nCode: "<< err << "\nInformation: \n" << cudaGetErrorString(err);
        throw std::runtime_error(rt_err.str());
    }

    return;
}

auto forward_drv_rois(spline *d_sp, float *d_rois, float *d_drv_rois, const int n, const int roi_size_x, const int roi_size_y,
    const float *d_x, const float *d_y, const float *d_z, const float *d_phot, const float *d_bg, const bool add_bg) -> void {

    // init cuda_err
    cudaError_t err = cudaSuccess;

    // start n blocks which itself start threads corresponding to the number of px childs (dynamic parallelism)
    kernel_derivative_roi<<<n, 1>>>(d_sp, d_rois, d_drv_rois, roi_size_x, roi_size_y, d_x, d_y, d_z, d_phot, d_bg, add_bg);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::stringstream rt_err;
        rt_err << "Error during ROI derivative computation computation.\nCode: "<< err << "\nInformation: \n" << cudaGetErrorString(err);
        throw std::runtime_error(rt_err.str());
    }

    return;
}

// Just a dummy for checking correct parsing from python
// ... had to learn the hard way ...
__global__
auto check_spline(spline *d_sp) -> void {
    printf("Checking spline ...\n");
    printf("\txs, ys, zs: %i %i %i\n", d_sp->xsize, d_sp->ysize, d_sp->zsize);
    printf("\toutside-roi value: %f\n", d_sp->roi_out_eps);
    printf("\toutside-roi derivative value: %f\n", d_sp->roi_out_deriv_eps);

    printf("\tDevice coeff: \n");
    for (int i = 0; i < 100; i++) {
        printf("\t\ti: %d coeff %f\n", d_sp->coeff[i]);
    }
    printf("\n");
}

// kernel to compute common term for spline function (for all pixels this will stay the same)
__device__
auto kernel_computeDelta3D(spline *sp, float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf,
    float x_delta, float y_delta, float z_delta) -> void {
    // Should be parallizable
    int i,j,k;
    float cx,cy,cz;
    // Computes the deltas that will be multiplied by the "a" parameters delta_f is 64 long
    cz = 1.0;
    for(i=0;i<4;i++){
        cy = 1.0;
        for(j=0;j<4;j++){
            cx = 1.0;
            for(k=0;k<4;k++){
                delta_f[i*16+j*4+k] = cz * cy * cx;
                if(k<3){
					delta_dxf[i*16+j*4+k+1] = ((float)k+1) * cz * cy * cx;
				}
				if(j<3){
					delta_dyf[i*16+(j+1)*4+k] = ((float)j+1) * cz * cy * cx;
				}
				if(i<3){
					delta_dzf[(i+1)*16+j*4+k] = ((float)i+1) * cz * cy * cx;
				}
                cx = cx * x_delta;
            }
            cy = cy * y_delta;
        }
        cz= cz * z_delta;
    }
}

// kernel to compute pixel-wise term
__global__
auto fAt3Dj(spline *sp, float* rois, const int roi_ix, const int npx, const int npy,
    int xc, int yc, int zc, float phot, float x_delta, float y_delta, float z_delta) -> void {
    
    // The kernel knows its index and the blockdim, i.e. number of threads per block (???)
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) / npx;
    const int j = (blockIdx.x * blockDim.x + threadIdx.x) % npx;

     // allocate space for df, dxf, dyf, dzf
    __shared__ float delta_f[64];
    __shared__ float dxf[64];
    __shared__ float dyf[64];
    __shared__ float dzf[64];

    // term common to all pixels, must be executed at least once per kernel block (since sync only syncs within block)
    // if (i == 0 and j == 0) {  // linear / C++ equivalent
    
    if (threadIdx.x == 0) {
        
        // 64 -- number of "a" parameters I guess this is a way of initializing the array
        for (int k = 0; k < 64; k++) {
            delta_f[k] = 0.0;
            dxf[k] = 0.0;
            dyf[k] = 0.0;
            dzf[k] = 0.0;
        }
        

        // This is different to the C library since we needed to rearrange a bit to account for the GPU parallelism
        // Could possibly be done in parallel but does not seem faster
        kernel_computeDelta3D(sp, delta_f, dxf, dyf, dzf, x_delta, y_delta, z_delta);
    }

    __syncthreads();  // wait so that all threads see the deltas. REMINDER: only works for within block 

    // kill excess threads (I think it needs to happen after syncthreads)
    if ((i >= npx) || (j >= npy)) {
        return;
    }

    // xc must originally be in like the top left corner? Unless i and j can be negative

    xc = xc + i;
    yc = yc + j;

    // If the lateral position is outside the calibration, return epsilon value
    if ((xc < 0) || (xc > sp->xsize-1) || (yc < 0) || (yc > sp->ysize-1)) {

        rois[roi_ix * npx * npy + i * npy + j] = sp->roi_out_eps;
        return;
    }

    zc = max(zc,0);
    zc = min(zc,sp->zsize-1);

    float fv = 0.0;
    // Loop but variables in shared memory means still fast
    for (int k = 0; k < 64; k++) {
        fv += delta_f[k] * sp->coeff[k * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
    }

    // This looks like the C code
    // write to global roi stack
    rois[roi_ix * npx * npy + i * npy + j] = phot * fv;
    
    // atomicAdd(&sum_array[roi_ix], phot * fv);
    return;
}

// kernel to compute psf for a single emitter
__global__
auto kernel_roi(spline *sp, float *rois, const int npx, const int npy, const float* xc_, 
const float* yc_, const float* zc_, const float* phot_) -> void {

    int r = blockIdx.x;  // roi number 'r'

    int x0, y0, z0;
    float x_delta,y_delta,z_delta;

    float xc = xc_[r];
    float yc = yc_[r];
    float zc = zc_[r];
    float phot = phot_[r];

    /* Compute delta. Will be the same for all following px */
    // floats use floorf I think, stolen from another branch
    x0 = (int)floorf(xc);
    x_delta = xc - x0;

    y0 = (int)floorf(yc);
    y_delta = yc - y0;

    z0 = (int)floorf(zc);
    z_delta = zc - z0;


    int n_threads = min(1024, npx * npy);  // max number of threads per block
    int n_blocks = ceil(static_cast<float>(npx * npy) / static_cast<float>(n_threads));

    fAt3Dj<<<n_blocks, n_threads>>>(sp, rois, r, npx, npy, x0, y0, z0, phot, x_delta, y_delta, z_delta);

    return;
}

__global__
auto kernel_derivative_roi(spline *sp, float *rois, float *drv_rois, const int npx, const int npy,
    const float *xc_, const float *yc_, const float *zc_, const float *phot_, const float *bg_, const bool add_bg) -> void {

    int r = blockIdx.x;  // roi number 'r'

    int x0, y0, z0;
    float x_delta,y_delta,z_delta;

    float xc = xc_[r];
    float yc = yc_[r];
    float zc = zc_[r];
    float phot = phot_[r];
    float bg = bg_[r];

    /* Compute delta. Will be the same for all following px */
    x0 = (int)floorf(xc);
    x_delta = xc - x0;

    y0 = (int)floorf(yc);
    y_delta = yc - y0;

    z0 = (int)floorf(zc);
    z_delta = zc - z0;

    int n_threads = min(1024, npx * npy);  // max number of threads per block
    int n_blocks = ceil(static_cast<float>(npx * npy) / static_cast<float>(n_threads));

    kernel_derivative<<<n_blocks, n_threads>>>(sp, rois, drv_rois, r, npx, npy, x0, y0, z0, phot, bg, x_delta, y_delta, z_delta, add_bg);

    return;
}

__global__
auto kernel_derivative(spline *sp, float *rois, float *drv_rois, const int roi_ix, const int npx, const int npy,
    int xc, int yc, int zc, const float phot, const float bg, const float x_delta, const float y_delta, const float z_delta, const bool add_bg) -> void {

    int i = (blockIdx.x * blockDim.x + threadIdx.x) / npx; // Number of threads I guess
    int j = (blockIdx.x * blockDim.x + threadIdx.x) % npx;

     // allocate space for df, dxf, dyf, dzf
    __shared__ float delta_f[64];
    __shared__ float dxf[64];
    __shared__ float dyf[64];
    __shared__ float dzf[64];

    float dudt[5] = { 0 };  // derivatives in this very pixel

    // term common to all pixels, must be executed at least once per kernel block (since sync only syncs within block)
    // if (i == 0 and j == 0) {  // linear / C++ equivalent
    if (threadIdx.x == 0) {

        for (int k = 0; k < 64; k++) {
            delta_f[k] = 0.0;
            dxf[k] = 0.0;
            dyf[k] = 0.0;
            dzf[k] = 0.0;
        }

        // This is different to the C library since we needed to rearrange a bit to account for the GPU parallelism
        kernel_computeDelta3D(sp, delta_f, dxf, dyf, dzf, x_delta, y_delta, z_delta);
    }
    __syncthreads();  // wait so that all threads see the deltas

    // kill excess threads
    if ((i >= npx) || (j >= npy)) {
        return;
    }

    // let each thread go to their respective pixel
    xc = xc + i;
    yc = yc + j;

    // set epsilon values outside of ROI
    if ((xc < 0) || (xc > sp->xsize-1) || (yc < 0) || (yc > sp->ysize-1)) {

        for (int k = 0; k < sp->n_par; k++) {
            dudt[k] = sp->roi_out_deriv_eps;
        }

        if (add_bg) {
            rois[roi_ix * npx * npy + i * npy + j] = sp->roi_out_eps + bg;
        }
        else {
            rois[roi_ix * npx * npy + i * npy + j] = sp->roi_out_eps;
        }
        return;
    }

    // safety for zc
    zc = max(zc,0);
    zc = min(zc,sp->zsize-1);

    float fv;  // taken from yiming, not entirely understood by myself

    // actual derivative computation
    for (int k = 0; k < 64; k++)
    {
        fv += delta_f[k] * sp->coeff[k * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
        dudt[0] += dxf[k] * sp->coeff[k * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
        dudt[1] += dyf[k] * sp->coeff[k * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
        dudt[4] += dzf[k] * sp->coeff[k * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
    }

    dudt[0] *= -1 * phot;
    dudt[1] *= -1 * phot;
    dudt[4] *= phot;
    dudt[2] = fv;
    dudt[3] = 1;

    // write to global roi and derivate stack
    if (add_bg) {
        rois[roi_ix * npx * npy + i * npy + j] = phot * fv + bg;
    }
    else {
        rois[roi_ix * npx * npy + i * npy + j] = phot * fv;
    }

    for (int k = 0; k < sp->n_par; k++) {
        drv_rois[roi_ix * sp->n_par * npx * npy + k * npx * npy + i * npy + j] = dudt[k];
    }

    return;
}

// accumulate rois to frames
__global__
auto roi_accumulate(float *frames, const int frame_size_x, const int frame_size_y, const int n_frames,
                    const float *rois, const int n_rois,
                    const int *frame_ix, const int *x0, const int *y0,
                    const int roi_size_x, const int roi_size_y) -> void {

        // kernel ix
        const long kx = (blockIdx.x * blockDim.x + threadIdx.x);
        if (kx >= n_rois * roi_size_x * roi_size_y) {
            return;
        }

        // roi index
        const long j = kx % roi_size_y;
        const long i = ((kx - j) / roi_size_y) % roi_size_x;
        const long r = (((kx - j) / roi_size_y) - i) / roi_size_x;

        const long ii = x0[r] + i;
        const long jj = y0[r] + j;

        if ((frame_ix[r] < 0) || (frame_ix[r] >= n_frames)) {  // if frame ix is outside
            return;
        }

        if ((ii < 0) || (jj < 0) || (ii >= frame_size_x) || (jj >= frame_size_y)) {  // if outside frame throw away
            return;
        }
        float val = rois[r * roi_size_x * roi_size_y + i * roi_size_y + j];
        
        // Possible reduction?
        atomicAdd(&frames[frame_ix[r] * frame_size_x * frame_size_y + ii * frame_size_y + jj], val);  // otherwise race condition

        return;
    }

__global__
void kernel_sum_up(spline *sp, float *rois, float* sum_array,const int npx, const int npy){
    
    int r = blockIdx.x; 

    const int np = npx * npy;
    int n_threads;
    if (np > 1024){n_threads = 1024;}
    else if(np > 512) {n_threads = 512;}
    else if(np > 256) {n_threads = 256;}
    else if(np > 128) {n_threads = 128;}
    else if(np > 64) {n_threads = 64;}
    else if(np > 32) {n_threads = 32;}
    else if(np > 16) {n_threads = 16;}
    else if(np > 8) {n_threads = 8;}
    else if(np > 4) {n_threads = 4;}

    // n_threads should be smallest power of 2 larger than half of npx*npy
    const int startPos = r*np;

    reduce_small<<<1,n_threads, 2*n_threads*sizeof(float)>>>(rois, sum_array, r, np, 1, startPos);
   

    }


__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
    }

__global__ void reduce_small(float *rois, float *g_odata, int r, const int np, int n_blocks, const int startPos) {
    extern __shared__ float sdata[];
    int tid = int(threadIdx.x);
    int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    if (i < (np)-blockDim.x) {
        sdata[tid] = rois[startPos + i] + rois[startPos + i + blockDim.x];
    } else {
        sdata[tid] = rois[startPos + i];
    }

    __syncthreads();


    // do reduction in shared mem
    for (int s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }       
        __syncthreads();
    }

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) {
        g_odata[r*n_blocks + blockIdx.x] = sdata[0];
        }
    }


__global__ void kernel_normalize(float *rois, float *factors, int npx, int npy){
    int r = blockIdx.x;
    float __shared__ factor; 
    if (threadIdx.x == 0){
        factor = factors[r];
    }

    int n_threads = min(1024, npx * npy);  // max number of threads per block
    int n_blocks = ceil(static_cast<float>(npx * npy) / static_cast<float>(n_threads));
    const int n_pixels = npx * npy;
    normalize<<<n_blocks, n_threads>>>(rois, factor, r, n_pixels);

}
__global__ void normalize(float *rois, float factor, int r, const int n_pixels){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int startPos = r*n_pixels;

    if (i < n_pixels) {rois[startPos + i] *= factor;}
    
}

__global__ void get_factor(const float* phot_, float *total_sum, float *factors){
    int r = blockIdx.x;
    float phot = phot_[r];
    

    factors[r] = phot / total_sum[r];

}

