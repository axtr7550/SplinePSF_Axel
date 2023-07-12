// Header file

#ifndef SPLINE_PSF_GPU_H_
#define SPLINE_PSF_GPU_H_

#ifndef DEBUG
#define DEBUG 0
#endif // DEBUG

namespace spline_psf_gpu {

    /* Spline Structure */
    /**
     * @brief defines the cubic spline and holds its coefficients
     *
     **/
    typedef struct {
        int xsize;  // size of the spline in x
        int ysize;  // size of the spline in y
        int zsize;  // size of the spline in z

        float roi_out_eps;  // epsilon value outside the roi
        float roi_out_deriv_eps; // epsilon value of derivative values outside the roi

        int n_par;  // number of parameters to fit
        int n_coeff;  // number of coefficients per pixel

        float *coeff;

        // bool add_bg_drv_roi;  // add background value in roi-wise calculation of the derivatives where the ROIs are output as well

    } spline;

    // Check cuda is available
    // Returns:
    //      bool: is available
    auto cuda_is_available(void) -> bool;


    // Initialisation of Spline Coefficients on Device
    // Args:
    //      xsize, ysize, zsize:  size of the coefficients in the respective axis
    //      h_coeff: coefficients on host
    // Returns:
    //      spline*:    pointer to spline struct living on the device (!)
    auto d_spline_init(const float *h_coeff, int xsize, int ysize, int zsize, int device_ix) -> spline*;

    auto destructor(spline *d_sp) -> void;


    // Wrapper function to compute the ROIs on the device.
    // Takes in all the host arguments and returns leaves the ROIs on the device
    //
    auto forward_rois_host2device(spline *d_sp, const int n, const int roi_size_x, const int roi_size_y,
        const float *h_x, const float *h_y, const float *h_z, const float *h_phot, const bool normalize) -> float*;

    auto forward_drv_rois_host2device(spline *d_sp, float *d_rois, float *d_drv_rois, const int n, const int roi_size_x, const int roi_size_y,
        const float *h_x, const float *h_y, const float *h_z, const float *h_phot, const float *h_bg, const bool add_bg) -> void;

    auto forward_frames_host2device(spline *d_sp, const int frame_size_x, const int frame_size_y, const int n_frames,
        const int n_rois, const int roi_size_x, const int roi_size_y,
        const int *h_frame_ix, const float *h_xr0, const float *h_yr0, const float *h_z0,
        const int *h_x_ix, const int *h_y_ix, const float *h_phot, const bool normalize) -> float*;

    // Wrapper function to compute the ROIs on the device and ships it back to the host
    // Takes in all the host arguments and returns the ROIs to the host
    // Allocation for rois must have happened outside
    //
    auto forward_rois_host2host(spline *d_sp, float *h_rois, const int n, const int roi_size_x, const int roi_size_y,
        const float *h_x, const float *h_y, const float *h_z, const float *h_phot, const bool normalize) -> void;

    auto forward_drv_rois_host2host(spline *d_sp, float *h_rois, float *h_drv_rois, const int n, const int roi_size_x, const int roi_size_y,
        const float *h_x, const float *h_y, const float *h_z, const float *h_phot, const float *h_bg, const bool add_bg) -> void;

    auto forward_frames_host2host(spline *d_sp, float *h_frames, const int frame_size_x, const int frame_size_y, const int n_frames,
        const int n_rois, const int roi_size_x, const int roi_size_y,
        const int *h_frame_ix, const float *h_xr0, const float *h_yr0, const float *h_z0,
        const int *h_x_ix, const int *h_y_ix, const float *h_phot, const bool normalize) -> void;

}

#endif  // SPLINE_PSF_GPU_H_