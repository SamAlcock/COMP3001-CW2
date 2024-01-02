/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#include "convolution_layer_2D.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))




int unoptimized_layer_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

    float temp, bias;
    __m256 mm, mmbias, mmMask_Y_dim, mmMask_X_dim, mmInput_depth_dim, mmoff_y, mmoff_x, mmd, mmfilter_subscript, mm0, mm1, mm1a, mm2, mm3, mm4, mm4a, mms, mms1, mmw, mmtemp, mmtemp1;
    __m256 mmout_subscript, mmb, mmOutput_depth_dim, mmOutput_X_dim, mmOutput_Y_dim, mmy, mmx, mm5, mm5a, mm5b, mm6, mm6a, mm7, mm8;

        // out_subscript, b, Output_depth_dim, Output_X_dim, Output_Y_dim, y, x, m
    mmMask_Y_dim = _mm256_set1_ps(Mask_Y_dim); // 8 copies of Mask_Y_dim
    mmMask_X_dim = _mm256_set1_ps(Mask_X_dim); // 8 copies of Mask_X_dim
    mmInput_depth_dim = _mm256_set1_ps(Input_depth_dim); // 8 copies of Input_depth_dim
    

    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
        for (unsigned int m = 0; m < Output_depth_dim; m+=8) { //channels
            mm = _mm256_set_ps(m, m + 1, m + 2, m + 3, m + 4, m + 5, m + 6, m + 7);
            
            mm0 = _mm256_setzero_ps();
            mm1 = _mm256_setzero_ps();
            mm2 = _mm256_setzero_ps();
            mm3 = _mm256_setzero_ps();
            mm4 = _mm256_setzero_ps();
            mm5 = _mm256_setzero_ps();
            mm6 = _mm256_setzero_ps();
            mm7 = _mm256_setzero_ps();
            mm8 = _mm256_setzero_ps();

            for (unsigned int y = 0; y < Output_Y_dim; y++) {			//Output height
                for (unsigned int x = 0; x < Output_X_dim; x++) {			//Output Width
                    /*
                    bias = bias_array_FP[m];
                    bias = bias_array_FP[m + 1];
                    bias = bias_array_FP[m + 2];
                    bias = bias_array_FP[m + 3];
                    */
                    
                    
                    mmbias = _mm256_load_ps(&bias_array_FP[m]); // load 8 elements of bias_array_FP[]
                    mmbias = _mm256_permute_ps(mmbias, _MM_SHUFFLE(0, 1, 2, 3));
                    mmbias = _mm256_permute2f128_ps(mmbias, mmbias, 1);


                    temp = 0.0f;
                    mmtemp = _mm256_setzero_ps();
                    for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
                        mmoff_y = _mm256_set1_ps(off_y); // 8 copies of off_y
                        for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                            mmoff_x = _mm256_set1_ps(off_x); // 8 copies of off_x
                            for (unsigned int d = 0; d < Input_depth_dim; d++) {
                                mmd = _mm256_set1_ps(d); // 8 copies of d


                                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                                    + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                                    + (x * Stride_X_dim + off_x) * Input_depth_dim
                                    + d;

                                // m
                                /*
                                    unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                                    + off_y * Mask_X_dim * Input_depth_dim
                                    + off_x * Input_depth_dim
                                    + d;
                                */


                                mm0 = _mm256_mul_ps(mmMask_X_dim, mmInput_depth_dim); // Mask_x_dim * Input_depth_dim, stored into mm0
                                mm1 = _mm256_mul_ps(mm0, mmMask_Y_dim); // Mask_x_dim * Input_depth_dim * Mask_Y_dim, stored into mm1
                                mm1a = _mm256_mul_ps(mm1, mm); // Mask_X_dim * Input_depth_dim * Mask_Y_dim * m, stored into mm1

                                mm2 = _mm256_mul_ps(mmoff_y, mm0); // off_y * Mask_x_dim * Input_depth_dim, stored into mm2

                                mm3 = _mm256_mul_ps(mmoff_x, mmInput_depth_dim); // off_x * Input_depth_dim, stored into mm3

                                // Add mm1, mm2, mm3 and mmd
                                mm4 = _mm256_add_ps(mm1a, mm2);
                                mm4a = _mm256_add_ps(mm4, mm3);
                                mmfilter_subscript = _mm256_add_ps(mm4a, mmd);
                                
                                __declspec(align(64)) float ref[8];
                                __declspec(align(64)) float filter_FPs[8]{};
                                _mm256_store_ps(ref, mmfilter_subscript);
                                
                                for (int i = 0; i < 8; i++) {
                                    filter_FPs[i] = filter_FP[(int)ref[i]];
                                }

                                mms = _mm256_set1_ps(in_FP[in_subscript]);
                                mmw = _mm256_load_ps(&filter_FPs[0]);

                                mms1 = _mm256_mul_ps(mmw, mms);
                                mmtemp = _mm256_add_ps(mmtemp, mms1);
                                /*
                                    float s = in_FP[in_subscript];
                                    float w = filter_FP[filter_subscript];
                                    temp = temp + s * w;
                                */



                            }
                        }
                    }

                    mmb = _mm256_set1_ps(b); // 8 copies of b
                    mmOutput_depth_dim = _mm256_set1_ps(Output_depth_dim);
                    mmOutput_X_dim = _mm256_set1_ps(Output_X_dim);
                    mmOutput_Y_dim = _mm256_set1_ps(Output_Y_dim);
                    mmy = _mm256_set1_ps(y);
                    mmx = _mm256_set1_ps(x);

                    mm5 = _mm256_mul_ps(mmOutput_depth_dim, mmOutput_X_dim); // Output_depth_dim * Output_X_dim, stored in mm5
                    mm5a = _mm256_mul_ps(mm5, mmOutput_Y_dim); // Output_depth_dim * Output_X_dim * Output_Y_dim, stored in mm5
                    mm5b = _mm256_mul_ps(mmb, mm5a); // b * (Output_depth_dim * Output_X_dim * Output_Y_dim), stored in mm5

                    mm6 = _mm256_mul_ps(mmOutput_depth_dim, mmOutput_X_dim); // Output_depth_dim * Output_X_dim, stored in mm6
                    mm6a = _mm256_mul_ps(mmy, mm6); // y * (Output_depth_dim * Output_X_dim), stored in mm6

                    mm7 = _mm256_mul_ps(mmx, mmOutput_depth_dim); // x * Output_depth_dim, stored in mm7

                    mm8 = _mm256_add_ps(mm5b, mm6a); // (b * (Output_depth_dim * Output_X_dim * Output_Y_dim)) + (y * (Output_depth_dim * Output_X_dim)), stored in mm8
                    mmout_subscript = _mm256_add_ps(mm7, mm8); // (x * Output_depth_dim) + (b * (Output_depth_dim * Output_X_dim * Output_Y_dim)) + (y * (Output_depth_dim * Output_X_dim)), stored in mmout_subscript
                    mmout_subscript = _mm256_add_ps(mmout_subscript, mm);
                    /*
                        unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
                        y * (Output_depth_dim * Output_X_dim) +
                        x * Output_depth_dim
                        + m;
                    */
                    

                   
                    mmtemp = _mm256_add_ps(mmtemp, mmbias);
                    
                    __declspec(align(64)) float temps[8];
                    __declspec(align(64)) float out_subscripts[8];
                    _mm256_store_ps(temps, mmtemp);
                    _mm256_store_ps(out_subscripts, mmout_subscript);
                    

                    for (int i = 0; i < 8; i++) {
                        //out_to_compare_with_FP[(int)out_subscripts[i]] = Relu_float(temps[i]);

                        if (temps[i] < 0.0f) {
                            out_to_compare_with_FP[(int)out_subscripts[i]] = 0.0f;
                        }
                        else {
                            out_to_compare_with_FP[(int)out_subscripts[i]] = temps[i];
                        }
                        
                    }
                    
                    // temp += bias;
                    // out_to_compare_with_FP[out_subscript] = Relu_float(temp);
                }
            }
            
        }
    }

    //printf("\n from unopt %d %d ",out_to_compare_with[0],out_to_compare_with[1]);
    return 0;

}


int unoptimized_layer_FP_openmp(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

    float temp, bias;

#pragma omp parallel for private(temp,bias)
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
            for (unsigned int od = 0; od < 1; od++) {	//Output Depth , for 3D convolution only
                for (unsigned int y = 0; y < Output_Y_dim; y++) {			//Output height
                    for (unsigned int x = 0; x < Output_X_dim; x++) {			//Output Width
                        bias = bias_array_FP[m];
                        temp = 0.0f;
                        for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
                            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
#pragma omp simd reduction(+:temp) aligned(in_FP,filter_FP:64)
                                for (unsigned int d = 0; d < Input_depth_dim; d++) {

                                    unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                                        + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                                        + (x * Stride_X_dim + off_x) * Input_depth_dim
                                        + d;
                                    unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                                        + off_y * Mask_X_dim * Input_depth_dim
                                        + off_x * Input_depth_dim
                                        + d;

                                    float s = in_FP[in_subscript];
                                    float w = filter_FP[filter_subscript];
                                    temp = temp + s * w;


                                }
                            }
                        }


                        unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
                            y * (Output_depth_dim * Output_X_dim) +
                            x * Output_depth_dim
                            + m;

                        temp += bias;
                        out_to_compare_with_FP[out_subscript] = Relu_float(temp);

                    }
                }
            }
        }
    }

    //printf("\n from unopt %d %d ",out_to_compare_with[0],out_to_compare_with[1]);
    return 0;

}

float Relu_float(const float temp) {


    if (temp < 0.0f)
        return 0.0f;
    else
        return temp;

}

