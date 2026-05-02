/**
 * @file mfcc_features.c
 * @brief Implementation of MFCC feature extraction functions.
 */

#include "mfcc_features.h"
#include "mfcc_config.h"

#define FFT_SIZE 256U /**< @brief FFT size for MFCC */

// Global instance to store the MFCC context
static arm_mfcc_instance_f32 mfcc_ctx;
static arm_mfcc_instance_q15 mfcc_ctx_q15;

// Internal buffer used during MFCC calculation by CMSIS-DSP
static float32_t mfcc_complex_buff[2 * FFT_SIZE];
static q31_t mfcc_complex_buff_q31[2 * FFT_SIZE];

/**
 * @brief Initializes the MFCC feature extractor using precomputed parameters.
 */
void mfcc_features_init_f32(void)
{
    arm_mfcc_init_f32(&mfcc_ctx,
                      FFT_SIZE, 
                      NB_MFCC_NB_FILTER_CONFIG_8K_F32, 
                      MFCC_COEFFS_NUM, 
                      mfcc_dct_coefs_config1_f32, 
                      mfcc_filter_pos_config_8k_f32, 
                      mfcc_filter_len_config_8k_f32,
                      mfcc_filter_coefs_config_8k_f32,
                      mfcc_window_coefs_config3_f32);
}

void mfcc_features_init_q15(void) 
{
  arm_mfcc_init_q15(
      &mfcc_ctx_q15, FFT_SIZE, NB_MFCC_NB_FILTER_CONFIG_8K_Q15, MFCC_COEFFS_NUM,
      mfcc_dct_coefs_config1_q15, mfcc_filter_pos_config_8k_q15,
      mfcc_filter_len_config_8k_q15, mfcc_filter_coefs_config_8k_q15,
      mfcc_window_coefs_config3_q15);
}

/**
 * @brief  Executes MFCC extraction from the audio sample.
 * @param  audio_buffer Pointer to the audio signal block.
 * @param  out_buffer   Pointer to the start of the buffer that will contain the calculated coefficients.
 */
void mfcc_features_compute(float32_t *audio_buffer, float32_t *out_buffer)
{
    arm_mfcc_f32(&mfcc_ctx, audio_buffer, out_buffer, mfcc_complex_buff);
}

/**
 * @brief  Executes MFCC extraction from the audio sample.
 * @param  audio_buffer Pointer to the audio signal block.
 * @param  out_buffer   Pointer to the start of the buffer that will contain the calculated coefficients.
 */
void mfcc_features_compute_q15(q15_t *audio_buffer, q15_t *out_buffer)
{
    arm_mfcc_q15(&mfcc_ctx_q15, audio_buffer, out_buffer, mfcc_complex_buff_q31);
}

void mfcc_features_build_feature_vector_q15(q15_t *mfcc_mat, q15_t *delta_mfcc_matrix, uint32_t num_hops, uint32_t num_coeffs, q15_t *feature_vector)
{
   
    // Delta computation
    compute_delta_mfcc_q15(mfcc_mat, 
                           delta_mfcc_matrix,
                           num_hops, num_coeffs,
                           2);

    // Interleave MFCC and Delta MFCC into final feature vector
    interleave_mfcc_delta_q15(mfcc_mat, 
                              delta_mfcc_matrix, 
                              num_hops, 
                              num_coeffs, 
                              feature_vector);
}

/**
 * @brief  Computes the mean and standard deviation to condense several MFCC frames into a single vector.
 * @param  mfcc_mat       Linear pointer to the MFCC result matrix of size [num_hops][num_coeffs].
 * @param  num_hops       The total number of iterations/subdivisions (hops) captured.
 * @param  num_coeffs     The number of coefficients extracted per frame.
 * @param  feature_vector Final output with means followed by standard deviations.
 */
void mfcc_features_mean_and_std(float32_t *mfcc_mat, uint32_t num_hops, uint32_t num_coeffs, float32_t *feature_vector)
{
    float32_t temp_col_buf[num_hops];
    
    for( uint16_t i = 0; i < num_coeffs; i++)
    {
        for( uint16_t j = 0; j < num_hops; j++)
        {
            // Extraer la columna i-ésima de la matriz MFCC
            temp_col_buf[j] = *(mfcc_mat + (j * num_coeffs) + i); 
        }
        arm_mean_f32(temp_col_buf, num_hops, &feature_vector[i]);
        arm_std_f32(temp_col_buf, num_hops, &feature_vector[i + num_coeffs]);
    }  
}

// Compute delta MFCCs for a q15_t MFCC matrix (HOPS_PER_FRAME x MFCC_COEFFS_NUM)
// Input:  mfcc_mat [HOPS_PER_FRAME][MFCC_COEFFS_NUM] (row-major)
// Output: delta_mat [HOPS_PER_FRAME][MFCC_COEFFS_NUM] (row-major)
// win_length: N (typically 2)
// Uses edge padding (replicates border values)
// High-performance, fixed-point, and SIMD-friendly delta MFCC computation for q15_t
void compute_delta_mfcc_q15(const q15_t *mfcc_mat, q15_t *delta_mat, uint32_t num_hops, uint32_t num_coeffs, uint16_t win_length)
{
  if (win_length == 0 || num_hops == 0) {
    return;
  }

  int32_t den = 0;
  for (uint16_t n = 1; n <= win_length; n++) {
    den += n * n;
  }
  den *= 2;

  for (uint32_t t = 0; t < num_hops; t++) {
    for (uint32_t c = 0; c < num_coeffs; c++) {
      int32_t num = 0;

      for (uint16_t n = 1; n <= win_length; n++) {
        int32_t t_plus = (int32_t)t + (int32_t)n;
        if (t_plus >= (int32_t)num_hops) {
          t_plus = (int32_t)num_hops - 1;
        }

        int32_t t_minus = (int32_t)t - (int32_t)n;
        if (t_minus < 0) {
          t_minus = 0;
        }

        int32_t val_plus  = mfcc_mat[t_plus * num_coeffs + c];
        int32_t val_minus = mfcc_mat[t_minus * num_coeffs + c];

        num += (int32_t)n * (val_plus - val_minus);
      }

      delta_mat[t * num_coeffs + c] = (q15_t)(num / den);
    }
  }
}

// Interleave MFCC and Delta MFCC matrices into a single feature vector
// Structure: [MFCCHop0][DeltaHop0]...[MFCCHop(N-1)][DeltaHop(N-1)]
void interleave_mfcc_delta_q15(const q15_t *mfcc_mat, const q15_t *delta_mat, uint32_t num_hops, uint32_t num_coeffs, q15_t *feature_vector)
{
  for (uint32_t t = 0; t < num_hops; t++) 
  {
    arm_copy_q15(&mfcc_mat[t * num_coeffs], &feature_vector[t * 2 * num_coeffs], num_coeffs);
    arm_copy_q15(&delta_mat[t * num_coeffs], &feature_vector[t * 2 * num_coeffs + num_coeffs], num_coeffs);
  }
}
