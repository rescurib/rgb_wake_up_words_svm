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
 * @brief Initializes the MFCC feature extractor (float32 version) using precomputed parameters.
 *
 * This function sets up the MFCC context for floating-point operations using the configuration
 * parameters defined in mfcc_config.h.
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


/**
 * @brief Initializes the MFCC feature extractor (q15 version) using precomputed parameters.
 *
 * This function sets up the MFCC context for fixed-point (q15) operations using the configuration
 * parameters defined in mfcc_config.h.
 */
void mfcc_features_init_q15(void) 
{
  arm_mfcc_init_q15(
      &mfcc_ctx_q15, FFT_SIZE, NB_MFCC_NB_FILTER_CONFIG_8K_Q15, MFCC_COEFFS_NUM,
      mfcc_dct_coefs_config1_q15, mfcc_filter_pos_config_8k_q15,
      mfcc_filter_len_config_8k_q15, mfcc_filter_coefs_config_8k_q15,
      mfcc_window_coefs_config3_q15);
}


/**
 * @brief Executes MFCC extraction from the audio sample (float32 version).
 *
 * @param[in]  audio_buffer Pointer to the audio signal block.
 * @param[out] out_buffer   Pointer to the buffer that will contain the calculated coefficients.
 */
void mfcc_features_compute(float32_t *audio_buffer, float32_t *out_buffer)
{
  arm_mfcc_f32(&mfcc_ctx, audio_buffer, out_buffer, mfcc_complex_buff);
}


/**
 * @brief Executes MFCC extraction from the audio sample (q15 version).
 *
 * @param[in]  audio_buffer Pointer to the audio signal block.
 * @param[out] out_buffer   Pointer to the buffer that will contain the calculated coefficients.
 */
void mfcc_features_compute_q15(q15_t *audio_buffer, q15_t *out_buffer)
{
  arm_mfcc_q15(&mfcc_ctx_q15, audio_buffer, out_buffer, mfcc_complex_buff_q31);
}


/**
 * @brief Builds a feature vector by interleaving MFCC and delta MFCC matrices (q15 version).
 *
 * This function computes the delta MFCCs and interleaves them with the original MFCCs to form a final feature vector.
 *
 * @param[in]  mfcc_mat           Pointer to the MFCC matrix [num_hops x num_coeffs].
 * @param[out] delta_mfcc_matrix  Pointer to the output delta MFCC matrix [num_hops x num_coeffs].
 * @param[in]  num_hops           Number of frames (hops).
 * @param[in]  num_coeffs         Number of coefficients per frame.
 * @param[out] feature_vector     Pointer to the output feature vector (interleaved MFCC and delta MFCC).
 */
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
 * @brief Computes the mean and standard deviation to condense several MFCC frames into a single vector.
 *
 * This function calculates the mean and standard deviation for each MFCC coefficient across all frames (hops),
 * and stores the results in the output feature vector (means followed by standard deviations).
 *
 * @param[in]  mfcc_mat       Pointer to the MFCC result matrix of size [num_hops][num_coeffs] (row-major).
 * @param[in]  num_hops       The total number of frames (hops).
 * @param[in]  num_coeffs     The number of coefficients extracted per frame.
 * @param[out] feature_vector Output vector: means [0..num_coeffs-1], stddevs [num_coeffs..2*num_coeffs-1].
 */
void mfcc_features_mean_and_std(float32_t *mfcc_mat, uint32_t num_hops, uint32_t num_coeffs, float32_t *feature_vector)
{
  float32_t temp_col_buf[num_hops];
    
  for( uint16_t i = 0; i < num_coeffs; i++)
  {
    for( uint16_t j = 0; j < num_hops; j++)
    {
      // Extract the i-th column from the MFCC matrix
      temp_col_buf[j] = *(mfcc_mat + (j * num_coeffs) + i); 
    }
    arm_mean_f32(temp_col_buf, num_hops, &feature_vector[i]);
    arm_std_f32(temp_col_buf, num_hops, &feature_vector[i + num_coeffs]);
  }  
}


/**
 * @brief Computes delta MFCCs for a q15_t MFCC matrix.
 *
 * This function calculates the delta (temporal derivative) of MFCC features for each frame using a windowed difference.
 * Edge values are padded by replicating border values. Optimized for fixed-point and SIMD-friendly operations.
 *
 * @param[in]  mfcc_mat   Pointer to the input MFCC matrix [num_hops x num_coeffs] (row-major).
 * @param[out] delta_mat  Pointer to the output delta MFCC matrix [num_hops x num_coeffs] (row-major).
 * @param[in]  num_hops   Number of frames (hops).
 * @param[in]  num_coeffs Number of coefficients per frame.
 * @param[in]  win_length Window length for delta computation (typically 2).
 */
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


/**
 * @brief Interleaves MFCC and delta MFCC matrices into a single feature vector (q15 version).
 *
 * The output feature vector is structured as [MFCCHop0][DeltaHop0]...[MFCCHop(N-1)][DeltaHop(N-1)].
 *
 * @param[in]  mfcc_mat       Pointer to the MFCC matrix [num_hops x num_coeffs] (row-major).
 * @param[in]  delta_mat      Pointer to the delta MFCC matrix [num_hops x num_coeffs] (row-major).
 * @param[in]  num_hops       Number of frames (hops).
 * @param[in]  num_coeffs     Number of coefficients per frame.
 * @param[out] feature_vector Pointer to the output feature vector (interleaved MFCC and delta MFCC).
 */
void interleave_mfcc_delta_q15(const q15_t *mfcc_mat, const q15_t *delta_mat, uint32_t num_hops, uint32_t num_coeffs, q15_t *feature_vector)
{
  for (uint32_t t = 0; t < num_hops; t++) 
  {
    arm_copy_q15(&mfcc_mat[t * num_coeffs], &feature_vector[t * 2 * num_coeffs], num_coeffs);
    arm_copy_q15(&delta_mat[t * num_coeffs], &feature_vector[t * 2 * num_coeffs + num_coeffs], num_coeffs);
  }
}
