/**
 * @file mfcc_features.h
 * @brief Header for MFCC feature extraction abstraction.
 */

#ifndef MFCC_FEATURES_H
#define MFCC_FEATURES_H

#include <stdint.h>
#include <arm_math.h>

/** @brief Number of MFCC coefficients to extract */
#define MFCC_COEFFS_NUM 13U

/**
 * @brief Initializes the MFCC extraction module.
 */
void mfcc_features_init(void);
void mfcc_features_init_f32(void);
void mfcc_features_init_q15(void);

/**
 * @brief  Computes MFCC features over an audio block.
 * @param  audio_buffer Pointer to the incoming audio block.
 * @param  out_buffer   Pointer to the buffer where MFCC results will be stored.
 */
void mfcc_features_compute(float32_t *audio_buffer, float32_t *out_buffer);
void mfcc_features_compute_q15(q15_t *audio_buffer, q15_t *out_buffer);

void mfcc_features_build_feature_vector_q15(q15_t *mfcc_mat, q15_t *delta_mfcc_matrix, uint32_t num_hops, uint32_t num_coeffs, q15_t *feature_vector);
void compute_delta_mfcc_q15(const q15_t *mfcc_mat, q15_t *delta_mat, uint32_t num_hops, uint32_t num_coeffs, uint16_t win_length);
void interleave_mfcc_delta_q15(const q15_t *mfcc_mat, const q15_t *delta_mat, uint32_t num_hops, uint32_t num_coeffs, q15_t *feature_vector);

/**
 * @brief  Computes the mean and standard deviation of a matrix of MFCC vectors.
 * @param  mfcc_matrix    Pointer to the matrix containing MFCC vectors for all frames.
 * @param  num_hops       Number of frames (hops) in the matrix.
 * @param  num_coeffs     Number of MFCC coefficients per frame.
 * @param  feature_vector Destination vector to store first the means and then the standard deviations.
 */
void mfcc_features_mean_and_std(float32_t *mfcc_matrix, uint32_t num_hops, uint32_t num_coeffs, float32_t *feature_vector);

#endif /* MFCC_FEATURES_H */
