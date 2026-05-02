/**
 * @file mfcc_features.h
 * @brief Cabecera para la abstracción de extracción de características MFCC.
 */

#ifndef MFCC_FEATURES_H
#define MFCC_FEATURES_H

#include <stdint.h>
#include <arm_math.h>

/** @brief Número de coeficientes MFCC a extraer */
#define MFCC_COEFFS_NUM 13U

/**
 * @brief Inicializa el módulo de extracción MFCC.
 */
void mfcc_features_init(void);
void mfcc_features_init_f32(void);
void mfcc_features_init_q15(void);

/**
 * @brief  Calcula las características MFCC sobre un bloque de audio.
 * @param  audio_buffer Puntero al bloque de audio entrante.
 * @param  out_buffer   Puntero al buffer donde se guardarán los resultados MFCC.
 */
void mfcc_features_compute(float32_t *audio_buffer, float32_t *out_buffer);
void mfcc_features_compute_q15(q15_t *audio_buffer, q15_t *out_buffer);

void mfcc_features_build_feature_vector_q15(q15_t *mfcc_mat, q15_t *delta_mfcc_matrix, uint32_t num_hops, uint32_t num_coeffs, q15_t *feature_vector);
void compute_delta_mfcc_q15(const q15_t *mfcc_mat, q15_t *delta_mat, uint32_t num_hops, uint32_t num_coeffs, uint16_t win_length);
void interleave_mfcc_delta_q15(const q15_t *mfcc_mat, const q15_t *delta_mat, uint32_t num_hops, uint32_t num_coeffs, q15_t *feature_vector);

/**
 * @brief  Calcula la media y desviación estándar de una matriz de vectores MFCC.
 * @param  mfcc_matrix    Puntero a la matriz que contiene los vectores MFCC para todos los frames.
 * @param  num_hops       Cantidad de frames (hops) en la matriz.
 * @param  num_coeffs     Número de coeficientes MFCC por frame.
 * @param  feature_vector Vector destino para guardar primero las medias y luego las desviaciones estándar.
 */
void mfcc_features_mean_and_std(float32_t *mfcc_matrix, uint32_t num_hops, uint32_t num_coeffs, float32_t *feature_vector);

#endif /* MFCC_FEATURES_H */
