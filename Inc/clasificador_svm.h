/**
 * @file clasificador_svm.h
 * @brief Definitions and prototypes for the Support Vector Machine (SVM) classification module.
 */

#ifndef CLASIFICADOR_SVM_H
#define CLASIFICADOR_SVM_H

#include <stdint.h>
#include <arm_math.h>

/**
 * @brief Initializes the contexts of all SVM classifiers "one vs the rest".
 */
void clasificador_svm_init(void);

/**
 * @brief  Predicts to which class a given feature vector belongs.
 * @param  feature_vector      Feature vector already compiled (means and variances).
 * @param  feature_vector_size The number of elements in the provided vector.
 * @param  label               Pointer to integer where the found label is written, or -1 if unknown.
 */
void clasificador_svm_predict(float32_t *feature_vector, uint32_t feature_vector_size, int32_t *label);

#endif /* CLASIFICADOR_SVM_H */