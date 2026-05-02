/**
 * @file clasificador_svm.c
 * @brief Implementation of functions to initialize and predict using polynomial SVMs.
 */

#include "clasificador_svm.h"
#include "svm_params.h"

// Contexts for each SVM classifier "One vs Rest"
static arm_svm_polynomial_instance_f32 svm_blue_ctx;
static arm_svm_polynomial_instance_f32 svm_green_ctx;
static arm_svm_polynomial_instance_f32 svm_red_ctx;

// Array to easily iterate SVM contexts during prediction
static arm_svm_polynomial_instance_f32* svm_ctx_array[NB_CLASSIFIERS];

/**
 * @brief Initializes each classifier indicating its precomputed support vectors and dual variables.
 */
void clasificador_svm_init(void)
{
    // blue vs rest
    arm_svm_polynomial_init_f32(&svm_blue_ctx,
                                NB_SV_BLUE_VS_REST,
                                VECTOR_DIMENSION,
                                INTERCEPT_BLUE_VS_REST,
                                dualCoefficients_blue_vs_rest,
                                supportVectors_blue_vs_rest,
                                classes_blue_vs_rest,
                                1, /* degree */
                                0.0f, /* coef0 */
                                1.0f /* gamma */);
    // green vs rest
    arm_svm_polynomial_init_f32(&svm_green_ctx,
                                NB_SV_GREEN_VS_REST,
                                VECTOR_DIMENSION,
                                INTERCEPT_GREEN_VS_REST,
                                dualCoefficients_green_vs_rest,
                                supportVectors_green_vs_rest,
                                classes_green_vs_rest,
                                1, /* degree */
                                0.0f, /* coef0 */
                                1.0f /* gamma */);    
    // red vs rest
    arm_svm_polynomial_init_f32(&svm_red_ctx,
                                NB_SV_RED_VS_REST,
                                VECTOR_DIMENSION,
                                INTERCEPT_RED_VS_REST,
                                dualCoefficients_red_vs_rest,
                                supportVectors_red_vs_rest,
                                classes_red_vs_rest,
                                1, /* degree */
                                0.0f, /* coef0 */
                                1.0f /* gamma */);
                        
    svm_ctx_array[0] = &svm_blue_ctx;
    svm_ctx_array[1] = &svm_green_ctx;
    svm_ctx_array[2] = &svm_red_ctx;
}

/**
 * @brief  Submits the feature vector to each configured classifier. Upon finding a positive match, exits and returns the label.
 * @param  feature_vector      Information prepared by the parameter extractor.
 * @param  feature_vector_size Dimension of the vector.
 * @param  label               Pointer where the winning class is injected if detected, or otherwise, -1.
 */
void clasificador_svm_predict(float32_t *feature_vector, uint32_t feature_vector_size, int32_t *label)
{
    (void)feature_vector_size; // Unused indicator if predefined VECTOR_DIMENSION is used.
    
    *label = -1;
    for(int i = 0; i < NB_CLASSIFIERS; i++)
    {
        arm_svm_polynomial_predict_f32(svm_ctx_array[i], feature_vector, label);
        if (*label == svm_ctx_array[i]->classes[1]) // If inference indicates recognized class
        {
            break; // Exit the iteration early
        }
    }
}
