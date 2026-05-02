/**
 * @file clasificador_svm.c
 * @brief Implementación de las funciones para inicializar y predecir usando SVM polinomiales.
 */

#include "clasificador_svm.h"
#include "svm_params.h"

// Contextos para cada clasificador SVM "Uno vs el Resto"
static arm_svm_polynomial_instance_f32 svm_blue_ctx;
static arm_svm_polynomial_instance_f32 svm_green_ctx;
static arm_svm_polynomial_instance_f32 svm_red_ctx;

// Arreglo para poder iterar fácilmente los contextos SVM durante la predicción
static arm_svm_polynomial_instance_f32* svm_ctx_array[NB_CLASSIFIERS];

/**
 * @brief Inicia cada clasificador indicando sus vectores de soporte y variables duales precomputadas.
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
 * @brief  Somete al vector característico a cada clasificador configurado. Al encontrar coincidencia positiva, escapa y retorna la etiqueta.
 * @param  feature_vector      Información preparada por el extractor de parámetros.
 * @param  feature_vector_size Dimensión del vector.
 * @param  label               Puntero donde se inyecta la clase ganadora si se detecta, o por consiguiente, un -1.
 */
void clasificador_svm_predict(float32_t *feature_vector, uint32_t feature_vector_size, int32_t *label)
{
    (void)feature_vector_size; // Indicador no utilizado si se utiliza el predefinido VECTOR_DIMENSION.
    
    *label = -1;
    for(int i = 0; i < NB_CLASSIFIERS; i++)
    {
        arm_svm_polynomial_predict_f32(svm_ctx_array[i], feature_vector, label);
        if (*label == svm_ctx_array[i]->classes[1]) // Si la inferencia indica clase reconocida
        {
            break; // Salir de la iteración prematuramente
        }
    }
}
