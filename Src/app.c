#if 0
/**
 * @file app.c
 * @brief Example project for real-time audio acquisition and wake word detection using I2S MEMS microphone and SVM classification on STM32F3.
 *
 * This project demonstrates how to acquire audio from a digital MEMS microphone (INMP441) using the STM32 I2S peripheral in DMA mode, extract MFCC features, and classify wake words using an SVM model. Audio samples are processed in real time, and results are sent via UART for monitoring or further processing. Recording is controlled by the user button (B1), and a status LED (LD2) indicates the current acquisition state.
 *
 * Features:
 *   - Real-time audio acquisition from MEMS microphone (INMP441, I2S interface)
 *   - DMA-based I2S data transfer for efficient sampling
 *   - MFCC and Delta-MFCC feature extraction
 *   - SVM-based wake word classification (multi-class, one-vs-rest)
 *   - UART output for detected words and system status
 *   - User button (B1) to start/stop recording
 *   - Status LED (LD2) for acquisition state
 *
 * Hardware:
 *   - STM32F3 series MCU
 *   - INMP441 MEMS microphone (I2S interface)
 *   - UART for serial output (460800 Bits/s)
 *   - User button (B1)
 *   - Status LED (LD2)
 *
 * Usage:
 *   - Press the user button to start or stop audio acquisition.
 *   - Detected words are sent via UART in a simple format.
 */
#endif

#include "main.h"
#include "app.h"

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

// CMSIS-DSP and feature extraction includes
#include <arm_math.h>
#include "mfcc_features.h"
#include "clasificador_svm.h"


/******** Definitions ******** */
#define SAMPLES_PER_HOP 256U /**< @brief 32 ms at 8 kHz */
#define HOPS_PER_FRAME   16U /**< @brief Total frame duration: 512 ms at 8 kHz */
#define FULL_BUFFER_SIZE (SAMPLES_PER_HOP * HOPS_PER_FRAME) /**< @brief 512 ms of audio at 8 kHz */
#define NOISE_ALPHA (q15_t)(0.01f * 32768)

/**
 * @brief Structure for sending float data via UART.
 */
typedef union {
    float32_t f32;
    uint8_t b[4];
} float_packet_t;

// Indicates if the microphone is currently recording
static volatile bool is_recording = false;

// Buffers for I2S stereo samples (2 words for left and right channels)
static uint8_t i2s_stereo_samples[SAMPLES_PER_HOP * 2 * 4]; /**< @brief Two channels, 4 bytes per sample. */
static uint8_t sample_buff[SAMPLES_PER_HOP * 2 * 4];        /**< @brief Buffer for one hop of mono samples (32 ms). */
static float32_t full_buff[FULL_BUFFER_SIZE];               /**< @brief 256 ms of mono samples at 8 kHz. */
static q15_t hop[SAMPLES_PER_HOP];                          /**< @brief Buffer for the current hop. */

// MFCC and Delta-MFCC matrices for the frame
q15_t mfcc_matrix[HOPS_PER_FRAME][MFCC_COEFFS_NUM]; 
q15_t delta_mfcc_matrix[HOPS_PER_FRAME][MFCC_COEFFS_NUM];

// Final feature vector with interleaved MFCCs and Delta-MFCCs
#define FEATURE_VECTOR_SIZE (2 * MFCC_COEFFS_NUM * HOPS_PER_FRAME) /**< @brief Final feature vector with interleaved MFCCs and Delta-MFCCs. */
static q15_t feature_vector_q15[FEATURE_VECTOR_SIZE]; 
static float32_t feature_vector_f32[FEATURE_VECTOR_SIZE];

// Global variables
q15_t g_noise_floor = (q15_t)(0.01f * 32768); // Initial background noise floor value (RMS)
volatile bool g_signal_detected  = false;  /**< @brief Indicates if a signal above the noise threshold has been detected */
volatile bool g_dma_data_ready   = false;  /**< @brief Indicates if the DMA transfer has completed */
static uint32_t g_last_turn_on_time = 0;

// External handlers for I2S and UART (defined elsewhere)
extern I2S_HandleTypeDef hi2s2;
extern UART_HandleTypeDef huart2;

// Internal function prototypes
/**
 * @brief Start microphone acquisition using I2S DMA.
 */
static void mic_start(void);
/**
 * @brief Stop microphone and DMA acquisition.
 */
static void mic_stop(void);
/**
 * @brief Convert a byte array to a 32-bit floating-point audio sample.
 * @param sample Pointer to the raw sample (8 bytes).
 * @return 32-bit float sample normalized to [-1.0, 1.0].
 */
static inline float32_t i2s_sample_to_float32(uint8_t* sample);
/**
 * @brief Convert a byte array to a Q15 audio sample.
 * @param sample Pointer to the raw sample (8 bytes).
 * @return Q15 sample.
 */
static inline q15_t i2s_sample_to_q15(uint8_t *sample);
/**
 * @brief Convert an array of Q8.7 fixed-point values to float32.
 * @param dst Destination float array.
 * @param src Source Q15 array.
 * @param length Number of elements.
 */
static inline void q8_7_to_float32(float32_t *dst, const q15_t *src, uint32_t length);

static bool rgb_timeout_expired(void);

// Function implementations

/**
 * @brief  Update the status LED to indicate the recording state.
 * @param  recording True if recording, false otherwise.
 */
static inline void update_status_led(bool recording)
{
    if (recording)
    {
        // Turn on the status LED (e.g. LD2)
        HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
    }
    else
    {
        // Turn off the status LED
        HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
    }
}

/**
 * @brief  Main application loop.
 *         Handles initialization and processes the captured audio signal.
 */
void app_run(void)
{

    // State initialization
    is_recording = false;
    update_status_led(is_recording);
    memset(i2s_stereo_samples, 0, sizeof(i2s_stereo_samples));
    uint16_t hop_index = 0;

    // Initialize MFCC feature extraction and SVM classifier
    mfcc_features_init_q15();
    clasificador_svm_init();

    while(1)
    {
        if(g_signal_detected)
        {
            if(g_dma_data_ready && hop_index < HOPS_PER_FRAME) // Only process if DMA data is ready and the buffer is not full
            {
                memcpy(full_buff + (hop_index * SAMPLES_PER_HOP), hop, sizeof(hop));
                g_dma_data_ready = false; 

                // Compute MFCCs for the current hop
                mfcc_features_compute_q15(hop, mfcc_matrix[hop_index]);
                hop_index++;

            } else if (hop_index == HOPS_PER_FRAME)
            {
                // Compute feature vector (MFCC + Delta MFCC) from the full frame
                mfcc_features_build_feature_vector_q15((q15_t *)mfcc_matrix, 
                                                       (q15_t *)delta_mfcc_matrix, 
                                                       HOPS_PER_FRAME, 
                                                       MFCC_COEFFS_NUM, 
                                                       feature_vector_q15);

                // Convert the feature vector from Q15 to float32 for the SVM classifier
                q8_7_to_float32(feature_vector_f32, feature_vector_q15, FEATURE_VECTOR_SIZE);

                // Classify the sample with the SVM model
                int32_t svm_result = -1;
                clasificador_svm_predict(feature_vector_f32, FEATURE_VECTOR_SIZE, &svm_result);
                hop_index = 0;
                g_signal_detected = false;

                // RGB control based on detected word
                switch(svm_result)
                {
                    case 'R': // Detected "RED"
                        HAL_GPIO_WritePin(LD_RED_GPIO_PORT,   LD_RED_PIN, GPIO_PIN_SET);
                        HAL_GPIO_WritePin(LD_GREEN_GPIO_PORT, LD_GREEN_PIN, GPIO_PIN_RESET);
                        HAL_GPIO_WritePin(LD_BLUE_GPIO_PORT,  LD_BLUE_PIN, GPIO_PIN_RESET);
                        g_last_turn_on_time = HAL_GetTick();
                        break;
                    case 'G': // Detected "GREEN"
                        HAL_GPIO_WritePin(LD_RED_GPIO_PORT,   LD_RED_PIN, GPIO_PIN_RESET);
                        HAL_GPIO_WritePin(LD_GREEN_GPIO_PORT, LD_GREEN_PIN, GPIO_PIN_SET);
                        HAL_GPIO_WritePin(LD_BLUE_GPIO_PORT,  LD_BLUE_PIN, GPIO_PIN_RESET);
                        g_last_turn_on_time = HAL_GetTick();
                        break;
                    case 'B': // Detected "BLUE"
                        HAL_GPIO_WritePin(LD_RED_GPIO_PORT,   LD_RED_PIN, GPIO_PIN_RESET);
                        HAL_GPIO_WritePin(LD_GREEN_GPIO_PORT, LD_GREEN_PIN, GPIO_PIN_RESET);
                        HAL_GPIO_WritePin(LD_BLUE_GPIO_PORT,  LD_BLUE_PIN, GPIO_PIN_SET);
                        g_last_turn_on_time = HAL_GetTick();
                        break;
                    default: // No word detected
                        // Do nothing
                        break;
                }

                // Send result via UART
                const char* msg = "Detected word (initial): ";
                if(svm_result != -1)
                {
                    HAL_UART_Transmit(&huart2, (uint8_t*)msg, strlen(msg), 100U);
                    HAL_UART_Transmit(&huart2, (uint8_t*)&svm_result, 1, 100U);
                    HAL_UART_Transmit(&huart2, (uint8_t*)"\r\n", 2, 100U);
                }
            }
        }

       // Turn off RGB after a non-blocking time-out
       if(rgb_timeout_expired())
        {
            HAL_GPIO_WritePin(LD_RED_GPIO_PORT,   LD_RED_PIN, GPIO_PIN_RESET);
            HAL_GPIO_WritePin(LD_GREEN_GPIO_PORT, LD_GREEN_PIN, GPIO_PIN_RESET);
            HAL_GPIO_WritePin(LD_BLUE_GPIO_PORT,  LD_BLUE_PIN, GPIO_PIN_RESET);
        }
    }
}

/**
 * @brief  Start microphone acquisition using I2S DMA. Updates state and notifies via UART.
 */
static void mic_start(void)
{
    if (!is_recording)
    {
          /* Start I2S DMA reception (2 words, 24 bits each)
              The size argument is uint16_t*, as required by the HAL API for 24/32-bit formats.
          */
        if (HAL_I2S_Receive_DMA(&hi2s2, (uint16_t*)i2s_stereo_samples, 2 * SAMPLES_PER_HOP) == HAL_OK)
        {
           is_recording = true;
           update_status_led(is_recording);

           // Notify that acquisition has started
           const char* msg = "Microphone ON\r\n";
           HAL_UART_Transmit(&huart2, (uint8_t*)msg, strlen(msg), 100U);
        } 
        else 
        {
           // Notify error
           const char* error_msg = "Acquisition error\r\n";
           HAL_UART_Transmit(&huart2, (uint8_t*)error_msg, strlen(error_msg), 100U);
        }
    } 
}

/**
 * @brief  Stop microphone and DMA acquisition. Updates state and notifies via UART.
 */
static void mic_stop(void)
{
    if (is_recording)
    {
        // Stop I2S DMA
        HAL_I2S_DMAStop(&hi2s2);
        is_recording = false;
        update_status_led(is_recording);

        // Notify that acquisition has finished
        const char* msg = "Microphone OFF\r\n";
        HAL_UART_Transmit(&huart2, (uint8_t*)msg, strlen(msg), 100U);
    }
}

/**
 * @brief  Callback for EXTI line detection (button press). Changes recording state when user button (B1) is pressed.
 * @param  GPIO_Pin Specifies the pin connected to the EXTI line.
 */
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if (GPIO_Pin == B1_Pin)
    {
        // Change recording state on button press
        if (is_recording)
        {
            mic_stop();
        }
        else
        {
            mic_start();
        }
    }
}

/**
 * @brief  Callback for I2S DMA reception completion. Invoked when a block of I2S data is received. Sends the received samples to the buffer for processing.
 * @param  hi2s I2S handle structure.
 */
void HAL_I2S_RxCpltCallback(I2S_HandleTypeDef *hi2s)
{

    // Only process if it's our I2S instance
    if(hi2s->Instance != hi2s2.Instance)
    {
        return; 
    }

    // Only process if recording
    if(is_recording == false)
    {
        return; 
    }

    memcpy(sample_buff, i2s_stereo_samples, sizeof(sample_buff));

    g_dma_data_ready = true; // Indicates that DMA data is ready to be processed
    
    q15_t result;

    for (uint32_t i = 0; i < SAMPLES_PER_HOP; i++)
    {
        // Convert the left channel sample (first 4 bytes of each stereo pair)
        uint8_t* sample_ptr = &sample_buff[i * 8]; // 8 bytes per stereo sample (4 left, 4 right)
        hop[i] = i2s_sample_to_q15(sample_ptr);
    }

    arm_rms_q15(hop, SAMPLES_PER_HOP, &result);

    if(result > 7.5 * g_noise_floor)
    {
        g_signal_detected = true;
    } else
    {
       // Update noise floor with exponential smoothing in Q15:
       // equivalent to g_noise_floor = (1.0f - NOISE_ALPHA) * g_noise_floor +
       // NOISE_ALPHA * result. Mathematically optimized to: g_noise_floor =
       // g_noise_floor + NOISE_ALPHA * (result - g_noise_floor)
       g_noise_floor = (q15_t)(g_noise_floor + ((((q31_t)result - g_noise_floor) * NOISE_ALPHA) >> 15));
    }
}

/**
 * @brief  Convert a byte array into a floating-point audio sample.
 * @param  sample Pointer to the raw sample (8 bytes).
 * @return 32-bit floating-point sample normalized to the range [-1.0, 1.0].
 */
static inline float32_t i2s_sample_to_float32(uint8_t* sample)
{
    int32_t reord_sample = (int32_t)( (sample[1] << 16 ) |
                                      (sample[0] << 8  ) |
                                      (sample[3]       )
							        );

    if (reord_sample & 0x00800000)   // If the 24-bit sign bit is set
    {
        reord_sample |= 0xFF000000;  // Sign-extend to 32 bits
    }

    // Convert the 24-bit sample to a float value in the range [-1.0, 1.0]
    return (float32_t) (reord_sample / 8388608.0f); // Divide by 2^23 to normalize
}


/**
 * @brief  Convert a byte array into a Q15 audio sample.
 * @param  sample Pointer to the raw sample (8 bytes).
 * @return Q15 sample.
 */
static inline q15_t i2s_sample_to_q15(uint8_t *sample) 
{
    return (q15_t)((sample[1] << 8) | sample[0]);
}


/**
 * @brief  Convert an array of Q8.7 fixed-point values to float32.
 * @param  dst Destination float array.
 * @param  src Source Q15 array.
 * @param  length Number of elements.
 */
static inline void q8_7_to_float32(float32_t *dst, const q15_t *src, uint32_t length)
{
    // MFCC output of arm_mfcc_q15 is interpreted as q8.7 fixed-point, so we need to divide by 128 to get the float value.
    for (uint32_t i = 0; i < length; i++) 
    {
        dst[i] = (float32_t)(q31_t)src[i] / 128.0f; // Convert from Q8.7 to float
    }
}

static bool rgb_timeout_expired(void)
{
    // Implement a non-blocking timeout mechanism (e.g., using HAL_GetTick())
    const uint32_t timeout_duration = 4000; // 4 seconds

    uint32_t current_time = HAL_GetTick();
    if ((current_time - g_last_turn_on_time) >= timeout_duration)
    {
    	g_last_turn_on_time = current_time; // Reset the timer
        return true; // Timeout expired
    }
    return false; // Timeout not yet expired
}
