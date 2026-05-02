# Words Classifier with SVMs

This project implements an SVM classifier for the detection of three wake up words: *red*, *green*, and *blue*, using an STM32F3 microcontroller. The system acquires audio through an INMP441 MEMS microphone, extracts MFCC features, and classifies the vowel using Support Vector Machines (SVM).

## Main Features

- **Audio Acquisition**: I2S interface with INMP441 MEMS microphone at 8 kHz.
- **Signal Processing**:
  - Extraction of 13 MFCC coefficients per hop in fixed point Q8.7 format.
  - Delta MFCC computing. 
- **Classification**:
  - Implementation of multiple "one-vs-rest" SVM classifiers using the CMSIS-DSP library.
- **User Interface**:
  - User button (B1) to start/stop recording.
  - Status LED indicating recording state.
  - UART communication for debugging and reporting results.

## Project Structure

```
clasificador_vocales_svm/
├── Inc/                      # Application headers
│   ├── app.h                 # Main application header
│   ├── clasificador_svm.h    # SVM classifier definitions
│   ├── mfcc_features.h       # MFCC extraction definitions
│   └── ...                   # Other system headers
├── Src/                      # Source code
│   ├── app.c                 # Main application logic
│   ├── clasificador_svm.c    # SVM classifier implementation
│   ├── mfcc_features.c       # MFCC extraction implementation
│   └── ...                   # Initialization and HAL code
├── STM32CubeIDE/             # IDE configuration
├── Drivers/                  # HAL and CMSIS drivers
└── README.md                 # Project documentation
```

## Hardware Requirements

- STM32F3xx microcontroller
- INMP441 MEMS microphone
- Status LED (LD2)
- User button (B1)

<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1k-mYVQVk2T5NjttuTqAW95a4pcD9Cpar" width="500">
<p>

## Configuration

### Sampling Frequency
The system is configured for a **8 kHz** sampling frequency.

### MFCC Parameters
- **Hop size**: 256 samples (32 ms)
- **Number of hops per frame**: 16
- **Total frame duration**: 512 ms
- **Number of MFCC coefficients**: 13

### Detection Parameters
- **Signal detection threshold**: 7.5 X noise level (RMS)
- **Noise filter**: Exponential smoothing with alpha = 0.01

See [Training README](https://github.com/rescurib/rgb_wake_up_words_svm/tree/main/Training#readme) for SVM model generation, training and dataset creation.

## Usage

1. **Compile and flash** the firmware to the microcontroller using STM32CubeIDE.
2. **Press the user button (B1)** to start audio acquisition. The message "Iniciando grabacion" (Recording started) or "Grabacion detenida" (Recording stopped) will be shown via UART each time the button is pressed.
3. The **status LED** will light up while the system is recording.
4. The system will process audio in 512 ms blocks and classify the detected words.
5. Results will be shown via **UART** in the following format:

   ```
   Detected word (initial): X
   ```
   Where `X` is the initial of the word.

### Serial Monitoring (UART)
To view the scans and results emitted by the microcontroller, you must properly configure your serial terminal. The system operates at a very high speed to transfer floating-point vectors in time. The port must be set specifically to a **Baudrate of 460800 bps** (8 data bits, no parity, 1 stop bit).

#### On Windows
You can use the following commonly recommended GUI programs:
- **PuTTY**: In *Connection type* select `Serial`. In *Serial line* specify your port (e.g., `COM3`) and change the *Speed* (baudrate) to `460800`.
- **Tera Term** or **RealTerm**: Make sure to select the corresponding COM port number and go to Serial Port settings to set the same baudrate (460800).

#### On Linux
On robust distributions like Ubuntu/Debian, your board's port will often appear as `/dev/ttyACM0` (or `ttyUSB0`). You can use terminal utilities like `screen` or `picocom`.

Example with `screen`:
```bash
sudo screen /dev/ttyACM0 460800
```
*(To exit the screen interface press `Ctrl+A` followed by `K`)*

Optionally, using the robust and lightweight `picocom`:
```bash
sudo picocom -b 460800 /dev/ttyACM0
```

## Implementation Notes

- The `app_run()` function is the main application loop.
- GPIO and I2S interrupts handle data acquisition and system control.
- The CMSIS-DSP library is used for efficient signal processing.

## Measurements

| Function | Execution time |
|----------|---------------|
| `mfcc_features_compute` | 850 µs |
| `clasificador_svm_predict` | 90 - 180 µs |

## References
* [MFCC´s](https://turing.iimas.unam.mx/~ivanvladimir/posts/mfcc/), Ivan Meza Ruiz
* [Intuitive understanding of MFCCs](https://medium.com/@derutycsl/intuitive-understanding-of-mfccs-836d36a1f779), Emmanuel Deruty
* [Mel-Frequency Cepstral Coefficients Explained Easily](https://www.youtube.com/watch?v=4_SH2nfbQZ8), Valerio Velardo
* [Mel Frequency Cepstral Coefficient and its Applications: A Review](https://ieeexplore.ieee.org/document/9955539), IEEE, Open Access
* [Support Vector Machines](https://www.youtube.com/watch?v=efR1C6CvhmE&list=PLblh5JKOoLUL3IJ4-yor0HzkqDQ3JmJkc), StatQuest with Josh Starmer