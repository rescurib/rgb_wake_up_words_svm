# SVM Model Training

This directory contains the collected datasets and the Python scripts needed to analyze, train, and format the Support Vector Machines (SVM) used by the firmware on the STM32 board.

## Feature Vector Structure

The feature vector computed by the STM32 board represents a frame of 16 audio hops. For each hop (32 ms), the firmware calculates the standard MFCC coefficients and their first temporal derivatives (Delta-MFCCs).

Both sets of coefficients are interleaved hop by hop into a final, flat 1-dimensional array. The resulting structure consists of `2 * MFCC_COEFFS_NUM * HOPS_PER_FRAME` elements mathematically arranged as follows:

```text
[ MFCCs Hop 1 ] [ Delta-MFCCs Hop 1 ] [ MFCCs Hop 2 ] [ Delta-MFCCs Hop 2 ] ... [ MFCCs Hop 16 ] [ Delta-MFCCs Hop 16 ]
```

Each block in the sequence contains `MFCC_COEFFS_NUM` elements. This combined feature vector acts as the input for the Support Vector Machine (SVM) classifier. Before prediction or UART transmission, the `q15_t` values (specifically `q8.7` used by CMSIS-DSP) are converted to `float32_t` decimal format.

## Collecting Samples (Dataset)
To collect your own samples, clone [this](https://github.com/rescurib/mfcc_mic_recorder) repository. That firmware will send a feature vector of a sequence of 16 hops every time you speak into the microphone.

Use the `serial_to_features.py` script (located in the _root directory_ of the project) to capture the incoming burst of floating-point data via UART and aggregate them into `.npy` files.

For example:

```bash
python serial_to_features.py --output Training/mfcc_blue.npy
```
*You should repeat this process and rename the `--output` parameter for each new class you want to record.*

## Training the Model (`train_svm_mfcc.py`)
Once you have the collected files in this directory (for example: `mfcc_a.npy`, `mfcc_e.npy`, `mfcc_i.npy`, `mfcc_o.npy`, `mfcc_u.npy`), you are ready to run the main training script:

```bash
python train_svm_mfcc.py
```

### How does `train_svm_mfcc.py` work internally?
1. **Automatic Loading**: It searches for any file named `mfcc_<label>.npy` within this directory. It concatenates them vertically and assigns each its own text label by dynamically reading the file name.
2. **Linear SVM under OVR rule**: Since the CMSIS-DSP library is efficient at evaluating binary linear models, Python splits the multiclass problem using `OneVsRestClassifier` instantiating `LinearSVC` from _Scikit-Learn_. It will internally generate $N$ independent classifiers.
3. **Model Metrics**: It evaluates and splits the dataset (80% training, 20% testing), and prints a detailed accuracy and recall report to the console.
4. **C Header Rendering**: It mathematically extracts the geometric configuration of the trained models (.coef_ and .intercept_) and builds/formats constant arrays in C language. Finally, it saves and prints the **`svm_params.h`** file, which is exactly the same one imported by `clasificador_svm.c` in the firmware.

## Visualization (`plot_mfcc_pca.py`)
To check the quality of your dataset or recovered audios, you can run this auxiliary visualization script:
```bash
python plot_mfcc_pca.py
```
The script will read the files in the same format, standardize them, and apply Principal Component Analysis (PCA) to compress the geometric dimensionality and show you a 2D figure. With this, you can visually confirm if the captured vowels manage to group (be separable) in the MFCC spectrum. Here is an example I made for the vowels:

<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1SPOT91A9TNcxonFPD9OUkRUxNS6wCCrs" width="500">
<p>
