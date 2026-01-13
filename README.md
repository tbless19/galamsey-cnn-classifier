Galamsey Image Classification Using CNNs and Apache Spark

Author: Prince Tawiah
Program: M.S. Data Science, Michigan Technological University
Course: SAT-5165 – Introduction to Big Data Analytics
Instructor: Dr. Neerav Kaushal

--------------------------------------------------

OVERVIEW

Galamsey (illegal gold mining) is a major environmental and socio-economic problem in Ghana, causing deforestation, water pollution, and long-term land degradation. Manual inspection of satellite imagery for galamsey detection is slow, expensive, and not scalable.

This project implements an end-to-end deep learning pipeline for detecting galamsey
(illegal small-scale gold mining) activities from satellite imagery. The complete
workflow is implemented in a single Jupyter notebook (final.ipynb), covering data
preparation, distributed preprocessing with Apache Spark, CNN model training, and
evaluation.

The task is formulated as a binary image classification problem:
- Galamsey
- Non-galamsey (natural environments)

--------------------------------------------------
RESULTS SUMMARY

- Designed and implemented an end-to-end CNN-based image classification pipeline for detecting illegal mining (galamsey) from satellite imagery using Python, TensorFlow/Keras, and Apache Spark.

- Processed 200 high-resolution satellite images into 15,128 labeled image patches using patch extraction with source-level train/validation/test splitting to prevent data leakage.

- Applied data augmentation, dropout, batch normalization, and early stopping to mitigate overfitting on a limited dataset.

- Achieved approximately 96.7 percent test accuracy with a ROC-AUC of approximately 0.99, demonstrating strong separability between galamsey and non-galamsey classes.

- Obtained high precision and recall for both classes (F1-score up to 0.97 for galamsey), validated through confusion matrix and ROC analysis.

- Integrated Apache Spark for distributed image preprocessing, enabling a scalable framework suitable for large-scale environmental monitoring applications.

-------------------------------------------------
WORKFLOW

All core logic is contained in:

final.ipynb

The notebook includes the following stages:
1. Data loading and inspection
2. Patch extraction from high-resolution satellite images
3. Source-level train/validation/test splitting
4. Data augmentation (training set only)
5. Distributed preprocessing using Apache Spark
6. CNN model definition and training
7. Model evaluation and visualization

--------------------------------------------------

DATASET DESCRIPTION

Base images:
- 200 total images
  - 100 galamsey
  - 100 non-galamsey

Sources:
- Publicly viewable Google Maps and Google Earth satellite imagery

Image resolution:
- Approximately 800x800 to over 3000x3000 pixels

Patch extraction:
- Patch size: 128 x 128 pixels
- Non-overlapping tiling
- Patches inherit labels from their source image

Extracted patches:
- Galamsey: 10,716
- Non-galamsey: 4,412
- Total: 15,128

--------------------------------------------------

TRAIN / VALIDATION / TEST SPLIT

Splitting is performed at the source image level to prevent data leakage:
- Training: 70 percent of base images
- Validation: 15 percent
- Test: 15 percent

All patches from the same base image remain in the same split.

--------------------------------------------------

DATA AUGMENTATION

Applied only to the training set:
- Random rotations up to ±20 degrees
- Horizontal and vertical flips
- Width and height shifts up to 20 percent
- Random zoom up to 20 percent
- Brightness and minor color variations

After augmentation:
- Training patches: 43,832
- Validation patches: 2,022
- Test patches: 2,148

--------------------------------------------------

APACHE SPARK INTEGRATION

Apache Spark is used within the notebook to demonstrate scalable preprocessing:
- Distributed image loading
- Parallel patch extraction
- Metadata mapping (label, source image, split)
- Caching of intermediate results

The pipeline is designed to scale to larger satellite image archives.

--------------------------------------------------

CNN ARCHITECTURE

Input:
- 128 x 128 x 3 RGB images

Model structure:
- Convolutional block 1:
  - Conv2D (32 filters, 3x3, ReLU)
  - Batch normalization
  - Max pooling (2x2)
- Convolutional block 2:
  - Conv2D (64 filters, 3x3, ReLU)
  - Batch normalization
  - Max pooling
  - Dropout (0.25)
- Convolutional block 3:
  - Conv2D (128 filters, 3x3, ReLU)
  - Batch normalization
  - Max pooling
  - Dropout (0.25)
- Dense head:
  - Flatten
  - Dense (128 units, ReLU)
  - Dropout (0.50)
  - Sigmoid output neuron

Loss function: Binary cross-entropy
Optimizer: Adam

--------------------------------------------------

TRAINING CONFIGURATION

- Learning rate: 0.001
- Batch size: 32
- Maximum epochs: 10

Callbacks:
- Early stopping (patience = 3, monitoring validation loss)
- Model checkpointing (best validation accuracy)
- Reduce learning rate on plateau (factor = 0.5)

--------------------------------------------------

RESULTS

The final CNN model demonstrates strong performance on the held-out test set.

Overall performance:
- Test accuracy: approximately 96.7 percent
- ROC-AUC: approximately 0.99

Classification metrics (test set):

Non-galamsey:
- Precision: 0.95
- Recall: 0.91
- F1-score: 0.93
- Support: 608

Galamsey:
- Precision: 0.96
- Recall: 0.98
- F1-score: 0.97
- Support: 1,540

Confusion matrix summary:
- 552 of 608 non-galamsey patches correctly classified
- 1,511 of 1,540 galamsey patches correctly classified

Most misclassifications occur in visually ambiguous regions such as exposed soil
or seasonal vegetation changes.

--------------------------------------------------

HOW TO RUN

1. Clone the repository:
git clone https://github.com/YOUR_USERNAME/galamsey-cnn-spark-classifier.git
cd galamsey-cnn-spark-classifier

2. Install dependencies:
pip install -r requirements.txt

3. Launch Jupyter Notebook:
jupyter notebook final.ipynb

4. Run all cells sequentially to reproduce results.

--------------------------------------------------

LIMITATIONS AND FUTURE WORK

- Dataset limited to 200 base images
- RGB imagery only (no multispectral bands)
- Binary classification only

Future work includes:
- Transfer learning (ResNet, EfficientNet)
- Multispectral imagery (Sentinel, Landsat)
- Multi-class severity classification
- Temporal monitoring of mining activity
- Deployment on larger Spark clusters

--------------------------------------------------

LICENSE

MIT License

--------------------------------------------------

CONTACT

Prince Tawiah
Graduate Student – Data Science
Michigan Technological University
Email: ptawia3@mtu.edu
