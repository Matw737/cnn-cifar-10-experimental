# CNN Architecture & Regularization Study

## Overview

This project explores how custom CNN architectures and training strategies affect image classification and generalization on CIFAR-10.

I built the models from scratch in PyTorch and developed them through a series of experiments. The work started with a residual CNN that reached around 90% validation accuracy but showed signs of overfitting. I then tested different convolutional structures, data augmentation schedules, and a new classification head.

The main goal was to investigate whether the models could learn more useful image features and improve validation performance without relying on a pretrained network.

## Experiment Goals

The initial model sometimes appeared to rely too much on background and context. This led to two main questions:

- Could parallel convolutional branches with different kernel sizes help the model learn features at different spatial scales?
- Could a gradual reduction of feature channels preserve useful spatial information better than applying global average pooling and a linear layer directly?

The experiments were exploratory. The results show how the tested designs performed, but they do not prove that one architectural change alone caused every improvement.

## Model Architectures

### BasicCNN — Baseline

BasicCNN served as the reference model. It uses seven residual blocks with standard 3×3 convolutions. The number of channels increases from 32 to 1024, with spatial downsampling at selected stages.

The model ends with global average pooling (GAP) and a linear classification layer.

- **Parameters:** approximately 31.5 million
- **Best validation accuracy after fine-tuning:** 90.61%

### DualKernelCNN

DualKernelCNN was designed to test whether using different receptive fields could improve feature extraction.

Each custom residual block has two parallel branches:

- A branch with consecutive 3×3 convolutions
- A branch with consecutive 5×5 convolutions

The branch outputs are combined, allowing the model to use features from different spatial scales. The architecture also delays its first downsampling stage to preserve spatial resolution for longer.

- **Parameters:** 26.72 million
- **Best validation accuracy:** approximately 91.04%

The model achieved a higher validation accuracy than the baseline despite having fewer parameters.

### Dual-Kernel ResNest — “Hellcut”

The Dual-Kernel ResNest extends the DualKernelCNN with nested parallel feature-extraction paths. The outputs are concatenated, mixed with 1×1 convolutions, and combined through residual connections.

The design was developed experimentally rather than from a fixed mathematical formulation. Its goal was to test whether more feature-extraction paths could improve on the DualKernelCNN results.

- **Parameters:** 36,980,522
- **Final validation accuracy:** 91.26%

The larger model did not produce a large accuracy gain over DualKernelCNN. However, its validation loss remained relatively stable during the later stages of training.

### Experimental Classification Head

The original classification head uses global average pooling followed by a linear layer. I suspected that this reduced the spatial feature maps too quickly.

To test this, I added a second classification path. It uses convolutional layers to gradually reduce the channel count from 1024 to 10 while keeping the spatial feature maps. Global average pooling is applied after this reduction.

The outputs from the original GAP + Linear path and the convolutional path are added to produce the final class scores.

The head was tested with the DualKernelCNN. It reached **91.86% validation accuracy at epoch 24**, outperforming the tested DualKernelCNN without the head and the Dual-Kernel ResNest.

The parameter count for this model was not recorded in the available results.

## Training and Regularization

### Progressive Data Augmentation

Strong augmentation from the beginning made it difficult for the model to learn useful features. In early experiments, the model sometimes collapsed to predicting a single class.

I introduced progressive augmentation to increase the strength of image transformations gradually during training. The pipeline includes horizontal flipping, color jitter, translation, and Gaussian blur.

### Regressive Data Augmentation

During fine-tuning, strong augmentation appeared to limit further improvement. Regressive augmentation gradually reduces the transformation strength, giving the model a chance to refine the features it has learned.

The goal was to improve validation performance while monitoring for overfitting.

### Optimizer and Learning-Rate Schedule

The models were trained using AdamW and a cosine learning-rate schedule. Training runs also used validation metrics to track performance and identify useful checkpoints.

## Results

| Model | Parameters | Validation accuracy |
|---|---:|---:|
| BasicCNN, after fine-tuning | ~31.5M | 90.61% |
| DualKernelCNN | 26.72M | ~91.04% |
| Dual-Kernel ResNest | 36,980,522 | 91.26% |
| DualKernelCNN + experimental head | Not recorded | 91.86% at epoch 24 |

These values come from the experiments recorded in this project. They should be read as results from the tested training runs, not as a controlled proof that a single architectural change caused the full difference.

### Training Behavior

For DualKernelCNN with the experimental head, the training accuracy reached **99.55%** at epoch 24, while validation accuracy was **91.86%**. The validation loss had increased to approximately **0.406**.

This shows a clear gap between training and validation performance. Although validation accuracy continued to improve slightly, overfitting remained a challenge.

### Confusion Matrices

The confusion matrices provide more detail than overall accuracy. The most common errors were between visually similar animal classes, especially cats and dogs. Some confusion also occurred between birds, cats, and deer.

Vehicle classes such as airplanes, automobiles, ships, and trucks were generally recognized more reliably.

## Main Findings

- The DualKernelCNN performed better than the baseline in the recorded experiments while using fewer parameters.
- The nested Dual-Kernel ResNest increased model capacity, but its accuracy improvement over DualKernelCNN was small.
- The experimental classification head achieved the highest validation accuracy recorded in these experiments.
- The training logs show that high training accuracy did not guarantee equally high validation accuracy.
- Similar-looking animal classes remained a major source of classification errors.

## Limitations and Future Work

These experiments were developed iteratively, so not every model was tested under exactly the same conditions. Parameter count and accuracy alone also do not describe the full cost of an architecture; training time and memory use matter too.

Possible next steps include:

- Compare the models using the same training setup and checkpoint-selection rule.
- Record parameter counts and training time for every model.
- Evaluate the selected checkpoint on the test set.
- Add Grad-CAM or other activation analysis to inspect which image regions influence predictions.
- Investigate whether the experimental head improves results consistently across multiple runs.

## Technologies

- Python
- PyTorch
- CIFAR-10
