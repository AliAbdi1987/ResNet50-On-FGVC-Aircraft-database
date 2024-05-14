# FGVC Aircraft ResNet50 Implementation

This project implements a ResNet50 model to classify images from the FGVC Aircraft dataset. Below is a detailed description of the steps and components used in the implementation.

## Project Structure

### Dataset Preparation

- **AircraftDataset class**: Custom dataset class to handle the FGVC Aircraft images and labels.
- **Image augmentations**: Using `torchvision.transforms` for training and validation.

### Model Definition

- **Pre-trained ResNet50 model**: From `torchvision.models` fine-tuned for aircraft classification.
- **Model modification**: Adjusted to fit the number of classes in the FGVC Aircraft dataset.

### Training and Validation

- **Data loaders**: For training and validation.
- **Loss function**: Cross-entropy loss.
- **Optimizer**: Adam optimizer with a learning rate scheduler.
- **Training loop**: Includes periodic checkpoint saving and logging.
- **Evaluation metrics**: Accuracy and loss.

### Checkpoints and Logging

- **Checkpoints**: Automatic checkpointing every 5 epochs and at the end of training.
- **Logging**: Training and validation loss and accuracy are logged to a CSV file.

### Visualization

- **Plots**: Training and validation loss and accuracy over epochs.

## Dependencies

- `torch`
- `torchvision`
- `pandas`
- `PIL`
- `matplotlib`

## Usage

### Clone the Repository

```bash
git clone https://github.com/yourusername/fgvc_aircraft_resnet50.git
cd fgvc_aircraft_resnet50
