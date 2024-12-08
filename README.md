# Handwritten Medicine Name Recognition

This project implements a deep learning-based solution to recognize medicine names from handwritten prescriptions. The application also provides detailed information about the identified medicines using an integrated dataset.

## Features

- **Medicine Name Classification**:
  - A trained CNN model is used to predict the names of medicines from prescription images.
- **Web Application**:
  - Built using Flask for uploading prescriptions and displaying results interactively.
- **Information Retrieval**:
  - Retrieves detailed medicine information (uses, side effects, etc.) from an Excel file.

## File and Directory Overview

### Main Scripts

- **`app.py`**:
  - The entry point of the Flask web application.
  - Handles image uploads and displays medicine recognition results.
- **`doctors's prescription.py`**:
  - A script for processing prescription images, possibly preprocessing or augmenting the dataset.

### Model and Resources

- **`prescription_classification_model.keras`**:
  - The trained CNN model used for classifying handwritten medicine names.
- **`label_encoder.pkl`**:
  - A serialized label encoder for mapping predicted indices to medicine names.
- **`medicine_information.xlsx`**:
  - An Excel sheet containing detailed information about medicines, including alternate names, uses, dosages, side effects, and more.

### Directories

- **`Medicine name dataset`**:
  - Contains the dataset used for training and testing the model.
- **`static/`**:
  - Stores static files for the web application (e.g., CSS, JavaScript, images).
- **`templates/`**:
  - Contains HTML templates for the Flask application.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/handwritten-medicine-recognition.git
   cd handwritten-medicine-recognition
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Ensure Tesseract OCR is installed if required for additional functionality.

## Usage

1. Start the Flask application:

   ```bash
   python app.py
   ```

2. Open the application in your browser at `http://127.0.0.1:5000/`.

3. Upload a prescription image to get the recognized medicine name and its details.

## Dataset

- Approximately 4,000 images of handwritten prescriptions.
- Labeled with medicine names, with multiple handwriting variations per name.

## Model Details

- A Convolutional Neural Network (CNN) was trained to classify images into medicine names.
- Preprocessing steps included resizing and normalization.
- Achieved an accuracy of \~70%.

## Challenges and Solutions

1. **Handwriting Variability**:
   - Addressed using data augmentation.
2. **Overfitting**:
   - Reduced with dropout layers and regularization.

## Future Work

- Enhance model accuracy with larger and more diverse datasets.
- Deploy as a standalone mobile or web application for real-time use.
- Integrate multilingual support for recognizing prescriptions in other languages.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset Sources**: Kaggle and manual annotations.
- **Libraries and Frameworks**: TensorFlow, Flask, OpenCV, Pandas.

## Contribution

Contributions are welcome! Please create a pull request or open an issue for suggestions or improvements.

