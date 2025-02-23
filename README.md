# Bikeshare Rental Model - ML Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚴 Overview
This project builds a machine learning model to predict bikeshare rentals based on various features such as weather conditions, time of day, and other relevant factors. The model is developed as part of a mini-project for **IISC**.

## 🛠 Tech Stack
The following libraries and tools are used in this project:
- **requests** - To handle API requests.
- **numpy** - For numerical computations.
- **pandas** - For data manipulation and analysis.
- **seaborn** - For statistical data visualization.
- **matplotlib** - For plotting and data visualization.
- **scikit-learn** - For building machine learning models.
- **joblib** - For model serialization and deserialization.
- **pydantic** - For data validation and settings management.
- **strictyaml** - For YAML-based configuration management.
- **ruamel.yaml** - For YAML parsing and writing.

## 📂 Project Structure
```bash
├── bikeshare_model        # Main package
│   ├── config            # Configuration files
│   │   ├── core.py
│   │   ├── __init__.py
│   ├── datasets          # Dataset storage
│   │   ├── bike-sharing-dataset.csv
│   │   ├── __init__.py
│   ├── processing        # Data processing pipeline
│   │   ├── data_manager.py
│   │   ├── features.py
│   │   ├── validation.py
│   │   ├── __init__.py
│   ├── trained_models    # Stored trained models
│   │   ├── config.yml
│   │   ├── pipeline.py
│   │   ├── predict.py
│   │   ├── train_pipeline.py
│   │   ├── VERSION
│   │   ├── __init__.py
├── requirements          # Dependency management
│   ├── requirements.txt
├── venv                  # Virtual environment
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
```

## ⚙️ Installation
```sh
git clone <repository_url>
cd bikeshare-rental-model
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements/requirements.txt
```

## 🚀 Usage
- **Preprocess the data**
  ```sh
  python bikeshare_model/processing/data_manager.py
  ```
- **Train the model**
  ```sh
  python bikeshare_model/trained_models/train_pipeline.py
  ```
- **Make predictions**
  ```sh
  python bikeshare_model/trained_models/predict.py --input sample_input.json
  ```

## ⚙️ Configuration
- Configurations are stored in **`config.yml`**.
- Strict YAML parsing is enforced using **strictyaml** and **ruamel.yaml**.

## 💾 Model Saving & Loading
- The trained model is saved using **joblib** and can be loaded for inference.

## 🤝 Contributions
Contributions are welcome! Feel free to submit pull requests or report issues.

## 📜 License
This project is for educational purposes under **IISC**. Licensing details to be determined.

---
**Author: Your Name**

