# 🛡️ ScamShield – AI-Powered Fake Job & Internship Detection System

<p align="center">
  <strong>Detect Fake Job Postings. Protect Job Seekers.</strong>
</p>

<p align="center">
  <a href="https://scamshield-xhxdsyjureq8wpv8zsycfp.streamlit.app/">
    🚀 Live Demo
  </a>
  |
  <a href="https://github.com/snehasuman019/ScamShield">
    💻 GitHub Repository
  </a>
</p>

---

## 🌐 Live Application

**Try ScamShield here:**

### [🚀 Launch ScamShield – Live Demo](https://scamshield-xhxdsyjureq8wpv8zsycfp.streamlit.app/)

ScamShield is an AI-powered web application designed to identify potentially fraudulent job and internship postings using machine learning and natural language processing.

---

## 📌 About the Project

Online recruitment fraud has become a serious concern for job seekers. Fraudulent recruiters often publish fake job advertisements to collect personal information, demand payments, or deceive applicants.

ScamShield aims to address this problem by analyzing job posting information and predicting whether a listing is potentially fraudulent or legitimate.

The application combines machine learning classification, NLP-based text analysis, an interactive Streamlit interface, and MongoDB integration for database operations.

**Academic Project:** CSE494 – Intelligent NoSQL Databases

---

## ✨ Key Features

- **AI-Based Job Scam Detection:** Predicts whether a job posting is fraudulent or legitimate.
- **NLP Text Analysis:** Uses TF-IDF and text-based feature extraction.
- **Machine Learning Models:** Includes Logistic Regression and Random Forest models.
- **Fraud Risk Assessment:** Displays prediction results and model evaluation information.
- **Prediction History:** Provides an interface for reviewing saved predictions.
- **Job Management:** Supports job listing management and dataset import workflows.
- **Analytics Dashboard:** Presents job and prediction statistics.
- **Model Evaluation:** Displays model performance metrics.
- **MongoDB Atlas Integration:** Supports persistent storage and database operations when Atlas is configured.

---

## 🧠 Machine Learning Models

ScamShield includes two trained machine learning models:

| Model | Approach |
|---|---|
| Logistic Regression | NLP-based text classification using TF-IDF |
| Random Forest | Classification using multiple job-related attributes |

### Logistic Regression

- Text preprocessing
- TF-IDF vectorization
- Unigram and bigram feature extraction
- Fraudulent versus legitimate job classification

### Random Forest

Uses job-related attributes to classify potentially fraudulent postings.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF |
| Data Processing | Pandas, NumPy |
| Database | MongoDB Atlas |
| Database Driver | PyMongo |
| Model Serialization | Joblib |
| Visualization | Plotly |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

---

## 📂 Project Structure

```text
ScamShield/
│
├── analytics/
│   ├── __init__.py
│   └── reports.py
│
├── data/
│   ├── README.md
│   └── download_dataset.py
│
├── database/
│   ├── __init__.py
│   ├── connection.py
│   ├── indexes.py
│   ├── operations.py
│   └── validation.py
│
├── ml/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
│
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── model_metrics.json
│
├── pages/
│   ├── 1_Job_Scam_Detector.py
│   ├── 2_Prediction_History.py
│   ├── 3_Job_Management.py
│   ├── 4_Analytics.py
│   └── 5_Model_Evaluation.py
│
├── tests/
│   ├── test_config_and_startup.py
│   ├── test_database_operations.py
│   ├── test_predictions.py
│   └── test_preprocessing.py
│
├── app.py
├── config.py
├── conftest.py
├── import_data.py
├── requirements.txt
├── ui_utils.py
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/snehasuman019/ScamShield.git
```

### 2. Navigate to the Project Directory

```bash
cd ScamShield
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project root using `.env.example` as a template.

```env
MONGODB_URI=your_mongodb_atlas_connection_string
DB_NAME=scamshield_db
APP_ENV=development
APP_DEBUG=True
```

**Security:** Never commit your `.env` file or expose your MongoDB credentials.

MongoDB Atlas configuration is optional for running the application in offline/demo mode. Database-dependent functionality requires a valid Atlas connection.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open locally at:

```text
http://localhost:8501
```

---

## 🗄️ Database Integration

ScamShield uses MongoDB Atlas for persistent data storage.

The application is designed around the following collections:

- `jobs` – Stores job posting information.
- `predictions` – Stores prediction records.
- `feedback` – Stores user feedback.
- `model_metrics` – Stores model-related metrics.

Database validation, CRUD operations, indexes, and analytics workflows are implemented in the `database/` and `analytics/` modules.

**Note:** Persistent database features require MongoDB Atlas credentials and a reachable Atlas cluster. Without them, the application can operate in local/offline mode for supported ML functionality.

---

## 📊 Application Modules

| Module | Description |
|---|---|
| Dashboard | Displays application statistics and model status |
| Job Scam Detector | Analyzes job postings and predicts fraud risk |
| Prediction History | Provides access to saved prediction records |
| Job Management | Supports job record management and dataset import |
| Analytics | Displays analytical insights |
| Model Evaluation | Presents machine learning performance metrics |

---

## 🧪 Testing

The project includes automated tests using Pytest.

Run the test suite:

```bash
pytest
```

---

## 🚀 Deployment

ScamShield is deployed using **Streamlit Community Cloud**.

- Repository: [GitHub – ScamShield](https://github.com/snehasuman019/ScamShield)
- Branch: `main`
- Main application file: `app.py`

### Live Deployment

[https://scamshield-xhxdsyjureq8wpv8zsycfp.streamlit.app/](https://scamshield-xhxdsyjureq8wpv8zsycfp.streamlit.app/)

---

## 🔮 Future Enhancements

- Docker containerization.
- Jenkins-based CI/CD automation.
- AWS EC2 deployment.
- Terraform-based infrastructure provisioning.
- Ansible configuration management.
- Puppet automation.
- Nagios server and application monitoring.
- Improved fraud detection using advanced NLP techniques.
- Enhanced database analytics and reporting.

---

## 🎯 Project Objective

The primary objective of ScamShield is to develop an intelligent system that helps job seekers identify suspicious recruitment opportunities through machine learning and data-driven analysis.

The project also demonstrates the practical integration of machine learning, NoSQL database technologies, web application development, and cloud deployment.

---

## 👩‍💻 Author

**Sneha Suman**

- GitHub: [@snehasuman019](https://github.com/snehasuman019)
- Project: [ScamShield](https://github.com/snehasuman019/ScamShield)
- Live Demo: [ScamShield Web Application](https://scamshield-xhxdsyjureq8wpv8zsycfp.streamlit.app/)

---

<p align="center">
  <strong>ScamShield – Making Online Recruitment Safer with AI.</strong>
</p>