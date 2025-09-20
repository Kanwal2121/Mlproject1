 Math Score Predictor

A web application that estimates a student's math score based on their academic & demographic profile. Built with Flask and a custom prediction pipeline.

---

 Purpose

This app helps users predict a student's math score using inputs such as gender, parental education, test preparation, reading score, and writing score. It’s intended for demonstration, learning, and educational analytics.

---

 Features

- Web interface (HTML templates) for entering student details  
- Backend prediction pipeline processing inputs and returning a math score  
- Notebook for exploratory data analysis  
- Deployment-ready configuration (support for AWS Elastic Beanstalk via `.ebextensions`)  

---

 Tools

| Component | Tool / Library |
|-----------|----------------|
| Backend | Python, Flask |
| Data Handling / Processing | Pandas, NumPy, scikit-learn |
| Web Templates | Jinja2, HTML |
| Environment | `requirements.txt`, `pyproject.toml` |
| Deployment | AWS Elastic Beanstalk (config files included) |

---

 Project Structure
 Mlproject1/
│
├── application.py              # Main Flask app, defines routes
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata (optional)
├── README.md                   # Project documentation
├── .ebextensions/              # AWS EB deployment configs
├── templates/                  # HTML templates (home, index, etc.)
├── src/                        # ML pipeline, preprocessing and utilities
├── notebook/                   # Exploratory data analysis (Jupyter notebooks)
├── artifacts/                  # Saved models, output files, logs, etc.
└── gitignore                   # Ignored files configuration


