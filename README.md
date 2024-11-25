# Loan Prediction Web Application

This project is a web-based application designed to predict the likelihood of loan approvals. It features:
- A **FastAPI** backend that handles machine learning predictions.
- A **Streamlit** frontend that provides a user-friendly interface for making loan predictions.

The application is containerized with Docker for easy deployment and scalability, but you can also run it locally with Python.



## **Features**
- Predict loan approval probabilities using machine learning.
- Interactive web interface for entering loan data and viewing results.
- API endpoint for integrating predictions into other applications.

## **Documentation**
- [Introduction to the project](https://medium.com/@muchaibriank/end-to-end-machine-learning-project-loan-approval-prediction-part-1-introduction-541e639b7c5c)
- [Development Documentation](https://medium.com/@muchaibriank/end-to-end-machine-learning-project-loan-approval-part-2-development-0e437b084ad5)
- [Deployment and CI/CD Documentation](https://medium.com/@muchaibriank/end-to-end-machine-learning-project-loan-approval-part-3-deployment-and-ci-cd-pipeline-0db5e8d7d078)
- [API Documentation](https://loanpredictor.gitbook.io/api-docs)

## **Getting Started**

You can set up and run this project in two ways: using **Docker** or installing dependencies manually with Python.

### **1. Running with Docker**
Ensure you have **Docker** and **Docker Compose** installed on your machine.

1. Clone the repository:
   ```bash
   git clone https://github.com/muchai254/loan-prediction
   cd loan-prediction
   ```
2. Build and run the containers
    ```bash
    docker-compose up --build
    ```

### **2. Running locally with Python**
Ensure you have **Python 3.8+** and **pip** installed.
1. Clone the repository:
   ```bash
   git clone https://github.com/muchai254/loan-prediction
   cd loan-prediction
   ```
2. Install dependencies
   ```bash
   cd app
   pip install -r requirements.txt
   cd server
   pip install -r requirements.txt
   ```
3. Start the fastAPI backend
   ```bash
   cd server
   fastapi dev server.py
   ```
4. Start the Streamlit frontend
   ```bash
   cd app
   streamlit run streamlit-app.py
   ```
### In both installations, the server will run on http://localhost:8000 and frontend at http://localhost:8501

## Contributing to the project
We welcome contributions! Follow the steps below to contribute:
### **1. Fork the Repository**
Click the Fork button at the top right of the repository page on GitHub.
### **2. Clone your fork**
```bash
  git clone https://github.com/muchai254/loan-prediction
  cd loan-prediction
```
### **3. Create a New Branch**
```bash
  git checkout -b your-feature-branch
```
### **4. Create a Pull Request**
Please **do not** push changes directly to the `master` branch. Always use a new branch for your contributions.

