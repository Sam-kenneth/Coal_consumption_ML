Coal Consumption & Power Generation Analysis (Machine Learning Project)

💡 Project Goal

This project aims to analyze daily power generation reports and coal stock data to build a machine learning model that predicts or forecasts  coal consumption by India on a daily basis. The full pipeline handles data ingestion, cleaning, feature engineering, model training, and deployment (via Docker).

🚀 Getting Started

The most reliable way to run this application is by using the provided Docker container. This ensures that all dependencies and environmental configurations are handled automatically.

Prerequisites

Git: To clone the repository.

Docker: To build and run the containerized application.

Python (Optional): If you wish to run the project locally without Docker.

1. Cloning the Repository

First, clone the project to your local machine:

git clone [https://github.com/Sam-kenneth/Coal_consumption_ML.git](https://github.com/Sam-kenneth/Coal_consumption_ML.git)
cd Coal_consumption_ML


2. Running with Docker (Recommended)

Follow these steps to build the image and run the entire pipeline:

A. Build the Docker Image

Run the build command from the root directory (Coal_consumption_ML/), making sure the Dockerfile and requirements.txt are in the current context:

docker build -t coal_consumption_app .


B. Run the Container

The application will first execute the data cleaning/ingestion script (Data_ingestion_and_cleaning.py) and then launch the main application (application.py), which typically hosts the trained model or a basic web service.

# This command runs the container and removes it automatically after exit
docker run --rm -p 8000:8000 coal_consumption_app


The application should now be running and accessible at http://localhost:8000 (if your application.py hosts a web server).

3. Local Setup (Alternative)

If you prefer to run the project directly on your machine:

Install Dependencies:

pip install -r requirements.txt


Run the Pipeline:

# Step 1: Ingest and clean the data
python src/components/Data_ingestion_and_cleaning.py

# Step 2: Run the main application
python application.py


📂 Project Structure  (File/Folder: Description)

Dockerfile: Defines the environment and steps for building the Docker image.

requirements.txt: Lists all necessary Python dependencies.

application.py: The main entry point for running the finalized application or model serving.

setup.py: Configuration file for package installation (used by pip install -e .).

src/components/: Contains core modular components of the ML pipeline.

src/components/Data_ingestion_and_cleaning.py: Script responsible for loading raw data, cleaning, and initial feature creation.

notebook/data/: Directory holding the raw input CSV files.

artifacts/: Directory where trained models, processed data, or reports are saved.

⚙️ Dependencies

All required libraries are listed in requirements.txt. Key dependencies likely include:

pandas (for data manipulation)

numpy (for numerical operations)

scikit-learn (for machine learning models)

flask or fastapi (for model deployment, if applicable)