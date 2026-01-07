# Retail AI Inventory Management System



An end-to-end Flask + Machine Learning based system for intelligent retail inventory management.



---



## Features



\- Demand Forecasting (30-day)

\- Pricing Optimization

\- Inventory Reorder Recommendation

\- Business Alerts

\- Single \& Bulk Predictions

\- Analysis Page \& History



---



## Project Structure



```

Retail-AI-Inventory-System/
│
├── app.py                       # Main Flask application entry point
├── models.py                    # SQLAlchemy models & ML interface
├── requirements.txt             # Python dependencies
├── setup.py                     # Project setup metadata
├── Dockerfile                   # Docker configuration
├── README.md                    # Project documentation
├── bulk_input_sample.csv        # Sample CSV for bulk prediction
├── .gitignore
├── .gitattributes
│
├── BACKEND/                     # Machine Learning & Data Processing
│   ├── TRAIN.py                 # Model training pipeline
│   ├── TEST.py                  # Testing script
│   ├── TEST1.PY
│   ├── datagenerator.py         # Data generation & augmentation
│   ├── create_dir.py            # Utility for directory creation
│   ├── retail_store_inventory.csv
│   ├── retail_store_inventory_augmented.csv
│   ├── test_data.csv
│   │
│   ├── *.pkl                    # Trained ML models (Git LFS)
│   ├── model_results.json       # Model evaluation metrics
│   ├── analysis_*.json          # Saved analysis outputs
│   │
│   ├── catboost_info/           # CatBoost training artifacts
│   │   ├── catboost_training.json
│   │   ├── learn_error.tsv
│   │   ├── time_left.tsv
│   │   └── learn/
│   │       └── events.out.tfevents
│   │
│   └── checkpoints/             # Deep learning checkpoints
│       ├── epoch=3-step=3656.ckpt
│       └── epoch=5-step=5484.ckpt
│
├── models/                      # Models used by Flask app
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── catboost_model.pkl
│   └── preprocessing_objects.pkl
│
├── instance/
│   └── retail.db                # SQLite database (users, history)
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── images/
│
├── templates/                   # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── prediction_type.html
│   ├── prediction.html
│   ├── bulk_prediction.html
│   ├── bulk_results.html
│   ├── analysis.html
│   └── history.html
│
└── __pycache__/                 # Python cache files

```
### BACKEND Folder
Contains all machine learning–related code, datasets, trained models, 
and experiment artifacts. This folder is used for training, evaluation, 
and offline experimentation.

### models Folder
Contains the final trained models and preprocessing objects used directly 
by the Flask web application during prediction.

### instance Folder
Holds the SQLite database (`retail.db`) used by Flask for authentication 
and prediction history.

### templates & static
Frontend UI built using Jinja2 templates, Tailwind-style CSS, and JavaScript.



---



## Local Setup (Without Docker)



### 1. Clone Repository



```bash

git clone https://github.com/Pranathi1184/Retail-AI-Inventory-System.git

cd Retail-AI-Inventory-System

```



### 2. Create Virtual Environment



```bash

python -m venv venv

```



Activate:



Windows

```bash

venv\\Scripts\\activate

```



Mac / Linux

```bash

source venv/bin/activate

```



### 3. Install Dependencies



```bash

pip install --upgrade pip

pip install -r requirements.txt

```



Recommended Python version: 3.9 or 3.10



### 4. Run Application



```bash

python app.py

```



### 5. Open in Browser



```text

http://127.0.0.1:5000

```



---



## Bulk Prediction CSV Format



```csv

Store ID,Product ID,Category,Price,Inventory Level,Competitor Pricing,Holiday/Promotion

S001,P001,Electronics,499.99,120,480,1

S002,P002,Grocery,79.99,300,75,0

```



---



## Docker Setup (Optional)



Build Image



```bash

docker build -t retail-ai .

```



Run Container



```bash

docker run -p 5000:5000 retail-ai

```



---



## Common Issues



\- Python 3.13 build errors → Use Python 3.9 / 3.10

\- Daily forecast not showing → Pass enhanced\_forecasts to analysis.html

\- Large .pkl files → Handled using Git LFS



---



## Author



Pranathi Doddamani  

Final Year Engineering Student  

GitHub: https://github.com/Pranathi1184



---



Academic and educational use only.



