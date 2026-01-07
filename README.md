\# Retail AI Inventory Management System



An end-to-end Flask + Machine Learning based system for intelligent retail inventory management.



---



\## Features



\- Demand Forecasting (30-day)

\- Pricing Optimization

\- Inventory Reorder Recommendation

\- Business Alerts

\- Single \& Bulk Predictions

\- Analysis Page \& History



---



\## Project Structure



```

Retail-AI-Inventory-System/

│

├── app.py

├── models.py

├── requirements.txt

├── Dockerfile

├── bulk\_input\_sample.csv

│

├── instance/

│   └── retail.db

│

├── models/

│   ├── random\_forest\_model.pkl

│   ├── xgboost\_model.pkl

│   ├── lightgbm\_model.pkl

│   ├── catboost\_model.pkl

│   └── preprocessing\_objects.pkl

│

├── templates/

│   ├── base.html

│   ├── dashboard.html

│   ├── prediction.html

│   ├── analysis.html

│   ├── history.html

│   ├── login.html

│   └── register.html

│

└── static/

&nbsp;   ├── css/style.css

&nbsp;   └── js/script.js

```



---



\## Local Setup (Without Docker)



\### 1. Clone Repository



```bash

git clone https://github.com/Pranathi1184/Retail-AI-Inventory-System.git

cd Retail-AI-Inventory-System

```



\### 2. Create Virtual Environment



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



\### 3. Install Dependencies



```bash

pip install --upgrade pip

pip install -r requirements.txt

```



Recommended Python version: 3.9 or 3.10



\### 4. Run Application



```bash

python app.py

```



\### 5. Open in Browser



```text

http://127.0.0.1:5000

```



---



\## Bulk Prediction CSV Format



```csv

Store ID,Product ID,Category,Price,Inventory Level,Competitor Pricing,Holiday/Promotion

S001,P001,Electronics,499.99,120,480,1

S002,P002,Grocery,79.99,300,75,0

```



---



\## Docker Setup (Optional)



Build Image



```bash

docker build -t retail-ai .

```



Run Container



```bash

docker run -p 5000:5000 retail-ai

```



---



\## Common Issues



\- Python 3.13 build errors → Use Python 3.9 / 3.10

\- Daily forecast not showing → Pass enhanced\_forecasts to analysis.html

\- Large .pkl files → Handled using Git LFS



---



\## Author



Pranathi Doddamani  

Final Year Engineering Student  

GitHub: https://github.com/Pranathi1184



---



Academic and educational use only.



