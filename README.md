# Predicting Student At-Risk Status Using Longitudinal Academic and Demographic Data

**Overview:**

This project aims to develop a predictive model for identifying students at risk of academic failure.  Using longitudinal academic and demographic data, we analyze student performance trends to build a robust model capable of proactively identifying at-risk students. This allows for the targeted allocation of support resources, ultimately improving overall student success rates. The analysis involves data preprocessing, feature engineering, model selection, and performance evaluation.

**Technologies Used:**

* Python 3
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn


**How to Run:**

1. **Clone the repository:**  `git clone <repository_url>`
2. **Install dependencies:**  `pip install -r requirements.txt`
3. **Run the main script:** `python main.py`

**Example Output:**

The script will print key analysis results to the console, including model performance metrics (e.g., accuracy, precision, recall, F1-score).  Additionally, the script generates several visualization files (e.g., plots showing feature importance, model performance curves) in the `output` directory.  These visualizations provide insights into the model's predictions and the factors contributing to student at-risk status.


**Directory Structure:**

* `data/`: Contains the input datasets.
* `src/`: Contains the source code for data preprocessing, model training, and evaluation.
* `output/`: Contains the generated output files (plots and reports).
* `models/`: Contains saved trained models (if applicable).
* `requirements.txt`: Lists the project's dependencies.
* `README.md`: This file.


**Future Work:**

* Explore advanced machine learning techniques for improved prediction accuracy.
* Incorporate additional data sources (e.g., attendance, behavioral data) to enrich the model.
* Develop a user-friendly interface for visualizing the results and interacting with the model.


**Contributing:**

Contributions are welcome! Please feel free to open issues or submit pull requests.  Before contributing, please review our [Contributing Guidelines](CONTRIBUTING.md) (if applicable).