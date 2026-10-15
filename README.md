from pathlib import Path

readme = r"""# SMS Spam Detection using Support Vector Machine (SVM)

## Overview
This project builds a machine-learning model to classify SMS messages into two categories:

- **Ham** — a legitimate, non-spam message
- **Spam** — an unwanted or suspicious message

The notebook `SMS_Spam_SVM_Updated.ipynb` covers data loading, data validation, exploratory data analysis (EDA), text preprocessing, model training, hyperparameter tuning, evaluation, and predictions for new messages.

## Objectives
- Explore and validate an SMS spam dataset.
- Clean and normalize message text.
- Convert text into numerical features using TF-IDF.
- Train a Linear Support Vector Machine (LinearSVC) classifier.
- Tune model hyperparameters with grid search and stratified cross-validation.
- Evaluate performance using accuracy, precision, recall, F1-score, a classification report, and a confusion matrix.
- Test the trained model on custom SMS messages.

## Technologies and Libraries
- Python
- Jupyter Notebook / Google Colab
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- WordCloud (optional)
- joblib (optional, for saving the trained model)

## Dataset
The notebook looks for one of these CSV filenames in the working directory:

- `SMS.csv`
- `spam.csv`
- `SMSSpamCollection.csv`
- `sms_spam.csv`

It supports either of these column formats:

| Format | Label column | Message column |
|---|---|---|
| Common SMS spam dataset | `v1` | `v2` |
| Standardized format | `label` | `message` |

Labels should be `ham` or `spam`. The notebook removes invalid labels, empty messages, and duplicate label/message rows. If the CSV is not found in Google Colab, the notebook prompts you to upload it.

**Note:** The CSV dataset is not included automatically with the notebook. Obtain the dataset separately and place it in the notebook's working directory, or upload it when prompted in Colab.

## Methodology
1. **Load and validate data** — read the CSV, standardize the label/message columns, and inspect missing values and class proportions.
2. **Exploratory data analysis** — compare ham and spam counts and examine message lengths and common words. Word clouds are optional.
3. **Text preprocessing** — lowercase text, decode HTML entities, replace URLs and email addresses with tokens, remove non-alphanumeric characters, and normalize whitespace.
4. **Train/test split** — split the data into training and test sets using an 80/20 split and stratification.
5. **Baseline model** — build a pipeline with TF-IDF features and `LinearSVC`.
6. **Hyperparameter tuning** — use `GridSearchCV` with 5-fold stratified cross-validation. The search considers TF-IDF feature limits, unigram/bigram settings, and SVM `C` values.
7. **Evaluation** — report accuracy, spam precision, spam recall, spam F1-score, macro F1-score, classification report, and confusion matrix.
8. **Additional analysis** — calculate cross-validation macro-F1 scores and inspect words/phrases that influence the linear SVM.
9. **Prediction** — classify example messages and a custom SMS message as HAM or SPAM.

## Installation
Install the required packages in a terminal or notebook cell:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

For optional word clouds:

```bash
pip install wordcloud
```

For optional model saving/loading:

```bash
pip install joblib
```

## How to Run

### Option 1: Google Colab
1. Upload `SMS_Spam_SVM_Updated.ipynb` to Google Colab.
2. Run the notebook cells from top to bottom.
3. When prompted, upload your SMS spam CSV dataset if it is not already available in the runtime.
4. Review the EDA charts, tuned model parameters, evaluation metrics, and prediction examples.

### Option 2: Jupyter Notebook
1. Place `SMS_Spam_SVM_Updated.ipynb` and one supported dataset CSV file in the same working directory.
2. Install the required libraries.
3. Open the notebook in Jupyter.
4. Run all cells from top to bottom.

## Example Prediction
The notebook includes examples similar to these:

```python
print(predict_sms("Congratulations! You have won a free iPhone. Click here now."))
print(predict_sms("Hi, are we meeting tomorrow at 3 PM?"))
```

The function returns the original message, the predicted class (`HAM` or `SPAM`), and the SVM decision score. The decision score is a model score, **not a probability**.

To test a different message, edit `my_message` in the prediction section and run that cell.

## Model Saving (Optional)
The notebook includes commented example code for saving and loading the fitted model with `joblib`. Uncomment the relevant lines after training if you want to save the model.

## Evaluation
The notebook calculates the evaluation metrics when it is run. Results depend on the dataset and should be copied from the actual notebook output; this README does not claim fixed accuracy or other performance values.

## Limitations
- Predictions depend on the dataset and the language/style of messages represented in it.
- A decision score is not a calibrated probability.
- This is a text classifier and may misclassify legitimate or unwanted messages.
- Review important messages before taking action based only on the prediction.

## Project Structure
```text
.
├── SMS_Spam_SVM_Updated.ipynb
├── README.md
└── SMS.csv                 # or another supported dataset filename
```

## License
No license is specified in this project. Add a license file if you intend to distribute or reuse the project publicly.
"""
path = Path("/mnt/data/README.md")
path.write_text(readme, encoding="utf-8")
print(f"Created: {path}")
