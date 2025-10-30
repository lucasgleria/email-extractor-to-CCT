# Training a Custom Model for the Email Extractor

This guide provides a step-by-step process for training your own custom machine learning model to improve the accuracy of the Email Screenshot Extractor. By using the data you've corrected and saved, you can create a model that is tailored to the specific formats of the emails you work with.

The entire process is free, using Google Colab for training and your browser for running the model.

## Step 1: Export Your Training Data

1.  Open the `index.html` file in your browser.
2.  Process at least 50-100 images, ensuring you correct any mistakes made by the initial rule-based extractor. The more high-quality data you provide, the better your model will be.
3.  Click the **Export Training Data** button.
4.  Save the downloaded `training_data.json` file to your computer.

## Step 2: Set Up Your Free Training Environment (Google Colab)

1.  Go to [Google Colab](https://colab.research.google.com/).
2.  Click **New notebook**.
3.  You now have a ready-to-use Python environment in the cloud.

## Step 3: Upload Your Data to Colab

1.  In the left-hand panel of your Colab notebook, click the **Files** icon.
2.  Click the **Upload to session storage** icon and select the `training_data.json` file you downloaded in Step 1.

## Step 4: The Training Script

Copy and paste the entire Python script below into a single cell in your Colab notebook. Then, run the cell by clicking the play button or pressing `Shift+Enter`.

```python
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import re

# --- 1. Load the Data ---
with open('training_data.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} training examples.")

# --- 2. Preprocess and Featurize ---
# We'll create a simple featurizer that looks at the text of each line
def featurize(raw_text):
    return raw_text.split('\n')

# Prepare the data for training
# We will train a separate model for each field
fields_to_train = ['REFERENCIA', 'MAWB', 'HAWB', 'DESTINO', 'DESTINO_FINAL', 'CONSIGNEE']
models = {}

for field in fields_to_train:
    print(f"\n--- Training model for: {field} ---")

    # Create training data for this field
    X_train = []
    y_train = []

    for item in data:
        lines = featurize(item['raw_text'])
        correct_value = item['fields'].get(field, '').strip()

        if not correct_value:
            continue

        for line in lines:
            X_train.append(line)
            # If the correct value is in this line, it's a positive example
            if correct_value in line:
                y_train.append(1)
            else:
                y_train.append(0)

    if sum(y_train) == 0:
        print(f"Skipping {field} as no positive examples were found.")
        continue

    # --- 3. Train the Model ---
    # We use a simple TF-IDF vectorizer and a Logistic Regression model
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(class_weight='balanced'))
    model.fit(X_train, y_train)

    models[field] = model
    print(f"Model for {field} trained successfully.")

# --- 4. Export the Model ---
# We need to export the vectorizer's vocabulary and the model's coefficients
exported_model = {}
for field, model in models.items():
    vectorizer = model.named_steps['tfidfvectorizer']
    classifier = model.named_steps['logisticregression']

    exported_model[field] = {
        'vocabulary': vectorizer.vocabulary_,
        'idf': list(vectorizer.idf_),
        'coef': list(classifier.coef_[0]),
        'intercept': list(classifier.intercept_)
    }

with open('model.json', 'w') as f:
    json.dump(exported_model, f)

print("\n--- Model exported to model.json ---")
```

## Step 5: Download Your Custom Model

1.  After the script finishes running, you will see a new file named `model.json` in the Colab file explorer.
2.  Click the three dots next to `model.json` and select **Download**.
3.  Save this file in the same directory as your `index.html` file.

You have now successfully trained and exported your custom model. The next step is to integrate this model into the application so it can be used for inference.
