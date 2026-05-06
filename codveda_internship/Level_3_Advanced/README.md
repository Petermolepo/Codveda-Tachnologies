# Level 3 – Advanced

## Task 2: NLP – Text Classification

**File:** `task2_nlp/nlp_classification.py`

**What it does:**
- Generates a sentiment dataset (600 customer reviews: positive/negative)
- Preprocesses: lowercase, remove punctuation, stopword removal, lemmatization
- Converts text to numerical form using TF-IDF (unigrams + bigrams, 5000 features)
- Trains and compares 4 classifiers: Logistic Regression, Naive Bayes, Linear SVM, Random Forest
- Evaluates using accuracy, precision, recall, F1, confusion matrix
- Visualises top predictive words per sentiment class

**Outputs:**
- `customer_reviews.csv`
- `plot_01_model_comparison.png`
- `plot_02_confusion_matrix.png`
- `plot_03_top_words.png`

**Run:**
```bash
python nlp_classification.py
```

---

## Task 3: Neural Networks with TensorFlow/Keras

**File:** `task3_neural_network/neural_network.py`

**What it does:**
- Loads MNIST (or generates tabular multi-class data as fallback)
- Builds a 3-hidden-layer feed-forward neural network with BatchNorm and Dropout
- Trains using Adam optimiser, sparse categorical crossentropy loss
- Uses EarlyStopping and ReduceLROnPlateau callbacks
- Evaluates accuracy and loss on test set
- Runs hyperparameter experiments (learning rate × batch size)
- Saves trained model as `.keras` file

**Architecture:**
```
Input → Dense(256) + BN + Dropout(0.3)
      → Dense(128) + BN + Dropout(0.3)
      → Dense(64)  + Dropout(0.2)
      → Output(n_classes) [softmax]
```

**Outputs:**
- `plot_01_training_history.png`
- `plot_02_confusion_matrix.png`
- `plot_03_hyperparameter_tuning.png`
- `neural_network_model.keras`

**Requirements:**
```bash
pip install tensorflow
```

**Run:**
```bash
python neural_network.py
```
