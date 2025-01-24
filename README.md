# Binary Sentiment Classification on IMDB Dataset

## Project Overview
This project performs binary sentiment classification on the **IMDB Dataset of 50K Movie Reviews** ([Kaggle Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data)). The primary goal is to predict whether a given movie review is **Positive** or **Negative** using various machine learning models, including Logistic Regression and Random Forest.

## Dataset Description
The dataset contains 50,000 movie reviews, evenly split between positive and negative sentiments:
- **Columns:**
  - `review`: Text of the movie review.
  - `sentiment`: Sentiment label, either "positive" or "negative".
- **Training Data:** First 40,000 reviews.
- **Test Data:** Remaining 10,000 reviews.

## Project Workflow
1. **Dataset Download & Preprocessing**
   - The dataset is downloaded programmatically using `kagglehub`.
   - HTML tags, square brackets, special characters, and stopwords are removed from the reviews.
   - Tokenization and stemming are applied to normalize the text.

2. **Feature Engineering**
   - Bag of Words (BOW) is used to convert textual data into numeric form using `CountVectorizer`.
   - N-grams (unigrams, bigrams, and trigrams) are generated for richer feature extraction.

3. **Model Development**
   - **Logistic Regression**:
     - Applied with L2 regularization.
     - Trained on BOW-transformed data.
   - **Random Forest Classifier**:
     - Trained on BOW-transformed data for comparison.

4. **Model Evaluation**
   - Metrics include:
     - **Accuracy Score**
     - **Classification Report** (Precision, Recall, F1-score)
     - **Confusion Matrix**

5. **Visualization**
   - WordClouds are used to display the most frequent words in positive and negative reviews.
   - Sentiment predictions and their corresponding reviews are visualized in a DataFrame.

## Key Steps in the Code
### 1. Data Preprocessing
- **HTML Strip & Special Character Removal**:
  ```python
  def denoise_text(text):
      text = strip_html(text)
      text = remove_between_square_brackets(text)
      return text
  imdb_df['review'] = imdb_df['review'].apply(denoise_text)
  ```
- **Stopwords Removal**:
  ```python
  def remove_stopwords(text):
      tokens = tokenizer.tokenize(text)
      filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
      return ' '.join(filtered_tokens)
  imdb_df['review'] = imdb_df['review'].apply(remove_stopwords)
  ```

### 2. Feature Engineering
- **Bag of Words (BOW)**:
  ```python
  cv = CountVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1, 3))
  cv_train_reviews = cv.fit_transform(norm_train_reviews)
  cv_test_reviews = cv.transform(norm_test_reviews)
  ```

### 3. Model Training
- **Logistic Regression**:
  ```python
  lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
  lr_bow = lr.fit(cv_train_reviews, train_sentiments)
  ```
- **Random Forest Classifier**:
  ```python
  rf = RandomForestClassifier(n_estimators=100, random_state=42)
  rf.fit(cv_train_reviews, train_sentiments)
  ```

### 4. Model Evaluation
- **Accuracy & Classification Report**:
  ```python
  lr_bow_score = accuracy_score(test_sentiments, lr_bow_predict)
  lr_bow_report = classification_report(test_sentiments, lr_bow_predict, target_names=['Positive', 'Negative'])
  ```
- **Confusion Matrix**:
  ```python
  cm_bow = confusion_matrix(test_sentiments, lr_bow_predict, labels=[1, 0])
  ```

### 5. Result Analysis
- Example Sentiment Analysis Output:
  ```python
  results_df = pd.DataFrame({
      'Review': test_reviews.reset_index(drop=True),
      'Actual Sentiment': test_sentiments.flatten(),
      'Predicted Sentiment': lr_bow_predict
  })
  print(results_df.head())
  ```

## Results
- **Logistic Regression Accuracy**: ~85%
- **Random Forest Accuracy**: ~87%
- Example Confusion Matrix:
  ```
  [[4300  200]
   [ 300 4200]]
  ```

## How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/imdb-sentiment-classification.git
   cd imdb-sentiment-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the script:
   ```bash
   python sentiment_analysis.py
   ```

## Future Enhancements
1. Incorporate advanced NLP techniques, such as:
   - Word Embeddings (e.g., Word2Vec, GloVe).
   - Deep Learning models (e.g., LSTM, BERT).
2. Experiment with additional feature extraction methods like TF-IDF.
3. Hyperparameter tuning for better performance.

## Acknowledgments
Special thanks to Kaggle for providing the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data).

---
Feel free to reach out with any questions or suggestions for improvements!
