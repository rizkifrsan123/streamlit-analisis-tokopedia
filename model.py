import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("model data ulasan hp oppo.csv")
print(df.head())

# Fungsi untuk mengkategorikan rate sebagai label sentimen
def categorize_sentiment(rate):
    if rate in [1, 2]:
        return 'Negatif'
    elif rate == 3:
        return 'Netral'
    else:
        return 'Positif'
    
# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(data['Ulasan'], data['Sentiment'], test_size=0.2, random_state=42)

# Vektorisasi data teks menggunakan TF-IDF
vectorizer = TfidfVectorizer(stop_words=indonesian_stop_words, max_features=None)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Parameter grid untuk mencari k terbaik
param_grid = {'n_neighbors': range(1, 51)}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X_train_tfidf, y_train)

print("K terbaik:", grid.best_params_)
print("Akurasi terbaik:", grid.best_score_)

# Latih classifier KNN
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train_tfidf, y_train)

pickle.dump(classifier, open("model.pkl","ab"))