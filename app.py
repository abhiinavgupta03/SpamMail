from flask import Flask, render_template, request, jsonify
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.utils import pad_sequences
from keras.preprocessing.text import one_hot
from keras.models import load_model
from keras.optimizers import Adam

nltk.download('stopwords')

app = Flask(__name__)

# Load and compile the model
model = load_model("classifier.h5", compile=False)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Preprocess the text
def preprocess_text(text):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    mail = request.form.get("mail")
    voc_size = 5000
    corpus = [preprocess_text(mail)]
    onehot_rep = [one_hot(words, voc_size) for words in corpus]
    sent_len = 20
    embedded_docs = pad_sequences(onehot_rep, padding='pre', maxlen=sent_len)

    # Predict
    y_pred = model.predict(embedded_docs)[0][0]  

    print(f"Raw prediction: {y_pred}")  

    result = "Spam" if y_pred > 0.5 else "Ham"
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
