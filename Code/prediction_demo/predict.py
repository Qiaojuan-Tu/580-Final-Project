import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import warnings
import time
warnings.filterwarnings("ignore")


print('Loading Text Cleaning Pipeline...')

# Cleaning Pipeline
import re
import spacy
from spacy.language import Language


pipeline = spacy.load('en_core_web_sm')
# http://emailregex.com/
email_re = r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""

# replace = [ (pattern-to-replace, replacement),  ...]
replace = [
    (r"<a[^>]*>(.*?)</a>", r"\1"),  # Matches most URLs
    (email_re, "email"),            # Matches emails
    (r"(?<=\d),(?=\d)", ""),        # Remove commas in numbers
    (r"\d+", "number"),              # Map digits to special token <numbr>
    (r"[\t\n\r\*\.\@\,\-\/\$]", " "), # Punctuation and other junk
    (r"\s+", " ")                   # Stips extra whitespace
]

@Language.component("Preprocessor")
def ng20_preprocess(doc):
    tokens = [token for token in doc
              if not any((token.is_stop, token.is_punct))]
    tokens = [token.lemma_.lower().strip() for token in tokens]
    tokens = [token for token in tokens if token]
    return " ".join(tokens)

pipeline.add_pipe("Preprocessor")





corpus = [input('Please type in the text: ')]

# Apply Pipeline
sentences = []
for i, d in enumerate(corpus):
    for repl in replace:
        d = re.sub(repl[0], repl[1], d)
    sentences.append(d)

docs = []
for sent in sentences:
    docs.append(pipeline(sent))


# Vectorizing
print('Loading Trained Vectorizer...')
vectorizer = joblib.load('vectorizer.pkl')
Xs  =  vectorizer.transform(docs)




# Model Prediction
model = joblib.load('logistic.pkl')
print('Loading Trained Model...')
# CONVERT TO DENSE MATRIX
X= np.array(Xs.todense())
df_X = pd.DataFrame(X)

print('---------------------Prediction Result--------------------')
result = model.predict(df_X)[0]
if result == 1:
    print('It is predicted that the NYSE will RISE in the next week.')
else: print('It is predicted that the NYSE will FALL in the next week.')
print('----------------------------------------------------------')
