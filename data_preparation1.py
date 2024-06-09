import pandas as pd
from transformers import BertTokenizer, BertModel
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# import self-build datasets, manually scrapping the articles
df = pd.read_excel("text_summary_datasets_v2.xlsx")

# tokenization and bert
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def preprocess_text(text):
    # case standardization
    text = text.lower()
    #text = re.sub(r"(^|[.!?]\s+)(\w+)", lambda match: match.group(1) + match.group(2).capitalize(), text)

    # puntuation removal
    text = text.replace('"', '')

    tokens = tokenizer.tokenize(text)
    
    #new_tokens = []
    ## stop word removal
    #for token in tokens:
    #    if token.lower() not in stop_words:
    #        new_tokens.append(token)

 i   # lemmatizer and stemmer
    # Lemmatization and Stemming
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # detokenize
    def detokenize(tokens):
        new_tokens = []
        for token in tokens:
            if token.startswith("##"):
                new_tokens[-1] += token[2:]
            else:
                new_tokens.append(token)
        text = " ".join(new_tokens)
        text = re.sub(r'\s([?.!,\'-](?:\s|$))', r'\1', text)
        return text

    text = detokenize(lemmatized_tokens)

    # capitalize first alphabet of each sentence
    text = re.sub(r"(^|[.!?]\s+)(\w+)", lambda match: match.group(1) + match.group(2).capitalize(), text)

    return text
    
def toBert(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = bert_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return outputs.last_hidden_state[0, 0, :]

data = {
    "Index": [],
    "Category": [],
    **{f"dim_{it+1}": [] for it in range(768)}
}

for p, row in df.iterrows():
    data["Index"].append(row["Index"])
    data["Category"].append(row["Category"])
    text = str(row["Summary"])
    #print(preprocess_text(text))
    text = preprocess_text(text)

    for it in range(768):
        data[f"dim_{it+1}"].append(outputs[it].item())

new_df = pd.DataFrame(data)

new_df.to_excel("training_data.xlsx", index=False)





