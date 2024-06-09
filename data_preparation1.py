import pandas as pd
from transformers import BertTokenizer, BertModel

df = pd.read_excel("text_summary_datasets_v2.xlsx")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def func1(text):
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
    outputs = func1(str(row["Summary"]))
    for it in range(768):
        data[f"dim_{it+1}"].append(outputs[it].item())
    print(str(row["Summary"]))
    print(outputs)
    break

new_df = pd.DataFrame(data)

new_df.to_excel("training_data.xlsx", index=False)





