import json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

# Define paths
path_terms = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/04_spy_project_FEARS/00_data/01_preprocessed/terms.json'

# Global initialization
#model_name = 'ProsusAI/finbert'
model_name = 'google/bert_uncased_L-6_H-128_A-2'
seq_len = 256

# Load dataset
with open(path_terms, 'r') as fr:
    terms = json.load(fr)[1:]

#%% Tokenize terms
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
terms_tokens = {}
for idx, term in tqdm(enumerate(terms),
                      desc = 'Tokenizing terms',
                      total = len(terms)):
    terms_tokens[idx] = bert_tokenizer(term,
                                       return_tensors = 'pt',
                                       padding = 'max_length',
                                       truncation = True,
                                       max_length = seq_len)

#%% Encode terms
bert_model = AutoModel.from_pretrained(model_name)
terms_enc = {}
for idx, tokens in tqdm(enumerate(terms_tokens.values()),
                        desc = 'Encoding terms',
                        total = len(terms)):
    terms_enc[idx] = bert_model(**tokens).pooler_output        # 1 x h_dim
    terms_enc[idx] = terms_enc[idx].squeeze(0)                 # h_dim

terms_enc = torch.stack(list(terms_enc.values()), dim = 1)     # h_dim x 30

#%% PCA decomposition
terms_enc = terms_enc.detach().numpy().transpose()
pca = PCA(n_components = 2)
PCA_output = pca.fit_transform(terms_enc)

X = PCA_output[:, 0]
Y = PCA_output[:, 1]

#%% Plot
plt.figure(figsize = (10, 8))
plt.scatter(X, Y, s = 100, color = "red")
plt.xlabel("X")
plt.ylabel("Y")
for i, label in enumerate(terms):
    plt.annotate(label, (X[i], Y[i]))
plt.show()


