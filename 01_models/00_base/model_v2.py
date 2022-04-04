# v2 -> Uses actual SP500 data. w-1 data included as X
#       Includes self-attention layer

# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

#%% DataClass definition
class FEARS_dataset(Dataset):
    def __init__(self, data_df):
        
        # Extract features
        terms_df = data_df.iloc[:, 1:31]
        sp500_wm1_df = data_df['sp500_return_wm1']    
        sp500_w_df = data_df['sp500_return_w']
        
        self.X_terms = torch.tensor(terms_df.values).unsqueeze(1).float()
        self.X_sp500 = torch.tensor(sp500_wm1_df.values).unsqueeze(1).float()
        self.Y_sp500 = torch.tensor(sp500_w_df.values).unsqueeze(1).float()
                                        
    def __len__(self):
        return len(self.X_terms)
        
    def __getitem__(self, idx):
        X_terms = self.X_terms[idx]
        X_sp500 = self.X_sp500[idx]
        Y_sp500 = self.Y_sp500[idx]
        return X_terms, X_sp500, Y_sp500

#%% Model definition
class FEARS_model(nn.Module):
            
    def __init__(self, args, terms):
        super(FEARS_model, self).__init__()

        # Variables        
        self.h_dim = args.hidden_dim
        self.seq_len = args.seq_len
        self.dropout = args.dropout
        self.alpha = args.alpha
        self.att_dim = args.att_dim
        self.query_v = nn.Parameter(torch.randn((self.att_dim, 1),
                                                requires_grad = True))

        # Term encodings
        terms = terms[1:]
        model_name = 'ProsusAI/finbert'
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
        self.terms_tokens = {}
        for idx, term in enumerate(terms):
            self.terms_tokens[idx] = bert_tokenizer(term,
                                                    return_tensors = 'pt',
                                                    padding = 'max_length',
                                                    truncation = True,
                                                    max_length = self.seq_len)
                     
        # Bert layer
        self.model_name = args.model_name
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        if args.freeze_BERT == True:
            for parameter in self.bert_model.parameters():
                parameter.requires_grad = False
              
        # Fully connected context
        self.fc_context = nn.Linear(in_features = self.h_dim,
                                    out_features = self.att_dim)
        
        # Fully connected #1
        self.fc_1 = nn.Linear(in_features = self.h_dim, out_features = 1)

        # Relu
        self.relu = nn.ReLU()

        # Dropout
        self.drops = nn.Dropout(self.dropout)
           
        # Batch normalizations
        self.bn1 = nn.BatchNorm1d(self.h_dim)

    def forward(self, X_terms, X_sp500):
        batch_size = len(X_terms)
        device = X_terms.get_device()
        if device == -1: device = 'cpu'
        
        # BERT term encoder
        terms_enc = {}
        for idx, tokens in enumerate(self.terms_tokens.values()):
            # Move to device
            for key in tokens.keys():
                tokens[key] = tokens[key].to(device)
			# Encode
            terms_enc[idx] = self.bert_model(**tokens).pooler_output
            terms_enc[idx] = terms_enc[idx].squeeze(0)
        terms_enc = torch.stack(list(terms_enc.values()), dim = 1)
        terms_enc = terms_enc.repeat(batch_size, 1, 1)
        
        # Multiply by change in search volume
        factor_1 = self.alpha * X_terms
        factor_2 = (1 - self.alpha) * X_sp500
        factor_2 = factor_2.unsqueeze(1).repeat(1,1,30)
        factor = factor_1 + factor_2
        out = torch.multiply(terms_enc, factor)
                     
        # Attention
        out = torch.transpose(out, 1, 2)
        proj = torch.tanh(self.fc_context(out))
        alpha = torch.matmul(proj, self.query_v)
        alpha = torch.softmax(alpha, dim = 1)
        att_output = out * alpha
        att_output = torch.sum(att_output, axis = 1)
        
        # Fully connected
        out = self.bn1(att_output)
        out = self.fc_1(out)
        out = self.relu(out)

        return out
    
