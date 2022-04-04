# Generate vocabulary and add random index column
#   v1 Appends real S&P Data

#%% Imports
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

#%% Path definitions
input_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/04_spy_project_FEARS/00_data/00_raw/'
output_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/04_spy_project_FEARS/00_data/01_preprocessed'

#input_folder = ''
#output_folder = ''

fears_filename = 'fears_index.csv'
sp500_filename = 'sp500.csv'

fears_path = os.path.join(input_folder, fears_filename)
sp500_path = os.path.join(input_folder, sp500_filename)

#%% Read input data
fears = pd.read_csv(fears_path)
sp500 = pd.read_csv(sp500_path)

#%% Extract vocabulary
vocab = json.dumps(list(fears.columns))

#%% Preprocess datasets
fears = fears.rename(columns = {'Date':'date'})
sp500 = sp500.drop(columns = ['Unnamed: 0'])

#%% Append SP500 price and return for week w
fears_full = pd.merge(left = fears, right = sp500, how = 'left', on = 'date')
fears_full = fears_full.rename(columns = {'sp500_price':'sp500_price_w',
                                          'sp500_return':'sp500_return_w'})

#%% Generate and append SP500 price and return for week w-1
sp500_price_wm1 = [9999] + list(fears_full['sp500_price_w'])[0:-1]
sp500_return_wm1 = [9999] + list(fears_full['sp500_return_w'])[0:-1]

fears_full['sp500_price_wm1'] = sp500_price_wm1
fears_full['sp500_return_wm1'] = sp500_return_wm1

#%% Drop 1st row with no w-1 data
fears_full = fears_full[1:]

#%% Split train - test datasets
fears_train, fears_test = train_test_split(fears_full,
                                           test_size = 0.2,
                                           shuffle = False)

#%% Check dataset sizes
print(f'Size full set:\t{len(fears_full)}')
print(f'Size train set:\t{len(fears_train)}\t{len(fears_train)/len(fears_full)*100:.2f}%')
print(f'Size test set:\t{len(fears_test)}\t{len(fears_test)/len(fears_full)*100:.2f}%')

#%% Save outputs
with open(os.path.join(output_folder, 'vocab.json'), 'w') as fw:
    fw.write(vocab)
fears_full.to_pickle(os.path.join(output_folder, 'full_dataset.pkl'))
fears_train.to_pickle(os.path.join(output_folder, 'model_train.pkl'))
fears_test.to_pickle(os.path.join(output_folder, 'model_dev.pkl'))
