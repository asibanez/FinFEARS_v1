# Computes metrics

# Imports
import os
import json
import matplotlib.pyplot as plt

#%% Path definitions
base_path = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/04_spy_project_FEARS/00_data/02_runs/04_test_v1_train-test_unfrozen'

#%% Read data json
train_results_path = os.path.join(base_path, 'train_results.json')
test_results_path = os.path.join(base_path, 'test_results.json')

with open(train_results_path) as fr:
    train_results = json.load(fr)  
    
#%% Plot learning curves
plt.plot(train_results['training_loss'], label = 'train')
plt.plot(train_results['validation_loss'], label = 'val')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
#plt.ylim(0.05, 0.75)
plt.grid()
plt.show()

#%% Plot validation curve only
plt.plot(train_results['validation_loss'], label = 'val')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
plt.grid()
plt.show()

