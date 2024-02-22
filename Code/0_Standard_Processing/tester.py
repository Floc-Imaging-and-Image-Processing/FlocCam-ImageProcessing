# ms = 1 # add a mud/sand identifier ms = 1 for "yes" ms = 0 for "no"
# ms_model = 'sand_mud_models/SVM-linear-2.pickle'

from svm_discriminator import *

import pandas as pd
import numpy as np
import pickle
import os

datafile = '/Users/strom-adm/Library/CloudStorage/Dropbox/Code_nonGit_sync/SandTrials/FlocsOnly/0_analysis_output/001.csv'
full_df = pd.read_csv(datafile)
inputdata = full_df[['Area','Perimeter','Major','Minor','Circularity','AR','Round','Solidity']].to_numpy()

ms_model = 'sand_mud_models/SVM-linear-2.pickle'
loaded_model = pickle.load(open(ms_model, 'rb'))
y_pred = loaded_model.predict(inputdata)
sum(y_pred)

model_path = os.path.abspath(r"./generated_model")
classifier = load_model(model_path)
y_pred = classifier.predict(inputdata)
sum(y_pred)

# inputdata = full_df.to_numpy()[:, 1:-1]

# sand_mud_df = full_df['sand']
# target = sand_mud_df.to_numpy()
