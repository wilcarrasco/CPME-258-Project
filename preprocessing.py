# %% 
import os

import numpy as np
import pandas as pd

# %%

# Location to our dataset
files = os.listdir("./data/train")
labels = []

# parse the files, and split label types between cats and dogs
for file in files:
    animal_type = file.split('.')[0]

    if animal_type == "cat":
        labels.append(0)
    else:
        labels.append(1)

# Verify that we have 25000 labels
print(len(labels))
assert(len(labels) == 25000)

# Combine the filename and label type
df = pd.DataFrame({'filename':files,
                   'label':labels})

print(df)

# %%
