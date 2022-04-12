import numpy as np

history = np.load("history.npy", allow_pickle=True)
history1 = np.load("history2.npy", allow_pickle=True)

history_cat = {'loss': [],
               'accuracy': [],
               'val_accuracy': []}

for key, value in history.item().items():

    history_cat[key] = np.concatenate((history.item()[key], history1.item()[key]))

np.save("history3.npy", history_cat)
print("Concat done!")