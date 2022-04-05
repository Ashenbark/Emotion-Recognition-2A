import numpy as np

history = np.load("historyLSTM30.npy", allow_pickle=True)
history1 = np.load("historyLSTM302.npy", allow_pickle=True)

history_cat = {'loss': [],
               'accuracy': [],
               'val_accuracy': []}

for key, value in history.item().items():

    history_cat[key] = np.concatenate((history.item()[key], history1.item()[key]))

np.save("concathist.npy", history_cat)
print("Concat done!")