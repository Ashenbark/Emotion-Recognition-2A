import numpy as np
import matplotlib.pyplot as plt

history = np.load("historyDoubleLSTM.npy", allow_pickle=True)

plt.figure()
plt.title("Loss")
plt.plot(history.item()['loss'])
plt.plot(history.item()['val_loss'])
plt.legend(['Training Loss', 'Validation Loss'])

plt.figure()
plt.title("Accuracy")
plt.plot(history.item()['accuracy'])
plt.plot(history.item()['val_accuracy'])
plt.legend(['Training accuracy', 'Validation accuracy'])

plt.show()