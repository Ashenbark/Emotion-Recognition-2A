import numpy as np
import matplotlib.pyplot as plt

history = np.load("history.npy", allow_pickle=True)

plt.figure()
plt.title("Loss vs Epochs")
plt.plot(history.item()['loss'])
plt.plot(history.item()['val_loss'])
plt.legend(['Training Loss', 'Validation Loss'])

plt.figure()
plt.title("Accuracy vs Epochs")
plt.plot(history.item()['accuracy'])
plt.plot(history.item()['val_accuracy'])
plt.legend(['Training accuracy', 'Validation accuracy'])

plt.show()
