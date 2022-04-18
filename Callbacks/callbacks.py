from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

#Saving the best epochs over the training loss and the validation loss

save_best = ModelCheckpoint(filepath="save/best_GRU", monitor='loss', save_best_only=True,
                            save_freq='epoch', save_weights_only=True, verbose=1)
save_val_best = ModelCheckpoint(filepath="save/best_val_GRU", monitor='val_loss', save_best_only=True,
                                save_freq='epoch', save_weights_only=True, verbose=1)

#Adding an early stooping feature to stop the model and prevent overfitting/unnecessary computations

stopping = EarlyStopping(monitor="val_loss", patience=50)