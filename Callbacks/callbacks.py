from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

save_best = ModelCheckpoint(filepath="save/best_LSTM", monitor='loss', save_best_only=True,
                            save_freq='epoch', save_weights_only=True, verbose=1)
save_val_best = ModelCheckpoint(filepath="save/best_val_LSTM", monitor='val_loss', save_best_only=True,
                                save_freq='epoch', save_weights_only=True, verbose=1)

stopping = EarlyStopping(monitor="val_loss", patience=50)
