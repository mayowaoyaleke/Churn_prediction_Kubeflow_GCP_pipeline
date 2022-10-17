import tensorflow_decision_forests as tfdf

model = tfdf.keras.RandomForestModel(check_dataset = False)
model = model_builder()
model.compile(metrics=['accuracy'])
    
model.feature_keys = _FEATURE_KEYS
model.label_key = _LABEL_KEY
model.fit(train_set, validation_steps = 32, validation_data = eval_set)