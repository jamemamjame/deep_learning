from keras.models import model_from_json, model_from_yaml


def save_model(model, model_path, weight_path):
    model_json = model.to_json()
    with open(model_path, 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weight_path)
    print('Saved model structure to  %s complete!' % (model_path))
    print('Saved model weight to  %s complete!' % (weight_path))

def load_model(model_path, weight_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(weight_path)
    print("Loaded model from disk complete!")
    return loaded_model
