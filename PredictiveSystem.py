import numpy as np
import pickle
# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

time = [3000]
sampl = np.random.uniform(low=-10, high=10, size=(28,))
amount = [500]
input_data = [*time, *sampl, *amount]
print(input_data)
#changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshaping the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)


if(prediction[0] == 0):
  print('the transaction is legitimate')
else:
  print('the transaction is fraudulent')