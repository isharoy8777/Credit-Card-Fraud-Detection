import numpy as np
import pickle
import streamlit as st
# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# creating a function for prediction
def ccfraud_detection(input_data):
    #dont need all this as input from user
  #  time = [3000]
   # sampl = np.random.uniform(low=-10, high=10, size=(28,))
    #amount = [500]
    #input_data = [*time, *sampl, *amount]
    #print(input_data)
    #changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    #reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)


    if(prediction[0] == 0):
      return 'the transaction is legitimate'
    else:
      return 'the transaction is fraudulent'
    

def main():

    #giving a title  
    st.title('Credit Card Fraud Detection Web App')

#getting data from user
#"Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"

# Getting data from user
    time = st.number_input('Transaction Time', min_value=0)
    amount = st.number_input('Transaction Amount', min_value=0.0, format="%.2f")

    # Placeholder for additional raw data required for PCA transformation
    raw_data_defaults = np.zeros((1, 30))  # Adjust the shape as per your original data

    # Integrate user input into the raw data
    raw_data_defaults[0, 0] = time  # Place the time in the first column
    raw_data_defaults[0, 29] = amount  # Place the amount in the 30th column

    # Predict button
    if st.button('Predict'):
        result = ccfraud_detection(raw_data_defaults[0])
        st.success(result)

if __name__ == '__main__':
    main()
