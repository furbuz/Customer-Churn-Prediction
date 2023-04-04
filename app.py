# Data Credit: https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m

import streamlit as st
import joblib
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
st.header('Predicting Bank Customer Churn')
st.write('With Bank Churners Data')

st.subheader("Goal: To Predict Customer :red[Attrition].")

joblibed_model = joblib.load('RandomForest_model.pkl')
dataframe = pd.read_csv('BankChurners.csv')
joblibed_scaler = joblib.load('age_scaler.pkl')
clean_dataframe = pd.read_csv("clean_data.csv")



#--------------- QUERY -------------------------------------#

query = {
    'Customer_Age' : [0],
    'Dependent_count' : [0], 
    'Months_on_book' : [0],
    'Total_Relationship_Count' : [0], 
    'Months_Inactive_12_mon' : [0],
       'Contacts_Count_12_mon' : [0], 
       'Credit_Limit' : [0], 
       'Total_Revolving_Bal' : [0],
       'Avg_Open_To_Buy' : [0], 
       'Total_Amt_Chng_Q4_Q1' : [0], 
       'Total_Trans_Amt' : [0],
       'Total_Trans_Ct' : [0], 
       'Total_Ct_Chng_Q4_Q1' : [0], 
       'Avg_Utilization_Ratio' : [0],
       'Gender_F' : [0], 
       'Gender_M' : [0], 
       'Marital_Status_Divorced' : [0],
       'Marital_Status_Married' : [0], 
       'Marital_Status_Single' : [0],
       'Marital_Status_Unknown' : [0], 
       'Education_Level_College' : [0],
       'Education_Level_Doctorate' : [0], 
       'Education_Level_Graduate' : [0],
       'Education_Level_High School' : [0],
       'Education_Level_Post-Graduate' : [0],
       'Education_Level_Uneducated' : [0], 
       'Education_Level_Unknown' : [0],
       'Income_Category_$120K +' : [0], 
       'Income_Category_$40K - $60K' : [0],
       'Income_Category_$60K - $80K' : [0], 
       'Income_Category_$80K - $120K' : [0],
       'Income_Category_Less than $40K' : [0], 
       'Income_Category_Unknown' : [0],
       'Card_Category_Blue' : [0], 
       'Card_Category_Gold' : [0], 
       'Card_Category_Platinum' : [0],
       'Card_Category_Silver' : [0] 
}

query = pd.DataFrame.from_dict(query)





page_names = ['Data', 'Model', 'Prediction']
page = st.radio("Navigation", page_names)

st.markdown("<hr>", unsafe_allow_html=True)

if page == 'Data':
    st.header("Bank Churners DataSet")

    col1, col2, col3 = st.columns(3)
    col1.metric("Features", "23")
    col2.metric("Entries", "10127")
    col3.metric("Attrition Rate", "16.06 %")

    st.dataframe(dataframe)
    st.write("[Data Source](https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m)")
    st.write("Data retrieved from [Kaggle](https://kaggle.com)")
    st.subheader("The data is not in a format that model can use")
    st.write("Click the button below to scale and transform data")

    data_button = st.button("Clean, Transform, and Scale")

    if data_button:
        with st.spinner("Transforming and Scaling entire dataset..."):
            time.sleep(3)
            st.dataframe(clean_dataframe.drop(columns = ['Unnamed: 0', 'Customer_Age.1'], axis = 1))

        st.subheader("Great, now a model can be built!")
        

elif page == 'Model':
    st.write("Choose an Algorithm: ")
    st.radio("Select Random Forest:", ['Random Forest'])
    model_button = st.button("Click to run and feed the model")
    if model_button:
        with st.spinner("Running the model..."):
            time.sleep(3)
            st.balloons()
            st.header(":green[DONE SUCCESSFULLY!!!]")

        st.subheader("Now, you can feed the model with data to make prediction!")

elif page == 'Prediction':
    
# ---------------------- LAYOUT --------------------------------------------------#
    col1, col2 = st.columns(2)

# ------- First Column -----------------------------------------------------------#

    col1.header("Personal Information")



    # ------- Second Column ----------------------------------------------------------#
    col2.header("Financial Information")


    # -------------------------------------------------------------------------------#
    age = col1.slider("Customer Age: ")
    scaled_age = joblibed_scaler.transform(np.array(age).reshape(-1, 1))

    query['Customer_Age'] = scaled_age

    #
    gender = col1.radio("Customer Gender: ", ["Male", "Female"])

    if gender == 'Male':
        query['Gender_M'] = 1
    else:
        query['Gender_F'] = 1
    #

    query['Dependent_count'] = col2.slider("Dependent Count (the number of dependents that customer has) ", 0, 10)

    #
    education_level = col1.selectbox("Education Level", ['Uneducated', 'High School', 'College', 'Graduate', 'Doctorate', 'Post-Graduate', 'Unknown'])

    #
    if education_level == 'Uneducated':
        query['Education_Level_Uneducated'] = 1
    elif education_level == 'High School':
        query['Education_Level_High School'] = 1
    elif education_level == 'College':
        query['Education_Level_College'] = 1
    elif education_level == 'Graduate':
        query['Education_Level_Graduate'] = 1
    elif education_level == 'Doctorate':
        query['Education_Level_Doctorate'] = 1
    elif education_level == 'Post-Graduate':
        query['Education_Level_Post-Graduate'] = 1
    elif education_level == 'Unknown':
        query['Education_Level_Unknown'] = 1
    else:
        raise Exception("There is something wrong about education!!!")
    #

    income_category = col1.selectbox('Income Category', ['$120K +', '$40K - $60K', '$60K - $80K', '$80K - $120K', 'Less than $40K', 'Unknown'])

    if income_category == '$120K +':
        query['Income_Category_$120K +'] = 1
    elif income_category == '$40K - $60K':
        query['Income_Category_$40K - $60K'] = 1
    elif income_category == '$60K - $80K':
        query['Income_Category_$60K - $80K'] = 1
    elif income_category == '$80K - $120K':
        query['Income_Category_$80K - $120K'] = 1
    elif income_category == 'Less than $40K':
        query['Income_Category_Less than $40K'] = 1
    elif income_category == 'Unknown':
        query['Income_Category_Unknown'] = 1
    else:
        raise Exception("There is something wrong about income category!!!")
    #

    martial_status = col1.radio('Martial Status: ', ['Single', 'Married', 'Divorced', 'Unknown'])
    if martial_status == 'Single':
        query['Marital_Status_Single'] = 1
    elif martial_status == 'Married':
        query['Marital_Status_Married'] = 1
    elif martial_status == 'Divorced':
        query['Marital_Status_Divorced'] = 1
    elif martial_status == 'Unknown':
        query['Marital_Status_Unknown'] = 1
    else:
        raise Exception("There is something wrong in the marital status")

    #

    card_category = col2.selectbox('Card Category: ', ['Blue', 'Silver','Gold', 'Platinum'])
    if card_category == 'Blue':
        query['Card_Category_Blue'] = 1
    elif card_category == 'Silver':
        query['Card_Category_Silver'] = 1
    elif card_category == 'Gold':
        query['Card_Category_Gold'] = 1
    elif card_category == 'Platinum':
        query['Card_Category_Platinum'] = 1
    else:
        raise Exception("There is something wrong in card category!!!")
    #
    months_on_book = col2.slider("Months on book: (How long customer has been on the books)", 0, 70)
    query['Months_on_book'] = months_on_book

    #
    total_relationship_count = col2.slider("Total Relationship Count: (Total number of relationships customer has with the credit card provider.)", 0, 15)
    query['Total_Relationship_Count'] = total_relationship_count

    #
    months_inactive = col2.slider("Inactive Months: (Number of months customer has been inactive in the last twelve months)", 0, 15)
    query['Months_Inactive_12_mon'] = months_inactive

    #
    contact_count = col2.slider("Contact Count: (Number of contacts customer has had in the last twelve months)", 0, 15)
    query['Contacts_Count_12_mon'] = contact_count

    #
    credit_limit = col2.slider("Credit Limit: (Credit limit of customer)")
    query['Credit_Limit'] = credit_limit

    #
    total_revolve = col2.slider("Total Revoling: (Total revolving balance of customer)")
    query['Total_Revolving_Bal'] = total_revolve

    #
    average_open_to_buy = col2.slider("Average open to buy ratio of customer")
    query['Avg_Open_To_Buy'] = average_open_to_buy

    #
    total_amount_q4 = col2.slider("Total amount changed from quarter 4 to quarter 1")
    query['Total_Amt_Chng_Q4_Q1'] = total_amount_q4

    #
    total_transaction_amount = col2.slider("Total Transaction Amount")
    query['Total_Trans_Amt'] = total_transaction_amount

    #
    total_transaction_count = col2.slider("Total Transaction Count")
    query['Total_Trans_Ct'] = total_transaction_count

    #
    total_count_q4 = col2.slider("Total count changed from quarter 4 to quarter 1")
    query['Total_Ct_Chng_Q4_Q1'] = total_count_q4

    #
    average_utilization = col2.slider("Average utilization ratio of customer")
    query['Avg_Utilization_Ratio'] = average_utilization

    ######

    result = int(joblibed_model.predict(query))


    if result == 1 :
        result = "Customer Retention"
    elif result == 0:
        result = "Customer Churn"

    prediction_button = st.button("Predict")
    st.subheader("The Result of Prediction is: ")
    if prediction_button and result == 'Customer Retention':
        st.subheader(":green["+result+"]")
    elif prediction_button:
        st.subheader(":red[" + result+"]")






# ---------------------------------------------------- END OF QUERY




#st.markdown("With **:blue[23]** features and **:red[10127]** entries")





















# hiding useless parts

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

