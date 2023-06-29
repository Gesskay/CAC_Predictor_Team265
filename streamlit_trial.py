import streamlit as st
import numpy as np
import pandas as pd


df=pd.read_csv("media prediction and its cost.csv")

features_to_drop = ['avg_cars_at home(approx).1', 'net_weight', 'meat_sqft', 'salad_bar', 'food_category', 'food_department', 'food_family', 'sales_country', 'marital_status', 'education', 'member_card', 'houseowner', 'brand_name']
df.drop(columns=features_to_drop, inplace=True)

df.dropna(inplace=True)

class output:
    def __init__(self,mse,r2,pred) -> None:
        self.mse = mse
        self.r2 = r2
        self.pred = pred

def Linear_reg(df,train_ratio):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    X = df.drop(columns='cost')
    y = df['cost']

    n_rows = df.shape[0]
    train_rows = int(n_rows * train_ratio)
    X_train = X[:train_rows]
    y_train = y[:train_rows]
    X_test = X[train_rows:]
    y_test = y[train_rows:]

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    
    return output(mean_squared_error(y_test, y_pred),r2_score(y_test, y_pred),y_pred)
    
def Lasso_reg(df,train_ratio):
    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_squared_error, r2_score

    X = df.drop(columns='cost')
    y = df['cost']
    
    n_rows = df.shape[0]
    train_rows = int(n_rows * train_ratio)
    X_train = X[:train_rows]
    y_train = y[:train_rows]
    X_test = X[train_rows:]
    y_test = y[train_rows:]
    

    lasso = Lasso(alpha=0.01)
    lasso.fit(X, y)

    y_pred = lasso.predict(X)
    
    return output(mean_squared_error(y, y_pred),r2_score(y, y_pred),y_pred)

def Random_Forest(df,train_ratio):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    
    X = df.drop(columns='cost')
    y = df['cost']
    
    n_rows = df.shape[0]
    train_rows = int(n_rows * train_ratio)
    X_train = X[:train_rows]
    y_train = y[:train_rows]
    X_test = X[train_rows:]
    y_test = y[train_rows:]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    
    return output(mean_squared_error(y_test, y_pred),r2_score(y_test, y_pred),y_pred)
    
if __name__ == "__main__":


    st.title("Customer Acquisition Cost Predictor")

    st.header("Team No. 265")

    st.markdown(""" Members: ### Anusha Garg    #### Bhavya Nagpal    #### Saatvik Gupta #### Gursehaj Singh    """)

    st.markdown(""" # Working of the project """)

    split=st.slider('Split the test and train data',0.0,1.0,0.7)

    model = st.radio(
        "Select the model on which you want to input",
        ('Linear Regression', 'Lasso Regression', 'Random Forests'))

    if st.button('Train Data'):
        match model:
            case 'Linear Regression':
                out=Linear_reg(df,split)
            case 'Lasso Regression':
                out=Lasso_reg(df,split)
            case 'Random Forests':
                out=Random_Forest(df,split)

    with st.expander("See Visualisations and Plots"):
        st.write("The chart above shows some numbers I picked for you.I rolled actual dice for these, so they're *guaranteed* to be random.")

    st.markdown(""" ## Enter Customer Data """)

    new_inp=pd.DataFrame(columns=df.columns)

    new_inp['store_sales(in millions)'].iloc[0] = st.number_input('Enter estimated Store Sales in months')
    
    new_inp['store_cost(in millions)'].iloc[0] = st.number_input('Enter estimated Store Cost (In Millions)')
    
    new_inp['unit_sales(in millions)'].iloc[0] = st.slider('Enter Unit Sales (In Millions)',1,5,3)
    
    
    new_inp['promotion_name'].iloc[0] = st.selectbox('Enter The Promotion Name',(df.promotion_name.unique()))

    new_inp['gender'].iloc[0] = st.selectbox('Gender of customer',('M','F'))
    
    new_inp['total_children'].iloc[0] = st.number_input('No. of children of the customer')
    
    new_inp['occupation'].iloc[0] = st.selectbox('Customer Occupation:',(df.occupation.unique()))
    
    new_inp['avg_cars_at home(approx)'].iloc[0] = st.slider('No. of Cars at home',0,6,1)
    
    
    new_inp['avg. yearly_income'].iloc[0]= st.selectbox('Average Yearly income',(df.promotion_name.unique()))
    
    new_inp['num_children_at_home'].iloc[0] = st.slider('No. of children at home',0,6,2)
    
    new_inp['SRP'].iloc[0] = st.number_input('SRP of product bought by the customer')
    
    new_inp['gross_weight'].iloc[0] = st.number_input('Gross weight of the product bought')
    
    new_inp['recyclable_package'].iloc[0] = st.selectbox('Recyclable Package or Not',(0,1))
    
    new_inp['low_fat'].iloc[0] = st.selectbox('Low fat or Not',(0,1))
    
    new_inp['units_per_case'].iloc[0] = st.number_input('Gross weight of the product bought')
    
    
## Label Encoding to be done at the end before the output is shown   
# categorical_cols = df.select_dtypes(include='object').columns
# from sklearn.preprocessing import LabelEncoder
# df[categorical_cols] = df[categorical_cols].apply(LabelEncoder().fit_transform)
      
    
