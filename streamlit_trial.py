# %%
import streamlit as st

# %%
st.title("Customer Acquisition Cost Predictor")

# %%
st.header("Team No. 265")

# %%
st.markdown("""Members:
#### Anusha Garg
#### Bhavya Nagpal
#### Saatvik Gupta
#### Gursehaj Singh
""")

st.markdown(""" # Working of the project """)

split=st.slider('Split the test and train data',0.0,1.0,0.7)
st.write(split)

model = st.radio(
    "Select the model on which you want to input",
    ('Linear Regression', 'Lasso Regression', 'Random Forests'))


match model:
    case 'Linear Regression':
        st.write('shit1')
    case 'Lasso Regression':
        st.write('shit2')
    case 'Random Forests':
        st.write('shit3')
        
if st.button('Predict CAC'):
    st.write(model)


# %%
