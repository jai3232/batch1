import os, mlflow
import mlflow.sklearn
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn import metrics, model_selection, preprocessing
from sklearn.pipeline import Pipeline
import streamlit as st

import streamlit as st

# Embed Bootstrap CSS and JS
bootstrap_css_js = """
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.2/dist/umd/popper.min.js"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
"""

st.markdown(bootstrap_css_js, unsafe_allow_html=True)
st.markdown("""
<nav class="navbar navbar-expand-lg navbar-light bg-light">
    <a class="navbar-brand" href="#">ML</a>
    <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav">
            <li class="nav-item active">
                <a class="nav-link" href="#">Home <span class="sr-only">(current)</span></a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#">Features</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#">Pricing</a>
            </li>
        </ul>
    </div>
</nav>
""", unsafe_allow_html=True)



# function to load pickle object
@st.cache_resource
def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        pickle_object = pkl.load(f)
    return pickle_object

def load_model(uri):
    model = mlflow.sklearn.load_model(uri)
    return model



# st.write("Welcome")
st.html("<h1>Titanic survival Predictor</h1>")

# st.html(os.getcwd())
# src\ordinal_enconder_1.pkl
encoder_1 = load_pickle("src/ordinal_enconder_1.pkl")
encoder_2 = load_pickle("src/ordinal_enconder_2.pkl")

model = load_model("models:/Titanic_Model_Production@champion")
# st.write(type(encoder_1))
# st.write(type(encoder_2))

with st.form(key='my_form'):

    pclass = st.number_input("Pclass", min_value=1, max_value=3)

    sex = st.selectbox("Sex", options= ['female', 'male'])

    sibsp = st.number_input("Number of siblings and spouse", min_value=1, max_value=10)

    parch = st.number_input("Parch", min_value=0, max_value=10)

    fare = st.number_input("Fare", min_value=0.00, max_value=600.00, step=0.01)

    embark = st.selectbox("Embarked", options= ['C', 'Q', 'S'])

    age_group = ['Unknown', 'Baby', 'Child', 'Teenage', 'Student', 'Yound Adult', 'Adult', 'Senior']

    age = st.selectbox("Age Group", options= age_group)

    submit_button = st.form_submit_button(label='Submit')

    # age2 = st.number_input("Age", min_value=0)
    age2 = st.text_input("Age", value="4", type="password")
    # , max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, *, placeholder=None, disabled=False, label_visibility="visible")

# Create a label map
label_map = {0: "No", 1: "Yes"}
columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
       'Embarked', 'AgeGroup']
user_input = pd.DataFrame(np.array([[pclass, sex, age2, sibsp, parch, fare, embark, age]]
), columns=columns)

# user_input['AgeGroup'] = encoder_1.transfor

# print(user_input)

user_input['AgeGroup'] = encoder_1.transform(user_input['AgeGroup'].values.reshape(-1, 1))
user_input[['Sex', 'Embarked']] = encoder_2.transform(user_input[['Sex', 'Embarked']].values)

prediction = model.predict(user_input.values)
prediction_class = label_map[prediction[0]]
st.html("<h3>Will you survive the Titanic?</h3>")
st.html(prediction_class)

st.write(user_input)
st.write(user_input.values.shape)





