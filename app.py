import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import altair as alt

sns.set_style('whitegrid')
sns.set_palette('viridis')

raw_df = pd.read_csv('data/app_data.csv',  index_col='Unnamed: 0')

'# Analyze Your Risk of Heart Disease'

option1 = st.sidebar.selectbox('Factor 1', (['-']+list(raw_df.columns))[:-1])
option2 = st.sidebar.selectbox('(Optional) Factor 2', (['-']+list(raw_df.columns))[:-1])
option3 = st.sidebar.selectbox('(Optional) Factor 3', (['-']+list(raw_df.columns))[:-1])

# with open('models/softvote.pkl', 'rb') as fp:
#     model = pickle.load(fp)

options = [option for option in [option1, option2, option3] if option !='-']

if len(options)==1:
    fig, ax = plt.subplots()
    for group in raw_df[options+['pred_attribute']].groupby('pred_attribute'):
        group[1].hist(options[0], label=group[0], alpha=.7, ax=ax)
    fig.legend()
    ax.set_xlabel(options[0])
    ax.set_ylabel('Count')
    st.pyplot(fig)

if len(options)==2:
    st.altair_chart(alt.Chart(raw_df[options+['pred_attribute']]).mark_circle(size=60).encode(
    x=options[0],
    y=options[1],
    color='pred_attribute',
    tooltip=[options[0], options[1], 'pred_attribute']).interactive(), True)

if len(options)==3:
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    sctt = ax.scatter3D(raw_df[option1], raw_df[option2], raw_df[option3], c=raw_df['pred_attribute'], cmap='viridis', marker='o')
    ax.set_xlabel(option1)
    ax.set_ylabel(option2)
    ax.set_zlabel(option3)
    fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 15, ticks=[0,1], boundaries = [-.5, .5, 1.5], label='pred_attribute')
    st.pyplot(fig)

if st.checkbox('Legend'):
    st.write('''
    |Key|Meaning|Interpretation|
    |---|---|---|
    |_age_|age|0-99|
    |_sex_|sex|0 for female, 1 for male|
    |_cp_|chest pain|from 0 to 4 for worst|
    |_trestbps_|blood pressure on admission to hospital|mm Hg|
    |_chol_|serum cholesterol|mg/dl|
    |_fbs_|fasting blood sugar > 120 mg/dl|0 for no, 1 for yes|
    |_restecg_|resting electrocardiographic results|0-2|
    |_thalach_|max heart rate achieved|––|
    |_exang_|exercise induced angina|0 for no, 1 for yes|
    |_oldpeak_|ST depression induced by exercise relative to rest|––|
    |_slop_|slope of the peak exercise ST segment|––|
    |_ca_|number of major vessels colored by fluoroscopy|0-3|
    |_thal_|––|3 is normal, 6 is fixed defect, 7 is reversable defect|
    |_pred_attribute_|heart disease diagnosis|0 is no disease; 1 for some degree of disease|
    ''')
if st.checkbox('Raw Data'):
    raw_df
