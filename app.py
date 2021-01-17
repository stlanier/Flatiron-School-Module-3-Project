import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

sns.set_style('whitegrid')
sns.set_palette('mako')

raw_df = pd.read_csv('data/Heart_Disease_Data.csv')

'# Analyze Your Risk of Heart Disease'

if st.sidebar.checkbox('Legend'):  
    st.sidebar.write('''
    |Key|Meaning|Interpretation|
    |---|---|---|
    |_age_|age|0-99|
    |_sex_|sex|0 for female, 1 for male|
    |_cp_|||
    |_trestbps_|||
    |_chol_|||
    |_fbs_|||
    |_restecg_|||
    |_thalach_|||
    |_exang_|||
    |_oldpeak_|||
    |_slop_|||
    |_ca_|||
    |_thal_|||
    |_pred_attribute_|||
    ''')
if st.checkbox('Raw Data'):
    raw_df

option1 = st.sidebar.selectbox('Factor 1', (['-']+list(raw_df.columns))[:-1])
option2 = st.sidebar.selectbox('(Optional) Factor 2', (['-']+list(raw_df.columns))[:-1])
option3 = st.sidebar.selectbox('(Optional) Factor 3', (['-']+list(raw_df.columns))[:-1])

with open('models/softvote.pkl', 'rb') as fp:
    model = pickle.load(fp)
    
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
    options+=['pred_attribute']
#     fig, ax = plt.subplots()
    st.altair_chart(alt.Chart(raw_df[options]).mark_circle(size=60).encode(
    x=options[0],
    y=options[1],
    color=options[2],
    tooltip=[options[0], options[1], options[2]]).interactive(), True)