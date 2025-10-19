
import streamlit as st
import pandas as pd
from sdv.tabular import CTGAN

st.title('NHTS Synthetic Household Generator (CTGAN)')

model_path = st.text_input('CTGAN model path (.ctgan):', 'results/ctgan_sample.ctgan')
n_rows = st.number_input('Rows to generate:', min_value=10, max_value=10000, value=200, step=10)

if st.button('Generate'):
    try:
        model = CTGAN.load(model_path)
        df = model.sample(n_rows=int(n_rows))
        st.success(f'Generated {len(df)} rows')
        st.dataframe(df.head(20))
        st.download_button('Download CSV', df.to_csv(index=False), 'synthetic_households.csv', 'text/csv')
        # Simple hist if numeric columns present
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if num_cols:
            st.subheader('Example Distribution (first numeric column)')
            st.bar_chart(df[num_cols[0]].value_counts().sort_index())
    except Exception as e:
        st.error(str(e))
