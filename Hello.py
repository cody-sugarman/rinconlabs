import streamlit as st
from streamlit.logger import get_logger
import numpy as np
import pandas as pd
import os

import tempfile

from utils import process_document_sample

LOGGER = get_logger(__name__)


def preprocess_value(value, entity):
    value = value.replace('.', '')
    value = value.replace('$ ', '$')
    value = value.replace('$-', '-$')
    value = value.replace('$', '')

    if entity.type == 'recourse_liabilities_ending':
        value = value.replace(' ', '')
        value = value.replace('.', '')

    if entity.type == 'withdrawals_and_distributions':
        value = value.replace('.', '')
        if ')' in value:
            value = value.replace(')', '')
            value = value.replace('(', '')
            value = '-' + value
        if '1 ' in value:
            value = value.replace('1 ', '')
    if entity.type == 'other_deductions':
        value = value.replace('W* ', '')
    return value

def get_data(path):
    # RinconLabs Model
    result = process_document_sample(
        project_id="125021716564",
        location="us",
        processor_id="f305627853b27ecf",
        file_path=path,
        mime_type="image/png",
    )
    document_data = []
    for entity in result.entities:
        value = entity.mention_text
        if value == "\342\230\221" or value == "☑": value = 'TRUE' # if it's a checkbox, change to true
        if value == '': value = 'FALSE'
        value = preprocess_value(value, entity)

        document_data.append({
            "Field":entity.type,
            "Value":value
        })
    document_df = pd.DataFrame(document_data)
    return document_df

def run():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="eighth-sensor-388122-3fd12de90c04.json"


    st.set_page_config(
        page_title="RinconLabs Demo",
        page_icon="🌊",
    )

    st.write("# Welcome to RinconLabs' K-1 Analyzer Demo! 👋")

    # uploaded_file = st.file_uploader("Upload any K-1 cover page here:")
    # if uploaded_file:
    #     temp_dir = tempfile.mkdtemp()
    #     path = os.path.join(temp_dir, uploaded_file.name)
    #     with open(path, "wb") as f:
    #         f.write(uploaded_file.getvalue())

    #     # Creating a button in the Streamlit interface
    #     if st.button("Extract Data", type="primary"):
    #         document_df = get_data(path)
    #         st.data_editor(
    #             document_df,
    #             hide_index=True,
    #         )


    uploaded_file = st.file_uploader("Upload any K-1 cover page here:")

    # Check for file upload and save path
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
    
    # Button to extract data
    if uploaded_file and st.button("Extract Data"):
        st.session_state.document_df = get_data(path)  # Process the file and save to session state

    # Display and allow editing of the dataframe if it's in the session state
    if 'document_df' in st.session_state and not st.session_state.document_df.empty:
        updated_df = st.data_editor(
            st.session_state.document_df,
            hide_index=True,
        )
        st.session_state.document_df = updated_df  # Update session state with changes

        # Select tax software to integrate with
        option = st.selectbox(
            'Please select your tax software (for export formatting purposes)',
            ('UltraTax CS', 'GoSystem Tax RS', 'CCH Axcess Tax', 'Lacerte')
        )

        # Convert DataFrame to CSV for download
        csv = st.session_state.document_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='k1_data.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    run()
