import streamlit as st
from streamlit.logger import get_logger
import numpy as np
import pandas as pd
import os
import boto3
from botocore.exceptions import NoCredentialsError
import tempfile
from openai import OpenAI


from utils import pdf_to_pngs, get_textract_tables_and_forms, get_k1_cover_gpt_output, gpt_to_json, get_k1_supplement_gpt_output

LOGGER = get_logger(__name__)
openai_api_key = st.secrets["general"]["openai_api_key"]
openai_organization = st.secrets["general"]["openai_organization"]
openai_project = st.secrets["general"]["openai_project"]

# Enter your credentials here
aws_access_key_id = st.secrets["general"]["aws_access_key_id"]
aws_secret_access_key = st.secrets["general"]["aws_secret_access_key"]
region_name = 'us-east-1'

# Setting environment variables
os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
os.environ['AWS_DEFAULT_REGION'] = region_name

def extract_leaf_nodes(data, parent_key=''):
    items = []
    for k, v in data.items():
        new_key = k
        if isinstance(v, dict):
            items.extend(extract_leaf_nodes(v, new_key))
        else:
            items.append({'Field': new_key, 'Value': v})
    return items

def run():
    st.set_page_config(
        page_title="RinconLabs Demo",
        page_icon="🌊",
    )

    st.write("# Welcome to RinconLabs' K-1 Analyzer Demo! 👋")

    uploaded_file = st.file_uploader("Upload any K-1 cover page here:")

    client = OpenAI(
        organization=openai_organization,
        project=openai_project,
        api_key=openai_api_key
    )

    # Check for file upload and save path
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
        file_name, images = pdf_to_pngs(path)
        
    # Button to extract data
    if uploaded_file and st.button("Extract Data"):
        tables, forms = get_textract_tables_and_forms(file_name, images)
        k1_cover_gpt_output = get_k1_cover_gpt_output(file_name+'1.png', tables, forms, client)
        k1_cover_gpt_output_json = gpt_to_json(k1_cover_gpt_output)
        k1_supplement_gpt_output = get_k1_supplement_gpt_output(file_name, images, client, tables, forms)
        k1_supplement_gpt_output_json = gpt_to_json(k1_supplement_gpt_output)        
        k1_cover_gpt_output_json['part_three']['11'].update(k1_supplement_gpt_output_json['11'])
        k1_cover_gpt_output_json['part_three']['13'].update(k1_supplement_gpt_output_json['13'])

        st.json(k1_cover_gpt_output_json, expanded=True)
        cover_leaf_nodes = extract_leaf_nodes(k1_cover_gpt_output_json)
        cover_leaf_nodes_df = pd.DataFrame(cover_leaf_nodes)
        st.session_state.cover_document_df = cover_leaf_nodes_df
        

    # Display and allow editing of the dataframe if it's in the session state
    if 'cover_document_df' in st.session_state and not st.session_state.cover_document_df.empty:
        # updated_df = st.data_editor(
        #     st.session_state.cover_document_df,
        #     hide_index=True,
        #     use_container_width=True
        # )
        # st.session_state.cover_document_df = updated_df  # Update session state with changes

        # Select tax software to integrate with
        option = st.selectbox(
            'Please select your tax software (for export formatting purposes)',
            ('UltraTax CS', 'GoSystem Tax RS', 'CCH Axcess Tax', 'Lacerte')
        )

        # Convert DataFrame to CSV for download
        csv = st.session_state.cover_document_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='k1_data.csv',
            mime='text/csv',
        )


if __name__ == "__main__":
    run()
