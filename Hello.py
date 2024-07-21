import streamlit as st
from streamlit.logger import get_logger
import numpy as np
import pandas as pd
import os
import boto3
from botocore.exceptions import NoCredentialsError
import tempfile
from openai import OpenAI
from pypdf import PdfReader
import csv
import io
import json


from utils import pdf_to_pngs, get_textract_tables_and_forms, get_k1_gpt_output

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

def flatten_json(y, parent_key=''):
    rows = []
    def flatten(x, name=''):
        if isinstance(x, dict):
            if all(k in x for k in ['passive', 'non-passive', 'total']):
                rows.append((name[:-1], x))
            else:
                for k, v in x.items():
                    flatten(v, name + (parent_key + '.' if parent_key else '') + k + '.')
        elif isinstance(x, list):
            for i, v in enumerate(x):
                flatten(v, name + (parent_key + '.' if parent_key else '') + str(i) + '.')
        else:
            rows.append((name[:-1], x))
    flatten(y)
    return rows

def run():
    st.set_page_config(
        page_title="RinconLabs Demo",
        page_icon="🌊",
    )

    st.write("# Welcome to RinconLabs' K-1 Scanner Demo! 👋")

    # Opening JSON file
    k1_json_file = open('k1_json_data.json')
    k1_json_data_all = json.load(k1_json_file)

    # Define out to split up multiple calls to GPT for data extraction
    k1_json_keys_groups = [
        ['K-1 Headers', 'Part I (Information About the Partnership)', 'Part II (Information About the Partner)'],
        ['1', '2', '3', '4a', '4b', '4c', '5', '6a', '6b', '6c', '7', '8', '9a', '9b', '9c', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
        ['199A', '163(j) Information']
    ]

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

        # Convert PDF to string (including embeddings)
        # Initialize PdfReader
        reader = PdfReader(path)
        number_of_pages = len(reader.pages)

        # Initialize an empty string to hold the text
        all_pdf_text = ""

        # Extract text from each page and append it to the string
        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text()
            all_pdf_text += text + "\n"
        
    # Button to extract data
    if uploaded_file and st.button("Extract Data"):
        table_data, forms_data = get_textract_tables_and_forms(file_name, images)
        k1_gpt_output_json = get_k1_gpt_output(client, k1_json_keys_groups, k1_json_data_all, file_name, all_pdf_text, images, table_data, forms_data)

        st.json(k1_gpt_output_json, expanded=True)
        st.session_state.k1_gpt_output_json_session = k1_gpt_output_json
        
    # Display and allow editing of the dataframe if it's in the session state
    if 'k1_gpt_output_json_session' in st.session_state:
        # Select tax software to integrate with
        option = st.selectbox(
            'Please select your tax software (for export formatting purposes)',
            ('UltraTax CS', 'GoSystem Tax RS', 'CCH Axcess Tax', 'Lacerte')
        )

        k1_gpt_output_json = st.session_state.k1_gpt_output_json_session
        st.json(k1_gpt_output_json, expanded=True)

        # Flatten and prepare data
        flattened_data = flatten_json(k1_gpt_output_json)

        # Process flattened data
        rows = []
        header_set = set()
        for path, value in flattened_data:
            keys = path.split('.')
            if keys[0] == "199A" or keys[0] == "163(j) Information":
                continue  # Skip 199A and 163(j) Information sections in the initial section
            if isinstance(value, dict) and all(k in value for k in ['passive', 'non-passive', 'total']):
                section, subsection = keys[0], keys[1]
                detail = keys[2] if len(keys) > 2 else ''
                row = [section, subsection, detail, value['passive'], value['non-passive'], value['total']]
                rows.append(row)
            else:
                section = keys[0]
                subsection = keys[1] if len(keys) > 1 else ''
                detail = keys[2] if len(keys) > 2 else ''
                row = [section, subsection, detail, '', '', value]
                rows.append(row)

        # Process 199A section separately
        section_199A_rows = []
        header_199A_list = []
        if "199A" in k1_gpt_output_json:
            headers_199A = k1_gpt_output_json["199A"][0]
            header_199A_list = headers_199A
            for entry in k1_gpt_output_json["199A"][1:]:
                section_199A_rows.append(entry)

        # Process 163(j) section separately
        section_163j_rows = []
        header_163j_list = []
        if "163(j) Information" in k1_gpt_output_json:
            headers_163j = k1_gpt_output_json["163(j) Information"][0]
            header_163j_list = headers_163j
            for entry in k1_gpt_output_json["163(j) Information"][1:]:
                section_163j_rows.append(entry)

        # Add headers for the initial fields
        header_list = ['Section', 'Subsection', 'Detail', 'Passive', 'Non-passive', 'Total']

        # Use StringIO to handle CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(header_list)
        writer.writerows(rows)
        writer.writerow([])  # Add an empty row for separation
        writer.writerow(["Section 199A Information"])  # Header for 199A section
        writer.writerow(header_199A_list)
        writer.writerows(section_199A_rows)
        writer.writerow([])  # Add an empty row for separation
        writer.writerow(["Section 163(j) Information"])  # Header for 163(j) section
        writer.writerow(header_163j_list)
        writer.writerows(section_163j_rows)

        output.seek(0)  # Reset the StringIO object for reading
        csv_content = output.getvalue()

        # Convert DataFrame to CSV for download
        st.download_button(
            label="Download Data as CSV",
            data=csv_content,
            file_name='k1_data.csv',
            mime='text/csv',
        )


if __name__ == "__main__":
    run()
