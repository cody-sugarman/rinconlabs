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
        tables, forms = get_textract_tables_and_forms(file_name, images)
        k1_cover_gpt_output = get_k1_cover_gpt_output(file_name+'1.png', tables, forms, client)
        k1_cover_gpt_output_json = gpt_to_json(k1_cover_gpt_output)
        k1_supplement_gpt_output = get_k1_supplement_gpt_output(file_name, images, client, tables, forms, all_pdf_text)
        k1_supplement_gpt_output_json = gpt_to_json(k1_supplement_gpt_output)        
        k1_cover_gpt_output_json.update(k1_supplement_gpt_output_json)

        st.json(k1_cover_gpt_output_json, expanded=True)
        cover_leaf_nodes = extract_leaf_nodes(k1_cover_gpt_output_json)
        cover_leaf_nodes_df = pd.DataFrame(cover_leaf_nodes)
        st.session_state.cover_document_df = cover_leaf_nodes_df
        

    # Display and allow editing of the dataframe if it's in the session state
    if 'cover_document_df' in st.session_state and not st.session_state.cover_document_df.empty:
        # Select tax software to integrate with
        option = st.selectbox(
            'Please select your tax software (for export formatting purposes)',
            ('UltraTax CS', 'GoSystem Tax RS', 'CCH Axcess Tax', 'Lacerte')
        )

        # Flatten and prepare data
        flattened_data = flatten_json(k1_cover_gpt_output_json)

        # Process flattened data
        rows = []
        header_set = set()
        for path, value in flattened_data:
            keys = path.split('.')
            if keys[0] == "199A" or keys[0].startswith("163(j) Information"):
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
        if "199A" in k1_supplement_gpt_output_json:
            for entry in k1_supplement_gpt_output_json["199A"]:
                row = []
                row.append(entry.get("business_name", ""))
                row.append(entry.get("EIN", ""))
                for key, value in entry.items():
                    if key not in ["business_name", "EIN"]:
                        header_set.add(key)
                        row.append(value)
                section_199A_rows.append(row)

        # Add headers for the 199A fields
        header_list = ['Section', 'Subsection', 'Detail', 'Passive', 'Non-passive', 'Total']
        header_199A_list = ['Trade or Business Name', 'EIN'] + list(header_set)

        # Process 163(j) section separately
        section_163j_rows = []
        header_163j_set = set()
        if "163(j) Information" in k1_supplement_gpt_output_json:
            for entry in k1_supplement_gpt_output_json["163(j) Information"]:
                row = []
                for key, value in entry.items():
                    header_163j_set.add(key)
                    row.append(value)
                section_163j_rows.append(row)

        # Add headers for the 163(j) fields
        header_163j_list = list(header_163j_set)

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
