import streamlit as st
from collections import OrderedDict
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
import concurrent.futures
from pdf2image import convert_from_path
from pypdf import PdfReader
import base64
import json
import re

from utils import pdf_to_pngs, get_textract_tables_and_forms, get_k1_gpt_output

LOGGER = get_logger(__name__)
openai_api_key = st.secrets["general"]["openai_api_key"]
openai_organization = st.secrets["general"]["openai_organization"]
openai_project = st.secrets["general"]["openai_project"]

# Enter your credentials here
aws_access_key_id = st.secrets["general"]["aws_access_key_id"]
aws_secret_access_key = st.secrets["general"]["aws_secret_access_key"]
print(aws_access_key_id)
print(aws_secret_access_key)
region_name = 'us-east-1'

# Setting environment variables
os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
os.environ['AWS_DEFAULT_REGION'] = region_name

s3_client = boto3.client('s3', region_name='us-east-1')  # Ensure the region is correct
bucket_name = 'rincon-labs'  # Updated bucket name
# Initialize Textract client
textract = boto3.client('textract')

# Opening JSON file
k1_json_file = open('k1_json_data.json')
k1_json_data_all = json.load(k1_json_file)

client = OpenAI(
    organization=openai_organization,
    project=openai_project,
    api_key=openai_api_key
)


def process_json_array(all_results, file_paths):
    # Use the first JSON object to determine the row headers
    first_json = all_results[0]
    flattened_headers = flatten_json(first_json)

    # Create a header list based on the keys from the first JSON object
    header_list = ['Section', 'Subsection', 'Detail']
    
    for i in range(len(all_results)):
        header_list.append(f'{os.path.basename(file_paths[i])}')

    # Initialize rows dictionary to store data for each key
    rows_dict = OrderedDict()

    for path, value in flattened_headers:
        keys = path.split('.')
        section = keys[0]
        subsection = keys[1] if len(keys) > 1 else ''
        detail = keys[2] if len(keys) > 2 else ''
        rows_dict[path] = [section, subsection, detail] + [''] * len(all_results)

    # Populate the rows_dict with values from each JSON object
    for i, json_obj in enumerate(all_results):
        flattened_data = flatten_json(json_obj)
        for path, value in flattened_data:
            if path in rows_dict:
                rows_dict[path][3 + i] = value

    # Write to CSV
    output_file = 'output_file.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_list)
        for row in rows_dict.values():
            writer.writerow(row)
    
    return output_file

def flatten_json(y, parent_key=''):
    """Flatten the JSON object while maintaining the order of keys."""
    rows = []
    def flatten(x, name=''):
        if isinstance(x, dict):
            for k, v in x.items():
                flatten(v, name + (parent_key + '.' if parent_key else '') + k + '.')
        elif isinstance(x, list):
            for i, v in enumerate(x):
                flatten(v, name + (parent_key + '.' if parent_key else '') + str(i) + '.')
        else:
            rows.append((name[:-1], x))
    flatten(y)
    return rows

def analyze_document(image_key, bucket_name, object_name):
    s3_client.upload_file(image_key, bucket_name, object_name)
    response = textract.analyze_document(
        Document={
            'S3Object': {
                'Bucket': bucket_name,
                'Name': object_name
            }
        },
        FeatureTypes=['TABLES', 'FORMS']
    )
    return response

def analyze_documents_in_parallel(image_keys, bucket_name, object_name):
    # Initialize a list of None to hold the results in order
    responses = [None] * len(image_keys)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks and store the futures in a list
        future_to_index = {
            executor.submit(analyze_document, key, bucket_name, object_name): i
            for i, key in enumerate(image_keys)
        }

        # As each future completes, place the result in the correct position
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                response = future.result()
                responses[index] = response
            except Exception as e:
                print(f"Error processing {image_keys[index]}: {e}")
    return responses

def get_form_data(responses):
    all_forms_data = []

    for response in responses:
        key_map = {}
        value_map = {}
        block_map = {}

        for block in response['Blocks']:
            block_id = block['Id']
            block_map[block_id] = block
            if block['BlockType'] == "KEY_VALUE_SET":
                if 'KEY' in block['EntityTypes']:
                    key_map[block_id] = block
                else:
                    value_map[block_id] = block

        # Get KeyValue relationship
        forms = {}
        for block_id, key_block in key_map.items():
            value_block = find_value_block(key_block, value_map, block_map)
            key_text = get_text(key_block, block_map)
            value_text = get_text(value_block, block_map)
            forms[key_text] = value_text

        all_forms_data.append(forms)

    return all_forms_data

def find_value_block(key_block, value_map, block_map):
    for relationship in key_block.get('Relationships', []):
        if relationship['Type'] == 'VALUE':
            for value_id in relationship['Ids']:
                if value_id in value_map:
                    return value_map[value_id]
    return None

def get_text(block, block_map):
    text = ''
    if 'Relationships' in block:
        for relationship in block['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    child = block_map[child_id]
                    if child['BlockType'] == 'WORD':
                        text += child['Text'] + ' '
                    if child['BlockType'] == 'SELECTION_ELEMENT':
                        if child['SelectionStatus'] == 'SELECTED':
                            text += 'X '
    return text.strip()

def get_table_data(responses):
    all_tables_data = []

    for response in responses:
        tables = {}
        table_blocks = [block for block in response['Blocks'] if block['BlockType'] == 'TABLE']

        for index, table in enumerate(table_blocks):
            rows = {}
            for rel in table['Relationships']:
                if rel['Type'] == 'CHILD':
                    for child_id in rel['Ids']:
                        cell = next(block for block in response['Blocks'] if block['Id'] == child_id)
                        if 'RowIndex' in cell and 'ColumnIndex' in cell:
                            row_index = cell['RowIndex']
                            col_index = cell['ColumnIndex']
                            if row_index not in rows:
                                rows[row_index] = {}
                            # Extract and combine words within each cell
                            cell_text = ''
                            if 'Relationships' in cell:
                                for cell_rel in cell['Relationships']:
                                    if cell_rel['Type'] == 'CHILD':
                                        for word_id in cell_rel['Ids']:
                                            word_info = next(word for word in response['Blocks'] if word['Id'] == word_id)
                                            if 'Text' in word_info:
                                                cell_text += word_info['Text'] + ' '
                            rows[row_index][col_index] = cell_text.strip()

            # Convert dictionary to CSV string
            if rows:
                df = pd.DataFrame.from_dict(rows, orient='index')
                df.sort_index(axis=0, inplace=True)
                df.sort_index(axis=1, inplace=True)
                csv_string = df.to_csv(index=False, header=False)
                tables[f'Table_{index}'] = csv_string

        all_tables_data.append(tables)

    return all_tables_data

def get_subset(original_dict, keys):
    subset_dict = dict((key, original_dict[key]) for key in keys if key in original_dict)
    return subset_dict

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def process_to_json(content):
    # Helper function to fix keys and strings
    def fix_json_keys(match):
        key = match.group(1)
        return f'"{key}":'

    def fix_json_strings(match):
        return f'"{match.group(1)}"'

    # Ensure all keys and string values are enclosed in double quotes
    content = re.sub(r"'(\w+)':", fix_json_keys, content)
    content = re.sub(r'\'([^\']*?)\'', fix_json_strings, content)
    content = re.sub(r'(\w+):', fix_json_keys, content)
    content = re.sub(r'(\w+\s[\w\s]+):', fix_json_keys, content)

    # Replace Python-style booleans with JSON-style booleans
    content = content.replace("True", "true").replace("False", "false")

    # Remove commas from numeric values
    content = re.sub(r'(\d),(\d)', r'\1\2', content)

    # Fix misplaced quotes in keys
    content = re.sub(r'"year":', 'year:', content)

    # Correctly escape single quotes within keys and values
    content = re.sub(r'(\w+)\'s', r'\1\'s', content)

    # Define a regex pattern to match the JSON object
    pattern = re.compile(r'\{.*\}', re.DOTALL)

    # Search for the JSON object in the content
    match = pattern.search(content)

    if match:
        # Extract the JSON string
        json_string = match.group(0)

        try:
            # Convert JSON string to Python dictionary
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            return None
    else:
        print("No valid JSON object found in the content.")
        return None

# 1. Validate both the data AND the breakdown of passive / non-passive matches across data sources where possible.

def extract_json_data_batch(k1_json_keys, k1_json_data_all, file_name, all_pdf_text, images, table_data, forms_data):
    k1_json_dict = get_subset(k1_json_data_all, k1_json_keys)

    k1_json_fields_prompt = f"""
    Your task is the following:

    Given the provided 1065 / K-1 document pages and corresponding data, extract values for following JSON keys:\n\n

    {str(k1_json_dict)}\n\n

    Use the following steps to complete this task:

    1. Be sure to use the context of the parent key of the JSON file when nested to identify the relevant value
    2. When there is a letter and number combination (e.g. "13AE"), use this as the key search term
    3. Start by locating the image page(s) with the relevant data. If you cannot find any relevant data, return an empty JSON object.
    4. Find the corresponding 'Table Data' and 'Table Form' entries for the OCR extracted data.
    5. Find the corresponding PDF Data from the associated string.
    6. For 199A and 163(j) data (if requested in the input JSON), leverage the 'Table Data' corresponding to the correct image pages. Include the first 5 entries of the table in the JSON output and not just the totals.
    7. The final output you provide should get parsed without issue by the json.loads() function (all keys must be strings and enclosed in double quote; any special characters inside the string must be escaped)

    Supplemental instructions:
    **Data Sources**:
        - OCR Data: Text extracted from the images using OCR.
        - Table data: OCR data from AWS Textract, structured as a dictionary of csvs ('Table_0', 'Table_1', etc) corresponding to the tables found on each PDF page.
        - Form Data: OCR data from AWS Textract, structured as a dictionary of key/values found on each page.
        - K1 Document Pages: Images of K1 document pages.

    **Accuracy**:
        - This data will be used for filing tax returns, so it is CRITICAL that the data is accurate.
        - Validate each output using the image, table, form, and string data provided.

    **Data Extraction**:
        - Maintain the original format of the JSON file. Add data where you find it, and add a value of 0 for missing data.
        - The pages may be oriented incorrectly. Ensure you correctly interpret the orientation.
        - Some tables might expand over multiple pages.
        - For table data (like '199A' and '163(j) Information'), the example JSON only includes one row but there are likely multiple rows in the actual table - extract the first 40 rows.

    **Boolean Values**:
        - Any key starting with 'Is' should map to a boolean value.

    **Output Format**:
        - If a number is inside of parenthesis in the PDF, it should be inputted as negative
        - Numbers should include all decimals but not include any commas or other thousands separators.
        - Percentages should be expressed out of 100 (e.g. 1.1% = 1.1)
        - If no data is found for a given field, use 0.
    """

    openai_input_content = [
        {"type": "text", "text": k1_json_fields_prompt},
        {"type": "text", "text": "PDF Data:\n" + all_pdf_text},
    ]

    for i, image in enumerate(images):
        temp_file_name = file_name.replace('.pdf', f'_{i + 1}.png')
        print('img temp name 1: ' + temp_file_name)
        base64_image = encode_image(temp_file_name)
        openai_input_content.append({"type": "text", "text": f"PDF Page {i+1} image:\n"})
        openai_input_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        openai_input_content.append({"type": "text", "text": f"Table Data for Page {i+1}:\n {table_data[i]}"})
        openai_input_content.append({"type": "text", "text": f"Form Data for Page {i+1}:\n {forms_data[i]}"})

    # response = client.beta.chat.completions.parse(
    response = client.chat.completions.create(
        # model="gpt-4o",
        model="chatgpt-4o-latest",
        # model="gpt-4o-mini",
        messages=[{"role": "user", "content": openai_input_content}],
        temperature=0,
        # response_format={ "type": "json_object" }
        # response_format = { "type": "json_schema", "json_schema": k1_json_data_all , "strict": true }
    )
    response_content = response.choices[0].message.content
    print(response_content)
    parsed_json = process_to_json(response_content)
    return parsed_json

# Function to handle individual API call
def extract_json_data_batch_wrapper(k1_json_keys, k1_json_data_all, file_name, all_pdf_text, images, table_data, forms_data):
    return extract_json_data_batch(k1_json_keys, k1_json_data_all, file_name, all_pdf_text, images, table_data, forms_data)

def process_pdf(path, output_dir, bucket_name, k1_json_keys_groups):
    file_name = path
    try:
        print(f"Processing {file_name}...")

        # Initialize PdfReader
        reader = PdfReader(file_name)
        number_of_pages = len(reader.pages)
        print(f"Number of pages in {file_name}: {number_of_pages}")  # Debug: Check the number of pages

        # Initialize an empty string to hold the text
        all_pdf_text = ""

        # Extract text from each page and append it to the string
        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text()
            all_pdf_text += text + "\n"

        print(f"Extracted text from {file_name}: {len(all_pdf_text)} characters")  # Debug: Check the text extraction

        # Convert PDF to PNGs
        if 'pdf' in file_name:
            # Convert PDF to list of images
            images = convert_from_path(file_name)
            os.makedirs(output_dir, exist_ok=True)

            # Ensure unique base file name
            base_file_name = file_name.replace('.pdf', '')
            image_keys = []
            for i, image in enumerate(images):
                print('img temp name 2: ' + base_file_name)
                output_path = os.path.join(output_dir, f'{base_file_name}_{i + 1}.png')
                image.save(output_path, 'PNG')
                image_keys.append(output_path)
                print(f"Saved image {output_path}")  # Debug: Confirm image saving

        print(f"{file_name}: PDF pages converted to PNG images successfully.")

        # Upload K1 cover to S3 bucket for Textract usage
        object_name = base_file_name  # The name under which the PDF will be stored in S3

        # Analyze documents in parallel and maintain order
        responses = analyze_documents_in_parallel(image_keys, bucket_name, object_name)
        print(f"Responses for {file_name}: {responses}")  # Debug: Check the content of responses

        forms_data = get_form_data(responses)
        print(f"Forms data for {file_name}: {forms_data}")  # Debug: Check the content of forms_data

        table_data = get_table_data(responses)
        print(f"Table data for {file_name}: {table_data}")  # Debug: Check the content of table_data

        # Main function to execute API calls in parallel and combine results
        combined_json = OrderedDict()  # Ensure this is initialized per file

        # Initialize combined_json with keys from nested arrays
        for group in k1_json_keys_groups:
            for key in group:
                combined_json[key] = None

        # Execute API calls in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all API calls
            futures = {executor.submit(extract_json_data_batch_wrapper, k1_json_keys, k1_json_data_all, file_name, all_pdf_text, images, table_data, forms_data): k1_json_keys for k1_json_keys in k1_json_keys_groups}

            # Combine results as they complete
            for future in concurrent.futures.as_completed(futures):
                group = futures[future]
                result = future.result()
                if result:
                    for key in group:
                        if key in result:
                            combined_json[key] = result[key]

        # Filter out None values to ensure only completed data is included
        combined_json = OrderedDict((k, v) for k, v in combined_json.items() if v is not None)
        return combined_json

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def process_all_pdfs_concurrently(file_paths, output_dir, bucket_name, k1_json_keys_groups):
    results = []

    # Print file paths to confirm they are distinct
    print("File paths to process:", file_paths)  # Debug: Check file paths

    # Use ThreadPoolExecutor to process PDFs in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_pdf, path, output_dir, bucket_name, k1_json_keys_groups): path for path in file_paths}

        # Collect results as they complete, along with the corresponding file path
        completed_results = []
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                if result is not None:
                    completed_results.append((path, result))
            except Exception as e:
                print(f"Error processing {path}: {e}")

    # Sort the results by the original file_paths order
    ordered_results = [result for path, result in sorted(completed_results, key=lambda x: file_paths.index(x[0]))]

    return ordered_results

def run():
    st.set_page_config(
        page_title="RinconLabs Demo",
        page_icon="🌊",
    )

    st.write("# Welcome to the F&H K-1 Scanner! 👋")

    k1_json_keys_groups = [
    [
        "K-1 Other Information",
        "Tax Basis and FMV Calculations",
        "FMV Adjusting Entry Calculation",
        "K-1 Line Items",
        "Other Income (Loss) (part one)",
        "Other Income (Loss) (part two)",
        "Other Income (Loss) (part three)",
        "Section 179 & Other Deductions",
        "Section 179 & Other Deductions (continued)",
        "Credits",
        "Alternative Minimum Tax (AMT) Items",
        "Tax-Exempt Income and Nondeductible Expenses",
        "Other Information",
        "Other Footnote Items (not broken out above)"
    ]
    ]
    uploaded_files = st.file_uploader("Upload upto 100x K-1s here:", accept_multiple_files=True)

    # Check for file upload and save path
    if uploaded_files:
        file_paths = []
        for file in uploaded_files: 
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, file.name)
            with open(path, "wb") as f:
                file_paths.append(path)
                f.write(file.getvalue())

        
    # Button to extract data
    if uploaded_files and st.button("Extract Data"):
        output_dir = '/'
        bucket_name = 'rincon-labs'
        all_results = process_all_pdfs_concurrently(file_paths, output_dir, bucket_name, k1_json_keys_groups)
        st.session_state.k1_gpt_output_json_session = all_results
        output_file = process_json_array(all_results, file_paths)

        # In your Streamlit app, offer the file for download
        with open(output_file, 'rb') as f:
            st.download_button(
                label="Download CSV",
                data=f,
                file_name=output_file,
                mime='text/csv'
            )

if __name__ == "__main__":
    run()
