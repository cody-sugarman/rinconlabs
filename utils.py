# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import textwrap

import streamlit as st

from typing import Optional

from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore

from pdf2image import convert_from_path
import os
import boto3
from botocore.exceptions import NoCredentialsError
import pandas as pd
from openai import OpenAI
import base64
import requests
import json
import re
import concurrent.futures

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))

def pdf_to_pngs(file_name):
    # Convert PDF to list of images
    images = convert_from_path(file_name)

    # Output directory
    output_dir = '/'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save each image as PNG
    for i, image in enumerate(images):
        file_name = file_name.replace('.pdf', '')
        output_path = os.path.join(output_dir, f'{file_name}{i + 1}.png')
        image.save(output_path, 'PNG')
    print("FILE_NAME: " + file_name)
    return file_name, images

    print("PDF pages converted to PNG images successfully.")

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

def analyze_document(image_key, s3_client, textract, bucket_name, object_name):
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

def analyze_documents_in_parallel(image_keys, s3_client, textract, bucket_name, object_name):
    # Initialize a list of None to hold the results in order
    responses = [None] * len(image_keys)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks and store the futures in a list
        future_to_index = {
            executor.submit(analyze_document, key, s3_client, textract, bucket_name, object_name): i
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

def get_textract_tables_and_forms(file_name, images):
    # Upload K1 cover to s3 bucket for Textract usage
    s3_client = boto3.client('s3', region_name='us-east-1')  # Ensure the region is correct
    bucket_name = 'rincon-labs'  # Updated bucket name
    object_name = file_name  # The name under which the PDF will be stored in S3

    # Run textract API
    textract = boto3.client('textract')
    image_keys = [f'{file_name}{i + 1}.png' for i in range(len(images))]
    print("image_keys: " + str(image_keys))

    responses = analyze_documents_in_parallel(image_keys, s3_client, textract, bucket_name, object_name)

    print("responses: " + str(responses))
    # Post-processing for Textract data
    forms_data = get_form_data(responses)
    table_data = get_table_data(responses)
    return table_data, forms_data

def fix_json_keys(match):
    return f'"{match.group(1)}":'

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

def get_subset(original_dict, keys):
    subset_dict = dict((key, original_dict[key]) for key in keys if key in original_dict)
    return subset_dict

def extract_json_data_batch(client, k1_json_keys, k1_json_data_all, file_name, all_pdf_text, images, table_data, forms_data):
    k1_json_dict = get_subset(k1_json_data_all, k1_json_keys)

    k1_json_fields_prompt = f"""
    Your task is the following:

    Given the provided 1065 / K-1 document pages and corresponding data, extract values for following JSON keys:\n\n

    {str(k1_json_dict)}\n\n

    Use the following steps to complete this task:

    1. Start by locating the image page(s) with the relevant data. If you cannot find any relevant data, return an empty JSON object.
    2. Find the corresponding 'Table Data' and 'Table Form' entries for the OCR extracted data.
    3. Find the corresponding PDF Data from the associated string.
    4. For 199A and 163(j) data (if requested in the input JSON), leverage the 'Table Data' corresponding to the correct image pages. Include the first 5 entries of the table in the JSON output and not just the totals.
    5. Validate both the data AND the breakdown of passive / non-passive matches across data sources where possible.

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
        - Break down the data of the deepest branches into passive, non-passive, and totals where applicable (instead of just 0 like the JSON template).
        - Some tables might expand over multiple pages.
        - For table data (like '199A' and '163(j) Information'), the example JSON only includes one row but there are likely multiple rows in the actual table - extract the first 40 rows.

    **Boolean Values**:
        - Any key starting with 'Is' should map to a boolean value.

    **Output Format**:
        - Numbers should be expressed as integers without any commas.
        - The output given should get parsed without issue by the json.loads() function.

    Explain how you arrived at the final output, including which data sources were used and how they were combined.

    """
    
    openai_input_content = [
        {"type": "text", "text": k1_json_fields_prompt},
        {"type": "text", "text": "PDF Data:\n" + all_pdf_text},
    ]

    for i, image in enumerate(images):
        temp_file_name = f'{file_name}{i+1}.png'
        base64_image = encode_image(temp_file_name)
        openai_input_content.append({"type": "text", "text": f"PDF Page {i+1} image:\n"})
        openai_input_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        openai_input_content.append({"type": "text", "text": f"Table Data for Page {i+1}:\n {table_data[i]}"})
        openai_input_content.append({"type": "text", "text": f"Form Data for Page {i+1}:\n {forms_data[i]}"})

    response = client.chat.completions.create(
        model="gpt-4o",
        # model="gpt-4o-mini",
        messages=[{"role": "user", "content": openai_input_content}],
        temperature=0,
    )
    response_content = response.choices[0].message.content
    print(response_content)
    parsed_json = process_to_json(response_content)
    return parsed_json

# Function to handle individual API call
def extract_json_data_batch_wrapper(client, k1_json_keys, k1_json_data_all, file_name, all_pdf_text, images, table_data, forms_data):
    return extract_json_data_batch(client, k1_json_keys, k1_json_data_all, file_name, all_pdf_text, images, table_data, forms_data)

def get_k1_gpt_output(client, k1_json_keys_groups, k1_json_data_all, file_name, all_pdf_text, images, table_data, forms_data):
    # Main function to execute API calls in parallel and combine results
    combined_json = {}

    # Execute API calls in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all API calls
        futures = {
            executor.submit(extract_json_data_batch_wrapper, client, k1_json_keys, k1_json_data_all, file_name, all_pdf_text, images, table_data, forms_data): k1_json_keys
            for k1_json_keys in k1_json_keys_groups
        }

        # Combine results as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                combined_json.update(result)

    # Print the combined JSON
    print(json.dumps(combined_json, indent=4))
    return combined_json