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

def get_form_data(response):
    # Extract key-value pairs
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

    return forms

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

def get_table_data(response):
    # Dictionary to hold table data, where each key will hold a DataFrame
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

        # Convert dictionary to DataFrame
        if rows:
            df = pd.DataFrame.from_dict(rows, orient='index')
            df.sort_index(axis=0, inplace=True)
            df.sort_index(axis=1, inplace=True)
            tables[f'Table_{index}'] = df

    return tables


def get_textract_tables_and_forms(file_name):
    file_name = file_name + '1.png'
    # Upload K1 cover to s3 bucket for Textract usage
    s3_client = boto3.client('s3', region_name='us-east-1')  # Ensure the region is correct
    bucket_name = 'rincon-labs-s3-bucket'  # Updated bucket name
    object_name = file_name  # The name under which the PDF will be stored in S3

    # Upload the PDF to S3
    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
        print("File uploaded successfully")
    except NoCredentialsError:
        print("Credentials not available")


    # Run textract API
    textract = boto3.client('textract')

    # Process the document
    response = textract.analyze_document(
        Document={
            'S3Object': {
                'Bucket': bucket_name,
                'Name': object_name
            }
        },
        FeatureTypes=['FORMS', 'TABLES', 'LAYOUT']  # Specify the features you need
    )
    forms_data = get_form_data(response)
    table_data = get_table_data(response)
    return table_data, forms_data

def get_k1_cover_gpt_output(file_name, table_data, forms_data, client):
    # "One-shot" format of the required output fields for gpt-4o:
    k1_cover_page_format = {
        'headers':{
            'beginning':'',
            'ending':'',
            'year':2020,
            'is_final_k-1':False,
            'is_amended_k-1':True,   
        },
        'part_one':{
            'partnerships_ein':'12-3456789',
            'partnership_name':'',
            'partnership_address':'',
            'irs_center_filed':'e-file',
            'is_ptp':False,
        },
        'part_two':{
            'partner_ssn':'12-3456789',
            'partner_name':'',
            'partner_address':'',
            'is_general_partner':False,
            'is_domestic_partner':False,
            'is_disregarded_entity':False,
            'is_limited_partner':False,
            'is_foreign_parnter':False,
            'disregarded_entity_TIN':'12-1232123',
            'disregarded_entity_name':'Name',
            'entity_type':'',
            'is_retirement_plan':False,
            'partner_share_profit_beginning':'0%',
            'partner_share_profit_ending':'0%',
            'partner_share_loss_beginning':'0%',
            'partner_share_loss_ending':'0%',
            'partner_share_capital_beginning':'0%',
            'partner_share_capital_ending':'0%',
            'decrease_sale_exchange_interest':False,
            'partner_share_liability_nonrecourse_beginning':0,
            'partner_share_liability_nonrecourse_ending':0,
            'partner_share_liability_qualified_nonrecourse_beginning':0,
            'partner_share_liability_qualified_nonrecourse_ending':0,
            'partner_share_liability_recourse_beginning':0,
            'partner_share_liability_recourse_ending':0,
            'is_item_k_includes_liability_amounts_lower_tier_partnerships':False,
            'beginning_capital_account':0,
            'capital_contributed_during_year':0,
            'current_year_net_income':0,
            'other_increase_decrease':0,
            'withdrawals_and_distributions':0,
            'ending_capital_account':0,
            'is_contribute_property_built_in_gain_loss':False,
            'partner_share_net_unrecognized_section_704c_beginning':0,
            'partner_share_net_unrecognized_section_704c_ending':0
        },
        'part_three':{
            '1':{'ordinary_business_income':0},
            '2':{'net_rental_real_estate_income':0},
            '3':{'other_net_rental_income':0},
            '4':{
                'a':{'guaranteed_payments_for_services':0},
                'b':{'guaranteed_payments_for_capital':0},
                'c':{'total_guaranteed_payments':0},
            },
            '5':{'interest_income':0},
            '6':{
                'a':{'ordinary_dividends':0},
                'b':{'qualified_dividends':0},
                'c':{'divident_equivalents':0},
            },
            '7':{'royalties':0},
            '8':{'net_short_term_capital_gains':0},
            '9':{
                'a':{'net_long_term_capital_gains':0},
                'b':{'collectibles_gain_loss':0},
                'c':{'unrecaptured_section_1250_gain':0},
            },
            '10':{'net_section_1231_gain':0},
            '11':{'other_income_loss':0},
            '12':{'section_179_deduction':0},
            '13':{'other_deductions':[0]},
            '14':{'self_employment_earnings':0},
            '15':{'credits':0},
            '16':{'schedule_k3_attached':False},
            '17':{'amt_items':[0]},
            '18':{'tax_exempt_income_and_nondeductable_expenses':[0]},
            '19':{'distributions':[0]},
            '20':{'other_information':[0]},
            '21':{'foreign_taxes_paid_or_accrued':0},
            '22':{'more_than_one_activity_at_risk':False},
            '23':{'more_than_one_activity_passive_activity':False},
        }
    }
    base64_image = encode_image(file_name)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            { "role": "user", "content": [
                {
                    "type": "text",
                    "text": "Give me the data from the uploaded K-1 as a JSON output. Use the following tabular and forms data from the form to validate and every output. The output should be formatted exactly as follows: " + str(k1_cover_page_format)
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                },
                {
                    "type": "text",
                    "text": str(table_data)
                },
                {
                    "type": "text",
                    "text": str(forms_data)
                },
            ] }
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content

def fix_json_keys(match):
    return f'"{match.group(1)}":'

def gpt_to_json(content):
    # Replace Python-style booleans with JSON-style booleans
    content = content.replace("True", "true").replace("False", "false")

    # Define a regex pattern to match the JSON object
    pattern = re.compile(r'\{.*\}', re.DOTALL)

    # Search for the JSON object in the content
    match = pattern.search(content)

    if match:
        # Extract the JSON string
        json_string = match.group(0)
        
        # Clean the JSON string (if necessary, depending on the content structure)
        json_string = json_string.strip()
        
        try:
            # Convert JSON string to Python dictionary
            k1_cover_page_json = json.loads(json_string)
            return k1_cover_page_json
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
    else:
        print("No valid JSON object found in the content.")

def get_k1_supplement_gpt_output(file_name, images, client):
    # K1 supplemental information prompt generation:
    supplement_output_format = {
        '11':{
            'a':{'other_portfolio_income':{}},
            'b':{'involuntary_conversions':{}},
            'c':{'section_1256_gain_loss':{}},
            'e':{'cancellation_of_debt':{}},
            'f':{'section_743b_positive_adjustments':{}},
            'h':{'section_951a_inclusion':{}},
            'i':{'other_income_loss':{}},
        },
        '13':{
            
        }
    }

    supplemental_info_prompt = "Given the following K1 document pages, pull out all data from Line 11a-11i and Line 13w. Make sure to include all subrows for each category. Where relevant, just use the total for the category but do not have the actual key in the key/value pair be 'total', 'passive' or 'non-passive'. Format this data as a JSON output as follow: " + str(supplement_output_format) + ".\n"

    input_content=[
        {
            "type": "text",
            "text": supplemental_info_prompt
        },
    ]

    # Save each image as PNG and append to input content
    for i, image in enumerate(images):
        temp_file_name = f'{file_name}{i+1}.png'
        base64_image = encode_image(temp_file_name)
        input_content.append({
            "type": "image_url",
            "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            { "role": "user", "content": input_content }
        ],
        max_tokens=2000
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content