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

            # Convert dictionary to DataFrame
            if rows:
                df = pd.DataFrame.from_dict(rows, orient='index')
                df.sort_index(axis=0, inplace=True)
                df.sort_index(axis=1, inplace=True)
                tables[f'Table_{index}'] = df

        all_tables_data.append(tables)

    return all_tables_data

def analyze_document(image_key, s3_client, textract, bucket_name, object_name):
    # Upload the PDF to S3
    s3_client.upload_file(image_key, bucket_name, object_name)
    response = textract.analyze_document(
        Document={
            'S3Object': {
                'Bucket': bucket_name, 
                'Name': object_name
            }
        },
        FeatureTypes=['TABLES', 'FORMS', 'LAYOUT']
    )
    return response



def get_textract_tables_and_forms(file_name, images):
    # Upload K1 cover to s3 bucket for Textract usage
    s3_client = boto3.client('s3', region_name='us-east-1')  # Ensure the region is correct
    bucket_name = 'rincon-labs-s3-bucket'  # Updated bucket name
    object_name = file_name  # The name under which the PDF will be stored in S3

    # Run textract API
    textract = boto3.client('textract')

    # Example usage for each image
    responses = []
    for i in range(len(images)):
        image_key = f'{file_name}{i + 1}.png'
        response = analyze_document(image_key, s3_client, textract, bucket_name, object_name)
        responses.append(response)

    forms_data = get_form_data(responses)
    table_data = get_table_data(responses)
    return table_data, forms_data

def get_k1_cover_gpt_output(file_name, table_data, forms_data, client):
    # "One-shot" format of the required output fields for gpt-4o:
    k1_cover_page_format = {
    'K-1 Headers':{
        'Beginning':'',
        'Ending':'',
        'Year':2020,
        'Is Final K-1':False,
        'Is Amended K-1':False,
    },
    'Part I (Information About the Partnership)':{
        'A': {'Partnerships employer identification number':'12-3456789'},
        'B': {'Partnerships name, address, city, state, and ZIP code':''},
        'C': {'IRS center where partnership filed return':'e-file'},
        'D': {'Is a publicly traded partnership (PTP)':False}
    },
    'Part II (Information About the Partner)':{
        'E': {'Partners SSN or TIN':'12-3456789'},
        'F': {'Name, address, city, state, and ZIP code for partner':''},
        'G': {'Is General Partner':False},
        'G': {'Is Limited Partner':False},
        'H1': {'Is Domestic Partner':False},
        'H1': {'Is Foreign Partner':False},
        'H2': {
        'Is Disregarded Entity':False,
        'TIN':'',
        'Name':''
        },
        'I2': {'Is a retirement plan':''},
        'J': {
            'Partners share of beginning profit':'0%',
            'Partners share of ending profit':'0%',
            'Partners share of beginning loss':'0%',
            'Partners share of ending loss':'0%',
            'Partners share of beginning capital':'0%',
            'Partners share of ending capital':'0%',
            'Is decrease due to sale or exchange of partnership interest':False
        },
        'K - Partners share of liabilities': {
            'Partners share of beginning nonrecourse':'0%',
            'Partners share of ending nonrecourse':'0%',
            'Partners share of beginning qualified nonrecourse financing':'0%',
            'Partners share of ending qualified nonrecourse financing':'0%',
            'Partners share of beginning recourse':'0%',
            'Partners share of ending recourse':'0%',
            'Is item K includes liability amounts from lower-tier partnerships':False
        },
        'L - Partners capital account analysis': {
            'Beginning capital account':'0',
            'Capital contributed during year':'0',
            'Current year net income':'0',
            'Other increase/decrease':'0',
            'Withdrawals and distributions':'0',
            'Ending capital account':'0',
        },
        'M': {'Did the partner contribute property with a built-in gain':False},
        'N': {
            'Partners share of net unrecognized section 704c beginning':'0%',
            'Partners share of net unrecognized section 704c ending':'0%',
        }
    }
    }
    base64_image = encode_image(file_name)

    k1_cover_prompt = f"""
    Give me the data from the uploaded K-1 as a JSON output.
    Use the following data (tabular, forms, and text) to validate and every output.
    The output should be formatted exactly as follows (with no other text in the output).
    Replace single quotes with double quotes for JSON compliance and ensure there are no trailing commas: \n
    {str(k1_cover_page_format)}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            { "role": "user", "content": [
                {
                    "type": "text",
                    "text": k1_cover_prompt
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
        temperature=0,
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

def get_k1_supplement_gpt_output(file_name, images, client, table_data, forms_data, all_pdf_text):
    # K1 supplemental information prompt generation:
    k1_output_format = {
        "1": {
            "Ordinary business income (loss)":0,
        },
        "2": {
            "Net rental real estate income (loss)":0
        },
        "3": {
            "Other net rental income (loss)":0
        },
        "4a": {
            "Guaranteed payments for services":0
        },
        "4b": {
            "Guaranteed payments for capital":0
        },
        "4c": {
            "Total guaranteed payments":0
        },
        "5": {
            "Interest income": {
            "US Government/treasury":0,
            "Interest income - portfolio":0
            }
        },
        "6a": {
            "Ordinary dividends":0
        },
        "6b": {
            "Qualified dividends":0
        },
        "6c": {
            "Dividend equivalents":0
        },
        "7": {
            "Royalties":0
        },
        "8": {
            "Net short-term capital gain (loss)":0
        },
        "9a": {
            "Net long-term capital gain (loss)":0
        },
        "9b": {
            "Collectibles (28%) gain (loss)":0
        },
        "9c": {
            "Unrecaptured section 1250 gain":0
        },
        "10": {
            "Net section 1231 gain (loss)":0
        },
        "11": {
            "A - Other portfolio income": {
            "Sch E other income / (loss)":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Section 988 income":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Section 988 (loss)":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Other interest (Portfolio)":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "US Treasury interest":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Non-Qualified dividends":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Qualified dividends":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Royalties":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Short-term capital gain / (loss)":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Long-term capital gain / (loss)":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Total line 11a":{
                "passive":50,
                "non-passive":50,
                "total":100
            }
            },
            "B - Involuntary conversions": {
            "Total line 11b":{
                "passive":50,
                "non-passive":50,
                "total":100
            }
            },
            "C - Section 1256 gain (loss)": {
            "Total line 11c":{
                "passive":50,
                "non-passive":50,
                "total":100
            }
            },
            "E - Cancellation of debt": {
            "Total line 11e":{
                "passive":50,
                "non-passive":50,
                "total":100
            }
            },
            "F - Section 743(b) positive adjustments": {
            "Total line 11f":{
                "passive":50,
                "non-passive":50,
                "total":100
            }
            },
            "H - Section 951(a) inclusion": {
            "Total line 11h":{
                "passive":50,
                "non-passive":50,
                "total":100
            }
            },
            "I - Other income / (loss)": {
            "Sch E other income/loss":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Net section 1231 gain/loss":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Ordinary gain/loss (form 4797)":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Cancel debt income - 108(i)":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Form 1040, p1 other inc/loss (ordinary)":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Other interest (portfolio)":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "US treasury interest":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Self-charged interest":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Non-qualified dividends":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Qualified dividends":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Royalties":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Short-term capital gain/(loss)":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Long-term capital gain/(loss)":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Long-term capital gain/(loss) - QSBS":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Collectible (28%) L/T G/T":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Total line 11i":{
                "passive":50,
                "non-passive":50,
                "total":100
            }
            }
        },
        "12": {
            "Section 179 deduction":0
        },
        "13": {
            "A - cash contributions":0,
            "C - non-cash contributions":0,
            "H - Investment interest expense":{
            "Trading activities":0,
            "Investing activities":0
            },
            "I - Deductions - royalty income":0,
            "J - section 59(e)(2) expenditures":0,
            "L - Other portfolio deductions":0,
            "V - section 743(b) negative adjustments":0,
            "W - other expenses": {
            "Other deductions":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Sec 179 Expense":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Depreciation / amortization":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Investment int exp - sch A":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Investment int exp - sch E":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Charitable cont - cash 60%":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Charitable cont - noncash 50%":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "State taxes":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Section 212 - portfolio deductions":{
                "passive":50,
                "non-passive":50,
                "total":100
            },
            "Total line 13w":{
                "passive":450,
                "non-passive":450,
                "total":900
            }
            }
        },
        "14": {
            "Self-employment earnings (loss)":0
        },
        "15": {
            "Credit":0
        },
        "16": {
            "Is schedule K-3 attached":False
        },
        "17": {
            "Alternative minimum tax (AMT) items": {
            "A - post-1986 depreciation adjustment":0,
            "B - adjusted gain or loss":0,
            "C - Depletion (Other than Oil & Gas)":0,
            "D - oil, gas, and geothermal-gross income":0,
            "E - oil, gas, and geothermal-deductions":0,
            "F - other amt items":0,
            "Total line 17":0
            }
        },
        "18": {
            "Tax exempt income and nondeductible expenses": {
            "A - Tax-exempt interest":0,
            "B - Tax-exempt income":0,
            "C - Nondeductible expenses":0,
            "Total line 18":0
            }
        },
        "19": {
            "Distributions":0
        },
        "20": {
            "Other information": {
            "N - Interest Expense for Corporate Partners":0,
            "O - §453(I)(3) Information":0,
            "P - §453A(c) Information":0,
            "V - Unrelated Business Taxable Income":0,
            "AA - Section 704(c) information":0,
            "AE - Excess Taxable Income":0,
            "AF - Excess business interest income":0,
            "AG - Gross Reciepts for Section 448(c)":0,
            "A - Investment income":0,
            "B - Investment expenses":0,
            "ZZ - 3 Years or Less":0
            }
        },
        "21": {
            "Foreign taxes paid or accrued":0
        },
        "22": {
            "Is more than one activity for at-risk purposes":False
        },
        "23": {
            "Is more than one activity for passive activity purposes":False
        },
        '199A':[
            {
                'business_name':'',
                'EIN':0,
                'REP Aggregation':0,
                'SSTB/Non-SSTB':0,
                'Line 20Z - Ordinary Income':0,
                'Line 20Z - Rental Income':0,
                'Line 20Z - Royalty Income':0,
                'Line 20Z - Section 1231':0,
                'Line 20Z - Other Income':0,
                'Line 20Z - Section 179 Deduction':0,
                'Line 20Z - Charitable Contributions':0,
                'Line 20Z - Other Deductions':0,
                'Line 20Z - Section 199A W-2 Wages':0,
                'Line 20Z - Section 199A Unadjusted Basis':0,
                'Line 20Z - Section 199A Dividends':0,
                'Line 20Z - PTP Income':0
            }
        ],
        "163(j) Information":[
            {
                "Line 13K - Business Interest Expense":0,
                "Line 20AE - Excess Taxable Income":0,
                "Line 20AF - Excess business Interest Income":0,
                "Adjustable Taxable Income":0,
                "Gross Receipts":0,
                "Trade or Business Interest Expense":0,
            }
        ]
    }

    items_to_find = [k1_output_format.keys()]

    for items in items_to_find:
        items = {k: k1_output_format[k] for k in items}

        # Given the provided 1065 / K-1 document pages and corresponding data, extract all relevant information for Line Item 13.
        # Explain where you found the data for all of the 13W dictionary entry.

        supplemental_info_prompt = f"""
        Your task is the following:

        Given the provided 1065 / K-1 document pages and corresponding data, extract all relevant information for the provided JSON file.
        Ignore all document pages related to a K-3 not K-1.
        Start by locating the image page(s) with the relevant data. Then, find the corresponding 'Table Data' and 'Table Form' entries for the OCR extracted data.
        Confirm both the data AND the breakdown of passive / non-passive matches.
        For 199A data, include ALL entries in the table in the JSON output.

        The JSON file to populate is as follows:

        {str(items)}\n\n

        Supplemental instructions:
        1. **Data Sources**:
            - OCR Data: Text extracted from the images using OCR.
            - Table data: OCR data from AWS Textract, structured as a dictionary of dataframes converted to csvs corresponding to the tables found on each PDF page.
            - Form Data: OCR data from AWS Textract, structured as a dictionary of key/values found on each page.
            - K1 Document Pages: Images of K1 document pages.

        2. **Accuracy**:
            - This data will be used for filing tax returns, so it is CRITICAL that the data is accurate.
            - Validate each output using the image, table, form, and string data provided.

        3. **Data Extraction**:
            - Maintain the original format of the JSON file. Add data where you find it, and add a value of 0 for missing data.
            - The pages may be oriented incorrectly. Ensure you correctly interpret the orientation.
            - Break down the data of the deepest branches into passive, non-passive, and totals where applicable (instead of just 0 like the JSON template).

        4. **Boolean Values**:
            - Any key starting with 'Is' should map to a boolean value.

        5. **Output Format**:
            - Ensure the JSON is valid with no additional text in the output.
            - Replace every SSN or EIN with 'xx-xxxxxxx' to protect PII

        Use the provided tabular data, key/value data, OCR data, and K1 document pages to extract and validate the information accurately.
        The output given should get parsed without issue by the json.loads() function. Reminder that numbers should be expressed as integers without any commas.
        """

    openai_input_content=[
    {
        "type": "text",
        "text": supplemental_info_prompt
    },
    {
        "type": "text",
        "text": "PDF Data:\n" + all_pdf_text
    },
    ]

    # Save each image as PNG and append to input content
    for i, image in enumerate(images):
        temp_file_name = f'{file_name}{i+1}.png'
        base64_image = encode_image(temp_file_name)
        openai_input_content.append(
            {
                "type": "text",
                "text": f"PDF Page {i+1} image:\n"
            },
        )
        openai_input_content.append(
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        )

        openai_input_content.append(
            {
                "type": "text",
                "text": f"Table Data for Page {i+1}:\n {table_data[i]}"
            },
        )
        openai_input_content.append(
            {
                "type": "text",
                "text": f"Form Data for Page {i+1}:\n {forms_data[i]}"
            },
        )

    # Get all of the non-cover pages from the K1 and convert/feed into gpt-4o:

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            { "role": "user", "content": openai_input_content }
        ],
        temperature=0,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content