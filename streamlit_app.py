import streamlit as st
import pandas as pd
import os
import tempfile
from typing import List, Dict, Tuple
import anthropic
import json
import asyncio
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PreProcessor
from haystack.schema import Document

# Set page config
st.set_page_config(page_title="Voice Notes to Spreadsheet Processor", layout="wide")

# Initialize session state variables if they don't exist
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'extracted_entities' not in st.session_state:
    st.session_state.extracted_entities = {}
if 'edited_texts' not in st.session_state:
    st.session_state.edited_texts = {}

# Function to create a prompt for extracting entities
def create_extraction_prompt(text: str) -> str:
    return f"""
    Extract all mentions of flowers or seeds and their associated quantities from the following text. 
    Return the result as a Python dictionary where keys are the flower/seed names and values are their quantities.
    If no quantity is mentioned for a flower/seed, set the value to None.
    
    TEXT:
    {text}
    
    INSTRUCTIONS:
    - Only extract flowers and seeds (no other plants or items)
    - Normalize flower/seed names to their common form (e.g., 'roses' -> 'rose')
    - Convert all quantities to numeric values where possible
    - Return the result as a Python dictionary formatted like: {{"rose": 12, "sunflower seeds": 500}}
    - Only return the dictionary, no additional text or explanation
    """

# Single async function to process one text with Claude
async def process_text_with_claude(text: str, client: anthropic.AsyncAnthropic) -> Dict:
    prompt = create_extraction_prompt(text)
    
    message = await client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    response_text = message.content[0].text
    
    # Extract dictionary from response
    try:
        # Look for dictionary-like patterns
        import re
        dict_pattern = r'\{[^{}]*\}'
        dict_match = re.search(dict_pattern, response_text)
        
        if dict_match:
            dict_str = dict_match.group(0)
            # Safely evaluate the dictionary string to convert it to a Python dict
            return eval(dict_str)
        else:
            # Try to parse the entire response as a dictionary
            response_text = response_text.strip()
            if response_text.startswith('{') and response_text.endswith('}'):
                return eval(response_text)
            return {}
    except Exception as e:
        st.warning(f"Error parsing Claude's response: {e}")
        return {}

# Function to process multiple texts concurrently
async def process_texts_concurrently(texts: List[str], api_key: str) -> List[Dict]:
    client = anthropic.AsyncAnthropic(api_key=api_key)
    
    # Create tasks for concurrent processing
    tasks = [process_text_with_claude(text, client) for text in texts]
    
    # Run all tasks concurrently and return results
    return await asyncio.gather(*tasks)

# Function to process uploaded text files using Haystack
def process_text_files(text_files) -> Dict[str, str]:
    document_store = InMemoryDocumentStore()
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=200,
        split_overlap=20
    )
    
    file_contents = {}
    
    for text_file in text_files:
        file_contents[text_file.name] = text_file.getvalue().decode("utf-8")
    
    return file_contents

# Function to merge extracted entities across multiple texts
def merge_entities(entities_list: List[Dict]) -> Dict:
    merged = {}
    for entities in entities_list:
        for entity, quantity in entities.items():
            if entity in merged and merged[entity] is not None and quantity is not None:
                merged[entity] = merged.get(entity, 0) + quantity
            else:
                # If either is None, use the non-None value
                merged[entity] = quantity if merged.get(entity) is None else merged[entity]
    return merged

# Main application
def main():
    st.title("Voice Notes to Spreadsheet Processor")
    st.markdown("Upload transcribed voice notes and a spreadsheet to extract flower/seed references")
    
    # Step 1: File Upload Section
    st.header("Step 1: Upload Files")
    col1, col2 = st.columns(2)
    
    with col1:
        voice_note_files = st.file_uploader(
            "Upload up to 10 transcribed voice notes (text files)",
            type=["txt"], 
            accept_multiple_files=True,
            key="voice_notes"
        )
        
        if voice_note_files and len(voice_note_files) > 10:
            st.warning("Maximum 10 files allowed. Only the first 10 will be processed.")
            voice_note_files = voice_note_files[:10]
    
    with col2:
        spreadsheet_file = st.file_uploader(
            "Upload your spreadsheet with flower/seed data",
            type=["xlsx", "xls", "csv"],
            key="spreadsheet"
        )
    
    # Step 2: Process Voice Notes
    if voice_note_files and spreadsheet_file:
        st.header("Step 2: Review and Edit Transcriptions")
        
        # Process text files
        file_contents = process_text_files(voice_note_files)
        
        # Display editable text areas for each file
        edited_texts = {}
        for filename, content in file_contents.items():
            st.subheader(f"Edit text from: {filename}")
            edited_text = st.text_area(
                f"Edit text for {filename}",
                value=content,
                height=150,
                key=f"edit_{filename}"
            )
            edited_texts[filename] = edited_text
        
        st.session_state.edited_texts = edited_texts
        
        # Process spreadsheet
        try:
            if spreadsheet_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(spreadsheet_file)
            else:  # CSV
                df = pd.read_csv(spreadsheet_file)
            
            st.session_state.spreadsheet_data = df
            st.session_state.spreadsheet_name = spreadsheet_file.name
        except Exception as e:
            st.error(f"Error reading spreadsheet: {e}")
            return
        
        # Button to process edited texts
        if st.button("Extract Flower and Seed References"):
            with st.spinner("Processing voice notes concurrently..."):
                try:
                    # Get API key from secrets or environment
                    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))
                    if not api_key:
                        st.error("Anthropic API key not found. Please set it in Streamlit secrets or environment variables.")
                        return
                    
                    # Get list of texts to process
                    texts_to_process = list(st.session_state.edited_texts.values())
                    
                    # Process concurrently using asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    entity_results = loop.run_until_complete(
                        process_texts_concurrently(texts_to_process, api_key)
                    )
                    loop.close()
                    
                    # Merge entities from all files
                    all_entities = merge_entities(entity_results)
                    
                    st.session_state.extracted_entities = all_entities
                    
                    # Show success message with stats
                    st.success(f"Successfully processed {len(texts_to_process)} files and extracted {len(all_entities)} flower/seed references.")
                    
                except Exception as e:
                    st.error(f"Error processing voice notes: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    # Step 3: Review Extracted Entities and Update Spreadsheet
    if hasattr(st.session_state, 'extracted_entities') and st.session_state.extracted_entities:
        st.header("Step 3: Review Extracted Flower and Seed References")
        
        entities = st.session_state.extracted_entities
        
        # Display extracted entities in an editable table
        st.subheader("Extracted Flower/Seed References")
        
        # Convert to dataframe for easier editing
        entities_df = pd.DataFrame([
            {"Item": k, "Quantity": v} for k, v in entities.items()
        ])
        
        # Make dataframe editable
        edited_entities = st.data_editor(
            entities_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Item": st.column_config.TextColumn("Flower/Seed Name"),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=0)
            }
        )
        
        # Convert back to dictionary
        st.session_state.final_entities = {row["Item"]: row["Quantity"] for _, row in edited_entities.iterrows()}
        
        # Display the spreadsheet
        if hasattr(st.session_state, 'spreadsheet_data'):
            st.subheader("Current Spreadsheet Data")
            st.dataframe(st.session_state.spreadsheet_data)
            
            # Update spreadsheet with extracted values
            if st.button("Update Spreadsheet with Extracted Values"):
                df = st.session_state.spreadsheet_data.copy()
                
                # Create a new row for the updated data
                new_row = {}
                
                # Normalize column names to make matching easier
                normalized_cols = {col.lower().strip(): col for col in df.columns}
                
                # Match extracted entities with spreadsheet columns
                matched_entities = []
                unmatched_entities = []
                
                for entity, quantity in st.session_state.final_entities.items():
                    found = False
                    # Try to find an exact column match
                    for norm_col, actual_col in normalized_cols.items():
                        if entity.lower() in norm_col or norm_col in entity.lower():
                            new_row[actual_col] = quantity
                            matched_entities.append((entity, actual_col, quantity))
                            found = True
                            break
                    
                    if not found:
                        unmatched_entities.append(entity)
                        # Add as a new column
                        df[entity] = 0
                        new_row[entity] = quantity
                
                # Fill in any remaining columns with NaN
                for col in df.columns:
                    if col not in new_row:
                        new_row[col] = None
                
                # Add new row to dataframe
                updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
                # Display matched and unmatched entities
                if matched_entities:
                    st.write("**Matched Entities:**")
                    for entity, column, quantity in matched_entities:
                        st.write(f"✓ '{entity}' matched to column '{column}' with quantity {quantity}")
                
                if unmatched_entities:
                    st.write("**New Columns Added:**")
                    for entity in unmatched_entities:
                        st.write(f"➕ Added new column '{entity}'")
                
                # Display updated spreadsheet
                st.subheader("Updated Spreadsheet")
                st.dataframe(updated_df)
                
                # Allow downloading the updated spreadsheet
                st.session_state.final_df = updated_df
                
                # Create a download button
                if hasattr(st.session_state, 'spreadsheet_name'):
                    output_name = "updated_" + st.session_state.spreadsheet_name
                    
                    # Save to a temporary file for download
                    with tempfile.NamedTemporaryFile(delete=False, suffix=output_name) as tmp:
                        if output_name.endswith(('.xlsx', '.xls')):
                            updated_df.to_excel(tmp.name, index=False)
                        else:  # CSV
                            updated_df.to_csv(tmp.name, index=False)
                        
                        # Read the saved file as bytes
                        with open(tmp.name, 'rb') as f:
                            file_bytes = f.read()
                    
                    # Create download button
                    st.download_button(
                        label="Download Updated Spreadsheet",
                        data=file_bytes,
                        file_name=output_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" 
                        if output_name.endswith(('.xlsx', '.xls')) else "text/csv"
                    )

# Run the application
if __name__ == "__main__":
    main()