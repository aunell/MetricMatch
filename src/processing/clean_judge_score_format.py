import json

def clean_evaluation_fields(input_file, output_file=None):
    """
    Clean JSON file by converting escaped JSON strings in 'evaluation' fields to proper JSON objects.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file (optional, defaults to overwriting input)
    """
    try:
        # Read the JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each detailed_results entry
        if 'detailed_results' in data:
            for result in data['detailed_results']:
                if 'evaluation' in result:
                    # Check if evaluation is a string (escaped JSON)
                    if isinstance(result['evaluation'], str):
                        try:
                            # Parse the escaped JSON string
                            result['evaluation'] = json.loads(result['evaluation'])
                            print(f"Cleaned evaluation for text_id: {result.get('text_id', 'unknown')}")
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse evaluation for text_id {result.get('text_id', 'unknown')}: {e}")
                            # Keep original if parsing fails
                            continue
                    else:
                        print(f"Evaluation for text_id {result.get('text_id', 'unknown')} is already a JSON object")
        
        # Write the cleaned data
        output_path = output_file if output_file else input_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Cleaned JSON saved to: {output_path}")
        return data
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}': {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None