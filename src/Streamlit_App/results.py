# results.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from a .env file
load_dotenv()

def get_sepsis_subtype_and_treatment(sepsis_type: str):  # Changed to accept sepsis_type
    """
    Retrieves a short description and basic treatment principles for a given sepsis type
    using the Gemini API.

    Args:
        sepsis_type (str): The name of the sepsis subtype (e.g., "Septic Shock").

    Returns:
        str: The API's response containing the description and treatment information,
             or an error message if the API call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "API error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('models/gemini-2.5-flash-preview-04-17')

    prompt_content = f"Provide a short description of '{sepsis_type}' sepsis and outline the basic treatment principles for a patient with this condition. Focus on key characteristics and standard initial management approaches."

    try:
        response = model.generate_content([
            {"role": "user", "parts": [prompt_content]}
        ])
        return response.text
    except Exception as e:
        return f"API error:\n{e}"

if __name__ == '__main__':

    sepsis_type_to_query = "Septic Shock" # Example Usage
    print(f"Querying information for: {sepsis_type_to_query}\n")
    description_and_treatment = get_sepsis_subtype_and_treatment(sepsis_type_to_query)
    print(description_and_treatment)

    print("-" * 10)

    sepsis_type_to_query_2 = "Hyperinflammatory Sepsis"  # Example Usage
    print(f"\nQuerying information for: {sepsis_type_to_query_2}\n")
    description_and_treatment_2 = get_sepsis_subtype_and_treatment(sepsis_type_to_query_2)
    print(description_and_treatment_2)

    print("-" * 10)

    sepsis_type_to_query_3 = "Resolved Sepsis"  # Example
    print(f"\nQuerying information for: {sepsis_type_to_query_3}\n")
    description_and_treatment_3 = get_sepsis_subtype_and_treatment(sepsis_type_to_query_3)
    print(description_and_treatment_3)


