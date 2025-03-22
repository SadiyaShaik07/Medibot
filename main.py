import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API details
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

def process_image(image_path, query):
    """
    Processes an image by encoding it and querying the AI model for an explanation and medication suggestions.

    Args:
        image_path (str): Path to the image file.
        query (str): User query about the image.

    Returns:
        dict: Response containing explanation and medication suggestions.
    """
    try:
        # Read the image
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()

        if not image_content:
            raise ValueError("The provided image file is empty.")

        # Encode the image
        encoded_image = base64.b64encode(image_content).decode("utf-8")

        # Validate image format
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            return {"error": f"Invalid image format: {str(e)}"}

        def make_api_request(model, modified_query):
            """
            Sends a request to the API with the given model and query.

            Args:
                model (str): Model name.
                modified_query (str): Modified query for the AI.

            Returns:
                Response object from the API request.
            """
            try:
                response = requests.post(
                    GROQ_API_URL,
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": modified_query},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                                ]
                            }
                        ],
                        "max_tokens": 1000
                    },
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    timeout=30
                )
                response.raise_for_status()
                return response
            except requests.RequestException as req_error:
                logger.error(f"API request failed: {str(req_error)}")
                return None

        # Request explanation and medications
        explanation_response = make_api_request("llama-3.2-11b-vision-preview", f"Explain the issue shown in this image based on this query: {query}")
        medication_response = make_api_request("llama-3.2-90b-vision-preview", f"Suggest medications for the issue related to this image and query: {query}")

        responses = {}

        for category, response in [("explanation", explanation_response), ("medications", medication_response)]:
            if response and response.status_code == 200:
                try:
                    result = response.json()
                    answer = result["choices"][0]["message"]["content"]
                    logger.info(f"Processed {category}: {answer[:100]}...")
                    responses[category] = answer
                except (KeyError, IndexError, ValueError) as parse_error:
                    logger.error(f"Error parsing {category} response: {str(parse_error)}")
                    responses[category] = "Error parsing API response."
            else:
                error_message = f"Error fetching {category}: {response.status_code if response else 'No Response'}"
                logger.error(error_message)
                responses[category] = error_message

        return responses

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    image_path = "pic1.jpg"
    query = "What condition is shown in this image?"
    result = process_image(image_path, query)
    print(result)
