from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up template rendering with Jinja2
templates = Jinja2Templates(directory="templates")

# API details for external service
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Ensure API key is available
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

# Store chat history
chat_history = []

# Home route to render the chat page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})

# Endpoint to handle image uploads and queries
@app.post("/upload_and_query")
async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
    try:
        # Read uploaded image file
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Encode image to Base64 format
        encoded_image = base64.b64encode(image_content).decode("utf-8")

        # Validate image format
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        # Function to make API request to external AI model
        def make_api_request(model, modified_query):
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
            return response

        # Get medical explanation from AI model
        explanation_response = make_api_request("llama-3.2-11b-vision-preview", f"Explain the issue shown in this image based on this query: {query}")
        # Get medication suggestions from AI model
        medication_response = make_api_request("llama-3.2-90b-vision-preview", f"Suggest medications for the issue related to this image and query: {query}")

        responses = {}

        # Process responses from AI model
        for category, response in [("explanation", explanation_response), ("medications", medication_response)]:
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                logger.info(f"Processed {category}: {answer[:100]}...")
                responses[category] = answer
            else:
                logger.error(f"Error fetching {category}: {response.status_code} - {response.text}")
                responses[category] = f"Error fetching {category}: {response.status_code}"

        # Store query and responses in chat history
        chat_history.append({
            "query": query,
            "explanation": responses.get("explanation", "No response"),
            "medications": responses.get("medications", "No response")
        })

        # Return results as JSON response
        return JSONResponse(status_code=200, content=responses)

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Run FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
