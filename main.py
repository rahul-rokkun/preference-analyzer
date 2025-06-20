from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize Gemini API
client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
app = FastAPI()

# Request model


class ProductAnalysisRequest(BaseModel):
    recommended_products: List[str]
    clicked_products: List[str]
    user_inputs: List[str]


class ResponseModel(BaseModel):
    category: str
    analysis: str


# Endpoint
@app.post("/analyze")
async def analyze_products(data: ProductAnalysisRequest):
    prompt = f"""
You are a product analysis AI assistant.

Input:
- Recommended Products: {data.recommended_products}
- Clicked Products: {data.clicked_products}
- User Queries: {data.user_inputs}

Tasks:
1. From the recommended and clicked product names, identify the most common product category(dress, pants, tops/blouses ,jackets/blazers).
2. From the user inputs, identify:
   - Common themes or patterns
   - Any weird, outlier, or unexpected queries
   - The general vibe (e.g., are people excited, confused, nostalgic, etc.)

Provide your answer as a structured Json output.
    category: <category>
    analysis: <analysis>

Keep the response concise with only the most relevant information within 15 - 20 words.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ResponseModel,
            },
        )
        return {
            "category": response.parsed.category,
            "analysis": response.parsed.analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
