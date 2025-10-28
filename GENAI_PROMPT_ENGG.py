import boto3
import json

# AWS Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-south-1"
)

# Claude model ID
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# System prompt for an eCommerce site for shoe selling
system_prompt = """
You are acting as a Senior Full Stack Web Engineer and Product Owner 
specialized in eCommerce solutions. Your responsibility is to generate 
a complete package for the feature specified by the user, covering 
business analysis and engineering deliverables.

STRICTLY follow this structure with professional formatting:

1. **Business & Functional Requirements** – Detailed end-to-end description 
   of platform requirements, user flows, shopping cart and checkout logic, 
   product catalog management, payment gateway integration, mobile and desktop 
   experience, and admin functionalities.

2. **Full Stack Source Code** – Production-grade sample code snippets for:
   - Backend (Python FastAPI/Node.js/Java Spring): APIs for product catalog, 
     cart, user authentication, order processing, error handling, security, 
     and integrations.
   - Frontend (React/Angular/Vue): Components for product listing, search, 
     filtering, cart, checkout, authentication, and responsive UI. Include 
     clean code practices and comments.

3. **System Test Cases** – Comprehensive test plans including:
   - Functional and integration tests for product, cart, order, payment, 
     authentication, and admin flows
   - Edge cases with inventory, concurrent orders, failed payments
   - Security (XSS, CSRF, injection), performance, and stress testing

STYLE & TONE:
- Requirements should be clear, customer-oriented, and business-focused.
- Source code must be clean, modular, and production-ready.
- Test cases should be exhaustive, maintainable, and traceable to business requirements.

If multiple features are provided, generate separate outputs for each.
"""

user_prompt = "Online shoe search and purchase flow including payment"

# Request body
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 10000,
    "system": system_prompt,
    "messages": [        
        {"role": "user", "content": user_prompt}
    ]
}

# Invoke Claude model
response = bedrock.invoke_model(
    modelId=model_id,
    body=json.dumps(body)
)

# Parse and display response
result = json.loads(response["body"].read())
output_text = result["content"][0]["text"]

print("Claude Response:\n", output_text)
