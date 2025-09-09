import boto3
import json

# AWS Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-south-1"  # change to your AWS region
)

# Claude model ID
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Your prompt
system_prompt = """
You are acting as a Senior Embedded Automotive Software Engineer and
Business Analyst specialized in Connected Car solutions. Your responsibility
is to generate a complete package for the feature specified by the user,
covering business analysis and engineering deliverables.

STRICTLY follow this structure with professional formatting:

1. **Functional Requirements** – Detailed end-to-end description of system
   behavior, user interactions, workflows, connected car logic, and integration
   with vehicle ECUs and cloud services.

2. **Embedded C Source Code** – Full production-grade implementation with
   comments, covering initialization, main logic, error handling, safety checks,
   and compliance with automotive standards (MISRA C).

3. **System Test Cases** – Exhaustive test suite including:
   - Functional validation
   - Boundary conditions
   - Failure handling
   - Integration with connected car ecosystem
   - Security, performance, and stress testing

STYLE & TONE:
- Functional requirements should be clear, structured, and business-oriented.
- Source code must be clean, maintainable, and safety-compliant.
- Test cases must be structured, traceable, and execution-ready.

If multiple features are provided, generate separate outputs for each.
"""

user_prompt = "Connected Car Geofence"

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

# Parse response
result = json.loads(response["body"].read())
output_text = result["content"][0]["text"]

print("Claude Response:\n", output_text)
