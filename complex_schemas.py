import os
from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field # pip install pydantic
from groq import Groq
import instructor # pip install instructor

# Set up the client with instructor
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
instructor_client = instructor.patch(client)

# Define a complex nested schema
class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    country: str

class ContactInfo(BaseModel):
    email: str
    phone: Optional[str] = None
    address: Address

class ProductVariant(BaseModel):
    id: str
    name: str
    price: float
    inventory_count: int
    attributes: Dict[str, str]

class ProductReview(BaseModel):
    user_id: str
    rating: float = Field(ge=1, le=5)
    comment: str
    date: str

class Product(BaseModel):
    id: str
    name: str
    description: str
    main_category: str
    subcategories: List[str]
    variants: List[ProductVariant]
    reviews: List[ProductReview]
    average_rating: float = Field(ge=1, le=5)
    manufacturer: Dict[str, Union[str, ContactInfo]]

# System prompt with clear instructions about the complex structure
system_prompt = """
You are a product catalog API. Generate a detailed product with ALL required fields.
Your response must be a valid JSON object matching the following schema:

{
  "id": "string",
  "name": "string",
  "description": "string",
  "main_category": "string",
  "subcategories": ["string"],
  "variants": [
    {
      "id": "string",
      "name": "string",
      "price": number,
      "inventory_count": number,
      "attributes": {"key": "value"}
    }
  ],
  "reviews": [
    {
      "user_id": "string",
      "rating": number (1-5),
      "comment": "string",
      "date": "string (YYYY-MM-DD)"
    }
  ],
  "average_rating": number (1-5),
  "manufacturer": {
    "name": "string",
    "founded": "string",
    "contact_info": {
      "email": "string",
      "phone": "string (optional)",
      "address": {
        "street": "string",
        "city": "string", 
        "state": "string",
        "zip_code": "string",
        "country": "string"
      }
    }
  }
}
"""

# Use instructor to create and validate in one step
product = instructor_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    response_model=Product,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Give me details about a high-end camera product"}
    ],
    max_retries=3
)

# Print the validated complex object
print(f"Product: {product.name}")
print(f"Description: {product.description[:100]}...")
print(f"Variants: {len(product.variants)}")
print(f"Reviews: {len(product.reviews)}")
print(f"Manufacturer: {product.manufacturer.get('name')}")
print("\nManufacturer Contact:")
contact_info = product.manufacturer.get('contact_info')
if isinstance(contact_info, ContactInfo):
    print(f"  Email: {contact_info.email}")
    print(f"  Address: {contact_info.address.city}, {contact_info.address.country}") 