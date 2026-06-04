import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="replicate",
    api_key=os.environ["HF_TOKEN"],
)

# output is a PIL.Image object
image = client.text_to_image(
    "A serene landscape with mountains and a lake",
    model="black-forest-labs/FLUX.1-dev",
)