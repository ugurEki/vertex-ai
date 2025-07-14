from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import vertexai
from vertexai.language_models import TextEmbeddingModel

api_key_path = "/Users/ugurekinci/Downloads/vertex-ai-465718-b7ad113c10f1.json"

credentials = Credentials.from_service_account_file(
    api_key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

PROJECT_ID = "vertex-ai-465718"
REGION="us-central1"

vertexai.init(
    project = PROJECT_ID,
    location = REGION,
    credentials = credentials
)

# Embeding model is text-embedding-005.
embedding_model = TextEmbeddingModel.from_pretrained(
    "text-embedding-005"
)

# Visualize the embeddings in 2D space

text_1 = "Machine learning is a powerful tool for data science."
text_2 = "Data science relies heavily on machine learning techniques."
text_3 = "The learning process of this machine is very complex."
