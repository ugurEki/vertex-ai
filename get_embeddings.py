## Google Cloud API'lerine güvenli ve kimlik doğrulamalı erişim sağlamak için.

# Kimlik doğrulama sırasında HTTP istekleri göndermek için kullanılır.
from google.auth.transport.requests import Request 
# Google Cloud servislerine bağlanmak için Service Account kullanarak kimlik doğrulaması yapmayı sağlar.
# Service Account JSON dosyasını kullanarak kimlik oluşturur. Bu kimlik Google API'lerine yetkili erişim sağlar.
from google.oauth2.service_account import Credentials 

# Path to the Json key 
key_path = "/Users/ugurekinci/Downloads/vertex-ai-465718-b7ad113c10f1.json"

# Create credentials object
credentials = Credentials.from_service_account_file(
    key_path)

if credentials.expired:
    credentials.refresh(Request())

    
PROJECT_ID = "vertex-ai-465718"
REGION="us-central1"

# --------------------------------------------------------
import vertexai
from vertexai.language_models import TextEmbeddingModel

# initialize vertex
vertexai.init(
    project=PROJECT_ID,
    location=REGION,
    credentials=credentials)

embedding_model = TextEmbeddingModel.from_pretrained(
    "text-embedding-005"
)

embedding = embedding_model.get_embeddings(["Hello"])

# print(embedding)

# print(type(embedding))

# print(len(embedding))

# print(help(embedding[0]))

print(len(embedding[0].values))
