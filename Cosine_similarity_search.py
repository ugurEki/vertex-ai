from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import vertexai
# cosine_similarity fonksiyonunu iki veya daha fazla vektör arasındaki benzerliği -
# ölçmek için kullanacağım. 
# Kısaca cosine similarity iki vektör arasındaki açısal benzerliği hesaplayan bir metriktir.
from sklearn.metrics.pairwise import cosine_similarity
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


# Embeddings for similarity search.
embedding_1 = embedding_model.get_embeddings([
    "Machine learning is a powerful tool for data science."
])

embedding_2 = embedding_model.get_embeddings([
    "Data science relies heavily on machine learning techniques"
])

embedding_3 = embedding_model.get_embeddings([
    "The learning process of this machine is very complex."
])

# Extracting vectors from the embeddings.
vector_embedding_1 = [embedding_1[0].values]
vector_embedding_2 = [embedding_2[0].values]
vector_embedding_3 = [embedding_3[0].values]

# Cosine similarity
print(cosine_similarity(vector_embedding_1, vector_embedding_2))
print(cosine_similarity(vector_embedding_1, vector_embedding_3))
print(cosine_similarity(vector_embedding_2, vector_embedding_3))


