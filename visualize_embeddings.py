from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import vertexai
from vertexai.language_models import TextEmbeddingModel
import numpy as np
# PCA : Principal Component Analysis
# Yüksek boyutlu veri setlerini daha düşük boyutlu bir uzaya dönüştürmek için kulla-
# nılan istatistiksel bir yöntemdir.
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
text_4 = "The sun sets slowly behind the mountain every evening."
text_5 = "The sentence uses 'machine' to refer to a physical device (a factory machine) and 'learning' to describe the process of gaining knowledge or skill."

text_list = [text_1, text_2, text_3, text_4, text_5]

embeddings = []

for text in text_list:
    emb_vector = embedding_model.get_embeddings([text])[0].values
    embeddings.append(emb_vector)

embeddings_array = np.array(embeddings)

# print(embeddings_array.shape) : output = (5, 768)

PCA_model = PCA(n_components=2) # 2D dimension
PCA_model.fit(embeddings_array)

new_embeddings = PCA_model.transform(embeddings_array)

# print(new_embeddings.shape) (5, 2)

# print(new_embeddings)

# create a scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(new_embeddings[:, 0], new_embeddings[:, 1])
# create labels and title
ax.set_title("Embedding visualization in 2D")
ax.set_xlabel("X_1")
ax.set_ylabel("Y_1")
plt.show()