from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import vertexai

api_key_path = "/Users/ugurekinci/Downloads/vertex-ai-465718-b7ad113c10f1.json"

credentials = Credentials.from_service_account_file(
    api_key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

PROJECT_ID = "vertex-ai-465718"
REGION="us-central1"

help(vertexai.init)

# GCP ye ait olan Vertex AI platformunu kullanmak için Python ortamında bir başlangıç -
# noktası oluşturur.
# Bu metod, Vertex AI hizmetleriyle çalışmaya başlamadan önce gerekli kimlik doğrulama-
# ve yapılandırma ayarlarını yapar.
vertexai.init(
    project = PROJECT_ID,
    location = REGION,
    credentials = credentials
)

