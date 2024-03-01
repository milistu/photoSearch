from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(location="localhost", port="6333")

if __name__ == "__main__":
    status = False
    try:
        status = client.create_collection(
            collection_name="test_collection",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
    except Exception as e:
        logger.error(f"Error occured: {e}")

    logger.info(f"Collection creation status: {status}")
