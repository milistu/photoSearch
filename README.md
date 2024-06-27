# Search your photos with text 🔎

This project focuses on image search using the CLIP (Contrastive Language–Image Pretraining) model. The key concept behind the CLIP model is its ability to embed both images and text into the same multidimensional space, enabling the use of text to search for images. By mapping images and text to the same vector space, the model allows for efficient and accurate image retrieval based on textual descriptions.

## Poetry
Add this to your `pyproject.toml` if you want to use Poetry only for dependency management.
```bash
[tool.poetry]
package-mode = false
```

## Qdrant

[Quickstart](https://qdrant.tech/documentation/quick-start/)

Pull the Image
```bash
docker pull qdrant/qdrant
```

Run the service:
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
