# Search your photos with text 🔎

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