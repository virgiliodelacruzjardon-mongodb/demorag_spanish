"""Demo RAG con MongoDB Atlas Vector Search y Ollama local.

Retrieval:
    Voyage AI -> embedding de consulta
    MongoDB Atlas Vector Search -> documentos relevantes

Generation:
    Ollama local -> respuesta usando el contexto recuperado
"""

import os
import time
from typing import Any

import ollama
import voyageai
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel


load_dotenv()

MONGODB_URI = os.environ["MONGODB_URI"]
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]

DB_NAME = os.getenv("MONGODB_DATABASE", "sample_mflix")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "embedded_movies")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "movie_plot_vector_index")
VECTOR_FIELD = os.getenv(
    "VECTOR_FIELD", "plot_embedding_voyage_3_large"
)
VECTOR_DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS", "2048"))
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3-large")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

mongo_client = MongoClient(MONGODB_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]
voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)


def ensure_vector_index() -> None:
    """Crea el índice vectorial si no existe."""

    indexes = list(collection.list_search_indexes())

    if any(index.get("name") == VECTOR_INDEX_NAME for index in indexes):
        print(f"Índice encontrado: {VECTOR_INDEX_NAME}")
        return

    index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": VECTOR_FIELD,
                    "numDimensions": VECTOR_DIMENSIONS,
                    "similarity": "dotProduct",
                }
            ]
        },
        name=VECTOR_INDEX_NAME,
        type="vectorSearch",
    )

    collection.create_search_index(model=index_model)
    print(f"Creando índice {VECTOR_INDEX_NAME}...")

    for _ in range(36):
        indexes = list(collection.list_search_indexes())
        current = next(
            (
                index
                for index in indexes
                if index.get("name") == VECTOR_INDEX_NAME
            ),
            None,
        )

        if current and current.get("queryable", True):
            print("Índice listo.")
            return

        time.sleep(5)

    raise TimeoutError(
        "El índice no estuvo listo. Revisa su estado en Atlas."
    )


def create_query_embedding(question: str) -> list[float]:
    """Convierte la pregunta en un vector compatible con el índice."""

    result = voyage_client.embed(
        [question],
        model=VOYAGE_MODEL,
        output_dimension=VECTOR_DIMENSIONS,
        input_type="query",
    )

    return result.embeddings[0]


def retrieve_movies(question: str, limit: int = 5) -> list[dict[str, Any]]:
    """Recupera películas semánticamente similares."""

    query_vector = create_query_embedding(question)

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": VECTOR_FIELD,
                "queryVector": query_vector,
                "numCandidates": max(limit * 20, 100),
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "title": 1,
                "plot": 1,
                "genres": 1,
                "year": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    return list(collection.aggregate(pipeline))


def build_context(movies: list[dict[str, Any]]) -> str:
    """Convierte los documentos recuperados en contexto legible para Ollama."""

    return "\n\n".join(
        [
            (
                f"[{index}] {movie.get('title', 'Sin título')}\n"
                f"Argumento: {movie.get('plot', 'Sin descripción')}\n"
                f"Géneros: {movie.get('genres', [])}"
            )
            for index, movie in enumerate(movies, start=1)
        ]
    )


def ask_ollama(prompt: str) -> str:
    """Envía un prompt al modelo local de Ollama."""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Responde de forma clara, breve y honesta.",
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.2},
    )

    return response["message"]["content"]


def answer_without_rag(question: str) -> str:
    """Respuesta de control: Ollama no recibe contexto de Atlas."""

    prompt = f"""
Responde esta pregunta sobre películas:

{question}

No tienes acceso a una base de datos externa.
Si no conoces la respuesta, dilo claramente.
"""

    return ask_ollama(prompt)


def answer_with_rag(question: str, movies: list[dict[str, Any]]) -> str:
    """Respuesta RAG: Ollama recibe los documentos recuperados como contexto."""

    context = build_context(movies)

    prompt = f"""
Responde la pregunta usando únicamente el contexto recuperado.

Pregunta:
{question}

Contexto recuperado desde MongoDB Atlas:
{context}

Reglas:
- No inventes información.
- Si el contexto no es suficiente, dilo claramente.
- Sé breve y útil.
- Incluye referencias como [1], [2] según corresponda.
"""

    return ask_ollama(prompt)


def print_retrieved_context(movies: list[dict[str, Any]]) -> None:
    """Imprime el contexto para hacer visible la parte retrieval del RAG."""

    print("\n=== CONTEXTO RECUPERADO ===")

    if not movies:
        print("No se recuperaron documentos.")
        return

    for index, movie in enumerate(movies, start=1):
        score = movie.get("score", 0.0)
        print(f"\n[{index}] {movie.get('title', 'Sin título')} — score: {score:.4f}")
        print(movie.get("plot", "Sin argumento"))


def main() -> None:
    print("Demo RAG: MongoDB Atlas Vector Search + Ollama")
    print(f"Modelo local: {OLLAMA_MODEL}")

    ensure_vector_index()

    question = input(
        "\nPregunta sobre películas "
        "(ejemplo: películas sobre viajes en el tiempo):\n> "
    ).strip()

    movies = retrieve_movies(question)
    print_retrieved_context(movies)

    print("\n=== RESPUESTA SIN RAG ===")
    print(answer_without_rag(question))

    print("\n=== RESPUESTA CON RAG ===")
    if movies:
        print(answer_with_rag(question, movies))
    else:
        print("No es posible generar una respuesta fundamentada sin contexto.")


if __name__ == "__main__":
    main()
