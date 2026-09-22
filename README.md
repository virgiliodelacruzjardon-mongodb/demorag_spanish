# Demo RAG con MongoDB Atlas Vector Search y Ollama

Este demo muestra la diferencia entre una respuesta de Ollama sin contexto y una respuesta RAG. En la versión RAG, MongoDB Atlas recupera películas relevantes y esos documentos se insertan en el prompt que recibe Ollama.

## Arquitectura

```text
Pregunta del usuario
        ↓
Embedding de la consulta con Voyage AI
        ↓
MongoDB Atlas Vector Search
        ↓
Contexto recuperado
        ↓
Prompt + contexto
        ↓
Ollama local
        ↓
Respuesta fundamentada
```

## Importante sobre el alcance local

La generación de texto es local porque la realiza Ollama. El embedding de la consulta se genera con Voyage AI para que sea compatible con los embeddings que ya existen en `sample_mflix.embedded_movies`.

Para que también los embeddings sean locales, habría que generar embeddings nuevos para los documentos y guardarlos en otro campo o colección, y después crear un índice vectorial con las dimensiones correspondientes.

## Requisitos

1. Tener un cluster de MongoDB Atlas.
2. Cargar el dataset de ejemplo `sample_mflix`.
3. Tener disponible la colección `sample_mflix.embedded_movies`.
4. Instalar Ollama desde [ollama.com](https://ollama.com/).
5. Tener una API key de Voyage AI para generar el embedding de la pregunta.
 
 
La documentación de MongoDB utiliza `sample_mflix.embedded_movies`, el campo `plot_embedding_voyage_3_large`, el modelo `voyage-3-large` y 2048 dimensiones para este flujo. [MongoDB Docs — Perform Hybrid Search](https://www.mongodb.com/docs/atlas/ai-integrations/langchain/hybrid-search/)

## Instalación

```bash
python3 -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
ollama pull llama3.2
```

Copia `.env.example` como `.env` y completa:
cp .env.example .env

```env
MONGODB_URI=mongodb+srv://<usuario>:<password>@<cluster>/
VOYAGE_API_KEY=<tu_api_key>
```

## Ejecución

```bash
python rag_ollama_atlas_movies.py

## Ejemplos de preguntas
“¿Qué película trata sobre un accidente que le da a un niño un brazo extraordinario para lanzar?”

¿Qué película trata sobre un jugador estrella de béisbol que se convierte en la obsesión de un vendedor?

¿Qué película trata sobre un granjero de Iowa que construye un campo de béisbol después de escuchar voces?


El programa imprime cuatro cosas importantes:

1. Los documentos recuperados por Atlas Vector Search.
2. Sus puntuaciones de similitud.
3. Una respuesta de Ollama sin RAG.
4. Una respuesta de Ollama con los documentos recuperados como contexto.

## Dónde está el RAG

La parte de retrieval está en `retrieve_movies()`:

```python
movies = retrieve_movies(question)
```

La parte de contexto está en `build_context()`:

```python
context = build_context(movies)
```

La parte que conecta retrieval con generación está en `answer_with_rag()`:

```python
Contexto recuperado desde MongoDB Atlas:
{context}
```

Después, ese prompt se envía a Ollama. Esa conexión es la mejora principal: el modelo genera la respuesta usando documentos recuperados en tiempo de consulta, no solo su conocimiento general.

## Índice vectorial

El script intenta crear automáticamente el índice `movie_plot_vector_index`. También puedes crearlo en Atlas usando el contenido de `vector_index_definition.json`.

El campo vectorial y la configuración deben coincidir con el embedding usado para la consulta.
