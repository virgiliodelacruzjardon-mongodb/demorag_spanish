# Slide — RAG con MongoDB Atlas Vector Search y Ollama

## Idea principal

RAG combina la recuperación de información con la generación de texto:

```text
Pregunta del usuario
        ↓
Embedding compatible con los datos
        ↓
MongoDB Atlas Vector Search
        ↓
Películas relevantes como contexto
        ↓
Prompt + contexto
        ↓
Ollama local
        ↓
Respuesta fundamentada
```

El modelo local no necesita memorizar toda la información. La aplicación busca primero los documentos relevantes en MongoDB Atlas y después se los entrega a Ollama para que redacte una respuesta basada en ese contexto.

## Qué demuestra el demo

* La diferencia entre una respuesta sin contexto y una respuesta con RAG.
* Cómo Atlas Vector Search recupera películas semánticamente relacionadas.
* Dónde aparece el contexto dentro del prompt.
* Cómo Ollama genera la respuesta localmente.
* Cómo mostrar las fuentes recuperadas para mejorar la trazabilidad.

## Nota técnica

El dataset de ejemplo `sample_mflix.embedded_movies` ya contiene embeddings de Voyage AI en el campo `plot_embedding_voyage_3_large`. Por eso, este demo utiliza Voyage AI para el embedding de la consulta y Ollama para la generación local. Para hacer también los embeddings completamente locales, habría que re-embebir los documentos y crear un índice compatible nuevo.

