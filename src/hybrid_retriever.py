from rank_bm25 import BM25Okapi


# --------------------------------
# Tokenization
# --------------------------------

def tokenize(text):
    return text.lower().split()


# --------------------------------
# Create BM25
# --------------------------------

def create_bm25(chunks):

    tokenized_chunks = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


# --------------------------------
# Min-Max Normalization
# --------------------------------

def min_max_normalize(scores):

    if len(scores) == 0:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0 for _ in scores]

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


# --------------------------------
# Hybrid Search
# --------------------------------

def hybrid_search(
    question,
    vector_store,
    chunks,
    k=5,
    semantic_weight=0.6,
    bm25_weight=0.4
):

    # --------------------------------
    # Create BM25 index
    # --------------------------------

    bm25 = create_bm25(chunks)

    # --------------------------------
    # Semantic Search
    # --------------------------------

    semantic_results = vector_store.similarity_search_with_score(
        question,
        k=len(chunks)
    )

    # --------------------------------
    # Convert semantic results
    # to chunk-index based scores
    # --------------------------------

    semantic_distances = []

    for document, distance in semantic_results:

        # Find matching chunk
        matching_index = None

        for i, chunk in enumerate(chunks):

            if (
                chunk.page_content
                == document.page_content
            ):
                matching_index = i
                break

        if matching_index is not None:

            semantic_distances.append(
                (matching_index, distance)
            )

    # --------------------------------
    # Convert distance to similarity
    #
    # Smaller distance = better
    # --------------------------------

    semantic_similarity = [0.0] * len(chunks)

    if semantic_distances:

        distances = [
            distance
            for index, distance in semantic_distances
        ]

        # Convert distance to similarity
        similarities = [
            1 / (1 + distance)
            for distance in distances
        ]

        # Normalize similarity scores
        normalized_similarities = min_max_normalize(
            similarities
        )

        for (
            (index, distance),
            similarity
        ) in zip(
            semantic_distances,
            normalized_similarities
        ):

            semantic_similarity[index] = similarity

    # --------------------------------
    # BM25 Search
    # --------------------------------

    query_tokens = tokenize(question)

    bm25_raw_scores = bm25.get_scores(
        query_tokens
    )

    # Normalize BM25 scores
    bm25_normalized = min_max_normalize(
        bm25_raw_scores
    )

    # --------------------------------
    # Combine Scores
    # --------------------------------

    combined_scores = []

    for i, document in enumerate(chunks):

        semantic_score = semantic_similarity[i]

        bm25_score = bm25_normalized[i]

        hybrid_score = (
            semantic_weight * semantic_score
            +
            bm25_weight * bm25_score
        )

        combined_scores.append(
            (
                hybrid_score,
                document
            )
        )

    # --------------------------------
    # Sort
    # --------------------------------

    combined_scores.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------
    # Return Top K
    # --------------------------------

    return [
        document
        for score, document
        in combined_scores[:k]
    ]