import json
import config

import tiktoken
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_anthropic import ChatAnthropic
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import umap
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_fireworks import ChatFireworks
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from enum import Enum
# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import MilvusClient, DataType
import requests


def recursive_embed_cluster_summarize(texts: List[str], level: int = 1, n_levels: int = 3) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame, int]]:
    """
    Recursively embeds, clusters, and summarizes texts up to a specified level or until
    the number of unique clusters becomes 1, storing the results at each level.

    Parameters:
    - texts: List[str], texts to be processed.
    - level: int, current recursion level (starts at 1).
    - n_levels: int, maximum depth of recursion.

    Returns:
    - Dict[int, Tuple[pd.DataFrame, pd.DataFrame, int]], a dictionary where keys are the recursion
      levels and values are tuples containing the clusters DataFrame, summaries DataFrame at that level and number of token.
    """
    results = {}  # Dictionary to store results at each level

    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary, total_tokens = embed_cluster_summarize_texts(texts=texts, level=level)

    # Store the results of the current level
    results[level] = (df_clusters, df_summary, total_tokens)

    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        # Use summaries as the input texts for the next level of recursion
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(texts=new_texts, level=level + 1, n_levels=n_levels)

        # Merge the results from the next level into the current results dictionary
        results.update(next_level_results)

    return results

def embed_cluster_summarize_texts(texts: List[str], level: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
    clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
    the content within each cluster.

    Parameters:
    - texts: A list of text documents to be processed.
    - level: An integer parameter that could define the depth or detail of processing.

    Returns:
    - Tuple containing two DataFrames:
      1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
      2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
         and the cluster identifiers.
    """

    # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
    df_clusters = embed_cluster_texts(texts=texts)

    # Prepare to expand the DataFrame for easier manipulation of clusters
    expanded_list = []

    # Expand DataFrame entries to document-cluster pairings for straightforward processing
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append({"text": row["text"], "embd": row["embd"], "cluster": cluster})

    # Create a new DataFrame from the expanded list
    expanded_df = pd.DataFrame(expanded_list)

    # Retrieve unique cluster identifiers for processing
    all_clusters = expanded_df["cluster"].unique()

    print(f"--Generated {len(all_clusters)} clusters--")

    # Summarization
    template = """Give a detailed summary of the documentation provided.
Documentation:
{context}
    """

    # Setting additional parameters: temperature, max_tokens, top_p
    fireworks_gpt = ChatFireworks(
        model="accounts/fireworks/models/dbrx-instruct",
        temperature=0,
        max_tokens=1000,
        fireworks_api_key="If28EuWQZJF34b2FR8VG3N5WRdWj7QasZZgJ1TP58vvCfxks"
    )

    # Format text within each cluster for summarization
    summaries = []
    total_tokens = 0
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = fmt_txt(df_cluster)
        human_message = HumanMessage(content=template.format(context=formatted_txt))
        while True:
            try:
                msg = fireworks_gpt.invoke([human_message])
                break
            except Exception as ex:
                print(ex)
        total_tokens += msg.response_metadata['token_usage']['total_tokens']
        txt = msg.content
        summaries.append(txt)

    # Create a DataFrame to store summaries with their corresponding cluster and level
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary, total_tokens

def embed_cluster_texts(texts):
    """
    Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

    This function combines embedding generation and clustering into a single step. It assumes the existence
    of a previously defined `perform_clustering` function that performs clustering on the embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be processed.

    Returns:
    - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
    """
    text_embeddings_np = emb(sentences=texts)  # Generate embeddings
    cluster_labels = perform_clustering(text_embeddings_np, 10, 0.1)  # Perform clustering on the embeddings
    df = pd.DataFrame()  # Initialize a DataFrame to store the results
    df["text"] = texts  # Store original texts
    df["embd"] = list(text_embeddings_np)  # Store embeddings as a list in the DataFrame
    df["cluster"] = cluster_labels  # Store cluster labels
    return df

def emb(sentences: list[str]):
    # Gửi yêu cầu POST đến server localhost:8001 để embedding
    response = requests.post(config.url_embedding_sentence, json={"sentences": sentences})

    # Kiểm tra kết quả trả về
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        embeddings_np = np.array(embeddings)
        return embeddings_np
    else:
        print("Lỗi khi gửi yêu cầu embedding.")
        return None

def fmt_txt(df: pd.DataFrame) -> str:
    """
    Formats the text documents in a DataFrame into a single string.

    Parameters:
    - df: DataFrame containing the 'text' column with text documents to format.

    Returns:
    - A single string where all text documents are joined by a specific delimiter.
    """
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)

def perform_clustering(embeddings: np.ndarray, dim: int, threshold: float) -> List[np.ndarray]:
    """
    Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
    using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for UMAP reduction.
    - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

    Returns:
    - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
    """
    if len(embeddings) <= dim + 1:
        # Avoid clustering when there's insufficient data
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # Global clustering
    global_clusters, n_global_clusters = GMM_cluster(reduced_embeddings_global, threshold)

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        # Extract embeddings belonging to the current global cluster
        global_cluster_embeddings_ = embeddings[np.array([i in gc for gc in global_clusters])]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            # Handle small clusters with direct assignment
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Local dimensionality reduction and clustering
            reduced_embeddings_local = local_cluster_embeddings(global_cluster_embeddings_, dim)
            local_clusters, n_local_clusters = GMM_cluster(reduced_embeddings_local, threshold)

        # Assign local cluster IDs, adjusting for total clusters already processed
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[np.array([j in lc for lc in local_clusters])]
            indices = np.where((embeddings == local_cluster_embeddings_[:, None]).all(-1))[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)

        total_clusters += n_local_clusters

    return all_local_clusters

def global_cluster_embeddings(embeddings: np.ndarray, dim: int, n_neighbors: Optional[int] = None, metric: str = "cosine") -> np.ndarray:
    """
    Perform global dimensionality reduction on the embeddings using UMAP.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - n_neighbors: Optional; the number of neighbors to consider for each point.
                   If not provided, it defaults to the square root of the number of embeddings.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)

def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - threshold: The probability threshold for assigning an embedding to a cluster.
    - random_state: Seed for reproducibility.

    Returns:
    - A tuple containing the cluster labels and the number of clusters determined.
    """
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 2105) -> int:
    """
    Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - max_clusters: The maximum number of clusters to consider.
    - random_state: Seed for reproducibility.

    Returns:
    - An integer representing the optimal number of clusters found.
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]

def local_cluster_embeddings(embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine") -> np.ndarray:
    """
    Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - num_neighbors: The number of neighbors to consider for each point.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    return umap.UMAP(n_neighbors=num_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)

def emb_sentences(collection_name:str, document_id:str, sentences: list[str]):
    embeddings = emb(sentences=sentences)
    if embeddings is None:
        print('cannot embedding sentences')
    else:
        ss = []
        es = []
        for i in range(len(sentences)):
            if check_existed(collection_name=collection_name, sentence=sentences[i], embedding=embeddings[i]):
                continue
            ss.append(sentences[i])
            es.append(embeddings[i])
        store_milvus(collection_name=collection_name, document_id=document_id, embeddings=es, sentences=ss)
        print("Dữ liệu embedding đã được đẩy vào Milvus.")

def check_existed(collection_name:str, sentence:str, embedding:list[float]) -> bool:
    return False

def store_milvus(collection_name:str, document_id:str, embeddings: list[list[float]], sentences: list[str]):
    # Kết nối đến Milvus
    conn = MilvusClient(uri=config.milvus_uri)

    dim = len(embeddings[0])  # Số chiều của vector embedding
    # Tạo collection trong Milvus nếu chưa tồn tại
    if not conn.has_collection(collection_name=collection_name, timeout=30):
        schema = conn.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name='document_id', datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name='sentence', datatype=DataType.VARCHAR, max_length=10240)
        schema.add_field(field_name='embedding', datatype=DataType.FLOAT_VECTOR, dim=dim)

        index_params = conn.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128}
        )

        conn.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

    # Đẩy dữ liệu embedding vào Milvus
    data = [{'document_id': document_id, 'sentence': sentence, 'embedding': embedding} for sentence, embedding in zip(sentences, embeddings)]
    conn.insert(collection_name=collection_name, data=data)

    conn.release_collection(collection_name=collection_name)
    conn.close()

def similar_search(collection_name:str, query:str, top_k:int=5, threshold_distance:float=1.0) -> list[str]:
    embeddings = emb(sentences=[query])
    # Kết nối đến Milvus
    conn = MilvusClient(uri=config.milvus_uri)

    conn.load_collection(collection_name=collection_name)
    # Single vector search
    # search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    search_params = {"metric_type": "IP", "params": {}}
    res = conn.search(
        collection_name=collection_name,  # Replace with the actual name of your collection
        # Replace with your query vector
        data=embeddings,
        limit=top_k,  # Max. number of search results to return
        search_params=search_params,  # Search parameters
        output_fields=['*'],
        anns_field='embedding'
    )
    conn.release_collection(collection_name=collection_name)
    conn.close()

    # Convert the output to a formatted JSON string
    return [doc['entity']['sentence'] for doc in res[0] if doc['distance'] < threshold_distance]


def num_tokens_from_string(string: str, encoding_name: str='cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

if __name__ == '__main__':
    # # LCEL docs
    # url = 'https://docs.lootbot.xyz/lootbot'
    # loader = RecursiveUrlLoader(
    #     url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
    # )
    # docs = loader.load()
    # docs_texts = [d.page_content for d in docs]
    #
    # # Doc texts concat
    # d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    # d_reversed = list(reversed(d_sorted))
    # concatenated_content = "\n\n\n --- \n\n\n".join(
    #     [doc.page_content for doc in d_reversed]
    # )
    # print(
    #     "Num tokens in all context: %s"
    #     % num_tokens_from_string(concatenated_content, "cl100k_base")
    # )
    #
    # # Doc texts split
    # chunk_size_tok = 1024
    # chunk_overlap_tok = 128
    # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=chunk_size_tok, chunk_overlap=chunk_overlap_tok
    # )
    # texts_split = text_splitter.split_text(concatenated_content)
    #
    # # Build tree
    # leaf_texts = texts_split
    # results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)
    #
    # # Đẩy dữ liệu vào vector DB
    # # Initialize all_texts with leaf_texts
    # all_texts = leaf_texts.copy()
    #
    # # Iterate through the results to extract summaries from each level and add them to all_texts
    # for level in sorted(results.keys()):
    #     # Extract summaries from the current level's DataFrame
    #     summaries = results[level][1]["summaries"].tolist()
    #     # Extend all_texts with the summaries from the current level
    #     all_texts.extend(summaries)
    #
    collection_name = 'test_lootbot'
    document_id = 'document_id'
    # emb_sentences(collection_name=collection_name, document_id=document_id, sentences=all_texts)

    docs = similar_search(collection_name=collection_name, query='lootbot', top_k=10)
    for doc in docs:
        print(doc)

