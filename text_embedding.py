import weaviate
import ollama
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader


@dataclass
class TextEmbeddingConfig:   
    chunk_size: int = 1000
    chunk_overlap: int = 200
    model: str = "all-minilm"
    batch_size: int = 200

def pdf_to_text(pdf_path: str) -> str:
    """
    将 PDF 文件转换为文本
    :param pdf_path: PDF 文件路径
    :return: 提取的文本
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    将文本分割为块
    :param text: 输入文本
    :param chunk_size: 块大小
    :param chunk_overlap: 块重叠大小
    :return: 文本块列表
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def get_embedding(text: str) -> list:
    """
    获取文本的嵌入向量
    :param text: 输入文本
    :return: 嵌入向量
    """
    embedding = ollama.embeddings(model=TextEmbeddingConfig.model, prompt=text).get("embedding")
    if embedding is None:
        raise ValueError("Failed to get embedding from Ollama API.")
    return embedding

def text_chunks_to_vectors(text_chunks: list):
    """
    将文本块转换为嵌入向量
    :param text_chunks: 文本块列表
    :return: 文本和向量
    """
    result = []
    for chunk in text_chunks:
        embedding = get_embedding(chunk)
        result.append({
            "content": chunk,
            "vector": embedding
        })
    return result

def weaviate_create_collection(collection_name: str):
    """
    创建 Weaviate 数据集合
    :param collection_name: 集合名称
    """
    client = weaviate.connect_to_local()
    try:  
        client.collections.create(
            collection_name,
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
        )
        print(f"Collection '{collection_name}' created successfully.")
    except weaviate.exceptions.WeaviateBaseError as e:
        print(f"Error creating collection: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        client.close()

def weaviate_import(collection_name: str, content_and_vector = [], batch_size: int = 100):
    """
    将嵌入向量和文本导入 Weaviate
    :param embedding: 嵌入向量
    :param text: 输入文本
    """
    client = weaviate.connect_to_local()
    try:
        
        client.collections.exists(collection_name)
        # 检查集合是否存在
        
        collection_exists = client.collections.exists(collection_name)
        if not collection_exists:
            print(f"Collection '{collection_name}' not found. Creating it now.")
            weaviate_create_collection(collection_name)
        
        collection = client.collections.get(collection_name)
        print(f"Using collection: {collection.name}")
        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for item in content_and_vector:
                batch.add_object(
                    properties={"content": item.get("content")},
                    vector=item.get("vector")
                )
                if batch.number_errors > 10:
                    print("Batch import stopped due to excessive errors.")
                    break

        failed_objects = collection.batch.failed_objects
        if failed_objects:
            print(f"Number of failed imports: {len(failed_objects)}")
            print(f"First failed object: {failed_objects[0]}")
        else:
            print(f"Successfully imported {len(content_and_vector)} objects to collection '{collection_name}'.")
    except weaviate.exceptions.WeaviateBaseError as e:
        print(f"Error importing to Weaviate: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        client.close()

