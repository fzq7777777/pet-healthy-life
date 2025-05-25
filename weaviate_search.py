import weaviate
from text_embedding import get_embedding

def weaviate_search(query: str = "", collection_name: str = "docs123", limit: int = 5):
    """
    在 Weaviate 中执行搜索查询
    :param query: 搜索查询
    :param collection_name: 集合名称
    :param limit: 返回结果的数量限制
    :return: 搜索结果
    """
    try:
        query_vector = get_embedding(query)
        client = weaviate.connect_to_local()
        # 执行搜索查询
        collection_exists = client.collections.exists(collection_name)
        if not collection_exists:
            print(f"Collection '{collection_name}' not found. Creating it now.")
        else:
            print(f"Collection '{collection_name}' exists. Proceeding with search.")
            results = client.collections.get(collection_name)
            response = results.query.near_vector(
                near_vector=query_vector,
                include_vector=True,
                limit=limit,
                return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
            )

            contents = ""

            for i,o in enumerate(response.objects):
                
                contents += "\n\n" + o.properties.get('content', 'No content found')

        return contents      
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close()  # 确保关闭连接
    



