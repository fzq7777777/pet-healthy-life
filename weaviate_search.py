import weaviate
from text_embedding import get_embedding

def content_search(query: str = "", collection_name: str = "docs123", limit: int = 5):
    """
    在 Weaviate 中执行搜索查询
    :param query: 搜索查询
    :param collection_name: 集合名称
    :param limit: 返回结果的数量限制
    :return: 搜索结果
    """
    client = weaviate.connect_to_local()
    try:
        query_vector = get_embedding(query)
        # 执行搜索查询
        collection_exists = client.collections.exists(collection_name)
        if not collection_exists:
            print(f"Collection '{collection_name}' not found. ")
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


def vector_search(query_vector, collection_name: str = "docs123", limit: int = 5):
    """
    在 Weaviate 中执行搜索查询
    :param query: 搜索查询
    :param collection_name: 集合名称
    :param limit: 返回结果的数量限制
    :return: 搜索结果
    """
    client = weaviate.connect_to_local()
    try:
        # 执行搜索查询
        collection_exists = client.collections.exists(collection_name)
        if not collection_exists:
            print(f"Collection '{collection_name}' not found")
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

  


if __name__ == "__main__":
    from image_embedding import image_to_vector
    image_path = "cifar_images/truck_16.png"
    query_vector = image_to_vector(image_path)
    results = vector_search(query_vector, collection_name="image_collection002", limit=3)
    print(f"Search results for image '{results}':")

