import dashscope
from http import HTTPStatus
import os
import base64
import glob
from text_embedding import weaviate_import,TextEmbeddingConfig


def image_to_vector(image_path):
    
    # 打开图片文件
    with open(image_path, "rb") as image_file:
        # 读取文件并转换为Base64
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    # 设置图像格式
    image_format = "png"  # 根据实际情况修改，比如jpg、bmp 等
    image_data = f"data:image/{image_format};base64,{base64_image}"
    # 输入数据
    inputs = [{'image': image_data}]

    # 调用模型接口
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=inputs
    )
    if resp.status_code == HTTPStatus.OK:
        # print(f"Embedding: {len(resp.output['embeddings'][0]['embedding'])}")
        return resp.output['embeddings'][0]['embedding']

def get_content_and_vector(folder_path):
    """
    获取内容和向量
    :param file_path: 文件路径
    :return: 包含内容和向量的字典
    """
    content_and_vector = []
    file_paths = glob.glob(folder_path + '/**')
    print(f"Total files found: {len(file_paths)}")
    for file_path in file_paths:
        if os.path.isfile(file_path):
            vector = image_to_vector(file_path)
            content_and_vector.append({
                "content": file_path,
                "vector": vector
            })
    return content_and_vector


if __name__ == "__main__":
    folder_path = "cifar_images_small" 
    content_and_vector = get_content_and_vector(folder_path)
    print(f"Total content and vector pairs: {len(content_and_vector)}")

    weaviate_import(
        collection_name="image_collection002",
        content_and_vector=content_and_vector,
        batch_size=TextEmbeddingConfig.batch_size
    )


