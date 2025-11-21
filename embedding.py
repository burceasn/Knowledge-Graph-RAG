# 导入所需的库
from openai import OpenAI, APIError  # 导入OpenAI客户端和API错误类用于异常处理
from graph_handle import GraphHandle
import numpy as np
import pandas as pd
import os

class EmbeddingClient:
    """
    一个用于获取实体嵌入向量的客户端。

    该客户端负责与OpenAI API交互，为知识图谱中的实体生成嵌入向量，
    并使用缓存机制来避免不必要的API调用。
    """
    def __init__(self, api_key: str, base_url: str, model: str, cache_file: str = "embedding_data.pkl", graph_file: str = "knowledge_graph.json"):
        """
        初始化EmbeddingClient。

        :param api_key: OpenAI API密钥。
        :param base_url: OpenAI API的基础URL。
        :param model: 用于生成嵌入的模型的名称。
        :param cache_file: 用于存储和加载嵌入向量缓存的文件路径。
        :param graph_file: 知识图谱数据的文件路径。
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.graph_handle = GraphHandle(graph_file)
        self.cache_file = cache_file

    def embedding_all_entities(self):
        """
        为图中的所有'Entity'类型的节点生成或加载嵌入向量。

        该方法首先检查是否存在缓存文件。如果存在，则直接从文件加载嵌入数据。
        如果不存在，它会从图中获取所有实体，分批为它们生成嵌入向量，
        然后将结果保存到缓存文件并返回。

        :return: 一个包含实体'id'和'embedding'的pandas DataFrame。
        """
        # 检查是否存在缓存文件，如果存在则直接加载
        if os.path.exists(self.cache_file):
            print(f"Loading embeddings from cached file: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                df = pd.read_pickle(f)
            return df
        
        # 如果没有缓存，则从图中获取所有实体
        entities = self.graph_handle.node_types('Entity')
        rows_list = []
        batch_size = 10  # 设置批量处理的大小，以提高API调用效率

        # 分批处理实体
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            # 提取每个实体的描述，如果描述不存在则使用空字符串
            descriptions = [entity.get('description', '') for entity in batch]
            ids = [entity.get('id', '') for entity in batch]
            
            # 为当前批次的描述生成嵌入向量
            embeddings = self.get_embedding(descriptions)
            
            # 将id和对应的embedding组织起来
            for j, id in enumerate(ids):
                # 确保id和描述都存在
                if id and descriptions[j]:
                    row = {'id': id, 'embedding': embeddings[j]}
                    rows_list.append(row)
        
        # 将结果转换为DataFrame
        df = pd.DataFrame(rows_list)
        
        # 将DataFrame保存到pickle文件作为缓存
        with open(self.cache_file, 'wb') as f:
            df.to_pickle(self.cache_file)
        return df
    
    def get_embedding(self, input_text: str|list):
        """
        调用OpenAI API为单个文本或文本列表生成嵌入向量。

        :param input_text: 单个字符串或字符串列表。
        :return: 单个numpy数组或由多个numpy数组垂直堆叠的二维数组。
        :raises APIError: 如果OpenAI API调用失败。
        :raises Exception: 如果发生其他预期之外的错误。
        """
        try:
            # 调用OpenAI的embeddings API
            response = self.client.embeddings.create(
                input=input_text,
                model=self.model
            )
            # 根据输入是单个文本还是列表，返回相应格式的numpy数组
            if isinstance(input_text, str):
                return np.array(response.data[0].embedding)
            else:
                return np.vstack([response.data[i].embedding for i in range(len(response.data))])
        except APIError as e:
            # 捕获并处理OpenAI API特定的错误
            print(f"An OpenAI API error occurred: {e}")
            # 重新抛出异常，中断程序执行，以便上层调用者能感知到错误
            raise
        except Exception as e:
            # 捕获其他所有可能的意外错误
            print(f"An unexpected error occurred during embedding: {e}")
            # 同样重新抛出异常
            raise
