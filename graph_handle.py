import json
from typing import Dict, List, Optional, Any

class GraphHandle:
    """
    一个用于加载和检索知识图谱数据的处理器类。
    该类旨在提供一个简洁的API，用于根据类型、ID或名称等属性方便地查询节点和边。
    """

    def __init__(self, json_file_path: str):
        """
        初始化GraphHandle并从指定的JSON文件加载图数据。

        Args:
            json_file_path (str): 知识图谱JSON文件的路径。
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                self.graph: Dict[str, List[Dict[str, Any]]] = json.load(f)
        except FileNotFoundError:
            print(f"错误：文件未找到 -> {json_file_path}")
            self.graph = {"nodes": [], "links": []}
        except json.JSONDecodeError:
            print(f"错误：无法解析JSON文件 -> {json_file_path}")
            self.graph = {"nodes": [], "links": []}
        
        self.nodes: List[Dict[str, Any]] = self.graph.get('nodes', [])
        self.links: List[Dict[str, Any]] = self.graph.get('links', [])

    def node_types(self, node_type: str = 'all') -> List[Dict[str, Any]]:
        """
        根据节点类型检索节点。

        Args:
            node_type (str, optional): 要检索的节点类型 (例如 'Paper', 'Entity', 'Author')。
                                     默认为 'all'，返回所有节点。

        Returns:
            List[Dict[str, Any]]: 符合条件的节点列表。
        """
        if node_type.lower() == 'all':
            return self.nodes
        
        return [node for node in self.nodes if node.get('node_type', '').lower() == node_type.lower()]

    def get_links(self, edge_type: str = 'all') -> List[Dict[str, Any]]:
        """
        根据边的类型检索边（关系）。

        Args:
            edge_type (str, optional): 要检索的边的类型 (例如 'Coauthor', 'PaperEntity')。
                                     默认为 'all'，返回所有边。

        Returns:
            List[Dict[str, Any]]: 符合条件的边列表。
        """
        if edge_type.lower() == 'all':
            return self.links
        
        return [link for link in self.links if link.get('edge_type', '').lower() == edge_type.lower()]

    def find_node(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        通过其唯一标识符（ID, name, display_name, 或 title）查找单个节点。
        此方法返回第一个匹配项，优先匹配ID。

        Args:
            identifier (str): 节点的ID, name, display_name, 或 title。

        Returns:
            Optional[Dict[str, Any]]: 找到的节点字典，如果未找到则返回 None。
        """
        # 优先通过ID精确查找
        for node in self.nodes:
            if node.get('id') == identifier:
                return node
        
        # 如果ID未匹配，则通过名称等模糊查找
        for node in self.nodes:
            if node.get('name') == identifier or \
               node.get('display_name') == identifier or \
               node.get('title') == identifier:
                return node
        
        return None

    def get_links_for_node(self, node_id: str, link_type:str = "all") -> List[Dict[str, Any]]:
        """
        检索连接到特定节点的所有边。

        Args:
            node_id (str): 节点的唯一ID。

        Returns:
            List[Dict[str, Any]]: 与该节点相关的所有边的列表。
        """
        if not self.find_node(node_id):
            print(f"警告：未找到ID为 '{node_id}' 的节点。")
            return []
            
        connected_links = []
        for link in self.links:
            if link.get('source') == node_id or link.get('target') == node_id:
                connected_links.append(link)
        if link_type == "all":
            return connected_links
        else:
            return [link for link in connected_links if link.get('edge_type', '').lower() == link_type.lower()]