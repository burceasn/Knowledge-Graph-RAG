import mistune
from typing import List, Dict, Any, Tuple, Optional

class MDParser:
    """
    一个强大且优雅的 Markdown 解析器，使用 mistune 的 AST 来精确查询文档结构。
    (版本 4.0: 针对 mistune v2+ 的 AST 结构进行了完全重写和修复)
    """

    def __init__(self, markdown_content: str):
        """
        初始化解析器，并将 Markdown 文本转换为抽象语法树 (AST)。
        """
        # 使用 mistune v2+ 的标准方式创建解析器
        self.parser = mistune.create_markdown(renderer='ast')
        self.ast = self.parser(markdown_content)

    def _get_text_from_children(self, children: List[Dict[str, Any]]) -> str:
        """
        (内部辅助方法) 递归地从子节点列表中重建完整的文本内容。
        此方法现在可以正确处理 mistune v2+ 的 AST 结构。
        """
        text_parts = []
        for child in children:
            # 文本节点现在使用 'raw' 键
            if child.get('type') == 'text':
                text_parts.append(child.get('raw', ''))
            # 递归地从其他包含文本的节点中提取
            elif 'children' in child:
                text_parts.append(self._get_text_from_children(child['children']))
        return "".join(text_parts)

    def _find_heading_node(self, title: str, level: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], int]:
        """
        (内部辅助方法) 在 AST 中查找匹配的标题节点。
        """
        search_title = title.strip().lower()
        for i, node in enumerate(self.ast):
            # --- 核心修复 1: 检查 'attrs' 中的 'level' ---
            node_attrs = node.get('attrs', {})
            node_level = node_attrs.get('level')

            if node.get('type') == 'heading':
                # 如果指定了 level，但节点级别不匹配，则跳过
                if level is not None and node_level != level:
                    continue
                
                # --- 核心修复 2: 使用新的辅助方法从 'children' 重建标题文本 ---
                full_heading_text = ""
                if 'children' in node:
                    full_heading_text = self._get_text_from_children(node['children']).strip().lower()

                # 如果不提供标题，则匹配该级别的第一个标题
                if not search_title:
                    if node_level is not None:
                         return node, i
                # 如果提供了标题，则进行精确匹配
                elif full_heading_text == search_title:
                    return node, i
        return None, -1

    def _reconstruct_text_from_nodes(self, nodes: List[Dict[str, Any]]) -> str:
        """
        (内部辅助方法) 从 AST 节点列表中重建可读的文本内容。
        """
        texts = []
        for node in nodes:
            node_type = node.get('type')
            if node_type == 'paragraph' and 'children' in node:
                texts.append(self._get_text_from_children(node['children']))
            elif node_type == 'list' and 'children' in node:
                list_items = []
                for item in node['children']:
                    # 列表项的文本现在也通过通用函数获取
                    item_text = self._get_text_from_children(item.get('children', []))
                    list_items.append(f"* {item_text}")
                texts.append("\n".join(list_items))
            # 可以根据需要添加对其他节点类型（如 block_code, block_quote）的支持
        return "\n\n".join(texts)

    def get_heading(self, title: str, level: Optional[int] = None) -> Optional[str]:
        """
        获取与给定文本和级别匹配的标题的完整原始内容。
        """
        node, _ = self._find_heading_node(title, level)
        if node and 'children' in node:
            return self._get_text_from_children(node['children'])
        return None

    def get_content(self, title: str, level: Optional[int] = None) -> Optional[str]:
        """
        获取指定标题下方的所有内容，直到遇见下一个同级或更高级别的标题为止。
        """
        start_node, start_index = self._find_heading_node(title, level)
        if not start_node:
            return None

        content_nodes = []
        heading_level = start_node.get('attrs', {}).get('level')
        if heading_level is None: return ""

        for i in range(start_index + 1, len(self.ast)):
            node = self.ast[i]
            # 同样需要检查 attrs['level']
            node_level = node.get('attrs', {}).get('level')
            if node.get('type') == 'heading' and node_level is not None and node_level <= heading_level:
                break
            content_nodes.append(node)
        return self._reconstruct_text_from_nodes(content_nodes)