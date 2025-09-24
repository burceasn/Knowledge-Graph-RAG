"""
Paper Cache Manager
论文信息缓存管理器，用于存储和管理已提取的论文信息
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PaperCache:
    """
    论文缓存管理器
    使用JSON文件缓存已提取的论文信息，避免重复调用LLM API
    """
    
    def __init__(self, cache_dir: str = "cache", cache_file: str = "papers_cache.json"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
            cache_file: 缓存文件名
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file_path = self.cache_dir / cache_file
        self.cache_data: Dict[str, Any] = self._load_cache()
        
    def _load_cache(self) -> Dict[str, Any]:
        """从文件加载缓存数据"""
        if self.cache_file_path.exists():
            try:
                with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"成功加载缓存，包含 {len(data)} 篇论文")
                    return data
            except Exception as e:
                logger.error(f"加载缓存失败: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """保存缓存数据到文件"""
        try:
            with open(self.cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"缓存已保存到 {self.cache_file_path}")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def _generate_paper_id(self, title: str) -> str:
        """
        生成论文的唯一标识符
        使用标题的MD5哈希值作为ID
        
        Args:
            title: 论文标题
            
        Returns:
            论文的唯一ID
        """
        return hashlib.md5(title.encode('utf-8')).hexdigest()
    
    def has_paper(self, title: str) -> bool:
        """
        检查缓存中是否存在该论文
        
        Args:
            title: 论文标题
            
        Returns:
            是否存在该论文的缓存
        """
        paper_id = self._generate_paper_id(title)
        return paper_id in self.cache_data
    
    def get_paper_data(self, title: str) -> Optional[Dict[str, Any]]:
        """
        获取论文的缓存数据
        
        Args:
            title: 论文标题
            
        Returns:
            论文数据字典，如果不存在则返回None
        """
        paper_id = self._generate_paper_id(title)
        return self.cache_data.get(paper_id)
    
    def has_author_metadata(self, title: str) -> bool:
        """
        检查是否已缓存作者元数据
        
        Args:
            title: 论文标题
            
        Returns:
            是否已缓存作者元数据
        """
        paper_data = self.get_paper_data(title)
        if paper_data:
            return 'author_metadata' in paper_data and paper_data['author_metadata'] is not None
        return False
    
    def has_entities(self, title: str) -> bool:
        """
        检查是否已缓存实体信息
        
        Args:
            title: 论文标题
            
        Returns:
            是否已缓存实体信息
        """
        paper_data = self.get_paper_data(title)
        if paper_data:
            return 'entities' in paper_data and paper_data['entities'] is not None
        return False
    
    def update_paper_data(self, 
                          title: str, 
                          abstract: Optional[str] = None, 
                          field: Optional[str] = None, 
                          author_metadata: Optional[Dict[str, Any]] = None,
                          entities: Optional[List[Dict[str, Any]]] = None,
                          relations: Optional[List[Dict[str, Any]]] = None,
                          **kwargs):
        """
        更新或创建论文缓存数据
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            field: 研究领域
            author_metadata: 作者元数据
            entities: 提取的实体列表
            relations: 实体关系列表
            **kwargs: 其他自定义字段
        """
        paper_id = self._generate_paper_id(title)
        
        # 如果论文不存在，创建新记录
        if paper_id not in self.cache_data:
            self.cache_data[paper_id] = {
                'title': title,
                'paper_id': paper_id,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
        
        # 更新论文数据
        paper_data = self.cache_data[paper_id]
        paper_data['updated_at'] = datetime.now().isoformat()
        
        # 更新各个字段（只更新非None的值）
        if abstract is not None:
            paper_data['abstract'] = abstract
        if field is not None:
            paper_data['field'] = field
        if author_metadata is not None:
            paper_data['author_metadata'] = author_metadata
            paper_data['author_metadata_extracted_at'] = datetime.now().isoformat()
        if entities is not None:
            paper_data['entities'] = entities
            paper_data['entities_extracted_at'] = datetime.now().isoformat()
        if relations is not None:
            paper_data['relations'] = relations
            paper_data['relations_extracted_at'] = datetime.now().isoformat()
        
        # 添加其他自定义字段
        for key, value in kwargs.items():
            if value is not None:
                paper_data[key] = value
        
        # 保存到文件
        self._save_cache()
        logger.info(f"已更新论文缓存: {title[:50]}...")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        total_papers = len(self.cache_data)
        papers_with_authors = sum(1 for p in self.cache_data.values() 
                                 if 'author_metadata' in p)
        papers_with_entities = sum(1 for p in self.cache_data.values() 
                                  if 'entities' in p)
        papers_with_relations = sum(1 for p in self.cache_data.values() 
                                   if 'relations' in p)
        
        return {
            'total_papers': total_papers,
            'papers_with_authors': papers_with_authors,
            'papers_with_entities': papers_with_entities,
            'papers_with_relations': papers_with_relations,
            'cache_file': str(self.cache_file_path),
            'cache_size_kb': self.cache_file_path.stat().st_size / 1024 if self.cache_file_path.exists() else 0
        }
    
    def clear_cache(self, confirm: bool = False):
        """
        清空缓存
        
        Args:
            confirm: 确认清空操作
        """
        if confirm:
            self.cache_data = {}
            self._save_cache()
            logger.warning("缓存已清空")
        else:
            logger.warning("需要确认才能清空缓存 (confirm=True)")
    
    def export_to_separate_files(self, export_dir: str = "cache/papers"):
        """
        将每篇论文导出为单独的JSON文件
        
        Args:
            export_dir: 导出目录
        """
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        for paper_id, paper_data in self.cache_data.items():
            file_path = export_path / f"{paper_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已导出 {len(self.cache_data)} 篇论文到 {export_path}")


class CachedAuthorExtractor:
    """
    带缓存功能的作者信息提取器
    """
    
    def __init__(self, cache: PaperCache, author_extractor):
        """
        Args:
            cache: PaperCache实例
            author_extractor: AuthorMetadataExtractor实例
        """
        self.cache = cache
        self.extractor = author_extractor
    
    def get_authors(self, title: str, content: str):
        """
        获取作者信息（优先从缓存读取）
        
        Args:
            title: 论文标题
            content: 包含作者信息的文本
            
        Returns:
            AuthorList对象或错误信息
        """
        paper_data = self.cache.get_paper_data(title)

        # 检查缓存
        if paper_data and paper_data.get('author_metadata'):
            logger.info(f"从缓存读取作者信息: {title[:50]}...")
            # 将缓存的字典转换回AuthorList对象
            from Author_metadata import AuthorList, Author
            author_dict = paper_data['author_metadata']
            authors = [Author(**a) for a in author_dict['Authors']]
            return AuthorList(Authors=authors)
        
        # 调用LLM提取
        logger.info(f"调用LLM提取作者信息: {title[:50]}...")
        result = self.extractor.get_authors(content)
        
        # 保存到缓存
        if result and hasattr(result, 'Authors'):
            author_dict = {
                'Authors': [author.model_dump() for author in result.Authors]
            }
            self.cache.update_paper_data(title=title, author_metadata=author_dict)
        
        return result


class CachedEntityExtractor:
    """
    带缓存功能的实体提取器
    """
    
    def __init__(self, cache: PaperCache, entity_extractor):
        """
        Args:
            cache: PaperCache实例
            entity_extractor: PaperEntityExtractor实例
        """
        self.cache = cache
        self.extractor = entity_extractor
    
    def extract_entities_from_abstract(self, paper):
        """
        从论文摘要提取实体（优先从缓存读取）
        
        Args:
            paper: Paper对象
            
        Returns:
            (entities, relations) 元组
        """
        title = paper.Title
        paper_data = self.cache.get_paper_data(title)

        # 检查缓存
        if paper_data and paper_data.get('entities'):
            logger.info(f"从缓存读取实体信息: {title[:50]}...")
            
            # 重建Entity和Edge对象
            from Node import Entity
            from Edge import EntityToEntityEdge
            
            entities = []
            for e_data in paper_data.get('entities', []):
                entity = Entity(
                    name=e_data['name'],
                    field=e_data.get('field'),
                    description=e_data.get('description'),
                    entity_type=e_data.get('entity_type')
                )
                entities.append(entity)
            
            # 创建实体名称到对象的映射
            entity_map = {e.name: e for e in entities}
            
            relations = []
            for r_data in paper_data.get('relations', []):
                source = entity_map.get(r_data['source_name'])
                target = entity_map.get(r_data['target_name'])
                if source and target:
                    relation = EntityToEntityEdge(
                        source_entity=source,
                        target_entity=target,
                        relationship_description=r_data['description'],
                        strength=r_data.get('strength', 0.0)
                    )
                    relations.append(relation)
            
            return entities, relations
        
        # 调用LLM提取
        logger.info(f"调用LLM提取实体信息: {title[:50]}...")
        entities, relations = self.extractor.extract_entities_from_abstract(paper)
        
        # 保存到缓存
        if entities or relations:
            entities_data = [
                {
                    'name': e.name,
                    'field': e.field,
                    'description': e.description,
                    'entity_type': e.entity_type
                }
                for e in entities
            ]
            
            relations_data = [
                {
                    'source_name': r.source.name,
                    'target_name': r.target.name,
                    'description': r.get_relationship_description(),
                    'strength': r.get_strength()
                }
                for r in relations
            ]
            
            self.cache.update_paper_data(
                title=title,
                abstract=paper.Abstract,
                field=paper.field,
                entities=entities_data,
                relations=relations_data
            )
        
        return entities, relations