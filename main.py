"""
主处理流程 - 集成缓存管理
使用缓存系统来避免重复调用LLM API，节省token消耗
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from PDFprocess import PDFProcessor
from Markdownparser import MDParser
from Author_metadata import AuthorMetadataExtractor, AuthorList
from entity_extraction import PaperEntityExtractor
from Node import Paper
from construct import KnowledgeGraphBuilder
from paper_cache import (
    PaperCache, 
    CachedAuthorExtractor, 
    CachedEntityExtractor
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PaperProcessor:
    """
    论文处理器 - 集成缓存功能
    """
    
    def __init__(self, 
                 source_dir: str = "source",
                 data_dir: str = "data",
                 cache_dir: str = "cache",
                 field_list: Optional[List[str]] = None):
        """
        初始化处理器
        
        Args:
            source_dir: PDF源文件目录
            data_dir: Markdown输出目录
            cache_dir: 缓存目录
            field_list: 研究领域列表
        """
        self.source_dir = Path(source_dir)
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.field_list = field_list or ["ML", "LLM", "Knowledge Graph"]
        
        # 初始化缓存
        self.cache = PaperCache(cache_dir=str(self.cache_dir))
        
        # 初始化PDF处理器
        self.pdf_processor = PDFProcessor(
            source_dir=str(self.source_dir),
            output_dir=str(self.data_dir),
            model="0.1.0-small"
        )
        
        # 初始化知识图谱构建器
        self.kg_builder = KnowledgeGraphBuilder()
        
        # 初始化提取器（这里需要配置你的API）
        self._init_extractors()
    
    def _init_extractors(self):
        """初始化各种提取器"""
        # 作者提取器
        base_author_extractor = AuthorMetadataExtractor(
            api_base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen3:4b"  # 或者你使用的模型
        )
        self.author_extractor = CachedAuthorExtractor(
            cache=self.cache,
            author_extractor=base_author_extractor
        )
        
        # 实体提取器
        base_entity_extractor = PaperEntityExtractor(
            llm_provider="deepseek",  # 或 "ollama"
            api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量获取
            api_base_url="https://api.deepseek.com/beta",
            model="deepseek-chat",
            entity_types=["CONCEPT", "METHOD", "DATASET", "METRIC", 
                         "APPLICATION", "TOOL", "PROBLEM", "RESULT"],
            temperature=0.1
        )
        self.entity_extractor = CachedEntityExtractor(
            cache=self.cache,
            entity_extractor=base_entity_extractor
        )
    
    def process_single_paper(self, md_file: Path) -> bool:
        """
        处理单篇论文
        
        Args:
            md_file: Markdown文件路径
            
        Returns:
            是否处理成功
        """
        try:
            logger.info(f"开始处理: {md_file.name}")
            
            # 读取Markdown内容
            with open(md_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # 解析Markdown
            parser = MDParser(markdown_content)
            
            # 获取标题
            title = parser.get_heading(title="", level=1)
            if not title:
                logger.warning(f"无法获取标题: {md_file.name}")
                return False
            
            # 获取摘要
            abstract = parser.get_content(title="Abstract")
            if not abstract:
                logger.warning(f"无法获取摘要: {md_file.name}")
                return False
            
            # 确定研究领域（这里简单地使用默认值，你可以根据内容判断）
            field = self.field_list[0]  # 或者使用更智能的分类方法
            
            # 创建Paper对象
            paper = Paper(title=title, abstract=abstract, field=field)
            
            # 检查缓存状态
            logger.info(f"缓存状态 - 作者: {self.cache.has_author_metadata(title)}, "
                       f"实体: {self.cache.has_entities(title)}")
            
            # 提取作者信息（使用缓存）
            content = parser.get_content(title="", level=1)
            if content:
                meta_data = content.split(abstract)[0].strip()
                author_result = self.author_extractor.get_authors(title, meta_data)
                
                if isinstance(author_result, AuthorList):
                    # 添加到知识图谱
                    self.kg_builder.process_author_list_and_paper(
                        author_result, paper
                    )
                    logger.info(f"成功提取 {len(author_result.Authors)} 个作者")
            
            # 提取实体和关系（使用缓存）
            entities, relations = self.entity_extractor.extract_entities_from_abstract(paper)
            
            if entities:
                logger.info(f"成功提取 {len(entities)} 个实体")
                # 添加实体到知识图谱
                for entity in entities:
                    self.kg_builder.add_paper_entity_relation(
                        paper, entity, weight=1.0
                    )
            
            if relations:
                logger.info(f"成功提取 {len(relations)} 个关系")
                # 添加实体间关系到知识图谱
                for relation in relations:
                    # 这里可以根据需要添加实体间关系到图中
                    pass
            
            return True
            
        except Exception as e:
            logger.error(f"处理 {md_file.name} 时出错: {e}")
            return False
    
    def process_all_papers(self):
        """处理所有论文"""
        # 1. 首先处理PDF到Markdown（如果需要）
        logger.info("步骤 1: 处理PDF文件...")
        self.pdf_processor.process_all_pdfs()
        
        # 2. 获取所有Markdown文件
        md_files = list(self.data_dir.glob("*.md"))
        logger.info(f"找到 {len(md_files)} 个Markdown文件")
        
        # 3. 显示缓存统计
        stats = self.cache.get_statistics()
        logger.info(f"缓存统计: {stats}")
        
        # 4. 处理每个文件
        success_count = 0
        for i, md_file in enumerate(md_files, 1):
            logger.info(f"\n[{i}/{len(md_files)}] 处理中...")
            if self.process_single_paper(md_file):
                success_count += 1
        
        # 5. 显示最终统计
        logger.info("\n" + "="*50)
        logger.info(f"处理完成: {success_count}/{len(md_files)} 成功")
        
        # 显示更新后的缓存统计
        final_stats = self.cache.get_statistics()
        logger.info(f"最终缓存统计:")
        for key, value in final_stats.items():
            logger.info(f"  {key}: {value}")
        
        # 6. 显示知识图谱统计
        kg_stats = self.kg_builder.get_graph_statistics()
        logger.info("\n知识图谱统计:")
        for key, value in kg_stats.items():
            logger.info(f"  {key}: {value}")
        
        # 7. 导出知识图谱
        output_file = "knowledge_graph"
        self.kg_builder.export_to_json(output_file)
        logger.info(f"知识图谱已导出到 {output_file}.graphml")
    
    def clear_cache_for_paper(self, title: str):
        """
        清除特定论文的缓存
        
        Args:
            title: 论文标题
        """
        if self.cache.has_paper(title):
            # 删除该论文的数据
            paper_id = self.cache._generate_paper_id(title)
            del self.cache.cache_data[paper_id]
            self.cache._save_cache()
            logger.info(f"已清除论文缓存: {title}")
        else:
            logger.info(f"缓存中不存在该论文: {title}")
    
    def show_cache_status(self):
        """显示缓存状态的详细信息"""
        stats = self.cache.get_statistics()
        print("\n" + "="*60)
        print("缓存状态报告")
        print("="*60)
        print(f"总论文数: {stats['total_papers']}")
        print(f"已提取作者信息: {stats['papers_with_authors']} "
              f"({stats['papers_with_authors']/max(stats['total_papers'],1)*100:.1f}%)")
        print(f"已提取实体: {stats['papers_with_entities']} "
              f"({stats['papers_with_entities']/max(stats['total_papers'],1)*100:.1f}%)")
        print(f"已提取关系: {stats['papers_with_relations']} "
              f"({stats['papers_with_relations']/max(stats['total_papers'],1)*100:.1f}%)")
        print(f"缓存文件: {stats['cache_file']}")
        print(f"缓存大小: {stats['cache_size_kb']:.2f} KB")
        print("="*60 + "\n")


def main():
    """主函数"""
    # 创建处理器
    processor = PaperProcessor(
        source_dir="source",
        data_dir="data",
        cache_dir="cache",
        field_list=["ML", "LLM", "Knowledge Graph"]
    )
    
    # 显示当前缓存状态
    processor.show_cache_status()
    
    # 处理所有论文
    processor.process_all_papers()
    
    # 显示最终缓存状态
    processor.show_cache_status()
    
    # 可选：导出每篇论文为单独的JSON文件
    processor.cache.export_to_separate_files()


if __name__ == "__main__":
    main()
