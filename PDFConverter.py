"""
这是一个用于自动化处理学术PDF的脚本。

功能流程:
1. 在 'source' 文件夹中查找所有 PDF 文件。
2. 使用本地 GROBID 服务将每个 PDF 转换为 TEI XML 格式（在内存中处理）。
3. 使用 TEIParser 类解析 XML 内容，提取元数据和正文。
4. 对提取的数据进行清理（移除引用、截断至结论）。
5. 在控制台打印格式化的信息，并将其保存为 Markdown 格式的 TXT 文件到 'data' 文件夹。
"""

import os
import re
import json
import requests
import glob
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

# ------------------------------------------------------------------
# 类 1: 处理 PDF 与 GROBID 服务交互
# ------------------------------------------------------------------
class GrobidProcessor:
    """
    负责与 GROBID 服务进行交互，将 PDF 文件转换为 TEI XML 文本。
    """
    def __init__(self, grobid_url: str = 'http://localhost:8070/api/processFulltextDocument'):
        """
        初始化 GROBID 处理器。

        Args:
            grobid_url (str): GROBID 服务的 API 地址。
        """
        self.grobid_url = grobid_url
        print(f"GrobidProcessor initialized for server at: {self.grobid_url}")

    def process_pdf_to_xml_content(self, pdf_path: str) -> Optional[str]:
        """
        处理单个 PDF 文件，并返回 TEI XML 格式的字符串内容。

        Args:
            pdf_path (str): 需要处理的 PDF 文件的路径。

        Returns:
            Optional[str]: 成功时返回 TEI XML 字符串，失败则返回 None。
        """
        print(f"\nProcessing PDF: {os.path.basename(pdf_path)}...")
        try:
            with open(pdf_path, 'rb') as pdf_file:
                files = {'input': pdf_file}
                response = requests.post(self.grobid_url, files=files, timeout=30) # 增加30秒超时

            if response.status_code == 200:
                print("Successfully converted PDF to TEI XML in memory.")
                return response.text
            else:
                print(f"Grobid server returned an error. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while connecting to Grobid server: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during PDF processing: {e}")
            return None

# ------------------------------------------------------------------
# 类 2: 解析 TEI XML 内容
# ------------------------------------------------------------------
class TEIParser:
    """
    一个用于解析 TEI XML 文件或文本内容的健壮、可复用的解析器。
    它只负责提取、解析和清理数据，并以 Python 字典和列表的形式返回。
    """
    def __init__(self, filepath: Optional[str] = None, xml_content: Optional[str] = None):
        """
        初始化解析器。可以从文件路径加载，也可以直接从 XML 字符串加载。

        Args:
            filepath (Optional[str]): TEI XML 文件的路径。
            xml_content (Optional[str]): TEI XML 的字符串内容。
        """
        self.ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        self.root = None

        if xml_content:
            try:
                # 直接从字符串解析 XML
                self.root = ET.fromstring(xml_content.encode('utf-8'))
            except ET.ParseError as e:
                print(f"Error: Failed to parse XML content. {e}")
        elif filepath:
            try:
                tree = ET.parse(filepath)
                self.root = tree.getroot()
            except FileNotFoundError:
                print(f"Error: The file '{filepath}' was not found.")
            except ET.ParseError as e:
                print(f"Error: The file '{filepath}' is not a valid XML file. {e}")
        else:
            raise ValueError("TEIParser must be initialized with either 'filepath' or 'xml_content'.")

    # ... TEIParser 的其他方法 (_get_full_text, _clean_text, get_metadata等) 保持不变 ...
    def _get_full_text(self, element: Optional[ET.Element]) -> str:
        if element is None: return ""
        return "".join(element.itertext()).strip()

    def _clean_text(self, text: str) -> str:
        cleaned_text = re.sub(r'\[[^\]]*\]', '', text)
        return ' '.join(cleaned_text.split())

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        if self.root is None: return None
        title_element = self.root.find('.//tei:titleStmt/tei:title', self.ns)
        title = self._get_full_text(title_element)
        header = self.root.find('./tei:teiHeader', self.ns)
        if header is None: return {"title": title, "authors": [], "affiliations": [], "abstract": "Could not find teiHeader."}
        authors = []
        author_elements = header.findall('.//tei:author', self.ns)
        for author_element in author_elements:
            forenames = [name.text for name in author_element.findall('.//tei:forename', self.ns)]
            surname_element = author_element.find('.//tei:surname', self.ns)
            surname = surname_element.text if surname_element is not None else ""
            full_name = ' '.join(forenames) + ' ' + surname
            authors.append(full_name.strip())
        affiliations = []
        unique_affiliations = set()
        aff_elements = header.findall('.//tei:affiliation/tei:orgName', self.ns)
        for org in aff_elements:
            if org.text and org.text not in unique_affiliations:
                unique_affiliations.add(org.text)
                affiliations.append(org.text)
        abstract_element = self.root.find('.//tei:profileDesc/tei:abstract/tei:div/tei:p', self.ns)
        abstract = self._get_full_text(abstract_element) or "Abstract not found."
        return {"title": title, "authors": authors, "affiliations": affiliations, "abstract": abstract}

    def get_body_content(self) -> Optional[List[Dict[str, str]]]:
        if self.root is None: return None
        body_content = []
        div_elements = self.root.findall('.//tei:body/tei:div', self.ns)
        for div in div_elements:
            head_element = div.find('tei:head', self.ns)
            if head_element is not None:
                section_number = head_element.get('n', '')
                title = self._get_full_text(head_element)
                full_title = f"{section_number} {title}".strip()
                paragraphs = [self._get_full_text(p) for p in div.findall('tei:p', self.ns)]
                if full_title and paragraphs:
                    body_content.append({'title': full_title, 'content': "\n\n".join(paragraphs)})
        return body_content
        
    def get_cleaned_data(self) -> Optional[Dict[str, Any]]:
        metadata = self.get_metadata()
        body_content = self.get_body_content()
        if not metadata or not body_content: return None
        conclusion_index = -1
        for i, section in enumerate(body_content):
            if 'conclusion' in section['title'].lower():
                conclusion_index = i
                break
        cleaned_body = body_content[:conclusion_index + 1] if conclusion_index != -1 else body_content
        cleaned_metadata = metadata.copy()
        cleaned_metadata['abstract'] = self._clean_text(metadata['abstract'])
        for section in cleaned_body:
            section['content'] = self._clean_text(section['content'])
        return {"metadata": cleaned_metadata, "body": cleaned_body}


# ------------------------------------------------------------------
# 外部函数: 用于处理和保存数据
# ------------------------------------------------------------------
def display_paper_info(data: Optional[Dict[str, Any]]):
    if not data: print("No data to display."); return
    metadata = data['metadata']; body = data['body']
    print("\n" + "="*70); print(f"Title: {metadata['title']}"); print("="*70)
    print(f"**Authors:** {', '.join(metadata['authors'])}")
    print(f"**Affiliations:** {', '.join(metadata['affiliations'])}")
    print("\n## Abstract\n" + metadata['abstract'])

def save_paper_info(data: Optional[Dict[str, Any]], output_dir: str = 'data'):
    if not data: print("No data to save."); return
    os.makedirs(output_dir, exist_ok=True)
    metadata = data['metadata']; body = data['body']
    markdown_content = [
        f"# {metadata['title']}\n",
        f"**Authors:** {', '.join(metadata['authors'])}\n",
        f"**Affiliations:** {', '.join(metadata['affiliations'])}\n",
        "---\n## Abstract\n",
        f"{metadata['abstract']}\n",
        "---\n## Body Content\n"
    ]
    for section in body:
        markdown_content.append(f"### {section['title']}\n")
        content_for_txt = section['content'].replace('\n\n', '\n')
        markdown_content.append(f"{content_for_txt}\n")
    final_text = "\n".join(markdown_content)
    title = data['metadata']['title']
    sanitized_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    filename = f"{sanitized_title}.txt"
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(final_text)
        print(f"\nSuccessfully saved Markdown-formatted data to: {filepath}")
    except IOError as e:
        print(f"Error saving file to {filepath}: {e}")


# ------------------------------------------------------------------
# 主流程控制函数
# ------------------------------------------------------------------
def main():
    """
    主函数，编排整个处理流程。
    """
    # --- 配置区 ---
    # 存放 PDF 的源文件夹
    SOURCE_FOLDER = 'source'
    # 保存 TXT 的目标文件夹
    OUTPUT_FOLDER = 'data'
    # 本地 GROBID 服务的地址
    GROBID_SERVER_URL = 'http://localhost:8070/api/processFulltextDocument'

    # --- 流程开始 ---
    print("="*25 + " PDF Processing Pipeline Started " + "="*25)
    
    # 检查 source 文件夹是否存在
    if not os.path.isdir(SOURCE_FOLDER):
        print(f"Error: Source folder '{SOURCE_FOLDER}' not found.")
        print("Please create a 'source' folder and place your PDF files inside it.")
        return

    # 初始化 GROBID 处理器
    grobid_processor = GrobidProcessor(GROBID_SERVER_URL)

    # 查找所有 PDF 文件
    pdf_paths = glob.glob(os.path.join(SOURCE_FOLDER, '*.pdf'))
    if not pdf_paths:
        print(f"No PDF files found in the '{SOURCE_FOLDER}' folder.")
        return
        
    print(f"Found {len(pdf_paths)} PDF file(s) to process.")

    # 遍历并处理每个 PDF
    for pdf_path in pdf_paths:
        # 1. 使用 GROBID 将 PDF 转换为 XML 字符串
        xml_content = grobid_processor.process_pdf_to_xml_content(pdf_path)

        if xml_content:
            # 2. 直接用 XML 内容初始化 TEIParser
            parser = TEIParser(xml_content=xml_content)
            
            if parser.root:
                # 3. 获取清理后的数据
                cleaned_data = parser.get_cleaned_data()
                
                # 4. 显示和保存数据
                save_paper_info(cleaned_data, output_dir=OUTPUT_FOLDER)
            else:
                print(f"Failed to parse the XML content for {os.path.basename(pdf_path)}.")
    
    print("\n" + "="*25 + " PDF Processing Pipeline Finished " + "="*25)


if __name__ == "__main__":
    main()