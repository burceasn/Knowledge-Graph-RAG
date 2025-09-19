import subprocess
import shutil
from pathlib import Path
from typing import Optional
import pypdfium2 as pdfium

class PDFProcessor:
    """
    使用 Nougat 命令行工具处理 PDF 文件的第一页并转换为 Markdown
    """
    
    def __init__(self, 
                 source_dir: str = "source", 
                 output_dir: str = "data",
                 model: str = "0.1.0-small"):  # small, base, 或不指定使用默认
        """
        初始化 PDF 处理器
        
        Args:
            source_dir: 源 PDF 文件夹路径
            output_dir: 输出 Markdown 文件夹路径  
            model: 模型标签 (0.1.0-small = 快速, 0.1.0-base = 默认)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.model = model
        
        # 确保目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.source_dir.exists():
            raise FileNotFoundError(f"源文件夹 '{self.source_dir}' 不存在，请创建并放入 PDF 文件。")
        
        # 检查 nougat 是否已安装
        if not self._check_nougat_installed():
            raise RuntimeError(
                "Nougat 未安装。请运行：\n"
                "pip install nougat-ocr\n"
                "或访问: https://github.com/facebookresearch/nougat"
            )
    
    def _check_nougat_installed(self) -> bool:
        """检查 nougat 命令是否可用"""
        try:
            result = subprocess.run(
                ["nougat", "--help"], 
                capture_output=True, 
                text=True
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    
    def process_single_pdf(self, pdf_path: Path) -> bool:
        """
        处理单个 PDF 文件的第一页
        
        Args:
            pdf_path: PDF 文件路径
            
        Returns:
            是否成功处理
        """
        md_path = self.output_dir / f"{pdf_path.stem}.md"
        
        # ✅ 增量更新逻辑：如果已经存在 md 文件则跳过
        if md_path.exists():
            print(f"✓ 已存在，跳过: {md_path}")
            return True

        print(f"正在处理: {pdf_path.name}")
        
        try:
            # 构建 nougat 命令
            cmd = [
                "nougat",
                str(pdf_path),                    # 输入 PDF
                "-o", str(self.output_dir),       # 输出目录
                "-p", "1",                         # 只处理第一页
                "--markdown",                      # 输出 Markdown 格式
                "--no-skipping",                   # 不跳过任何内容
            ]
            
            # 添加模型参数（如果指定）
            if self.model:
                cmd.extend(["-m", self.model])
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode != 0:
                print(f"✗ Nougat 处理失败: {result.stderr}")
                return False
            
            # 重命名生成的 .mmd 文件为 .md
            mmd_path = self.output_dir / f"{pdf_path.stem}.mmd"
            
            if mmd_path.exists():
                # 读取内容
                content = mmd_path.read_text(encoding='utf-8')
                
                # 清理内容（移除多余空行）
                cleaned_content = self.clean_markdown(content)
                
                # 写入 .md 文件
                md_path.write_text(cleaned_content, encoding='utf-8')
                
                # 删除原始 .mmd 文件
                mmd_path.unlink()
                
                print(f"✓ 成功保存到: {md_path}")
                return True
            else:
                print(f"✗ 未找到输出文件")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"✗ 处理超时: {pdf_path.name}")
            return False
            
        except Exception as e:
            print(f"✗ 处理出错: {e}")
            return False
    
    def clean_markdown(self, text: str) -> str:
        """
        清理 Markdown 文本，移除多余的空行
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            line = line.rstrip()
            is_empty = len(line) == 0
            
            if is_empty and prev_empty:
                continue
                
            cleaned_lines.append(line)
            prev_empty = is_empty
        
        return '\n'.join(cleaned_lines).strip()
    
    def process_all_pdfs(self):
        """
        处理源文件夹中的所有 PDF 文件
        """
        # 清理可能存在的 pdfs 文件夹
        self.cleanup_temp_files()
        
        # 获取所有 PDF 文件
        pdf_files = list(self.source_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"在 '{self.source_dir}' 文件夹中没有找到 PDF 文件。")
            return
        
        print(f"找到 {len(pdf_files)} 个 PDF 文件")
        print("="*50)
        
        # 统计
        success_count = 0
        failed_files = []
        
        # 处理每个 PDF
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{idx}/{len(pdf_files)}]", end=" ")
            
            if self.process_single_pdf(pdf_path):
                success_count += 1
            else:
                failed_files.append(pdf_path.name)
        
        # 最终清理
        self.cleanup_temp_files()
        
        # 打印总结
        print("\n" + "="*50)
        print(f"处理完成！")
        print(f"成功: {success_count}/{len(pdf_files)}")
        
        if failed_files:
            print(f"失败的文件: {', '.join(failed_files)}")
        
        print("="*50)
    
    def cleanup_temp_files(self):
        """清理 Nougat 可能生成的临时文件夹"""
        pdfs_dir = Path("./pdfs")
        if pdfs_dir.exists():
            try:
                shutil.rmtree(pdfs_dir)
                print("已清理 ./pdfs 临时文件夹")
            except Exception as e:
                print(f"清理 pdfs 文件夹失败: {e}")


if __name__ == "__main__":
    processor = PDFProcessor(
        source_dir="source",      
        output_dir="data",        
        model="0.1.0-small"       
    )
    processor.process_all_pdfs()