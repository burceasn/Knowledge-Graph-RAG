import os
import requests
import pypdfium2 as pdfium
import io
import hashlib
import shutil
import time

class PDFProcessor:
    def __init__(self, nougat_api_url, source_dir="source", output_dir="data", server_cache_dir="pdfs"):
        self.nougat_api_url = nougat_api_url
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.server_cache_dir = server_cache_dir

    def ensure_directories_exist(self):
        """确保必要的目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.source_dir):
            raise FileNotFoundError(f"源文件夹 '{self.source_dir}' 不存在，请创建并放入PDF文件。")

    def get_pdf_files(self):
        """获取所有PDF文件名列表"""
        return [f for f in os.listdir(self.source_dir) if f.lower().endswith(".pdf")]

    def extract_first_page(self, source_path):
        """提取PDF的第一页，并返回其字节流"""
        pdf_first_page_bytes = io.BytesIO()
        doc = pdfium.PdfDocument(source_path)
        try:
            new_pdf = pdfium.PdfDocument.new()
            try:
                new_pdf.import_pages(doc, pages=[0])
                new_pdf.save(pdf_first_page_bytes)
            finally:
                new_pdf.close()
        finally:
            doc.close()

        pdf_first_page_bytes.seek(0)
        return pdf_first_page_bytes

    def calculate_md5(self, data_bytes):
        """计算数据的MD5值"""
        return hashlib.md5(data_bytes).hexdigest()

    def send_to_api(self, filename, pdf_data):
        """将PDF第一页发送给Nougat API"""
        files = {'file': (filename, pdf_data, 'application/pdf')}
        response = requests.post(self.nougat_api_url, files=files, timeout=300)
        return response

    def handle_response(self, response, md5_hash, output_path):
        """根据API响应进行后续操作：保存或移动结果"""
        if response.status_code == 200:
            print("API 处理成功。")
            time.sleep(0.5)

            server_result_dir = os.path.join(self.server_cache_dir, md5_hash)
            server_output_file = os.path.join(server_result_dir, 'doc.mmd')

            if os.path.exists(server_output_file):
                print(f"移动结果文件到: {output_path}")
                shutil.move(server_output_file, output_path)
                print(f"清理服务端缓存: {server_result_dir}")
                shutil.rmtree(server_result_dir, ignore_errors=True)
                print("清理完毕。")
            else:
                print(f"未在服务端找到预期的结果文件 '{server_output_file}'。")
                print("将直接保存API返回的内容。")
                with open(output_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(response.text)
        else:
            print(f"处理失败，状态码: {response.status_code}")
            print(f"服务器返回信息: {response.text}")

    def process_all_pdfs(self):
        """主处理逻辑"""
        self.ensure_directories_exist()
        pdf_files = self.get_pdf_files()

        if not pdf_files:
            print(f"在 '{self.source_dir}' 文件夹中没有找到PDF文件。")
            return

        print(f"找到 {len(pdf_files)} 个PDF文件，准备只处理每个文件的第一页...")

        for filename in pdf_files:
            source_path = os.path.join(self.source_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".mmd"
            output_path = os.path.join(self.output_dir, output_filename)

            print(f"\n--- 正在处理: {filename} ---")

            try:
                pdf_first_page_bytes = self.extract_first_page(source_path)
                pdf_data = pdf_first_page_bytes.read()
                md5_hash = self.calculate_md5(pdf_data)
                pdf_first_page_bytes.seek(0)

                response = self.send_to_api(filename, pdf_first_page_bytes)
                self.handle_response(response, md5_hash, output_path)

            except requests.exceptions.RequestException as e:
                print(f"请求API时发生网络错误: {e}")
            except Exception as e:
                print(f"处理文件时发生未知错误: {filename}")
                print(f"错误详情: {e}")

        print("\n--- 所有文件处理完毕 ---")


# --- 使用示例 ---
if __name__ == "__main__":
    NOUGAT_API_URL = "http://127.0.0.1:8503/predict/"
    processor = PDFProcessor(nougat_api_url=NOUGAT_API_URL)
    processor.process_all_pdfs()