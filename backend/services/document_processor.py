"""文档解析与切分服务

功能：
1. 解析多种格式文档（PDF、TXT、Markdown、DOCX）
2. 文档切分（支持递归切分、父子文档切分）
3. 元数据提取
"""
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml


class DocumentProcessor:
    """文档处理器 - 解析 + 切分"""

    def __init__(self, config_path: str = None):
        """初始化文档处理器

        Args:
            config_path: 配置文件路径，默认使用项目根目录下的 config/config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.chunking_config = self.config.get('chunking', {})

    def parse_file(self, file_path: str) -> str:
        """解析单个文件，提取纯文本内容

        Args:
            file_path: 文件路径

        Returns:
            提取的文本内容
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == '.pdf':
            return self._parse_pdf(file_path)
        elif suffix == '.txt':
            return self._parse_txt(file_path)
        elif suffix == '.md':
            return self._parse_txt(file_path)  # Markdown 当纯文本处理
        elif suffix == '.docx':
            return self._parse_docx(file_path)
        else:
            raise ValueError(f"不支持的文件格式：{suffix}")

    def _parse_pdf(self, file_path: str) -> str:
        """解析 PDF 文件"""
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    def _parse_txt(self, file_path: str) -> str:
        """解析纯文本 / Markdown 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _parse_docx(self, file_path: str) -> str:
        """解析 Word 文档"""
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        return "\n\n".join(paragraphs)

    def parse_bytes(self, file_bytes: bytes, filename: str) -> str:
        """从字节流解析文件（用于 API 上传场景）

        Args:
            file_bytes: 文件字节内容
            filename: 文件名（用于判断格式）

        Returns:
            提取的文本内容
        """
        import tempfile

        # 写入临时文件再解析
        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            return self.parse_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def chunk_documents(self, text: str, source: str = "unknown") -> List[Document]:
        """将文本切分为文档块

        根据配置文件中的切分策略进行切分：
        - recursive: 递归字符切分（默认）
        - parent_child: 父子文档切分（小子块检索，大父块返回）

        Args:
            text: 要切分的文本
            source: 来源文件名（写入元数据）

        Returns:
            切分后的 Document 列表
        """
        strategy = self.chunking_config.get('strategy', 'recursive')

        if strategy == 'parent_child':
            return self._parent_child_chunk(text, source)
        else:
            return self._recursive_chunk(text, source)

    def _recursive_chunk(self, text: str, source: str) -> List[Document]:
        """递归字符切分"""
        parent_config = self.chunking_config.get('parent', {})
        chunk_size = parent_config.get('chunk_size', 800)
        chunk_overlap = parent_config.get('chunk_overlap', 200)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )

        chunks = splitter.split_text(text)
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_type": "parent"
                }
            )
            documents.append(doc)

        return documents

    def _parent_child_chunk(self, text: str, source: str) -> List[Document]:
        """父子文档切分

        大父块用于返回完整上下文，小子块用于精确检索。
        每个子块通过 parent_id 关联到对应的父块。

        Args:
            text: 要切分的文本
            source: 来源文件名

        Returns:
            切分后的 Document 列表（包含父块和子块）
        """
        parent_config = self.chunking_config.get('parent', {})
        child_config = self.chunking_config.get('child', {})

        parent_size = parent_config.get('chunk_size', 800)
        parent_overlap = parent_config.get('chunk_overlap', 200)
        child_size = child_config.get('chunk_size', 200)
        child_overlap = child_config.get('chunk_overlap', 50)

        # 第一步：切大父块
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )
        parent_chunks = parent_splitter.split_text(text)

        # 第二步：每个父块再切小子块
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )

        documents = []
        for parent_id, parent_text in enumerate(parent_chunks):
            # 添加父块文档
            parent_doc = Document(
                page_content=parent_text,
                metadata={
                    "source": source,
                    "parent_id": parent_id,
                    "chunk_type": "parent",
                    "doc_id": f"{source}_parent_{parent_id}"
                }
            )
            documents.append(parent_doc)

            # 切子块
            child_chunks = child_splitter.split_text(parent_text)
            for child_id, child_text in enumerate(child_chunks):
                child_doc = Document(
                    page_content=child_text,
                    metadata={
                        "source": source,
                        "parent_id": parent_id,
                        "child_id": child_id,
                        "chunk_type": "child",
                        "doc_id": f"{source}_parent_{parent_id}_child_{child_id}"
                    }
                )
                documents.append(child_doc)

        return documents

    def process_file(self, file_path: str) -> List[Document]:
        """完整处理流程：解析 + 切分

        Args:
            file_path: 文件路径

        Returns:
            处理后的 Document 列表
        """
        # 解析文件
        text = self.parse_file(file_path)
        if not text.strip():
            return []

        # 切分文档
        source = Path(file_path).name
        documents = self.chunk_documents(text, source)

        return documents

    def process_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """完整处理流程（字节流版本）：解析 + 切分

        Args:
            file_bytes: 文件字节内容
            filename: 文件名

        Returns:
            处理后的 Document 列表
        """
        text = self.parse_bytes(file_bytes, filename)
        if not text.strip():
            return []

        documents = self.chunk_documents(text, filename)
        return documents
