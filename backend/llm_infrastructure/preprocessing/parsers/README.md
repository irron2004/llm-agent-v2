# PDF Parser Infrastructure

This package provides a unified interface for parsing PDF documents with support for multiple parsing backends, including RAGFlow's DeepDoc engine.

## Architecture

```
parsers/
├── base.py                 # Core interfaces and dataclasses
├── registry.py             # Parser registration and discovery
├── engines/                # Parsing engines (implementation logic)
│   ├── pdf_plain_engine.py      # Simple text extraction (pdfplumber)
│   └── pdf_deepdoc_engine.py    # Advanced DeepDoc/RAGFlow parsing
└── adapters/              # Registry integration adapters
    ├── pdf_plain.py             # Plain PDF adapter
    └── pdf_deepdoc.py           # DeepDoc PDF adapter
```

### Design Principles

1. **Layered Architecture**: Clear separation between interfaces (base), implementations (engines), and integrations (adapters)
2. **Registry Pattern**: Dynamic parser registration and retrieval
3. **Graceful Fallback**: Automatic fallback to simpler parsers when advanced features are unavailable
4. **Type Safety**: Comprehensive type annotations and dataclasses

## Usage

### Basic Usage

```python
from llm_infrastructure.preprocessing.parsers import get_parser
from llm_infrastructure.preprocessing.parsers.base import PdfParseOptions

# Get a parser from registry
parser = get_parser("pdf_deepdoc")

# Parse a PDF file
with open("document.pdf", "rb") as f:
    result = parser.parse(f, options=PdfParseOptions(
        ocr=True,
        layout=True,
        tables=True,
        max_pages=10
    ))

# Access parsed content
for block in result.blocks:
    print(f"Page {block.page}: {block.text}")

# Access tables
for table in result.tables:
    print(f"Table on page {table.page}: {table.html}")

# Access figures
for figure in result.figures:
    print(f"Figure on page {figure.page}: {figure.caption}")
```

### Available Parsers

#### 1. `pdf_plain` - Simple Text Extraction

Uses `pdfplumber` for basic text extraction without OCR or layout analysis.

**Pros:**
- Fast and lightweight
- No GPU required
- No model downloads

**Cons:**
- Cannot handle scanned PDFs
- No layout recognition
- No table structure extraction

**Use when:**
- You only need plain text
- PDFs have embedded text
- Speed is critical

```python
parser = get_parser("pdf_plain")
result = parser.parse(file, options=PdfParseOptions(preserve_layout=True))
```

#### 2. `pdf_deepdoc` - Advanced DeepDoc Parsing

Uses RAGFlow's DeepDoc engine for comprehensive PDF parsing with OCR, layout analysis, and table structure recognition.

**Pros:**
- Handles scanned PDFs (OCR)
- Recognizes document structure (titles, paragraphs, tables, figures)
- Extracts table structure with high accuracy
- Maintains spatial relationships (bounding boxes)
- Automatic text merging and cleanup

**Cons:**
- Requires `deepdoc` or `ragflow` package
- Slower than plain extraction
- May require GPU for best performance
- Requires model downloads

**Use when:**
- PDFs contain complex layouts
- You need table structure extraction
- Handling scanned documents
- Preserving document hierarchy is important

```python
parser = get_parser("pdf_deepdoc")
result = parser.parse(file, options=PdfParseOptions(
    ocr=True,              # Enable OCR for scanned PDFs
    layout=True,           # Recognize document layout
    tables=True,           # Extract table structures
    merge=True,            # Merge fragmented text blocks
    scrap_filter=True,     # Remove noise (page numbers, etc.)
    model_root="/path/to/models",  # Model cache directory
    device="cuda",         # Use GPU (cuda) or CPU
    max_pages=None,        # Process all pages
    fallback_to_plain=True # Fallback to plain if DeepDoc fails
))
```

#### 3. `pdf_deepseek_vlm` - Vision-Language Model Parsing (In Development)

Uses large multimodal language models (like DeepSeek-VL2) to directly understand document images and extract content without explicit OCR steps.

**Pros:**
- No explicit OCR required - model "sees" and understands images directly
- Excellent handling of complex formulas, equations, and technical content
- Natural understanding of document layout and reading order
- Flexible output format (Markdown, LaTeX, structured text)
- Leverages model's pre-trained knowledge for accurate recognition
- Fewer layout reconstruction errors compared to traditional OCR

**Cons:**
- Requires large GPU memory for model inference
- Slower processing compared to traditional methods
- Higher computational cost per page
- No bounding box/coordinate information in output
- Potential for model hallucination (incorrect content generation)
- Results are continuous text rather than structured data

**Use when:**
- Documents contain complex mathematical formulas or equations
- High accuracy is more important than speed
- Processing scanned documents with difficult-to-OCR content
- Need natural language understanding of document context
- Working with specialized technical or scientific documents

```python
# Note: This parser is currently in development
parser = get_parser("pdf_deepseek_vlm")
result = parser.parse(file, options=PdfParseOptions(
    model_name="deepseek-vl2",      # Vision-Language model
    output_format="markdown",        # Output format (markdown, latex, text)
    max_tokens=4096,                 # Max tokens per page
    temperature=0.1,                 # Lower for more deterministic output
    device="cuda",                   # GPU required for large models
    max_pages=None,                  # Process all pages
))

# Access continuous text output
print(result.merged_text())  # Full document as markdown text

# Output preserves structure but lacks coordinates
for block in result.blocks:
    print(block.text)  # No bbox information available
```

## DeepSeek VLM Parser Details

DeepSeek VLM (Vision-Language Model) parsing represents a modern approach to document understanding using large multimodal AI models instead of traditional computer vision pipelines.

### How It Works

Unlike traditional OCR-based approaches, DeepSeek VLM models process document images directly:

1. **Direct Image Understanding**: The model receives PDF pages as images and "sees" the content like a human would
2. **No Explicit OCR Stage**: Text recognition happens implicitly within the model's neural networks
3. **Contextual Comprehension**: The model understands document structure, layout, and semantics simultaneously
4. **Text Generation**: Output is generated as natural language (Markdown, LaTeX, etc.) rather than raw OCR text

This approach is fundamentally different from the multi-stage CV pipeline used in DeepDoc.

### Processing PDF and PPT Documents

**PDF Processing:**
- Each PDF page is converted to an image
- The image is fed to the Vision-Language model (e.g., DeepSeek-VL2)
- The model outputs text content in the requested format (Markdown, LaTeX, plain text)
- Formulas, tables, and special characters are recognized using the model's pre-trained knowledge
- No separate OCR, layout analysis, or table structure recognition stages

Example workflow:
```
PDF Page → Image → DeepSeek-VL2 → Markdown Text
                   (single model)
```

**PPT Processing:**
- Slides are rendered as images
- Each slide image is processed by the VLM
- Text in shapes, diagrams, and images is extracted
- Layout and visual hierarchy are understood contextually
- Diagrams and charts can optionally be described in natural language

**Key Characteristics:**
- **OCR-free**: No Tesseract or traditional OCR tools involved
- **Context-aware**: Understands reading order even in complex multi-column layouts
- **Formula-friendly**: Excellent at recognizing mathematical notation and equations
- **Table understanding**: Can interpret table structure and convert to Markdown/LaTeX
- **Minimal fragmentation**: Produces coherent paragraphs without word-level splitting

### Output Format: Extraction vs. Summarization

**Primary Goal: Faithful Extraction**

DeepSeek VLM parsing aims to **extract original content as accurately as possible**, not to summarize or transform it. The model is prompted to:

- Reproduce all text from the document
- Preserve document structure (headings, paragraphs, lists)
- Maintain formulas and equations (in LaTeX format)
- Convert tables to Markdown/HTML tables
- Keep figure captions and references

**Output Characteristics:**
```python
# Typical output structure
{
    "text": """
# Document Title

## Section 1

This is the first paragraph with complete original text...

The equation is: $E = mc^2$

| Header 1 | Header 2 |
|----------|----------|
| Value 1  | Value 2  |

Figure 1: Sample chart description
    """,
    "format": "markdown"
}
```

**Not Included by Default:**
- Content summarization
- Information filtering or reduction
- Semantic restructuring
- Question generation
- Keyword extraction

These post-processing tasks can be performed **after** parsing in a separate pipeline stage, but are not part of the VLM parsing itself.

**Potential Deviations:**
While the goal is faithful extraction, VLM parsing may differ from source in these ways:
- Formatting standardization (e.g., all tables → Markdown tables)
- Reading order interpretation (multi-column layouts → linear text)
- Formula normalization (image formulas → LaTeX notation)
- Rare cases of model hallucination (incorrect text generation)

### Comparison with DeepDoc

| Aspect | DeepDoc (CV Pipeline) | DeepSeek VLM (LLM-based) |
|--------|----------------------|--------------------------|
| **Architecture** | Multi-stage pipeline (OCR → Layout → TSR → Merging) | Single large multimodal model |
| **Models Used** | Specialized CV models (Tesseract, LayoutLM, Table-Transformer, XGBoost) | One Vision-Language Model (DeepSeek-VL2) |
| **Processing Flow** | Sequential stages with intermediate data | End-to-end image → text generation |
| **OCR Method** | Explicit OCR tools (Tesseract, YOLOv8-based) | Implicit within model weights |
| **Output Structure** | Structured JSON with coordinates and metadata | Continuous text (Markdown/LaTeX) |
| **Bounding Boxes** | ✅ Precise coordinates for all elements | ❌ No spatial information |
| **Table Extraction** | Structured cell-by-cell data + HTML | Markdown table or natural language |
| **Formula Handling** | OCR text (may have errors) | LaTeX notation (high accuracy) |
| **Layout Understanding** | Rule-based bbox analysis | Semantic understanding from training |
| **Speed** | Moderate (multiple model inferences) | Slow (large model inference) |
| **GPU Requirements** | Optional (faster with GPU) | Required for practical use |
| **Memory Usage** | ~2GB for models + page buffers | ~8-24GB for VLM model |
| **Accuracy on Scans** | Good (depends on OCR quality) | Excellent (human-like comprehension) |
| **Complex Formulas** | Poor (OCR struggles) | Excellent (trained on LaTeX) |
| **Searchability** | High (structured blocks with positions) | Moderate (continuous text only) |
| **UI Highlighting** | Possible (bbox coordinates available) | Difficult (no position data) |
| **Error Types** | OCR misrecognition, layout errors | Rare hallucinations, no coords |
| **Extensibility** | Add/replace specific CV models | Swap entire VLM model |
| **Cost** | Lower (smaller models, CPU-friendly) | Higher (large model, GPU-intensive) |

**When to Choose DeepDoc:**
- Need precise element coordinates for UI highlighting
- Require structured data output (JSON, HTML tables)
- Processing large volumes of documents (cost-sensitive)
- Want proven, stable production pipeline
- Need to run on CPU or limited GPU resources

**When to Choose DeepSeek VLM:**
- Documents contain complex mathematical formulas
- Scanned documents with difficult OCR scenarios
- Require highest accuracy on technical/scientific content
- Can afford GPU compute costs
- Prefer natural language output (Markdown, LaTeX)
- Don't need spatial coordinate information

### Processing Pipeline Comparison

**DeepDoc Pipeline (Multi-Stage):**
```
PDF → Image Rendering
  ↓
OCR (YOLOv8/Tesseract) → Text boxes with coordinates
  ↓
Layout Analysis (LayoutLM) → Classify blocks (title, paragraph, table, etc.)
  ↓
Table Structure Recognition → Cell boundaries, merged cells
  ↓
Text Merging (XGBoost) → Connect fragments across columns/pages
  ↓
Cleanup & Filtering → Remove headers/footers/page numbers
  ↓
Structured Output → JSON with blocks, tables, figures, coordinates
```

**DeepSeek VLM Pipeline (End-to-End):**
```
PDF → Image Rendering (per page)
  ↓
Vision-Language Model (DeepSeek-VL2)
  - Visual encoding: Image → visual embeddings
  - Text generation: Visual embeddings → text tokens
  - Format adherence: Follow Markdown/LaTeX structure
  ↓
Markdown/LaTeX Output → Continuous text with preserved structure
```

### Model Configuration

**Supported Models:**
- DeepSeek-VL2 (recommended)
- DeepSeek-R1 (text-only, requires separate OCR)
- Other compatible Vision-Language models (Qwen-VL, GPT-4-Vision, etc.)

**Configuration Example:**
```python
from llm_infrastructure.preprocessing.parsers.base import VlmParseOptions

options = VlmParseOptions(
    model_name="deepseek-vl2",
    model_path="/path/to/model",  # Local model path or HuggingFace repo
    device="cuda",                 # GPU device
    max_tokens=4096,               # Max output tokens per page
    temperature=0.1,               # Lower = more deterministic
    output_format="markdown",      # or "latex", "text"
    system_prompt="Extract all text from this document page, preserving structure and formulas.",
    batch_size=1,                  # Pages to process in parallel
)
```

### Use Cases

**Ideal Use Cases:**
1. **Academic Papers**: Complex formulas, equations, citations
2. **Scientific Reports**: Technical diagrams, mathematical notation
3. **Scanned Documents**: Historical documents, handwritten notes
4. **Technical Manuals**: Specifications with tables and formulas
5. **Research Documents**: High accuracy required, cost acceptable

**Not Recommended For:**
1. **High-Volume Processing**: Thousands of simple documents (too slow/expensive)
2. **UI Highlighting Needs**: Applications requiring element coordinates
3. **Structured Data Extraction**: When JSON/database output is needed
4. **Real-Time Processing**: Interactive applications with latency constraints
5. **Resource-Constrained Environments**: Limited GPU memory or budget

### Performance Considerations

**Processing Speed:**
- **DeepSeek-VL2**: ~5-15 seconds per page (GPU)
- **Factors affecting speed**:
  - Model size (14B-320B parameters)
  - Image resolution
  - Output length (formulas, tables increase generation time)
  - GPU compute capability

**Memory Requirements:**
- **Model Loading**: 8-24GB GPU VRAM (depending on model size)
- **Inference**: +2-4GB per concurrent request
- **Recommended**: A100 (40GB) or H100 (80GB) for production

**Cost Comparison (per 1000 pages):**
```
DeepDoc:      $1-5   (CPU/small GPU)
DeepSeek VLM: $50-200 (large GPU inference)
```

### Current Development Status

⚠️ **This parser is currently in development and not yet available in production.**

**Completed:**
- Research and design documentation
- Architecture planning
- Integration strategy with existing parser infrastructure

**In Progress:**
- Engine implementation (`pdf_deepseek_vlm_engine.py`)
- Model integration and prompt engineering
- Output format standardization
- Performance optimization

**Planned:**
- Registry adapter (`adapters/pdf_deepseek_vlm.py`)
- Comprehensive test suite
- Batch processing support
- Fallback strategies
- Production deployment

## DeepDoc Engine Details

DeepDoc is RAGFlow's core PDF parsing engine that uses deep learning models to understand document structure.

### Processing Pipeline

1. **OCR (Optical Character Recognition)**
   - Converts PDF pages to images
   - Uses YOLOv8-based OCR model to detect text regions
   - Extracts text from both embedded text layers and images
   - Output: Text boxes with bounding boxes

2. **Layout Recognition**
   - Analyzes document structure using Vision models
   - Classifies text blocks into categories:
     - Title / Headers
     - Paragraphs
     - Table captions
     - Figure captions
     - Footnotes
     - References
     - Equations
   - Output: Labeled text blocks with hierarchical structure

3. **Table Structure Recognition (TSR)**
   - Detects table regions
   - Recognizes cell boundaries and merged cells
   - Extracts table structure (rows, columns, headers)
   - Converts tables to HTML and natural language descriptions
   - Output: Structured table data with images

4. **Text Merging & Refinement**
   - Merges fragmented text blocks
   - Removes hyphenation and line breaks
   - Connects text across page boundaries using XGBoost models
   - Filters out noise (page numbers, headers, footers)
   - Output: Clean, continuous text chunks

5. **Figure Extraction**
   - Detects image regions
   - Extracts figure captions
   - Saves figure snapshots
   - Output: Figure metadata with image references

### Backend Discovery

The DeepDoc engine automatically searches for compatible parsers:

```python
# Attempts to import (in order):
1. deepdoc.parser.pdf_parser.RAGFlowPdfParser
2. deepdoc.parser.pdf_parser.PdfParser
3. deepdoc.parser.pdf_parser.PlainParser
4. ragflow.deepdoc.parser.pdf_parser.RAGFlowPdfParser
5. ragflow.deepdoc.parser.pdf_parser.PdfParser
```

### Output Structure Compatibility

DeepDoc engines can return various output formats, which are automatically normalized:

```python
# Supported backend output formats:
{
    "chunks": [...],      # Text blocks (primary)
    "blocks": [...],      # Alternative to chunks
    "pages": [...],       # Page-level content
    "tables": [...],      # Table structures
    "figures": [...],     # Figure/image metadata
    "images": [...]       # Alternative to figures
}
```

### Bounding Box Formats

Multiple coordinate formats are supported:

```python
# Format 1: Dict with x0/y0/x1/y1
{"x0": 10, "y0": 20, "x1": 100, "y1": 200}

# Format 2: Dict with left/top/right/bottom
{"left": 10, "top": 20, "right": 100, "bottom": 200}

# Format 3: List/Tuple
[10, 20, 100, 200]
(10, 20, 100, 200)
```

### Model Configuration

#### HuggingFace Configuration

```python
PdfParseOptions(
    model_root="/path/to/models",       # Cache directory for models
    hf_endpoint="https://hf-mirror.com", # Mirror for regions with restrictions
    allow_download=True,                 # Auto-download models
    ocr_model="yolov8/ocr",             # OCR model repository
    layout_model="layoutlm/base",        # Layout recognition model
    tsr_model="table-transformer/v1"     # Table structure model
)
```

Environment variables set automatically:
- `HF_ENDPOINT` - HuggingFace endpoint URL
- `HF_HOME` - Model cache directory
- `HUGGINGFACE_HUB_CACHE` - Hub cache directory
- `TRANSFORMERS_CACHE` - Transformers cache directory

#### Device Selection

```python
PdfParseOptions(device="cuda")   # Use GPU (faster, requires CUDA)
PdfParseOptions(device="cpu")    # Use CPU (slower, no GPU required)
```

### Fallback Mechanism

DeepDoc engine automatically falls back to PlainPdfEngine in these scenarios:

1. **Backend Not Available**: DeepDoc/RAGFlow not installed
2. **Import Errors**: Missing dependencies
3. **Parse Errors**: DeepDoc parsing fails
4. **Model Loading Errors**: Models cannot be loaded

```python
# Fallback can be disabled:
PdfParseOptions(fallback_to_plain=False)  # Raises error instead

# Fallback metadata is recorded:
result.metadata["used_fallback"]      # True if fallback was used
result.metadata["fallback_reason"]    # Error message
```

## Data Structures

### ParsedDocument

Main output structure containing all parsed content:

```python
@dataclass
class ParsedDocument:
    pages: List[ParsedPage]          # Pages with full text
    blocks: List[ParsedBlock]        # Text blocks with positions
    tables: List[ParsedTable]        # Extracted tables
    figures: List[ParsedFigure]      # Extracted figures
    metadata: Dict[str, Any]         # Parser metadata
    errors: List[str]                # Error messages
    content_type: str                # MIME type

    def merged_text(self, separator="\n\n") -> str:
        """Merge all blocks into single text"""
```

### ParsedBlock

Individual text block with metadata:

```python
@dataclass
class ParsedBlock:
    text: str                        # Block content
    page: int                        # Page number (1-indexed)
    bbox: Optional[BoundingBox]      # Position on page
    label: str                       # Block type (paragraph, title, etc.)
    confidence: Optional[float]      # Detection confidence (0-1)
    metadata: Dict[str, Any]         # Additional metadata
```

### ParsedTable

Table structure with multiple representations:

```python
@dataclass
class ParsedTable:
    page: int                        # Page number
    bbox: Optional[BoundingBox]      # Table position
    html: Optional[str]              # HTML representation
    text: Optional[str]              # Natural language description
    image_ref: Optional[str]         # Path to table image
    metadata: Dict[str, Any]         # Cell structure, etc.
```

### ParsedFigure

Figure/image metadata:

```python
@dataclass
class ParsedFigure:
    page: int                        # Page number
    bbox: Optional[BoundingBox]      # Figure position
    caption: Optional[str]           # Figure caption
    image_ref: Optional[str]         # Path to saved image
    metadata: Dict[str, Any]         # Additional metadata
```

### BoundingBox

Spatial coordinates on page:

```python
@dataclass
class BoundingBox:
    x0: float  # Left edge
    y0: float  # Top edge
    x1: float  # Right edge
    y1: float  # Bottom edge
```

## Advanced Usage

### Custom Parser Registration

```python
from llm_infrastructure.preprocessing.parsers import register_parser
from llm_infrastructure.preprocessing.parsers.base import BaseParser

class MyCustomParser(BaseParser):
    content_type = "application/pdf"

    def parse(self, file, options=None):
        # Custom parsing logic
        return ParsedDocument(...)

register_parser("my_parser", MyCustomParser)

# Use custom parser
parser = get_parser("my_parser")
```

### Processing Large Documents

```python
# Process in chunks
options = PdfParseOptions(max_pages=10)
for page_offset in range(0, total_pages, 10):
    chunk_result = parser.parse(file, options)
    # Process chunk...
```

### Combining Parsers

```python
# Try DeepDoc first, fallback manually
try:
    parser = get_parser("pdf_deepdoc")
    result = parser.parse(file, PdfParseOptions(fallback_to_plain=False))
except Exception:
    # Custom fallback logic
    parser = get_parser("pdf_plain")
    result = parser.parse(file)
```

### Accessing Metadata

```python
result = parser.parse(file)

# Parser information
print(result.metadata["parser"])           # "pdf_deepdoc"
print(result.metadata["backend"])          # "RAGFlowPdfParser"
print(result.metadata["backend_keys"])     # ["chunks", "tables", "figures"]

# Parse options
print(result.metadata["ocr"])              # True
print(result.metadata["layout"])           # True
print(result.metadata["max_pages"])        # 10

# Fallback information (if used)
if result.metadata.get("used_fallback"):
    print(result.metadata["fallback_reason"])
```

## Installation

### Minimal (Plain Parser Only)

```bash
pip install pdfplumber
```

### Full (DeepDoc Support)

```bash
# Option 1: Install RAGFlow
pip install ragflow

# Option 2: Install DeepDoc separately
pip install deepdoc

# Additional dependencies for GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Testing

Comprehensive test suite with 81 tests:

```bash
# Run all parser tests
pytest tests/preprocessing/parsers/ -v

# Run specific test modules
pytest tests/preprocessing/parsers/test_pdf_deepdoc_engine.py -v
pytest tests/preprocessing/parsers/test_base.py -v

# Run with coverage
pytest tests/preprocessing/parsers/ --cov=llm_infrastructure.preprocessing.parsers
```

## Performance Considerations

### DeepDoc Performance

- **OCR Stage**: 1-3 seconds per page (GPU), 5-10 seconds (CPU)
- **Layout Analysis**: 0.5-1 second per page (GPU)
- **Table Recognition**: 1-2 seconds per table
- **Memory**: ~2GB for models + page buffers

### Optimization Tips

1. **Use GPU**: 5-10x faster than CPU
2. **Limit Pages**: Process only needed pages
3. **Disable Unnecessary Features**:
   ```python
   PdfParseOptions(
       ocr=False,    # Skip if PDF has text layer
       tables=False, # Skip if no tables needed
       layout=False  # Skip if only need text
   )
   ```
4. **Batch Processing**: Process multiple documents in parallel
5. **Model Caching**: Reuse model instances across documents

## Troubleshooting

### "DeepDoc backend is not available"

```bash
# Install RAGFlow or DeepDoc
pip install ragflow
# or
pip install deepdoc
```

### "pdfplumber is required for PlainPdfEngine"

```bash
pip install pdfplumber
```

### Models not downloading

```python
# Check HuggingFace connectivity
PdfParseOptions(
    hf_endpoint="https://hf-mirror.com",  # Use mirror
    allow_download=True,
    model_root="/path/to/writable/directory"
)
```

### Out of memory errors

```python
# Reduce memory usage
PdfParseOptions(
    device="cpu",      # Use CPU instead of GPU
    max_pages=10,      # Process fewer pages at once
)
```

### Incorrect table extraction

```python
# Try different options
PdfParseOptions(
    tables=True,
    preserve_layout=True,  # Keep original layout
    merge=False            # Don't merge cells
)
```

## References

- [RAGFlow Documentation](https://github.com/infiniflow/ragflow)
- [DeepDoc Paper](https://arxiv.org/abs/deepdoc)
- [pdfplumber Documentation](https://github.com/jsvine/pdfplumber)

## License

This infrastructure is part of the LLM Agent project. See main LICENSE for details.
