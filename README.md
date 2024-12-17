# Sithfal_Tasks

# Task 1 Work Flow 

# Sithafal_Task1

#### Simple Workflow for Task 1
1. Input PDF Upload:
The user uploads a PDF document (e.g., `sample.pdf`).
2. Data Extraction:
Text Extraction:
• Extract text directly using `pdfplumber` for text-based PDFs.
• For image-based or empty-text pages, perform OCR using `pytesseract`.
Table Extraction:
• Use `pdfplumber` to identify and extract tables.
• Clean the extracted tables for better alignment.
3. Text Preprocessing:
Normalize the extracted text to remove:
• Extra spaces, line breaks, and unwanted characters.
• Non-ASCII characters.
• Ensure consistent formatting.
4. Dynamic Formatting:
Parse and dynamically format the data:
• Extract degrees (e.g., Doctoral, Master's) and align with their corresponding percentages or 
monetary values.
Present the cleaned text line by line.
5. Output Generation:
Text Output: Display formatted textual data for the requested page(s).
Tabular Output: Show the extracted table as a structured DataFrame.

Technologies and Modules Used
# Technologies:
1. PDF Parsing and OCR:
Extract text from PDFs with both text-based and image-based pages.
Preprocess images for better OCR accuracy.
2. Data Preprocessing:
Normalize and clean data for structured output.
3. Dynamic Formatting:
Use regex patterns to dynamically parse and align data.
4. Tabular Processing:
Extract, clean, and display tables using Python libraries.



# Python Modules:
1. `pdfplumber`:
Extracts text and tables from PDF files.
2. `pytesseract`:
Performs OCR for image-based PDF pages.
3. `Pillow (PIL)`:
Enhances images (grayscale conversion, resizing, contrast) for better OCR.
4. `pandas`:
Formats tabular data into structured DataFrames.
5. `re`:
Dynamically parses text with regular expressions.

## Simplified Workflow Visualization
1. Upload PDF 
 ↓ 
2. Extract Data 
Text: Use `pdfplumber` or OCR (`pytesseract`). 
Tables: Extract with `pdfplumber`. 
 ↓ 
3. Preprocess Data 
Clean text and normalize tables. 
 ↓ 
4. Format Output 
Text: Align degrees with values dynamically. 
Tables: Display clean DataFrame. 
 ↓ 
5. Display Results 
This workflow ensures accurate extraction, preprocessing, and structured display of both textual and 
tabular data



# Task 2 Work Flow

To implement Task 2: Chat with Website Using RAG Pipeline, you can follow this clear, structured approach:
 Workflow for RAG Pipeline Implementation

 # 1. Data Ingestion
   - Input: URLs or a list of websites to crawl/scrape.
   - Steps:

     1. Web Scraping:  
        - Use libraries like `BeautifulSoup`, `Scrapy`, or `requests` to scrape data (text and metadata) from the target websites.  
        - Extract key textual content from web pages (e.g., headings, paragraphs, metadata).
     2. Content Chunking:  
        - Split the extracted data into smaller chunks for better granularity using libraries like `nltk`, `langchain`, or `textwrap`.
     3. Vector Embedding:  
        - Convert the content chunks into vector embeddings using a pre-trained embedding model.  
        - Tools like `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) or OpenAI’s `text-embedding-ada` can be used.
     4. Store in Vector Database:  
        - Save the embeddings along with metadata into a vector database like Pinecone, FAISS, or Weaviate for efficient retrieval.

 # 2. Query Handling
   - Input: User’s natural language query.
   - Steps:
     1. Convert Query to Embeddings:  
        - Use the same pre-trained embedding model to convert the user’s query into a vector representation.
     2. Similarity Search:  
        - Perform a similarity search in the vector database to retrieve the most relevant content chunks.  
        - Use similarity metrics like cosine similarity or Euclidean distance.

     3. Pass to LLM:  
        - Combine the retrieved content chunks with a system prompt.  
        - Send the input to a Large Language Model (LLM) such as OpenAI GPT-4 or Hugging Face models.

 # 3. Response Generation
   - Input: Relevant content chunks and user query.
   - Steps:
     1. Prompt LLM with Context:  
        - Format the retrieved chunks and the user query as a prompt to the LLM.  
        - Ensure the prompt includes clear instructions to generate accurate and context-rich responses.
     2. Generate Response:  
        - Use the LLM to produce a response that integrates factual data from the retrieved chunks.  
        - Post-process the response to refine the format if needed (e.g., remove irrelevant content).

 # Tools and Technologies to Use
   1. Web Scraping:
      - `BeautifulSoup`, `Scrapy`, `requests`
   2. Text Processing & Chunking:
      - `nltk`, `spacy`, `langchain`
   3. Vector Embedding:
      - `sentence-transformers`, `OpenAI embeddings API`
   4. Vector Database:
      - `Pinecone`, `FAISS`, `Weaviate`, or `ChromaDB`
   5. Large Language Model (LLM):
      - OpenAI GPT-4 API or Hugging Face Transformers  
   6. Backend & Integration:
      - Python, `Flask`/`FastAPI`, `streamlit` (for simple UI)



 Example Steps to Build

1. Set Up Environment:

   Install necessary libraries using pip:
   
pip install beautifulsoup4 requests sentence-transformers pinecone-client openai
   

2. Implement Web Scraper:

Use `requests` and `BeautifulSoup` to scrape content from target URLs.

3. Embed the Data:

Convert scraped text into embeddings using `sentence-transformers`.

4. Store Embeddings in Pinecone:

Push the vector embeddings into Pinecone with their associated metadata.

5. Handle User Queries:
Accept queries via an input field or API.
Perform a similarity search using Pinecone and send the retrieved data to the LLM.

6. Generate and Return Response:
Use GPT-4 (or Hugging Face models) to generate a context-rich response.
