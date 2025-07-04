# Construct AI: Intelligent Document Summarization for Large-Scale Content

## Overview

Construct AI is a sophisticated Python-based system designed to overcome one of the most significant practical limitations of modern open-source Large Language Models (LLMs): their finite context windows. This project enables deep, context-aware summarization of extensive documents (text, PDFs, and Google Docs) that would otherwise be too large for a standard LLM to process in a single pass.

This tool is not just a summarizer; it's an intelligent content processor. It employs a multi-pass, hierarchical strategy to first understand the document's structure and then build a comprehensive, detailed summary, effectively giving the LLM "unlimited" context.

## The Problem: The Context Window Bottleneck

The AI landscape is rich with powerful, locally-runnable open-source LLMs like Llama 3, Mistral, and Gemma. While these models are incredibly capable, they share a fundamental constraint: the context window. This is the maximum amount of text (tokens) the model can "remember" or consider at one time. For most models, this ranges from 4,000 to 32,000 tokens.

What does this mean in practice?

Imagine you're a developer and you want an LLM to help you understand a large, unfamiliar codebase.

### The Goal

You want to ask the LLM, "What are the core modules in this project, how do they interact, and what is the main business logic?"

### The Problem

The entire codebase is 200,000 tokens. A standard LLM with an 8,000-token context window can't see the whole picture.

### The Failure Case

You could feed it small pieces of the code, but the model would lack the overarching context. Asking it about the "main business logic" would be impossible because it can't see how module-A.py connects to module-Z.py. You would get fragmented, out-of-context, and often incorrect answers.

This limitation applies to any large-scale document: analyzing a 100-page financial report, summarizing a dense academic paper, or understanding the full transcript of a series of user interviews. The LLM's inability to see the entire context at once makes it ineffective for these critical, real-world tasks.

## The Solution: Hierarchical & Incremental Summarization

Construct AI solves this problem by acting as an intelligent "executive function" for the LLM. Instead of naively feeding a massive document to the model, it uses a strategic, multi-pass approach that mirrors how a human would tackle the same task.

### How does it work conceptually?

Let's take the example of the 100-page financial report (e.g., 150,000 tokens).

#### Strategic Analysis (Instead of Brute Force)

The system first analyzes the document's size and determines the best strategy. It recognizes that the document is too large for a single pass.

##### First Pass - Structure Extraction

The system sends the entire document to the LLM with a specific, targeted prompt: "Read this entire document and extract its high-level structure. Identify the main sections, subsections, and key themes. Do not summarize yet, just give me the outline." This is a crucial first step. The LLM creates a "table of contents" or a structural map of the document. This map is small enough to fit in any context window.

##### Second Pass - Incremental, Context-Aware Summarization

Now, the system processes the document in intelligent, overlapping chunks.

- **Chunk 1:** It sends the first 10,000 tokens to the LLM along with the structural map and asks, "Summarize this chunk, keeping in mind it belongs to the 'Q2 Financial Performance' section of the overall report."
- **Chunk 2:** It takes the next 10,000 tokens. It sends this new chunk to the LLM along with the structural map AND the summary of the previous chunk. The prompt is now: "Here is the summary of what we've covered so far. Now, summarize this new text and integrate it into the existing summary, following the overall document structure."

This process continues iteratively. With each step, the LLM builds upon the previous summary, always guided by the high-level structure extracted in the first pass. The final output is a single, coherent, and comprehensive summary that reflects an understanding of the entire document, not just isolated parts.

## Key Features

- **Dynamic Strategy Selection:** Automatically analyzes document length and chooses the appropriate summarization strategy (simple, detailed, or hierarchical).
- **Multi-Format Support:** Ingests and processes .txt, .pdf, and Google Docs files seamlessly.
- **Hierarchical Structure Extraction:** Performs a dedicated first pass to understand the document's layout and key themes before summarization begins.
- **Incremental Summarization:** Processes large documents in overlapping chunks, feeding the summary of previous chunks back into the model to maintain context.
- **Intelligent Prompt Engineering:** Utilizes highly detailed, role-based prompts that guide the LLM to perform specific tasks (e.g., "You are a document structure analyst," "You are a professional document summarizer").
- **Local First:** Designed to work with locally-hosted LLMs via an Ollama-compatible API endpoint.
- **Automated File Discovery:** Scans specified directories and automatically processes new documents that don't yet have a summary.

## How It Works: A Technical Look

### Configuration

The user specifies the target folders to watch and the local LLM server details (BASE_URL, MODEL_NAME).

### File Discovery

The `discover_files_to_process()` function scans the target directories, comparing them against the ai summaries output folder to find new or unprocessed documents.

### Content Ingestion

The `read_file_content()` function dynamically reads the content from TXT, PDF, or Google Docs (using the Google Docs API for the latter).

### Strategy Selection

The `get_strategy_config()` function counts the tokens in the document and returns a `SummaryConfig` object based on predefined thresholds (SHORT_DOC_THRESHOLD, MEDIUM_DOC_THRESHOLD, etc.). This config determines if a simple, one-pass summary is sufficient or if the full hierarchical process is needed.

### Structure Pass

If required, `extract_structure()` is called. It uses a specialized prompt from `build_structure_prompt()` to ask the LLM to return only the document's outline.

### Summarization Pass

The `create_summary()` function orchestrates the final summary.

- For short documents, it performs a single-pass summary.
- For long documents, it uses `chunk_text()` to split the document into overlapping segments. It then iterates through these chunks, calling the LLM with a prompt from `build_summary_prompt()` that includes the document structure, the new chunk of text, and the running summary from previous chunks.

### Output

The final, consolidated summary is saved to a .txt file in the ai summaries directory, mirroring the original folder structure.

## Getting Started

### Prerequisites

- Python 3.x
- An Ollama-compatible LLM server running.
- Required Python packages: `requests`, `PyPDF2`, `google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib`.

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/construct-ai.git
cd construct-ai

### Prerequisites

- Python 3.x
- An Ollama-compatible LLM server running.
- Required Python packages: `requests`, `PyPDF2`, `google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib`.

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/construct-ai.git
cd construct-ai
```

Install the dependencies:


pip install -r requirements.txt



Usage
Configure the Script

Open construct_ai.py and set the following variables:

FOLDERS_TO_PROCESS: A list of directories you want the script to scan (e.g., ["my_research_papers", "project_docs"]). Leave empty to scan everything.
BASE_URL: The URL of your local LLM server (default is http://127.0.0.1:11434).

Run the Script

```bash
python construct_ai.py
```

The script will automatically find new documents, process them, and save the summaries in the ai summaries/ directory.