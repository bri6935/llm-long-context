# =============================================================================
# FOLDERS TO PROCESS CONFIGURATION
# =============================================================================
# List the folders you want to process (including all their subfolders)
# Use relative paths from the current working directory. Leave empty to process all folders.
# Example: ["documents", "research/papers", "data/reports"]
FOLDERS_TO_PROCESS = [
    "AI Ethics"
]

# =============================================================================
# SERVER CONFIGURATION - CHANGE THESE AS NEEDED
# =============================================================================
BASE_URL = "http://127.0.0.1:11434"
MODEL_NAME = "gemma3:12b-it-qat"

# =============================================================================
# IMPORTS
# =============================================================================
import os
import re
import pandas as pd
from datetime import datetime
import json
import requests
import glob
import PyPDF2
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

# --- Google API Imports ---
# Make sure to install the required libraries:
# pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# =============================================================================
# PROMPT DEFINITIONS (Integrated from prompts.py)
# =============================================================================
@dataclass
class SummaryConfig:
    document_type: str
    detail_level: str
    max_sections: int
    summary_length: str
    needs_structure: bool
    use_incremental: bool
    chunk_size: int
    overlap_size: int

def build_structure_prompt(config: SummaryConfig) -> str:
    base_instruction = """You are a document structure analyst. Your job is to carefully read the document and extract its organizational structure, key themes, and logical flow.

CRITICAL INSTRUCTIONS:
1. Read the ENTIRE document before creating any structure
2. Identify the main sections, subsections, and their relationships
3. Note key themes, arguments, and supporting evidence
4. Present the structure in a clear, hierarchical format
5. Use consistent formatting and numbering

"""
    
    if config.document_type == "short":
        return f"""{base_instruction}

TASK: Create a simple outline for a short document with up to {config.max_sections} main points.

FORMAT REQUIRED:
I. [Main Topic 1]
   - Key detail or subtopic
   - Key detail or subtopic

II. [Main Topic 2]
   - Key detail or subtopic
   - Key detail or subtopic

III. [Main Topic 3]
   - Key detail or subtopic
   - Key detail or subtopic

EXAMPLE OUTPUT:
I. Introduction to Machine Learning
   - Definition and basic concepts
   - Historical development

II. Types of Machine Learning
   - Supervised learning examples
   - Unsupervised learning applications

III. Practical Applications
   - Business use cases
   - Future trends

YOUR ANALYSIS MUST:
- Identify exactly {config.max_sections} main sections
- Include 2-3 key details under each section
- Use Roman numerals for main sections
- Use bullet points for subsections
- Keep each point concise but descriptive

=== DOCUMENT TO ANALYZE ===
Analyze the following document and extract its structure according to all requirements above:

"""

    elif config.document_type == "medium":
        return f"""{base_instruction}

TASK: Create a detailed outline for a medium-length document with up to {config.max_sections} main sections.

FORMAT REQUIRED:
I. [Main Section Title]
   A. [Subsection Title]
      - Key point or detail
      - Key point or detail
   B. [Subsection Title]
      - Key point or detail
      - Key point or detail

II. [Main Section Title]
   A. [Subsection Title]
      - Key point or detail
      - Key point or detail
   B. [Subsection Title]
      - Key point or detail
      - Key point or detail

THEMES & ARGUMENTS:
- Main Theme 1: [Description]
- Main Theme 2: [Description]
- Primary Argument: [Statement]
- Supporting Evidence: [Key evidence points]

EXAMPLE OUTPUT:
I. Problem Definition and Scope
   A. Current State Analysis
      - Market conditions and challenges
      - Existing solution limitations
   B. Stakeholder Impact
      - Customer pain points
      - Business implications

II. Proposed Solution Framework
   A. Technical Architecture
      - Core components and design
      - Integration requirements
   B. Implementation Strategy
      - Phased rollout plan
      - Resource requirements

THEMES & ARGUMENTS:
- Main Theme 1: Digital transformation necessity
- Main Theme 2: Risk mitigation strategies
- Primary Argument: Current systems inadequate for future needs
- Supporting Evidence: Performance metrics, user feedback, competitive analysis

YOUR ANALYSIS MUST:
- Create exactly {config.max_sections} main sections with Roman numerals
- Include 2-3 subsections per main section with capital letters
- Add 2-4 key points under each subsection
- Identify major themes and arguments separately
- Use consistent hierarchical formatting
- Extract specific evidence and examples mentioned

=== DOCUMENT TO ANALYZE ===
Analyze the following document and extract its structure according to all requirements above:

"""

    else:
        return f"""{base_instruction}

TASK: Create a comprehensive hierarchical outline for a long document with up to {config.max_sections} main sections.

FORMAT REQUIRED:
I. [Main Section Title]
   A. [Major Subsection]
      1. [Minor subsection]
         a. Specific detail or evidence
         b. Specific detail or evidence
      2. [Minor subsection]
         a. Specific detail or evidence
         b. Specific detail or evidence
   B. [Major Subsection]
      1. [Minor subsection]
         a. Specific detail or evidence
         b. Specific detail or evidence

DOCUMENT ANALYSIS:
- Document Type: [Academic paper/Report/Manual/etc.]
- Primary Purpose: [What the document aims to achieve]
- Target Audience: [Who this is written for]
- Methodology: [How information is presented]

THEMES & ARGUMENTS:
- Central Thesis: [Main argument or purpose]
- Supporting Arguments:
  1. [Argument 1 with evidence]
  2. [Argument 2 with evidence]
  3. [Argument 3 with evidence]
- Key Evidence Types: [Data, case studies, research, etc.]
- Counterarguments Addressed: [If any]

RELATIONSHIPS & FLOW:
- Section Dependencies: [How sections build on each other]
- Logical Progression: [How ideas develop through document]
- Cross-References: [Key connections between sections]

EXAMPLE OUTPUT:
I. Executive Summary and Strategic Context
   A. Current Market Position
      1. Competitive Landscape Analysis
         a. Direct competitors and market share data
         b. Emerging threats and opportunities
      2. Internal Capability Assessment
         a. Strengths in technology and talent
         b. Resource constraints and gaps
   B. Strategic Imperatives
      1. Short-term Objectives (6-12 months)
         a. Revenue growth targets and initiatives
         b. Operational efficiency improvements
      2. Long-term Vision (2-3 years)
         a. Market expansion strategies
         b. Innovation and R&D investments

DOCUMENT ANALYSIS:
- Document Type: Strategic business plan
- Primary Purpose: Guide organizational decision-making and resource allocation
- Target Audience: Executive leadership and board members
- Methodology: Data-driven analysis with financial projections

THEMES & ARGUMENTS:
- Central Thesis: Company must transform digitally to maintain competitive advantage
- Supporting Arguments:
  1. Market data shows 40% shift to digital platforms
  2. Customer survey indicates demand for mobile solutions
  3. Competitor analysis reveals technology gaps
- Key Evidence Types: Financial data, market research, customer feedback
- Counterarguments Addressed: Cost concerns and implementation risks

YOUR ANALYSIS MUST:
- Create exactly {config.max_sections} main sections with full hierarchy
- Use Roman numerals → Capital letters → Numbers → Lowercase letters
- Include document type analysis and purpose identification
- Extract all major themes, arguments, and evidence
- Map relationships between sections
- Note logical flow and dependencies
- Identify methodology and target audience
- Be thorough but organized in presentation

=== DOCUMENT TO ANALYZE ===
Analyze the following document and extract its structure according to all requirements above:

"""

def build_summary_prompt(config: SummaryConfig, structure: str = "", previous_summary: str = "") -> str:
    
    context_section = ""
    if structure:
        context_section += f"\n\nDOCUMENT STRUCTURE TO FOLLOW:\n{structure}"
    if previous_summary:
        context_section += f"\n\nPREVIOUS CONTENT ALREADY SUMMARIZED:\n{previous_summary}\n\nIMPORTANT: Build upon this previous summary by adding new information from the current text section. Do not repeat information already covered."

    if config.document_type == "short":
        return f"""You are a professional document summarizer. Create a comprehensive, detailed summary that captures ALL essential information, examples, data, and insights from the document.

CRITICAL REQUIREMENTS - READ CAREFULLY:
1. COMPREHENSIVE COVERAGE: Include every important detail, example, statistic, quote, and insight
2. PRESERVE SPECIFICITY: Maintain all specific numbers, dates, names, technical terms, and examples
3. DETAILED EXPLANATIONS: Don't just list points - explain the significance and context
4. COMPLETE CONTEXT: Include background information and reasoning behind conclusions
5. ACTIONABLE INSIGHTS: Extract and elaborate on practical implications and recommendations

FORMAT TEMPLATE:
**[DOCUMENT TITLE/MAIN TOPIC]**

**Executive Overview:** [4-6 sentences providing comprehensive explanation of the document's purpose, methodology, scope, key findings, and overall significance. Include specific context about why this document matters and what makes it unique or important.]

**Detailed Content Analysis:**

**[Section 1 Title]**
[Write 3-5 detailed paragraphs covering this section. Include:]
- All key concepts with full explanations and context
- Specific examples, case studies, or scenarios mentioned
- Any data, statistics, percentages, or measurements
- Direct quotes or important statements
- Technical details and methodologies
- Connections to broader themes or implications
- Step-by-step processes or procedures if applicable

**[Section 2 Title]**
[Write 3-5 detailed paragraphs covering this section. Include:]
- Complete explanation of all main points
- Supporting evidence and reasoning
- Specific examples and real-world applications
- Any challenges, limitations, or considerations mentioned
- Detailed analysis of cause and effect relationships
- Quantitative and qualitative data presented

**[Continue for all major topics...]**

**Key Insights and Detailed Implications:**
- **[Insight 1]:** [Provide 2-3 sentences explaining the insight, its significance, supporting evidence, and practical implications]
- **[Insight 2]:** [Comprehensive explanation including context, evidence, and real-world applications]
- **[Insight 3]:** [Detailed analysis of the insight's importance and how it connects to broader themes]
- **[Insight 4]:** [Full explanation with supporting details and actionable implications]

**Comprehensive Conclusions and Recommendations:**
[Write 2-3 detailed paragraphs that:]
- Synthesize all major findings and their interconnections
- Provide specific, actionable recommendations with implementation details
- Discuss future implications and potential developments
- Address any limitations or areas for further investigation
- Connect conclusions back to the document's original purpose

**Supporting Evidence Summary:**
[List all key data points, statistics, research findings, expert opinions, case studies, and empirical evidence mentioned in the document]

QUALITY REQUIREMENTS:
- Summary must be 800-1500 words minimum
- Include ALL specific details: numbers, percentages, dates, names, technical terms
- Preserve ALL examples, case studies, and scenarios
- Maintain ALL quotes and important statements
- Explain the significance and context of every major point
- Use clear, professional language with appropriate technical detail
- Ensure every paragraph adds substantive value and new information
- Connect ideas and show relationships between concepts{context_section}

=== DOCUMENT TO ANALYZE ===
Analyze and summarize the following document according to all requirements above:

"""

    elif config.document_type == "medium":
        return f"""You are a professional document summarizer. Create an extensive, highly detailed summary that comprehensively covers all sections, arguments, evidence, examples, and insights while maintaining the document's logical structure and analytical depth.

CRITICAL REQUIREMENTS - MAXIMUM DETAIL EXPECTED:
1. EXHAUSTIVE COVERAGE: Include every significant point, argument, example, data point, and insight
2. PRESERVE ALL SPECIFICITY: Maintain every number, statistic, quote, technical term, methodology, and example
3. COMPREHENSIVE EXPLANATIONS: Provide full context, background, and significance for all major points
4. DETAILED EVIDENCE: Include all supporting research, case studies, expert opinions, and empirical data
5. THOROUGH ANALYSIS: Explain reasoning, implications, connections, and broader significance
6. COMPLETE METHODOLOGY: Describe all processes, procedures, frameworks, and analytical approaches

FORMAT TEMPLATE:
**[DOCUMENT TITLE] - COMPREHENSIVE DETAILED SUMMARY**

**Executive Overview:** [6-8 sentences providing complete explanation of the document's purpose, scope, methodology, key arguments, major findings, implications, and significance. Include specific context about the document's contribution to its field and why it matters.]

**Document Context and Methodology:**
[Detailed paragraph explaining the document's background, objectives, target audience, analytical framework, data sources, and methodological approach. Include any limitations or scope considerations mentioned.]

**COMPREHENSIVE SECTION-BY-SECTION ANALYSIS:**

**[Section 1 Title from Structure]**
[Write 4-6 detailed paragraphs covering every aspect of this section:]

*Key Concepts and Definitions:*
[Comprehensive explanation of all main concepts, including technical definitions, theoretical frameworks, and contextual background]

*Detailed Analysis and Arguments:*
[Thorough coverage of all arguments presented, including supporting logic, evidence, and reasoning. Include any counterarguments or alternative perspectives discussed]

*Specific Evidence and Examples:*
[Complete documentation of all case studies, examples, data points, statistics, research findings, expert opinions, and empirical evidence presented]

*Methodological Details:*
[Full explanation of any processes, procedures, analytical methods, or frameworks used]

*Implications and Significance:*
[Detailed analysis of what this section means for the broader document objectives and real-world applications]

**[Section 2 Title from Structure]**
[Follow the same comprehensive format as Section 1, ensuring complete coverage of:]
- All theoretical and practical concepts
- Every piece of supporting evidence and data
- Complete explanation of methodologies and processes
- Detailed analysis of arguments and reasoning
- Comprehensive coverage of examples and applications
- Full discussion of implications and significance

**[Continue this detailed format for ALL major sections identified in the structure...]**

**Comprehensive Thematic Analysis:**

**Central Arguments and Thesis:**
[Detailed explanation of the document's main thesis, including how it's developed, supported, and defended throughout the document]

**Supporting Framework and Evidence Base:**
- **Primary Evidence Category 1:** [Detailed description of evidence type, specific examples, quality assessment, and how it supports main arguments]
- **Primary Evidence Category 2:** [Comprehensive analysis including methodology, findings, limitations, and significance]
- **Primary Evidence Category 3:** [Complete coverage of all supporting data, research, and expert opinions]

**Methodological Analysis:**
[Detailed examination of all analytical approaches, research methods, data collection techniques, and evaluation frameworks used]

**CRITICAL INSIGHTS AND COMPREHENSIVE IMPLICATIONS:**

**Major Findings:**
1. **[Finding 1]:** [Provide 4-5 sentences explaining the finding in detail, including supporting evidence, methodology used to reach this conclusion, significance for the field, and practical implications]
2. **[Finding 2]:** [Comprehensive explanation including context, supporting data, analytical process, limitations, and broader significance]
3. **[Finding 3]:** [Detailed analysis covering background, evidence base, implications, and connections to other findings]
4. **[Finding 4]:** [Complete coverage including methodology, results, interpretation, and real-world applications]

**Strategic and Practical Implications:**
- **[Implication Area 1]:** [Detailed 3-4 sentence explanation of specific implications, including who is affected, how they should respond, timeline considerations, and potential outcomes]
- **[Implication Area 2]:** [Comprehensive analysis of practical applications, implementation considerations, resource requirements, and expected benefits]
- **[Implication Area 3]:** [Thorough discussion of long-term implications, strategic considerations, and broader industry or field impacts]

**Detailed Recommendations and Action Items:**
[Comprehensive list of all recommendations with implementation details, resource requirements, timeline considerations, success metrics, and potential challenges]

**COMPREHENSIVE CONCLUSIONS AND FUTURE DIRECTIONS:**
[Write 3-4 detailed paragraphs that:]
- Synthesize all major findings and demonstrate their interconnections
- Provide specific, actionable recommendations with detailed implementation guidance
- Discuss comprehensive future implications and potential developments
- Address all limitations, constraints, and areas requiring further investigation
- Connect all conclusions back to the document's original objectives and broader significance

**Complete Evidence and Data Summary:**
[Comprehensive listing of ALL quantitative data, statistics, research findings, expert opinions, case studies, examples, and empirical evidence presented, organized by category and significance]

QUALITY REQUIREMENTS:
- Summary must be 1500-3000 words minimum
- Cover every major section proportionally and comprehensively
- Include ALL specific details: numbers, percentages, dates, names, technical terms, methodologies
- Preserve ALL examples, case studies, scenarios, and applications
- Maintain ALL quotes, expert opinions, and important statements
- Explain the full significance and context of every major point
- Show detailed logical connections between sections and ideas
- Use professional, technical language appropriate to the source material
- Ensure every paragraph provides substantial new information and analysis
- Demonstrate deep understanding of the document's contribution and significance{context_section}

=== DOCUMENT TO ANALYZE ===
Analyze and summarize the following document according to all requirements above:

"""

    else:
        return f"""You are an expert document analyst creating an exhaustive, comprehensive summary that preserves the full scope, hierarchical organization, analytical depth, and scholarly rigor of a complex document.

CRITICAL REQUIREMENTS - MAXIMUM SCHOLARLY DETAIL:
1. COMPLETE COMPREHENSIVE COVERAGE: Include every significant argument, piece of evidence, methodology, example, and insight
2. PRESERVE ALL TECHNICAL DETAIL: Maintain every statistic, formula, technical term, process, methodology, and specialized knowledge
3. EXHAUSTIVE ANALYSIS: Provide complete context, background, theoretical framework, and analytical significance
4. COMPREHENSIVE EVIDENCE DOCUMENTATION: Include all research findings, case studies, expert opinions, empirical data, and supporting materials
5. THOROUGH METHODOLOGICAL ANALYSIS: Describe all analytical approaches, research methods, evaluation frameworks, and validation processes
6. COMPLETE SCHOLARLY CONTEXT: Explain theoretical foundations, literature connections, and contribution to the field

FORMAT TEMPLATE:
**[DOCUMENT TITLE] - EXHAUSTIVE SCHOLARLY ANALYSIS AND SUMMARY**

**Comprehensive Executive Summary:** [8-10 sentences providing complete overview of the document's purpose, theoretical framework, methodology, scope, key arguments, major findings, contributions to the field, limitations, and overall significance. Include specific context about how this work advances knowledge and why it represents an important contribution.]

**Document Context and Scholarly Framework:**
- **Document Classification:** [Detailed explanation of document type, academic discipline, theoretical tradition, and methodological approach]
- **Primary Objectives:** [Comprehensive statement of all research questions, hypotheses, and analytical goals]
- **Theoretical Foundation:** [Complete explanation of underlying theories, conceptual frameworks, and scholarly traditions]
- **Methodological Approach:** [Detailed description of all analytical methods, data collection techniques, validation processes, and quality assurance measures]
- **Scope and Limitations:** [Thorough discussion of what is and isn't covered, methodological constraints, and acknowledged limitations]

**EXHAUSTIVE SECTION-BY-SECTION ANALYSIS:**

**I. [Main Section 1 Title]**

**Conceptual Framework and Theoretical Foundation:**
[Comprehensive explanation of all theoretical concepts, definitions, frameworks, and scholarly context that underlies this section]

*[Subsection A Title]:*
[Write 4-5 detailed paragraphs providing exhaustive coverage including:]
- Complete explanation of all concepts, theories, and analytical frameworks
- Detailed description of methodology, processes, and analytical approaches
- Comprehensive documentation of all evidence, data, examples, and case studies
- Thorough analysis of arguments, reasoning, and logical development
- Complete discussion of implications, significance, and connections to broader themes
- Detailed examination of any challenges, limitations, or alternative perspectives

*[Subsection B Title]:*
[Write 4-5 detailed paragraphs with comprehensive coverage of:]
- All technical details, specifications, and methodological considerations
- Complete evidence base including quantitative and qualitative data
- Exhaustive analysis of findings, interpretations, and significance
- Detailed discussion of practical applications and theoretical implications
- Comprehensive examination of relationships to other sections and broader document objectives

**II. [Main Section 2 Title]**

**Analytical Framework and Evidence Base:**
[Detailed explanation of the analytical approach, evidence evaluation criteria, and methodological considerations specific to this section]

*[Subsection A Title]:*
[Provide comprehensive 4-5 paragraph analysis covering all aspects of content, methodology, evidence, and implications]

*[Subsection B Title]:*
[Complete detailed coverage following the same exhaustive approach]

**[Continue this comprehensive pattern for ALL major sections, ensuring each receives proportional detailed treatment...]**

**COMPREHENSIVE THEMATIC AND ANALYTICAL SYNTHESIS:**

**Central Thesis and Argument Structure:**
[Detailed explanation of the document's main thesis, including theoretical foundation, evidence base, argument development, logical structure, and defense against potential counterarguments]

**Complete Evidence Framework:**
- **Primary Research Evidence:** [Exhaustive description of all original research, including methodology, sample sizes, analytical techniques, findings, statistical significance, and limitations]
- **Secondary Source Analysis:** [Comprehensive coverage of all cited works, expert opinions, scholarly consensus, and literature review findings]
- **Empirical Data and Case Studies:** [Detailed documentation of all quantitative data, qualitative findings, case study methodology, and analytical results]
- **Theoretical and Conceptual Evidence:** [Complete explanation of theoretical support, conceptual frameworks, and scholarly foundations]

**Comprehensive Methodological Analysis:**
[Exhaustive examination of all research methods, analytical techniques, data collection procedures, validation processes, quality assurance measures, and methodological innovations or contributions]

**EXHAUSTIVE FINDINGS AND IMPLICATIONS ANALYSIS:**

**Major Research Findings:**
1. **[Primary Finding 1]:** [Provide comprehensive 6-8 sentence analysis including the finding's theoretical significance, methodological foundation, supporting evidence, statistical or analytical validation, implications for existing knowledge, practical applications, limitations, and suggestions for future research]

2. **[Primary Finding 2]:** [Complete detailed explanation covering background context, analytical methodology, evidence base, significance for theory and practice, connections to other findings, broader implications, and future research directions]

3. **[Primary Finding 3]:** [Exhaustive analysis including theoretical framework, empirical support, methodological validation, practical significance, scholarly contribution, limitations, and recommendations]

4. **[Continue for all major findings with equivalent comprehensive detail...]**

**Strategic and Scholarly Implications:**
- **Theoretical Contributions:** [Detailed 4-5 sentence explanation of how findings advance theoretical understanding, challenge existing paradigms, suggest new conceptual frameworks, and contribute to scholarly knowledge]
- **Methodological Innovations:** [Comprehensive analysis of methodological contributions, analytical innovations, and improvements to research practices]
- **Practical Applications:** [Exhaustive discussion of real-world applications, implementation considerations, stakeholder implications, and practical benefits]
- **Policy and Strategic Implications:** [Detailed analysis of policy recommendations, strategic considerations, and institutional implications]

**Complete Recommendations Framework:**
[Comprehensive presentation of all recommendations organized by category (theoretical, methodological, practical, policy), including detailed implementation guidance, resource requirements, timeline considerations, success metrics, potential challenges, and evaluation frameworks]

**COMPREHENSIVE CONCLUSIONS AND SCHOLARLY CONTRIBUTION:**
[Write 4-6 detailed paragraphs that:]
- Synthesize all major findings and demonstrate their theoretical and practical interconnections
- Explain the document's complete contribution to existing knowledge and scholarly understanding
- Provide detailed discussion of how findings advance the field and suggest new research directions
- Address all methodological contributions and innovations
- Discuss comprehensive implications for theory, practice, and policy
- Acknowledge all limitations and suggest specific areas for future investigation
- Connect all conclusions to the broader scholarly context and significance

**Complete Research and Evidence Appendix:**
[Exhaustive organizational listing of ALL research findings, statistical data, case study details, expert opinions, theoretical references, methodological innovations, and empirical evidence, categorized by type, significance, and methodological approach]

**Future Research Framework:**
[Detailed suggestions for future research including specific research questions, methodological approaches, theoretical frameworks, and expected contributions to knowledge]

QUALITY REQUIREMENTS:
- Summary must be 3000-5000 words minimum for comprehensive coverage
- Every major section and subsection must receive detailed proportional treatment
- Include ALL technical details, methodologies, statistical data, and specialized knowledge
- Preserve ALL examples, case studies, research findings, and expert opinions
- Maintain ALL theoretical frameworks, analytical methods, and scholarly contributions
- Explain complete significance and context of every major finding and argument
- Demonstrate deep scholarly understanding of theoretical foundations and methodological rigor
- Show detailed connections between all sections, arguments, and findings
- Use appropriate academic and technical language reflecting the source material's scholarly level
- Ensure every paragraph contributes substantial analytical value and advances understanding
- Provide comprehensive treatment worthy of the document's scholarly contribution and complexity{context_section}

=== DOCUMENT TO ANALYZE ===
Analyze and summarize the following document according to all requirements above:

"""

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
# TOKEN THRESHOLDS
SHORT_DOC_THRESHOLD = 10000
MEDIUM_DOC_THRESHOLD = 15000
LONG_DOC_THRESHOLD = 20000

# DIRECTORIES
AI_SUMMARIES_DIR = "ai summaries"

# GOOGLE API CONFIGURATION
SCOPES = [
    'https://www.googleapis.com/auth/documents',       # To read and write content inside docs
    'https://www.googleapis.com/auth/drive.file'       # To rename files and manage them
]
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'


# =============================================================================
# GOOGLE DOCS API FUNCTIONS
# =============================================================================

def get_docs_service() -> Optional[Any]:
    """
    Handles Google API authentication flow and returns a service object
    for the Google Docs API.
    """
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            print(f"Error loading {TOKEN_FILE}: {e}. Will try to re-authenticate.")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing token: {e}. Please re-authenticate.")
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE)
                return get_docs_service()
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"ERROR: Google API credentials file ('{CREDENTIALS_FILE}') not found.")
                print("Please enable the Google Docs API, create OAuth 2.0 credentials for a Desktop App,")
                print(f"and download and save the file as '{CREDENTIALS_FILE}' in this directory.")
                return None
            
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                print(f"Failed to run authentication flow: {e}")
                return None

        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('docs', 'v1', credentials=creds)
        print("Google Docs API service successfully created.")
        return service
    except HttpError as err:
        print(f"An error occurred building the service: {err}")
        return None

def read_gdoc_file(file_path: str, service: Any) -> str:
    """Reads content from a .gdoc file by fetching it from Google Docs API"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                gdoc_data = json.load(file)
                doc_id = gdoc_data.get('doc_id')
            except json.JSONDecodeError:
                raise Exception(f"Could not decode JSON from {file_path}.")

        if not doc_id:
            raise Exception("'.gdoc' file does not contain a 'doc_id'.")

        print(f"Fetching Google Doc with ID: {doc_id}")
        document = service.documents().get(documentId=doc_id).execute()
        body = document.get('body', {})
        content = body.get('content', [])

        text_content = ""
        for value in content:
            if 'paragraph' in value:
                elements = value.get('paragraph').get('elements', [])
                for elem in elements:
                    if 'textRun' in elem:
                        text_run = elem.get('textRun', {})
                        text_content += text_run.get('content', '')
        
        if not text_content.strip():
            print(f"Warning: Fetched Google Doc '{document.get('title', doc_id)}' appears to be empty.")

        return text_content.strip()

    except HttpError as err:
        if err.resp.status == 404:
            raise Exception(f"Google Doc with ID '{doc_id}' not found.")
        elif err.resp.status == 403:
            raise Exception(f"Permission denied for Google Doc with ID '{doc_id}'. Ensure you have access and the Google Docs API is enabled.")
        else:
            raise Exception(f"Error fetching Google Doc: {err}")
    except Exception as e:
        raise Exception(f"Error reading .gdoc file: {e}")

# =============================================================================
# CORE SUMMARIZATION FUNCTIONS
# =============================================================================

def get_available_models(base_url: str) -> List[str]:
    try:
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return [model['name'] for model in data.get('models', [])]
    except requests.RequestException as e:
        print(f"Error getting models: {e}")
        return []

def get_llm_response(base_url: str, model_name: Optional[str], prompt: str, images: Optional[List[str]] = None) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt, "images": images}
        ],
        "model": model_name,
        "options": {
            "num_ctx": 128000,
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
            "min_p": 0.0
        },
        "stream": True
    }

    response = requests.post(f"{base_url}/api/chat", json=payload)
    response.raise_for_status()

    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_line = json.loads(line.decode('utf-8'))
                if "message" in json_line and "content" in json_line["message"]:
                    content = json_line["message"]["content"]
                    if content: 
                        full_response += content
            except json.JSONDecodeError: 
                continue

    return full_response

def read_pdf_file(file_path: str) -> str:
    """Read text content from a PDF file"""
    try:
        text_content = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n"
            
        return text_content.strip()
    except Exception as e:
        raise Exception(f"Error reading PDF file: {e}")

def read_text_file(file_path: str) -> str:
    """Read text content from a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Error reading text file: {e}")

def read_file_content(file_path: str, file_type: str, docs_service: Optional[Any] = None) -> str:
    """Read content from file based on file type"""
    file_type_lower = file_type.lower()
    if file_type_lower == 'pdf':
        return read_pdf_file(file_path)
    elif file_type_lower == 'txt':
        return read_text_file(file_path)
    elif file_type_lower == 'gdoc':
        if not docs_service:
            raise Exception("Google Docs service is not available for reading .gdoc files.")
        return read_gdoc_file(file_path, docs_service)
    else:
        raise Exception(f"Unsupported file type: {file_type}")

def count_tokens(text: str) -> int:
    # Simple token approximation: 1 token ≈ 4 characters
    return len(text) // 4

def get_strategy_config(token_count: int) -> SummaryConfig:
    if token_count <= SHORT_DOC_THRESHOLD:
        return SummaryConfig(
            document_type="short",
            detail_level="concise",
            max_sections=3,
            summary_length="brief",
            needs_structure=False,
            use_incremental=False,
            chunk_size=token_count,
            overlap_size=0
        )
    elif token_count <= MEDIUM_DOC_THRESHOLD:
        return SummaryConfig(
            document_type="medium",
            detail_level="detailed",
            max_sections=5,
            summary_length="comprehensive",
            needs_structure=True,
            use_incremental=True,
            chunk_size=8000,
            overlap_size=1000
        )
    else:
        return SummaryConfig(
            document_type="long",
            detail_level="hierarchical",
            max_sections=8,
            summary_length="extensive",
            needs_structure=True,
            use_incremental=True,
            chunk_size=10000,
            overlap_size=1500
        )

def extract_structure(text: str, config: SummaryConfig) -> str:
    if not config.needs_structure:
        return ""
    
    structure_prompt = build_structure_prompt(config)
    full_prompt = f"{structure_prompt}\n\nDocument to analyze:\n{text}"
    
    print(f"Extracting document structure...")
    structure = get_llm_response(BASE_URL, MODEL_NAME, full_prompt)
    
    print(f"\n{'='*60}")
    print("DOCUMENT STRUCTURE EXTRACTED:")
    print(f"{'='*60}")
    print(structure)
    print(f"{'='*60}\n")
    
    return structure

def chunk_text(text: str, chunk_size: int, overlap_size: int) -> List[str]:
    if chunk_size >= len(text):
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
            
        start = end - overlap_size
    
    return chunks

def create_summary(text: str, structure: str, config: SummaryConfig) -> str:
    if not config.use_incremental:
        # Simple summarization for short documents
        summary_prompt = build_summary_prompt(config, structure)
        full_prompt = f"{summary_prompt}\n\nDocument to summarize:\n{text}"
        return get_llm_response(BASE_URL, MODEL_NAME, full_prompt)
    
    # Convert token-based sizes to character-based sizes for chunking
    char_chunk_size = config.chunk_size * 4
    char_overlap_size = config.overlap_size * 4
    
    # Incremental summarization for longer documents
    chunks = chunk_text(text, char_chunk_size, char_overlap_size)
    running_summary = ""
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        
        summary_prompt = build_summary_prompt(config, structure, running_summary)
        full_prompt = f"{summary_prompt}\n\nText to process:\n{chunk}"
        
        if running_summary:
            full_prompt += f"\n\nPlease update and expand the previous summary with information from this new text section."
        else:
            full_prompt += f"\n\nThis is the first section. Create an initial summary."
        
        chunk_result = get_llm_response(BASE_URL, MODEL_NAME, full_prompt)
        
        print(f"\n{'-'*50}")
        print(f"CHUNK {i+1} SUMMARY RESULT:")
        print(f"{'-'*50}")
        print(chunk_result)
        print(f"{'-'*50}\n")
        
        running_summary = chunk_result
        
        # Compress summary if it's getting too long
        if count_tokens(running_summary) > 8000:
            print("Compressing summary (too long)...")
            compress_prompt = f"Compress this summary to focus on the most essential information while preserving key details:\n\n{running_summary}"
            running_summary = get_llm_response(BASE_URL, MODEL_NAME, compress_prompt)
            
            print(f"\n{'-'*50}")
            print("COMPRESSED SUMMARY:")
            print(f"{'-'*50}")
            print(running_summary)
            print(f"{'-'*50}\n")
    
    return running_summary

def summarize_document(text: str) -> str:
    print("Analyzing document...")
    
    token_count = count_tokens(text)
    config = get_strategy_config(token_count)
    
    print(f"Document length: {token_count} tokens")
    print(f"Using {config.document_type} document strategy")
    
    # Structure extraction pass
    structure = extract_structure(text, config)
    
    # Summary creation pass
    print("Creating summary...")
    summary = create_summary(text, structure, config)
    
    return summary

def ensure_ai_summaries_dir():
    """Ensure the AI summaries directory exists"""
    if not os.path.exists(AI_SUMMARIES_DIR):
        os.makedirs(AI_SUMMARIES_DIR)
        print(f"Created directory: {AI_SUMMARIES_DIR}")
    else:
        print(f"Using existing directory: {AI_SUMMARIES_DIR}")

def clean_ai_summaries():
    """
    Scans the 'ai summaries' directory, reads each .txt file, removes special characters
    (keeping letters, numbers, and basic punctuation), and overwrites the file.
    """
    print(f"\n--- Starting AI Summary Cleanup Utility ---")
    
    if not os.path.isdir(AI_SUMMARIES_DIR):
        print(f"Directory '{AI_SUMMARIES_DIR}' not found. Skipping cleanup.")
        print(f"--- AI Summary Cleanup Finished ---\n")
        return

    print(f"Scanning directory: '{AI_SUMMARIES_DIR}' for .txt files to clean.")
    
    cleaned_files_count = 0
    total_files_scanned = 0
    
    for root, _, files in os.walk(AI_SUMMARIES_DIR):
        for filename in files:
            if filename.endswith('.txt'):
                total_files_scanned += 1
                file_path = os.path.join(root, filename)
                
                print(f"  - Processing: {file_path}")
                try:
                    original_content = ""
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        original_content = f.read()
                    
                    cleaned_content = re.sub(r'[^a-zA-Z0-9.,!?\s]', '', original_content)
                    
                    if cleaned_content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(cleaned_content)
                        print(f"    -> Content cleaned and file overwritten.")
                        cleaned_files_count += 1
                    else:
                        print(f"    -> No changes needed.")

                except Exception as e:
                    print(f"    [ERROR] Could not process file {filename}: {e}")
    
    print(f"\nScanned {total_files_scanned} .txt file(s).")
    print(f"Cleanup complete. Cleaned and overwrote {cleaned_files_count} file(s).")
    print(f"--- AI Summary Cleanup Finished ---\n")

def discover_files_to_process(root_dir: str, target_folders: List[str]) -> List[str]:
    """
    Recursively finds all supported files (txt, pdf, gdoc) that need summarization
    by checking for the absence of a corresponding summary file.
    """
    supported_extensions = ('.txt', '.pdf', '.gdoc')
    files_found = []
    
    abs_summaries_dir = os.path.abspath(AI_SUMMARIES_DIR)
    
    # Create a list of absolute paths for the target folders for efficient checking
    abs_target_dirs = []
    if target_folders and any(tf.strip() for tf in target_folders):
        abs_target_dirs = [os.path.abspath(os.path.join(root_dir, f.strip())) for f in target_folders if f.strip()]
        print(f"Scanning only within specified target folders: {[f.strip() for f in target_folders if f.strip()]}")
    else:
        print("Scanning all subdirectories for processable files...")

    for root, _, files in os.walk(root_dir):
        abs_root = os.path.abspath(root)

        # 1. Skip the 'ai summaries' directory itself to avoid processing summaries
        if abs_root.startswith(abs_summaries_dir):
            continue

        # 2. If targets are specified, skip any directories not in the target list
        if abs_target_dirs:
            if not any(abs_root.startswith(target_dir) for target_dir in abs_target_dirs):
                continue
        
        # 3. Check files in the current valid directory
        for file_name in files:
            if file_name.lower().endswith(supported_extensions):
                files_found.append(os.path.join(root, file_name))

    if not files_found:
        return []

    print(f"\nDiscovered {len(files_found)} total supported files. Checking which ones are new...")
    files_to_process = []
    for source_path in files_found:
        # 4. Check if a summary file already exists for this source file
        source_dir = os.path.dirname(source_path)
        source_filename = os.path.basename(source_path)
        base_name = os.path.splitext(source_filename)[0]
        summary_filename = f"{base_name}_ai_summary.txt"
        
        relative_dir = os.path.relpath(source_dir, root_dir)
        
        target_summary_dir = os.path.join(abs_summaries_dir, relative_dir) if relative_dir != '.' else abs_summaries_dir
        summary_path = os.path.join(target_summary_dir, summary_filename)
        
        if not os.path.exists(summary_path):
            files_to_process.append(source_path)

    return files_to_process

def process_discovered_files():
    """
    New main processing function that discovers files directly without using a CSV.
    """
    print("Starting file discovery process...")
    
    # Discover files that need processing from the current working directory
    files_to_process = discover_files_to_process(os.getcwd(), FOLDERS_TO_PROCESS)
    
    if not files_to_process:
        print("\nNo new files to summarize at this time. All summaries are up to date.")
        return
        
    print(f"\nFound {len(files_to_process)} new file(s) to process.")
    
    ensure_ai_summaries_dir()
    
    docs_service = None
    # Check if any .gdoc files need processing before initializing the service
    if any(f.lower().endswith('.gdoc') for f in files_to_process):
        print("\nFound Google Doc files to process, initializing Google Docs API service...")
        docs_service = get_docs_service()
        if not docs_service:
            print("Failed to initialize Google Docs service. Skipping .gdoc files for this run.")

    for file_path in files_to_process:
        file_name = os.path.basename(file_path)
        file_type = os.path.splitext(file_name)[1].lstrip('.').lower()
        
        if file_type == 'gdoc' and not docs_service:
            print(f"Skipping .gdoc because Google service is unavailable: {file_name}")
            continue

        print(f"\n{'='*50}")
        print(f"Processing: {file_name} ({file_type.upper()})")
        print(f"Path: {file_path}")
        print(f"{'='*50}")
        
        try:
            # --- 1. Read file content ---
            try:
                content = read_file_content(file_path, file_type, docs_service=docs_service)
            except Exception as e:
                print(f"Error reading file: {e}")
                continue
            
            if not content.strip():
                print(f"Skipping empty file: {file_name}")
                continue
            
            # --- 2. Summarize the document ---
            summary = summarize_document(content)
            
            # --- 3. Construct the output path for the summary ---
            base_name = os.path.splitext(file_name)[0]
            summary_filename = f"{base_name}_ai_summary.txt"
            
            original_dir = os.path.dirname(file_path)
            
            # Get the directory relative to the CWD to replicate it inside "ai summaries"
            relative_dir = os.path.relpath(original_dir, os.getcwd())

            if relative_dir == '.':
                 target_summary_dir = os.path.abspath(AI_SUMMARIES_DIR)
            else:
                target_summary_dir = os.path.join(os.path.abspath(AI_SUMMARIES_DIR), relative_dir)
            
            os.makedirs(target_summary_dir, exist_ok=True)
            
            summary_path = os.path.join(target_summary_dir, summary_filename)

            # --- 4. Save the summary ---
            with open(summary_path, 'w', encoding='utf-8') as file:
                file.write(summary)
            
            print(f"Summary saved to: {summary_path}")
            
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_name}: {e}")
            continue
    
    print(f"\nAll processing complete!")

def main():
    print("Document Summarization System - Text, PDF, and Google Doc Processing")
    print(f"Server: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"AI Summaries Directory: {AI_SUMMARIES_DIR}")
    print("Supported file types: TXT, PDF, GDOC")
    print("Mode: Direct file discovery (ignoring any CSV files).")
    
    print("\nIMPORTANT: To process Google Docs (.gdoc files), you must:")
    print("1. Enable the Google Docs API in your Google Cloud project.")
    print("2. Create OAuth 2.0 credentials for a Desktop App.")
    print(f"3. Download the credentials and save them as '{CREDENTIALS_FILE}' in the same directory as this script.")
    print("4. The first time you run this, you will be prompted to authorize access in your browser.")
    
    if FOLDERS_TO_PROCESS and any(tf.strip() for tf in FOLDERS_TO_PROCESS):
        print(f"\nTarget folders to process: {[f.strip() for f in FOLDERS_TO_PROCESS if f.strip()]}")
    else:
        print("\nProcessing all folders (no specific folders configured).")
    
    print("\nTesting server connection...")
    models = get_available_models(BASE_URL)
    if not models:
        print("Cannot connect to server or no models found.")
        return
    
    if MODEL_NAME not in models:
        print(f"Warning: Model '{MODEL_NAME}' not found on server.")
        print(f"Available models: {models}")
        return
    
    print("Server connection successful.")
    
    # Process files by discovering them directly
    process_discovered_files()

    # Clean the newly created summaries after processing is complete
    clean_ai_summaries()
    
    print("\nAll operations complete!")

if __name__ == "__main__":
    main()