# Academic RAG Mapping Assistant

An LLM-powered academic intelligence platform that combines Retrieval-Augmented Generation (RAG), knowledge graph construction, and evidence-based reasoning to map researcher expertise, collaborations, grants, and external partnerships from heterogeneous public data sources.

## Motivation

Finding collaborators, understanding a researcher's expertise, or identifying relevant funding often requires manually searching publications, grant databases and institutional websites.

Academic RAG Mapping Assistant automates this process by combining document retrieval, large language models and knowledge graph construction into a single AI-assisted workflow.

## Features

✓ Research interest extraction

✓ Grant discovery

✓ Funding source identification

✓ Collaboration network construction

✓ External organisation mapping

✓ Interactive knowledge graph

✓ Evidence-backed responses

✓ Citation-aware RAG

✓ Multi-document retrieval

## Pipeline

Step 1. (At this stage) manually collect raw researcher documents, e.g. grants, publication pdfs and html webpages from 3 sources: previous place of work, google scholar and current place of work.
Step 2. Parsed the above files to extract raw text. 
Step 3. Cut the texts above to chunks of 1200 characters.
Step 4. Create local retrieval index (TF-IDF similarity for now)
Step 5. Evidence pack
Step 6. LLM-generated structured profile
Step 7. Create profile graph JSON
Step 8. Local browser graph visualization 

## Outputs

The system generates

- researcher profile

- research interests

- projects

- grants

- collaborators

- external organisations

- interactive knowledge graph

- evidence table

- confidence score
