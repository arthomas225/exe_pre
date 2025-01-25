import os
import re
import logging
import threading

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import PyPDF2
import docx

# Set up logging
logging.basicConfig(
    filename='resume_search.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def setup_nltk():
    """Ensure required NLTK data is downloaded."""
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

def read_resume(file_path):
    """Read the resume file and return its raw content."""
    extension = os.path.splitext(file_path)[1].lower()
    content = ""

    try:
        if extension == ".pdf":
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
        elif extension == ".docx":
            doc = docx.Document(file_path)
            content = '\n'.join([para.text for para in doc.paragraphs])
        elif extension == ".txt":
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
        else:
            logging.error(f"Unsupported file type: {file_path}")
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")

    return content

def preprocess_text(text):
    """Preprocess text by lowercasing, removing non-alpha chars, removing stopwords, and lemmatizing."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmas)

def preprocess_tokens(text):
    """Return a set of lemma tokens for keyword or skill matching."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    lemmas = {lemmatizer.lemmatize(word) for word in tokens}
    return lemmas

def segment_resume(content):
    """Segment the resume into sections based on common headings."""
    pattern = re.compile(
        r'\n\s*(Experience|Work Experience|Professional Experience|Skills|Education|Projects|Qualifications|Summary)\s*\n',
        re.I
    )
    sections = pattern.split(content)
    section_dict = {}
    i = 1
    while i < len(sections):
        heading = sections[i].strip().lower()
        body = sections[i + 1].strip()
        section_dict[heading] = body
        i += 2
    return section_dict

def filter_by_criteria(
    file_path, content, sections, keyword_tokens,
    degree_level, required_skills_list, required_certs_list
):
    """Apply additional filters: keyword, degree, required skills, certifications."""
    # Keyword filtering
    if keyword_tokens:
        resume_tokens = preprocess_tokens(content)
        if not keyword_tokens.issubset(resume_tokens):
            return False

    # Degree Level filter (if provided)
    if degree_level:
        degree_level_lower = degree_level.lower()
        education_section = sections.get('education', content)
        if degree_level_lower not in education_section.lower():
            return False

    # Required Skills filter
    if required_skills_list:
        skills_section = sections.get('skills', content)
        skills_tokens = preprocess_tokens(skills_section)
        for skill in required_skills_list:
            skill_lemmas = preprocess_tokens(skill)
            if not skill_lemmas.issubset(skills_tokens):
                return False

    # Required Certifications filter
    if required_certs_list:
        cert_tokens = preprocess_tokens(content)
        for cert in required_certs_list:
            cert_lemmas = preprocess_tokens(cert)
            if not cert_lemmas.issubset(cert_tokens):
                return False

    return True

def compute_sentence_snippet(content, query_embedding, model):
    """Find the most relevant sentence snippet for the query."""
    sentences = sent_tokenize(content)
    if not sentences:
        return ""
    candidate_sentences = [s for s in sentences if len(s.strip()) > 3]

    if not candidate_sentences:
        return ""

    preprocessed_sents = [preprocess_text(s) for s in candidate_sentences]
    sent_embeddings = model.encode(preprocessed_sents)
    sims = cosine_similarity([query_embedding], sent_embeddings)[0]
    max_idx = sims.argmax()
    best_sentence = candidate_sentences[max_idx].strip()

    return best_sentence

def precompute_embeddings(resume_folder, model):
    """Precompute embeddings and store them to speed up searches."""
    file_paths = []
    for root, dirs, files in os.walk(resume_folder):
        for file in files:
            if file.endswith(('.pdf', '.docx', '.txt')):
                file_paths.append(os.path.join(root, file))

    resume_contents = {}
    resume_sections = {}
    resume_embeddings = {}

    for file_path in file_paths:
        content = read_resume(file_path)
        if not content:
            continue
        resume_contents[file_path] = content
        sections = segment_resume(content)
        resume_sections[file_path] = sections
        preprocessed_content = preprocess_text(content)
        if preprocessed_content.strip():
            embedding = model.encode(preprocessed_content)
            resume_embeddings[file_path] = embedding
        else:
            resume_embeddings[file_path] = None
    return resume_contents, resume_sections, resume_embeddings

def handle_search_resumes(
    query, section, keyword, degree_level,
    required_skills, required_certs, score_cutoff,
    top_n, resume_contents, resume_sections,
    resume_embeddings, model
):
    """Handle the logic for searching resumes using precomputed embeddings and additional filters."""
    required_skills_list = [s.strip() for s in required_skills.split(',') if s.strip()]
    required_certs_list = [c.strip() for c in required_certs.split(',') if c.strip()]

    preprocessed_query = preprocess_text(query)
    query_embedding = model.encode(preprocessed_query)

    keyword_tokens = set()
    if keyword.strip():
        keyword_tokens = preprocess_tokens(keyword.strip())

    all_files = list(resume_contents.keys())
    total_files = len(all_files)
    resume_similarity = []
    
    # Create a Streamlit progress bar
    progress_bar = st.progress(0)
    for idx, file_path in enumerate(all_files):
        content = resume_contents[file_path]
        sections = resume_sections[file_path]

        # Apply filters
        if not filter_by_criteria(
            file_path, content, sections, keyword_tokens,
            degree_level, required_skills_list, required_certs_list
        ):
            # Update progress
            progress_bar.progress(int((idx + 1) / total_files * 100))
            continue

        # Section-based content if requested
        if section != 'All':
            section_key = section.lower()
            selected_content = sections.get(section_key, '')
            if not selected_content:
                logging.warning(
                    f"Section '{section}' not found in {file_path}. Using entire content."
                )
                selected_content = content
            compare_text = preprocess_text(selected_content)
            if compare_text.strip():
                embedding = model.encode(compare_text)
            else:
                progress_bar.progress(int((idx + 1) / total_files * 100))
                continue
        else:
            embedding = resume_embeddings[file_path]
            if embedding is None:
                progress_bar.progress(int((idx + 1) / total_files * 100))
                continue

        similarity_score = cosine_similarity([query_embedding], [embedding])[0][0]
        resume_similarity.append((file_path, similarity_score))

        # Update progress
        progress_bar.progress(int((idx + 1) / total_files * 100))

    # Filter by score cutoff
    sorted_resumes = sorted(resume_similarity, key=lambda x: x[1], reverse=True)
    if score_cutoff > 0.0:
        sorted_resumes = [item for item in sorted_resumes if item[1] >= score_cutoff]

    # Top N results
    sorted_resumes = sorted_resumes[:top_n]

    results_with_snippets = []
    for (rfile, rscore) in sorted_resumes:
        snippet = compute_sentence_snippet(resume_contents[rfile], query_embedding, model)
        results_with_snippets.append((rfile, rscore, snippet))

    return results_with_snippets

def main():
    setup_nltk()

    # Initialize model once
    st.title("Enhanced Resume Search Tool with Streamlit")
    st.write("A tool to load and search resumes using semantic similarity and additional filters.")

    # We need a global model. We can store it in a session state if we want to load it once.
    if 'model' not in st.session_state:
        with st.spinner("Loading SentenceTransformer model..."):
            st.session_state.model = SentenceTransformer('all-mpnet-base-v2')
    
    # State variables to hold loaded resumes and embeddings
    if 'resume_contents' not in st.session_state:
        st.session_state.resume_contents = {}
    if 'resume_sections' not in st.session_state:
        st.session_state.resume_sections = {}
    if 'resume_embeddings' not in st.session_state:
        st.session_state.resume_embeddings = {}

    st.subheader("1. Load Resumes")
    resume_folder = st.text_input("Enter the path to the folder containing resumes")
    if st.button("Load Resumes"):
        if not os.path.isdir(resume_folder):
            st.error("Invalid folder path!")
        else:
            with st.spinner("Loading and indexing resumes. Please wait..."):
                contents, sections, embeddings = precompute_embeddings(resume_folder, st.session_state.model)
                st.session_state.resume_contents = contents
                st.session_state.resume_sections = sections
                st.session_state.resume_embeddings = embeddings
            if contents:
                st.success("Resumes loaded successfully.")
            else:
                st.warning("No valid resumes found in the specified folder.")

    st.write("---")
    st.subheader("2. Search Parameters")

    query = st.text_input("Enter Search Query:", "")
    keyword = st.text_input("Enter Keyword (optional):", "")
    degree_level = st.text_input("Required Degree Level (optional):", "")
    required_skills = st.text_input("Required Skills (comma-separated, optional):", "")
    required_certs = st.text_input("Required Certifications (comma-separated, optional):", "")

    section = st.selectbox("Focus on section (optional):", ['All', 'Experience', 'Education', 'Skills'])

    score_cutoff = st.text_input("Similarity Score Cutoff (0 to 1):", "0.5")
    top_n = st.text_input("Top N Results:", "10")

    if st.button("Search Resumes"):
        # Validate input
        if not st.session_state.resume_contents:
            st.error("No resumes loaded. Please load resumes first.")
        elif not query.strip():
            st.error("Search query cannot be empty!")
        else:
            try:
                score_cutoff_val = float(score_cutoff)
                if not 0 <= score_cutoff_val <= 1:
                    raise ValueError
            except ValueError:
                st.error("Please enter a valid similarity score cutoff between 0 and 1.")
                return

            try:
                top_n_val = int(top_n)
                if top_n_val <= 0:
                    raise ValueError
            except ValueError:
                st.error("Please enter a valid positive integer for top N results.")
                return

            # Perform the search
            with st.spinner("Searching resumes..."):
                results = handle_search_resumes(
                    query, section, keyword, degree_level,
                    required_skills, required_certs,
                    score_cutoff_val, top_n_val,
                    st.session_state.resume_contents,
                    st.session_state.resume_sections,
                    st.session_state.resume_embeddings,
                    st.session_state.model
                )

            if results:
                st.success("Search completed.")
                st.write("### Ranked Resumes")
                result_text_lines = []
                for i, (file, score, snippet) in enumerate(results):
                    filename = os.path.basename(file)
                    line = f"{i+1}. **{filename}** - Similarity: {score:.2f}\n\nSnippet: {snippet}\n"
                    result_text_lines.append(line)
                    st.markdown(line)

                # Combine the text for saving
                combined_text = "\n".join(result_text_lines)
                st.download_button(
                    label="Save Results to File",
                    data=combined_text,
                    file_name="search_results.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No resumes matched the query with the given criteria.")

if __name__ == '__main__':
    main()
