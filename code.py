# Libraries import
import os
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from umap import UMAP
import hdbscan
from bertopic import BERTopic
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import unicodedata
import re

# Download stopwords
nltk.download('stopwords')

# Get Spanish stop words
spanish_stopwords = stopwords.words('spanish')

# We use our own list of custom stopwords, specific to our corpus to further reduce unwanted noise
custom_stopwords = ["the", "issn", "http", "licencia", "anclajes", "año", "and", "cuyo", "permitir", "bajo", "creative", "commons", "caso", "siglo", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre", "xix", "xx", "xxi","cuadernos","auster","eissn","saga","orbis","tertius", "cilha", "obra", "texto", "vida", "literatura", "historia", "forma", "literario", "argentino", "ano", "interfaz", "tipo", "ejemplo", "rubir", "rubir dario", "casal", "aires", "martel", "maromero", "chileno", "bync", "cuaderno bync", "marin", "castelnouovo", "mendoza"]

# Combine custom stopwords with Spanish stop words and convert to list
combined_stopwords = list(set(spanish_stopwords + custom_stopwords))

# Load spaCy model for Spanish
nlp = spacy.load("es_core_news_sm")

# Normalize text
def normalize_text(text):
    # Normalize text by removing accents and converting to lowercase.
    # Normalize unicode text to NFD (Normalization Form D)
    text = unicodedata.normalize('NFD', text)
    # Remove diacritics (accents)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # Convert to lowercase
    text = text.lower()
    return text

#Reconstruct hyphenated words by joining parts split across lines.
def reconstruct_hyphenated_words(text):
    pattern = re.compile(r'(\w+)-\s*(\w+)')
    def replace_match(match):
        return f"{match.group(1)}{match.group(2)}"
    return pattern.sub(replace_match, text)

# Tokenize text with specific parameters. We use lemmatization and work with only words longer than two characters
def tokenize_text(text):
    text = reconstruct_hyphenated_words(text)
    text = normalize_text(text)  # Normalize text first
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_space and not token.is_stop and not token.is_punct and token.lemma_.lower() not in combined_stopwords and not token.like_num and len(token.text) > 2 and token.is_alpha]
    return tokens

# Process our directory, pre processing and tokenizing all files
def process_directory(directory):
    filenames = []
    documents = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    tokens = tokenize_text(text)
                    filenames.append(file_name)  # Store the filename
                    documents.append(" ".join(tokens))  # Join tokens to form a document
    return filenames, documents

# Load documents
directory_path = 'path to out locally saved files'
filenames, documents = process_directory(directory_path)

# Embedding documents
embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
embeddings = embedding_model.encode(documents, batch_size=64, show_progress_bar=True)

# Apply PCA (Principal Component Analysis) for initial noise reduction
pca = PCA(n_components=50)  # Reduce to 50 dimensions
pca_embeddings = pca.fit_transform(embeddings)

# Apply UMAP (Uniform Manifold Approximation and Projection) for further reduction
umap_model = UMAP(n_neighbors=15, min_dist=0.0)
umap_embeddings = umap_model.fit_transform(pca_embeddings)

# Initialize TF-IDF Vectorizer with combined stop words
tfidf_vectorizer = TfidfVectorizer(stop_words=combined_stopwords, ngram_range=(1, 3))

# Initialize and fit BERTopic
bertopic = BERTopic(
    embedding_model=None,  # UMAP embeddings are already processed
    umap_model=umap_model,
    hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=30, min_samples=5), # Set parameters for clustering
    vectorizer_model=tfidf_vectorizer # Use TF-IDF vectorizer to convert text into numerical form
)

# Fit the model and transform the documents
topics, probs = bertopic.fit_transform(documents)

# Display and evaluate results
topics_info = bertopic.get_topic_info()
print(topics_info)


# Save topics as a .xlsx file
import pandas as pd
df_topics = pd.DataFrame(topics_info)
df_topics.to_excel('topics_info.xlsx', index=False)


# Creates a word cloud for each topic and saves it as a png image
from wordcloud import WordCloud
def save_wordclouds(bertopic_model, topics, output_dir='topic_wordclouds'):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Get the words for each topic
    for topic_id in set(topics):
        if topic_id == -1:  # Skip the noise topic (if any)
            continue
        # Get the top words for the topic
        words = bertopic_model.get_topic(topic_id)
        # Convert the list of words into a dictionary with words and their frequencies
        word_freq = {word: freq for word, freq in words}
        # Create the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        # Save the word cloud as a PNG file
        output_path = os.path.join(output_dir, f"topic_{topic_id}_wordcloud.png")
        wordcloud.to_file(output_path)
        print(f"Word cloud for Topic {topic_id} saved as: {output_path}")

save_wordclouds(bertopic, topics, output_dir='topic_wordclouds')


# Creates a topic hierarchy and saves it as a png image
import plotly.io as pio
def show_and_save_topic_hierarchy(bertopic_model, output_path='topic_hierarchy.png'):
    # Generate the topic hierarchy visualization (this returns a Plotly figure)
    fig = bertopic_model.visualize_hierarchy()
    # Show the figure in the browser or notebook (interactive plot)
    fig.show()
    # Optionally save the figure as a static image after viewing
    try:
        pio.write_image(fig, output_path)
        print(f"Topic hierarchy saved as: {output_path}")
    except Exception as e:
        print(f"Error saving the figure: {e}")

show_and_save_topic_hierarchy(bertopic, output_path='topic_hierarchy.png')


# Plots the number of documents per topic and the score distribution, and save the images as png files
import seaborn as sns
import matplotlib.pyplot as plt
# Create a DataFrame where each document is assigned a topic and its associated score
document_topic_df = pd.DataFrame({
    'Document': range(len(documents)),
    'Topic': topics,
    'Score': probs
})
# Count the number of documents per topic
topic_counts = pd.Series(topics).value_counts()
plt.figure(figsize=(10, 6))
topic_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Documents per Topic', fontsize=16)
plt.xlabel('Topic', fontsize=14)
plt.ylabel('Number of Documents', fontsize=14)
output_path = 'documents_per_topic.png'
plt.savefig(output_path, format='png')
# Plot the distribution of scores for each topic
plt.figure(figsize=(12, 8))
sns.boxplot(data=document_topic_df, x='Topic', y='Score', palette='viridis')
plt.title('Score Distribution per Topic')
plt.xlabel('Topic')
plt.ylabel('Score')
plt.savefig('score_distribution_per_topic.png', dpi=300, bbox_inches='tight')


# Visualize and save document distribution
fig_documents = bertopic.visualize_documents(documents)
fig_documents.write_html("documents_visualization.html")


# Generate and save the topics visualization
fig = bertopic.visualize_topics()
fig.write_html("topics_visualization.html")




