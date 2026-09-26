Agriculture NLP Question Answering and Summarization



Overview



This project implements an NLP system for working with a collection of agriculture-related e-books from Project Gutenberg.



The system combines:



Dense Passage Retrieval (DPR) for semantic passage retrieval



BERT-based extractive Question Answering



T5-based abstractive summarization



FAISS for efficient vector similarity search



Streamlit for the user interface



The project provides two main functions:



Question Answering



Topic Summarization



Dataset



The system uses six unique agriculture books from Project Gutenberg:



Agriculture for Beginners — Charles William Burkett, Frank Lincoln Stevens, and Daniel Harvey Hill



Science and Practice in Farm Cultivation — James Buckman



The Farm That Won't Wear Out — Cyril G. Hopkins



Dry-Farming: A System of Agriculture for Countries under a Low Rainfall — John Andreas Widtsoe



Pleasant Talk About Fruits, Flowers and Farming — Henry Ward Beecher



Field, Forest and Farm — Jean-Henri Fabre



One of the original assignment links was duplicated, so six unique books were processed.



Preprocessing



The Project Gutenberg texts are cleaned before being used by the NLP models.



The preprocessing includes:



Removing Project Gutenberg header and footer markers



Removing production and transcription notes



Removing formatting artifacts



Normalizing whitespace



Preserving the actual book content



The cleaned books are then divided into overlapping passages.



Each passage contains approximately 200 words with an overlap of 50 words between consecutive passages.



The final dataset contains approximately 2,973 passages.



System Architecture



Agriculture E-books

&#x20;       |

&#x20;       v

&#x20;  Preprocessing

&#x20;       |

&#x20;       v

&#x20; Passage Creation

&#x20;       |

&#x20;       v

&#x20;  DPR Context Encoder

&#x20;       |

&#x20;       v

&#x20;     FAISS

&#x20;  Vector Database

&#x20;       |

&#x20;       +----------------------+

&#x20;       |                      |

&#x20;       v                      v

&#x20;Question Answering      Topic Summarization

&#x20;       |                      |

&#x20;       v                      v

&#x20;   DPR Retrieval          DPR Retrieval

&#x20;       |                      |

&#x20;       v                      v

&#x20;   BERT-QA                T5-small

&#x20;       |                      |

&#x20;       v                      v

&#x20;Extractive Answer       Topic Summary



Question Answering



For Question Answering, the system follows these steps:



The user enters a question.



DPR converts the question into a dense vector representation.



FAISS retrieves the most relevant passages.



The retrieved passages are given to a BERT-based Question Answering model.



BERT-QA extracts an answer span directly from the retrieved passage.



The system displays the answer and its source passage.



Models



DPR:



facebook/dpr-question\_encoder-single-nq-base

facebook/dpr-ctx\_encoder-single-nq-base



Question Answering:



deepset/bert-base-cased-squad2



Topic Summarization



For topic summarization:



The user enters a topic.



DPR retrieves the most relevant passages.



T5 generates a summary for each retrieved passage.



The generated passage summaries are combined.



T5 processes the combined summaries again to produce a final topic-focused summary.



Model



t5-small



Vector Search



The DPR passage embeddings have 768 dimensions.



FAISS is used with an inner-product index after L2 normalization. This makes the similarity measure equivalent to cosine similarity.



The final FAISS index contains 2,973 passage vectors.



Streamlit Application



The application provides two modes:



Question Answering



Users can enter questions such as:



How can soil fertility be maintained?



The system retrieves relevant agriculture passages and extracts an answer using BERT-QA.



Topic Summarization



Users can enter topics such as:



soil fertility and how farmers can maintain fertile soil



The system retrieves relevant passages and generates a topic-focused summary using T5.



Project Structure



agriculture-nlp-qa-summarization/

|

├── data/

│   ├── raw/

│   ├── processed/

│   ├── passages.json

│   ├── dpr\_passages.faiss

│   └── dpr\_metadata.pkl

|

├── models/

|

├── notebooks/

|

├── src/

│   ├── preprocess.py

│   ├── chunk\_books.py

│   ├── dpr\_retriever.py

│   ├── test\_dpr.py

│   ├── test\_bert\_qa.py

│   ├── qa\_system.py

│   ├── test\_summarization.py

│   ├── summarization\_system.py

│   └── app.py

|

├── requirements.txt

├── requirements-freeze.txt

└── README.md



Installation



Create and activate a virtual environment:



python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1



Install the dependencies:



pip install -r requirements.txt



Running the Application



From the project root:



streamlit run src\\app.py



The application will open in a browser.



Main Components



preprocess.py



Cleans the original Project Gutenberg texts.



chunk\_books.py



Divides the processed books into overlapping passages.



dpr\_retriever.py



Creates DPR embeddings for the passages and builds the FAISS index.



test\_dpr.py



Tests DPR retrieval independently.



test\_bert\_qa.py



Tests the BERT Question Answering model independently.



qa\_system.py



Combines DPR retrieval with BERT-based extractive Question Answering.



test\_summarization.py



Tests T5 summarization independently.



summarization\_system.py



Combines DPR retrieval with T5-based topic summarization.



app.py



Provides the complete Streamlit interface.



Example



Question



How can soil fertility be maintained?



The system retrieves relevant passages and uses BERT-QA to extract an answer from the retrieved text.



Topic



soil fertility and how farmers can maintain fertile soil



DPR retrieves relevant passages and T5 generates a topic-focused summary.



Limitations



The system uses pretrained models rather than models trained specifically on the agriculture books.



T5-small can produce incomplete or simplified summaries for longer or complex passages.



BERT-QA performs extractive Question Answering, so answers must come from the retrieved text.



Retrieval quality depends on the DPR model and the passage segmentation.



No formal benchmark accuracy evaluation has been performed yet.



Technologies



Python



PyTorch



Hugging Face Transformers



Dense Passage Retrieval (DPR)



BERT



T5



FAISS



Streamlit



NumPy

