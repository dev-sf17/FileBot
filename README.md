# FileBot
## AI Chatbot with LLM and LPU - Hobby Project

FileBot is a powerful AI chatbot designed to search and retrieve information from customized data, including CSV and PDF uploads. It is developed using the LangChain framework, enabling seamless integration with advanced natural language processing capabilities.

### Key Features
- **Custom Data Search:** Upload CSV and PDF files to search and extract relevant information.
- **LangChain Framework:** Built using the LangChain framework for flexible and efficient management of chatbot functionalities.
- **Embedding Model:** Utilizes the HuggingFace `all-MiniLM-L6-v2` embedding model to generate text embeddings for enhanced search precision.
- **Vector Embeddings:** Leverages **ChromaDB** as the vector store to manage embeddings and support fast, scalable search across large datasets.
- **Large Language Model (LLM):** Integrated with the `mixtral-8x7b-32768` LLM model via the Groq API, providing high-quality natural language understanding and response generation.

### Usage
Users can upload their own CSV and PDF documents to the chatbot and perform customized searches on their data. The chatbot returns precise and relevant information based on user queries.

To use the chatbot, visit the site at , https://filebot-a2emfc4epdvzs9jxx69xpj.streamlit.app/

### Limitations
- **Embedding Model Performance:** While the HuggingFace `all-MiniLM-L6-v2` embedding model provides solid performance for many use cases, more sophisticated embedding models may yield even more accurate and contextually rich responses. Depending on the complexity of your data, experimenting with other embedding models could improve the quality of search results.
