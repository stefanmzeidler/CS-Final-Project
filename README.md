## CS Final Project
Stefan Zeidler
Department of Computer Science
University of Wisconsin - Milwaukee

Notice: This repository is a proof of concept only and should not be used in any production capacity.

#Introduction
This project is intended as a proof of concept for a generative-AI based system to combat medical dis- and misinformation. One of the main barriers for the average person to understand medical information is the high reading level and expert-level domain knowledge required to understand medical journal articles. While traditional NLP models can summarize articles, they struggle to make high-level text easier to read and are completely unable to infer what a reader does and does not know. 

By leveraging the generative capaiblities of large language models, we can both change the reading level to the match the target audience and provide additional context as needed to help the reader understand the material. To hinder large language models' penchant for hallucinations, a two-level information retrieval system has been implemented for retrieval augmented generation to provide relevant journal articles as supporting documentation. 

The (barebones) UI employs "human-in-the-loop" design principles to avoid alienating users and to promote user agency. The system should not dictate what a user needs, but allow them to choose.

#Design
This system is designed to summarize articles from the [PubMed Central Open Access Subset]([url](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)) from the National Library of Medicine. 

The system retrieves the article to summarize and creates an embedding from the text. Using this embedding, the system retrieves the top K most relevant articles from a database of previously bi-encoded articles. Once the top K articles have been retrieved, they are re-ranked using cross-encoding. As noted on the [SentenceTransformers site]([url](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)) there are trade-offs between bi-encoding systems and cross-encoding systems that can be mitigated by using a retrieve and re-rank pipeline. While bi-encoding is quite fast for retrieving articles given that the article embeddings can be pre-generated, it is not as accurate as cross-encoding. Cross-encoding, while much more accurate than bi-encoding, must generate embeddings at run time, meaning that its performance suffers greatly for large datasets. To leverage the performance of bi-encoding and the accuracy of cross-encoding, this system uses retrieval and re-ranking for complex semantic search tasks. 

Once the documents have been retrieved and re-ranked, the top three are selected and sent along with the original article to gemini-flash-2.5 for summarization. This is done three times, with different prompts for different reading levels. The temperature of the model is kept intentionally high to avoid as much creativity when generating summaries. If the stated goal of this project is combat medical misinformation, limiting the amount of freedom the LLM has is critical. However, at the same time, too high of a temperature may impede the model's ability to change reading levels and provide context. 

Once the summaries are retrieved, all three are provided to the user so that they can select what is most appropriate for themselves. The user is given agency instead of a system that dictates their level of knowledge, keeping in line with human-in-the-loop design principes as described by [Stanford's HAI]([url](https://hai.stanford.edu/news/humans-loop-design-interactive-ai-systems)). 

#Limitations
While this system seeks to address large language model's propensity to hallucinate, it does not remove this challenge entirely. Based on analaysis by Kim et al., further improvements could be made to the pipeline through evidence filtering and query reformulation [1]. Each of the individual steps can and should be analyzed for performance instead of only the final output. 

Most importantly, there has been no analysis of the accuracy of the information provided by the system. Any further research should include this metric, particularily in the context of a system for improving understanding and combatting misinformation. A specific avenue of research would be to test the relationship between LLM temperature, reading level, and accuracy. Preliminary results indicated that the current high temperature LLM used in this program provides a 12th grade reading level for lay users of the system, well above the accepted average of a 6th to 8th grade reading level. However, any adjustments to allow for a lower reading level may also negatively impact accuracy. 

[1] H. Kim et al., “Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights,” 2025.
