**AMWAL: Arabic Financial Named Entity Recognition System**

**Introduction:**
Financial Named Entity Recognition (NER) is a critical task for extracting structured information from unstructured financial data, especially when extending its application to languages beyond English. In this project, we introduce AMWAL, a named entity recognition system tailored for Arabic financial news.

**Data Collection and Annotation:**
We compiled a specialized corpus from three major Arabic financial newspapers spanning from 2000 to 2023. Entities were extracted from this corpus using a semi-automatic process that included manual annotation and review to ensure accuracy. The total number of entities identified amounts to 17.1k tokens, distributed across 21 categories, providing comprehensive coverage of financial entities.

**Entity Standardization:**
To standardize the identified entities, we adopted financial concepts from the Financial Industry Business Ontology (FIBO, 2020), aligning our framework with industry standards.

**Model Development:**
We developed the model using SpaCy's custom NER pipeline and employed Arabert Large for processing the data. Our approach involved training the model on the annotated corpus to recognize financial entities accurately.

**Evaluation Results:**
The evaluation results of the model on the test data demonstrated strong performance metrics, with precision at 96.08%, recall at 95.87%, and an F1-score at 95.97%. These scores outperform several other systems for financial NER in other languages.

**Future Directions:**
For future directions, we aim to expand the size of the corpus as well as the number of entity types. This involves restructuring the entities into more intricate hierarchical structures. Additionally, we plan to expand the scope of the model to encompass not only entity types but also their interrelations, with the ultimate objective of building an Arabic financial knowledge graph to better inform various stakeholders in the field of Finance.
