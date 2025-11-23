# CLAUDE.md - AMWAL Repository Guide for AI Assistants

## Project Overview

**AMWAL** is an Arabic Financial Named Entity Recognition (NER) system designed to extract structured information from Arabic financial news articles. The system achieves state-of-the-art performance with 96.08% precision, 95.87% recall, and 95.97% F1-score.

### Key Characteristics
- **Domain**: Financial NER for Arabic text
- **Framework**: SpaCy with custom NER pipeline
- **Base Model**: AraBERT Large (aubmindlab/bert-large-arabertv02)
- **Dataset**: 17.1k tokens from Arabic financial newspapers (2000-2023)
- **Ontology**: Aligned with FIBO (Financial Industry Business Ontology)
- **Language**: Arabic (lang code: "ar")

## Repository Structure

```
AMWAL/
├── README.md                      # Project documentation
├── CLAUDE.md                      # This file - AI assistant guide
├── config.cfg                     # Main SpaCy training configuration
├── Arabert.py                     # AraBERT training script (alternative approach)
│
├── Data Annotation/               # Annotation tools and configurations
│   ├── annotator.ipynb           # Data annotation notebook
│   ├── tokenizer_exceptions.json # Custom tokenizer rules
│   └── lemmatization_exceptions.json # Custom lemmatization rules
│
├── Model/                         # Trained models
│   ├── model-best/               # Best performing model checkpoint
│   │   ├── config.cfg
│   │   ├── meta.json             # Model metadata and performance metrics
│   │   ├── tokenizer
│   │   ├── ner/                  # NER component
│   │   ├── tok2vec/              # Token-to-vector component
│   │   └── vocab/                # Vocabulary files
│   └── model-last/               # Last training checkpoint
│       └── [same structure as model-best]
│
├── Train.ipynb                    # Main training notebook (SpaCy approach)
├── Wojood.ipynb                   # Wojood dataset experiments
├── camel.ipynb                    # CAMeL Tools experiments
├── train_data.spacy              # Training data in SpaCy format (52MB)
└── test.spacy                    # Test data in SpaCy format (30MB)
```

## Entity Categories (21 Classes)

The system recognizes 21 entity types organized into financial categories:

### Financial Institutions & Organizations
1. **BANK** - Banking institutions (F1: 91.66%)
2. **CORP** - Corporations (F1: 83.20%)
3. **ORG** - General organizations (F1: 86.61%)
4. **STOCK_EXCHANGE** - Stock exchanges (F1: 98.18%)
5. **GOVERNMENT_ENTITY** - Government entities (F1: 97.60%)

### People & Roles
6. **PERSON** - Person names (F1: 76.33%)
7. **OFFICIAL** - Officials (F1: 89.01%)
8. **ROLE** - Job roles/positions (F1: 89.16%)

### Location Entities
9. **COUNTRY** - Countries (F1: 97.50%)
10. **CITY** - Cities (F1: 81.64%)
11. **GPE** - Geopolitical entities (F1: 90.35%)
12. **NATIONALITY** - Nationalities (F1: 97.25%)

### Financial Instruments & Metrics
13. **FINANCIAL_INSTRUMENT** - Financial instruments (F1: 97.60%)
14. **CURRENCY** - Currencies (F1: 99.20%)
15. **PRODUCT_OR_SERVICE** - Products/services (F1: 95.56%)
16. **Metrics** - Financial metrics (F1: 95.20%)
17. **FinMarket** - Financial markets (F1: 92.87%)

### Temporal & Quantitative
18. **TIME** - Temporal expressions (F1: 98.97%)
19. **QUNATITY_OR_UNIT** - Quantities/units (F1: 96.01%) [Note: typo in original]

### Media & Events
20. **MEDIA** - Media organizations (F1: 95.84%)
21. **Events** - Financial events (F1: 98.03%)

## Development Workflows

### 1. Data Preparation Workflow

**Input Format**: JSON files with tokens and entities
```json
{
  "tokens": [...],
  "entities": [
    {"text": "...", "type": "BANK", "start": 0, "end": 5}
  ]
}
```

**Processing Steps** (see Train.ipynb):
1. Load JSON files from annotation directory
2. Reconstruct text from tokens
3. Reformat entities with start/end positions
4. Convert to SpaCy DocBin format
5. Split into train/dev sets (80/20)
6. Save as `.spacy` files

### 2. Training Workflow (SpaCy Approach)

**Primary Method** (Train.ipynb):
```bash
# 1. Initialize config
python -m spacy init fill-config base_config.cfg config.cfg

# 2. Train model
python -m spacy train config.cfg \
  --output ./output \
  --paths.train training_data.spacy \
  --paths.dev test.spacy

# 3. Evaluate
spacy evaluate output/model-best/ dev.spacy
```

**Training Configuration**:
- Pipeline: tok2vec → ner
- Transformer: aubmindlab/bert-large-arabertv02
- Max steps: 20,000
- Eval frequency: 200 steps
- Batch size: 50
- Optimizer: Adam (lr=0.001)
- Dropout: 0.1

### 3. Alternative Training Workflow (AraBERT + Transformers)

**Alternative Method** (Arabert.py):
- Uses Hugging Face Transformers
- Model: aubmindlab/bert-base-arabertv02
- Token classification approach
- 3 epochs, learning rate 2e-5
- Batch size: 16 (train), 64 (eval)

### 4. Text Preprocessing

Arabic text normalization (see Train.ipynb cell 9):
```python
text = re.sub("[إأٱآا]", "ا", text)  # Normalize alef variants
text = re.sub("ى", "ي", text)        # Normalize yeh
text = re.sub("ًٌَُ'ٍِـ،ْ", "", text)  # Remove diacritics
text = re.sub("ؤ", "ء", text)        # Normalize waw hamza
text = re.sub("ئ", "ء", text)        # Normalize yeh hamza
text = re.sub("ة", "ه", text)        # Normalize teh marbuta
```

### 5. Inference Workflow

```python
import spacy

# Load trained model
nlp = spacy.load("Model/model-best")

# Preprocess text (apply normalization)
text = preprocess_arabic(text)

# Run inference
doc = nlp(text)

# Extract entities
for ent in doc.ents:
    print(f"{ent.text} → {ent.label_}")
```

## Key Files and Their Purpose

### Training Files
- **Train.ipynb**: Primary training notebook using SpaCy
  - Cell 0-1: Imports and data loading
  - Cell 2: JSON to training data conversion
  - Cell 3-5: DocBin creation and entity processing
  - Cell 6-7: Config initialization and training
  - Cell 8-9: Evaluation and inference demo

- **Arabert.py**: Alternative training script using Transformers
  - Uses token classification approach
  - 21-class label map (lines 22-44)
  - Custom tokenization and label alignment (lines 56-81)

### Configuration Files
- **config.cfg**: SpaCy training configuration
  - Defines tok2vec and NER architecture
  - Transformer settings (bert-large-arabertv02)
  - Training hyperparameters
  - Batch scheduling and optimization

### Data Files
- **train_data.spacy**: Training dataset (52MB, ~80% of data)
- **test.spacy**: Test dataset (30MB, ~20% of data)
- **Data Annotation/*.json**: Custom tokenizer/lemmatization rules

### Model Files
- **Model/model-best/**: Best model checkpoint
  - **meta.json**: Contains performance metrics per entity type
  - **config.cfg**: Model-specific configuration
  - **ner/**: NER component weights
  - **tok2vec/**: Token vectorization weights
  - **tokenizer**: Tokenizer configuration

## Development Guidelines for AI Assistants

### When Modifying Code

1. **Preserve Arabic Text Processing**
   - Always maintain Arabic normalization steps
   - Don't modify tokenizer_exceptions.json without consultation
   - Test with Arabic text samples

2. **Entity Type Changes**
   - Update ALL occurrences: Arabert.py label_map, config.cfg, meta.json
   - Retrain model after entity schema changes
   - Update documentation

3. **Model Training**
   - Prefer SpaCy approach (Train.ipynb) as primary method
   - Use Arabert.py approach only for transformer-specific experiments
   - Always evaluate on test set before deployment

4. **Performance Expectations**
   - Overall F1 score should be ≥95%
   - High-frequency entities (TIME, CURRENCY, Events) should be ≥98%
   - Lower-frequency entities (PERSON, CITY) may be 75-85%

### When Adding Features

1. **New Entity Types**
   - Align with FIBO ontology when possible
   - Ensure sufficient training examples (≥100 instances)
   - Update meta.json with expected performance

2. **Model Improvements**
   - Keep model-best as baseline for comparison
   - Document performance changes in meta.json
   - Preserve backward compatibility with .spacy format

3. **Data Augmentation**
   - Maintain 80/20 train/dev split
   - Ensure entity distribution remains balanced
   - Validate with domain experts

### Common Pitfalls to Avoid

1. **Character Encoding Issues**
   - Always use UTF-8 encoding for Arabic text
   - Test with right-to-left (RTL) text rendering
   - Preserve Arabic-specific Unicode characters

2. **Entity Boundary Issues**
   - Use `filter_spans()` to handle overlapping entities
   - Validate start/end positions match token boundaries
   - Handle multi-word entities correctly

3. **Path Dependencies**
   - Arabert.py has hardcoded paths (lines 17, 119)
   - config.cfg uses relative paths (lines 2-3)
   - Update paths when moving files

4. **Model Version Mismatches**
   - SpaCy version: ≥3.7.4, <3.8.0 (see meta.json)
   - AraBERT versions: bert-base vs bert-large
   - Transformer library compatibility

## Testing and Validation

### Unit Testing Checklist
- [ ] Arabic text normalization works correctly
- [ ] Entity boundaries align with token positions
- [ ] All 21 entity types are recognized
- [ ] Model loads without errors
- [ ] Inference produces valid SpaCy Doc objects

### Integration Testing
- [ ] Full training pipeline completes
- [ ] Evaluation metrics match expected ranges
- [ ] Model serialization/deserialization works
- [ ] GPU/CPU compatibility verified

### Performance Benchmarks
- [ ] Precision ≥96%
- [ ] Recall ≥95%
- [ ] F1-Score ≥95%
- [ ] Inference speed: <100ms per document (CPU)

## Git Workflow Conventions

### Branch Strategy
- Work on feature branches prefixed with `claude/`
- Branch names include session IDs for tracking
- Current branch: `claude/claude-md-mic2p8y0q2zu0jg3-01XDikarVaivFZ2ADsXnnGyF`

### Commit Guidelines
- Clear, descriptive commit messages
- Reference issue/PR numbers when applicable
- Group related changes in single commits

### Before Committing
- Run evaluation: `spacy evaluate Model/model-best/ test.spacy`
- Verify all notebooks run without errors
- Update documentation if schema changes
- Check file sizes (avoid committing large data files >100MB)

## Resources and References

### External Dependencies
- **SpaCy**: https://spacy.io/
- **AraBERT**: https://github.com/aub-mind/arabert
- **CAMeL Tools**: https://github.com/CAMeL-Lab/camel_tools
- **FIBO**: https://spec.edmcouncil.org/fibo/

### Dataset Sources
- Arabic financial newspapers (2000-2023)
- Wojood dataset (for experiments)
- Custom annotated corpus (17.1k tokens)

### Related Work
- SinaLab/ArabicWojood-FlatNER (recent commit)
- bert-base-arabic-camelbert-msa-ner (experiments)

## Performance Metrics (Model-Best)

| Entity Type | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| CURRENCY | 99.16% | 99.23% | 99.20% |
| TIME | 98.94% | 98.99% | 98.97% |
| STOCK_EXCHANGE | 97.78% | 98.58% | 98.18% |
| Events | 98.01% | 98.05% | 98.03% |
| GOVERNMENT_ENTITY | 97.76% | 97.43% | 97.60% |
| FINANCIAL_INSTRUMENT | 97.91% | 97.29% | 97.60% |
| COUNTRY | 97.62% | 97.37% | 97.50% |
| NATIONALITY | 97.12% | 97.37% | 97.25% |
| QUNATITY_OR_UNIT | 95.92% | 96.11% | 96.01% |
| MEDIA | 95.78% | 95.90% | 95.84% |
| PRODUCT_OR_SERVICE | 95.42% | 95.71% | 95.56% |
| Metrics | 96.07% | 94.36% | 95.20% |
| FinMarket | 88.29% | 97.94% | 92.87% |
| BANK | 91.32% | 92.01% | 91.66% |
| ROLE | 89.01% | 89.30% | 89.16% |
| OFFICIAL | 84.61% | 93.90% | 89.01% |
| GPE | 86.47% | 94.58% | 90.35% |
| ORG | 85.99% | 87.23% | 86.61% |
| CORP | 87.30% | 79.47% | 83.20% |
| CITY | 79.05% | 84.40% | 81.64% |
| PERSON | 75.01% | 77.70% | 76.33% |
| **Overall** | **96.29%** | **96.26%** | **96.27%** |

## Future Directions

As noted in README.md, planned improvements include:
1. Expanding corpus size and entity types
2. Restructuring entities into hierarchical structures
3. Adding entity relationship extraction
4. Building Arabic financial knowledge graph

When implementing these features, ensure backward compatibility with existing model and maintain performance standards.

---

**Last Updated**: 2025-11-23
**SpaCy Version**: 3.7.4
**Model Version**: 0.0.0
**Primary Contact**: See README.md for project information
