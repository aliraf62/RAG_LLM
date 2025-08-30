# Usage Examples: core.classify

This guide demonstrates how to use the classification utilities for question type and subject domain.

## Example: Classify a Question

```python
from core.classify import classify_question, get_primary_domain, get_subject_domains

question = "How do I create a sourcing event in Coupa?"

# Classify the question
classification = classify_question(question)
print("Prompt type:", classification["prompt_type"])
print("Primary domain:", classification["primary_domain"])
print("All domains:", classification["domains"])
print("Is multi-domain?", classification["multi_domain"])
```

**Output:**
```
Prompt type: How to
Primary domain: sourcing
All domains: [('sourcing', 2.0), ('cso', 0.5)]
Is multi-domain? True
```

## Example: Get Primary Domain

```python
domain = get_primary_domain("What is a scenario in CSO?")
print(domain)
```

**Output:**
```
cso
```

## Example: Get Subject Domains with Scores

```python
domains = get_subject_domains("How do I invite a supplier to my event?")
print(domains)
```

**Output:**
```
[('sourcing', 0.6), ('cso', 0.5)]
```

See also: [docs/architecture/classify.md](../architecture/classify.md)
