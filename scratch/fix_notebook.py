import json

with open('AnnotatedTransformer.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Add language_info metadata
nb['metadata']['language_info'] = {
    "codemirror_mode": {
        "name": "ipython",
        "version": 3
    },
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
    "version": "3.13.0"
}

with open('AnnotatedTransformer.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print("Done. Updated metadata:")
print(json.dumps(nb['metadata'], indent=2, ensure_ascii=False))
