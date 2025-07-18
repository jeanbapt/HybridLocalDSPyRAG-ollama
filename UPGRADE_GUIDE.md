# Upgrade Guide - Dependency Updates

## Overview

This guide documents the dependency updates made to ensure the project uses the latest stable versions of all packages.

## Updated Dependencies

### Core ML Packages

1. **PyTorch**: `>=2.1.0` → `>=2.6.0`
   - Latest stable release with improved performance and features
   - Includes better MPS (Apple Silicon) support
   - Enhanced torch.compile functionality

2. **TorchVision**: `>=0.17.0` → `>=0.21.0`
   - Updated to match PyTorch 2.6 compatibility
   - Improved transforms and models

3. **Transformers**: `>=4.36.0` → `>=4.53.2`
   - Latest Hugging Face transformers
   - Better model support and performance optimizations
   - Enhanced tokenizer features

4. **SentencePiece**: `==0.1.99` → `==0.2.0`
   - Latest stable version
   - Improved tokenization performance

### RAG and NLP Packages

1. **NLTK**: `>=3.8.1` → `>=3.9.1`
   - Latest version with bug fixes
   - Enhanced language processing features

2. **DSPy**: `>=2.3.4` → `>=2.6.27`
   - Major version update
   - New features and optimizations
   - Better integration with language models

### Other Dependencies

1. **NumPy**: `>=1.24.0` → `>=2.0.0`
   - Major version update
   - Improved performance and new features
   - Full compatibility with PyTorch 2.6

2. **Scikit-learn**: `>=1.3.0` → `>=1.5.0`
   - Latest stable version
   - Enhanced ML algorithms and utilities

## Migration Steps

### 1. Update Dependencies

```bash
# Create a backup of your environment
pip freeze > requirements_old.txt

# Update all packages
pip install --upgrade -r requirements.txt
```

### 2. Code Compatibility

The code has been reviewed and is compatible with the updated dependencies. No changes are required to the existing codebase.

### 3. Testing

After updating, run the following tests:

```bash
# Test the main script
python hybrid_rag.py

# Run evaluation
python run_evaluation.py

# Test with Jupyter notebook
jupyter notebook notebooks/hybrid_local_dspy_rag.ipynb
```

## Breaking Changes

### NumPy 2.0

NumPy 2.0 includes some breaking changes, but our code is compatible. Key changes to be aware of:
- Some deprecated functions have been removed
- Type promotion rules have been updated
- Array printing has been improved

### DSPy Updates

DSPy 2.6.x includes significant improvements:
- Better prompt optimization
- Enhanced model integration
- Improved performance

## Performance Improvements

With the updated dependencies, you can expect:
- Faster model inference with PyTorch 2.6
- Better memory efficiency
- Improved tokenization speed
- Enhanced retrieval performance

## Troubleshooting

### Installation Issues

If you encounter issues during installation:

1. **Clear pip cache**: `pip cache purge`
2. **Update pip**: `pip install --upgrade pip`
3. **Install in order**: Install PyTorch first, then other dependencies

### Compatibility Issues

If you experience compatibility problems:

1. Check Python version (3.9+ required)
2. Verify CUDA/MPS availability for GPU support
3. Ensure all dependencies are properly installed

### Performance Issues

If performance degrades after update:

1. Clear model caches
2. Re-download model weights
3. Check device allocation (CPU/GPU/MPS)

## Rollback Instructions

If you need to rollback to previous versions:

```bash
# Restore old requirements
pip install -r requirements_old.txt
```

## Additional Resources

- [PyTorch 2.6 Release Notes](https://pytorch.org/blog/pytorch2-6/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [DSPy Documentation](https://github.com/stanfordnlp/dspy)
- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)