# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.1.0] - 2025-07-18

### Changed
- Updated all dependencies to latest stable versions:
  - PyTorch: 2.1.0 → 2.6.0
  - TorchVision: 0.17.0 → 0.21.0
  - Transformers: 4.36.0 → 4.53.2
  - SentencePiece: 0.1.99 → 0.2.0
  - NLTK: 3.8.1 → 3.9.1
  - DSPy: 2.3.4 → 2.6.27
  - NumPy: 1.24.0 → 2.0.0
  - Scikit-learn: 1.3.0 → 1.5.0

### Added
- UPGRADE_GUIDE.md with detailed migration instructions
- Compatibility notes in README.md

### Improved
- Better performance with PyTorch 2.6
- Enhanced MPS support for Apple Silicon
- Improved tokenization speed

## [1.0.0] - 2024-03-15

### Added
- Initial release of HybridLocalDSPyRAG-ollama 