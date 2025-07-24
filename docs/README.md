# nanoGPT MoE Documentation

This directory contains all documentation for the nanoGPT Mixture-of-Experts implementation.

📋 **[DOCS_ORGANIZATION.md](DOCS_ORGANIZATION.md)** - Summary of documentation reorganization

## 📚 Documentation Structure

### 🔧 Fixes & Solutions (`fixes/`)
- **[CHECKPOINT_FIX.md](fixes/CHECKPOINT_FIX.md)** - Fixes for checkpoint loading with DDP/compilation prefixes
- **[🚀 NEW: ADVANCED_CHECKPOINT_LOADING.md](fixes/ADVANCED_CHECKPOINT_LOADING.md)** - **Enhanced checkpoint loading with intelligent prefix handling**
- **[CHECKPOINT_LOADING_ENHANCEMENT.md](fixes/CHECKPOINT_LOADING_ENHANCEMENT.md)** - Enhanced checkpoint loading for advanced MoE training
- **[CHECKPOINT_LOADING_IMPLEMENTATION.md](fixes/CHECKPOINT_LOADING_IMPLEMENTATION.md)** - Implementation details and troubleshooting
- **[GRADIENT_CHECKPOINTING_FIX.md](fixes/GRADIENT_CHECKPOINTING_FIX.md)** - Gradient checkpointing implementation and fixes
- **[OOM_SOLUTION.md](fixes/OOM_SOLUTION.md)** - Out-of-memory issue solutions and memory optimization
- **[TRAIN_ADVANCED_FIX.md](fixes/TRAIN_ADVANCED_FIX.md)** - Fix for UnboundLocalError in advanced training script
- **[FIX_SUMMARY.md](fixes/FIX_SUMMARY.md)** - Summary of all fixes applied to the codebase

### 📖 User Guides (`guides/`)
- **[SAMPLING_GUIDE.md](guides/SAMPLING_GUIDE.md)** - 🚀 **Enhanced** - Complete guide to sampling with KV cache performance optimizations
- **[🚀 NEW: KV_CACHE_GUIDE.md](guides/KV_CACHE_GUIDE.md)** - **Fast text generation with 2-10x speedup using KV cache**
- **[🚀 NEW: CHECKPOINT_LOADING_QUICKREF.md](guides/CHECKPOINT_LOADING_QUICKREF.md)** - **Quick reference for checkpoint loading issues**
- **[OPTIMIZATION_GUIDE.md](guides/OPTIMIZATION_GUIDE.md)** - Performance optimization strategies and configurations
- **[MODEL_ARCHITECTURE.md](guides/MODEL_ARCHITECTURE.md)** - Comprehensive model architecture guide with V100 configurations
- **[ADVANCED_TRAINING_ENHANCEMENTS.md](guides/ADVANCED_TRAINING_ENHANCEMENTS.md)** - Enhanced training script features and usage
- **[TEST_ORGANIZATION.md](guides/TEST_ORGANIZATION.md)** - Guide to the test suite organization and usage

### 🔗 Integration (`integration/`)
- **[WANDB_INTEGRATION.md](integration/WANDB_INTEGRATION.md)** - Weights & Biases logging integration guide
- **[WANDB_INTEGRATION_SUMMARY.md](integration/WANDB_INTEGRATION_SUMMARY.md)** - Summary of wandb integration features

## 🚀 Quick Navigation

### For Users
- **New to sampling?** → [SAMPLING_GUIDE.md](guides/SAMPLING_GUIDE.md)
- **🚀 Want FAST generation?** → [KV_CACHE_GUIDE.md](guides/KV_CACHE_GUIDE.md)
- **Checkpoint loading issues?** → [CHECKPOINT_LOADING_QUICKREF.md](guides/CHECKPOINT_LOADING_QUICKREF.md)
- **Understanding the architecture?** → [MODEL_ARCHITECTURE.md](guides/MODEL_ARCHITECTURE.md)
- **Using advanced training?** → [ADVANCED_TRAINING_ENHANCEMENTS.md](guides/ADVANCED_TRAINING_ENHANCEMENTS.md)
- **Performance issues?** → [OPTIMIZATION_GUIDE.md](guides/OPTIMIZATION_GUIDE.md)
- **Want to track experiments?** → [WANDB_INTEGRATION.md](integration/WANDB_INTEGRATION.md)

### For Developers
- **Checkpoint loading issues?** → [CHECKPOINT_FIX.md](fixes/CHECKPOINT_FIX.md) or [ADVANCED_CHECKPOINT_LOADING.md](fixes/ADVANCED_CHECKPOINT_LOADING.md)
- **Enhanced checkpoint features?** → [CHECKPOINT_LOADING_ENHANCEMENT.md](fixes/CHECKPOINT_LOADING_ENHANCEMENT.md)
- **Memory problems?** → [OOM_SOLUTION.md](fixes/OOM_SOLUTION.md)
- **Running tests?** → [TEST_ORGANIZATION.md](guides/TEST_ORGANIZATION.md)

### For Troubleshooting
- **General fixes** → [FIX_SUMMARY.md](fixes/FIX_SUMMARY.md)
- **Gradient checkpointing** → [GRADIENT_CHECKPOINTING_FIX.md](fixes/GRADIENT_CHECKPOINTING_FIX.md)

## 📋 Documentation Categories

| Category | Purpose | Audience |
|----------|---------|----------|
| **Fixes** | Bug fixes and solutions | Developers experiencing issues |
| **Guides** | How-to guides and tutorials | All users |
| **Integration** | Third-party integrations | Users wanting to extend functionality |

## 🔄 Keeping Documentation Updated

When adding new features or fixes:
1. Add documentation to the appropriate category
2. Update this README.md index
3. Link from relevant guides
4. Update the main project README.md if needed

---

📂 **File Organization:**
```
docs/
├── README.md                           # This index
├── fixes/                              # Bug fixes and solutions
│   ├── CHECKPOINT_FIX.md
│   ├── GRADIENT_CHECKPOINTING_FIX.md
│   ├── OOM_SOLUTION.md
│   └── FIX_SUMMARY.md
├── guides/                             # User guides
│   ├── SAMPLING_GUIDE.md
│   ├── KV_CACHE_GUIDE.md
│   ├── CHECKPOINT_LOADING_QUICKREF.md
│   ├── OPTIMIZATION_GUIDE.md
│   └── TEST_ORGANIZATION.md
└── integration/                        # Third-party integrations
    ├── WANDB_INTEGRATION.md
    └── WANDB_INTEGRATION_SUMMARY.md
```
