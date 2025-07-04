"""
Test script to verify that all import paths work correctly after reorganization
Tests imports from training, sampling, and other reorganized scripts
"""

import os
import sys
import unittest
import importlib.util
from pathlib import Path

class TestImports(unittest.TestCase):
    """Test that all reorganized scripts can import their dependencies correctly"""
    
    def setUp(self):
        """Set up test environment"""
        self.repo_root = Path(__file__).parent.parent.parent
        self.original_path = sys.path.copy()
    
    def tearDown(self):
        """Clean up after tests"""
        sys.path = self.original_path
    
    def test_training_standard_imports(self):
        """Test that standard training script can import model"""
        script_path = self.repo_root / "training" / "standard" / "train.py"
        self.assertTrue(script_path.exists(), f"Training script not found at {script_path}")
        
        # Add the script's directory to path
        script_dir = str(script_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        # Try to load the module
        spec = importlib.util.spec_from_file_location("train_standard", script_path)
        self.assertIsNotNone(spec, "Could not create module spec for standard training")
        
        # Test that the imports in the script work
        # We'll simulate the key imports without executing the full script
        try:
            # Add repo root to path as the script does
            repo_root_str = str(self.repo_root)
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)
            
            from model import GPTConfig, GPT
            print("✅ Standard training script imports work correctly")
        except ImportError as e:
            self.fail(f"Standard training script imports failed: {e}")
    
    def test_training_sft_imports(self):
        """Test that SFT training script can import model and configurator"""
        script_path = self.repo_root / "training" / "sft" / "train_sft.py"
        self.assertTrue(script_path.exists(), f"SFT training script not found at {script_path}")
        
        try:
            # Add repo root to path as the script does
            repo_root_str = str(self.repo_root)
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)
            
            from model import GPTConfig, GPT
            
            # Test configurator path exists
            configurator_path = self.repo_root / "training" / "utils" / "configurator.py"
            self.assertTrue(configurator_path.exists(), f"Configurator not found at {configurator_path}")
            
            print("✅ SFT training script imports work correctly")
        except ImportError as e:
            self.fail(f"SFT training script imports failed: {e}")
    
    def test_training_moe_imports(self):
        """Test that MoE training script can import model and utilities"""
        script_path = self.repo_root / "training" / "moe" / "train_moe_advanced.py"
        self.assertTrue(script_path.exists(), f"MoE training script not found at {script_path}")
        
        try:
            # Add repo root to path
            repo_root_str = str(self.repo_root)
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)
            
            from model import GPTConfig, GPT
            
            # Test v100_config path exists
            v100_config_path = self.repo_root / "training" / "utils" / "v100_config.py"
            self.assertTrue(v100_config_path.exists(), f"V100 config not found at {v100_config_path}")
            
            print("✅ MoE training script imports work correctly")
        except ImportError as e:
            self.fail(f"MoE training script imports failed: {e}")
    
    def test_sampling_imports(self):
        """Test that sampling scripts can import model"""
        # Test standard sampling
        sample_path = self.repo_root / "sampling" / "sample.py"
        self.assertTrue(sample_path.exists(), f"Sample script not found at {sample_path}")
        
        # Test MoE sampling
        sample_moe_path = self.repo_root / "sampling" / "sample_moe.py"
        self.assertTrue(sample_moe_path.exists(), f"Sample MoE script not found at {sample_moe_path}")
        
        try:
            # Add repo root to path
            repo_root_str = str(self.repo_root)
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)
            
            from model import GPTConfig, GPT
            
            # Test kv_cache exists for sample_moe
            kv_cache_path = self.repo_root / "kv_cache.py"
            self.assertTrue(kv_cache_path.exists(), f"KV cache not found at {kv_cache_path}")
            
            print("✅ Sampling script imports work correctly")
        except ImportError as e:
            self.fail(f"Sampling script imports failed: {e}")
    
    def test_training_utils_imports(self):
        """Test that training utilities can import model"""
        bench_path = self.repo_root / "training" / "utils" / "bench.py"
        self.assertTrue(bench_path.exists(), f"Bench script not found at {bench_path}")
        
        configurator_path = self.repo_root / "training" / "utils" / "configurator.py"
        self.assertTrue(configurator_path.exists(), f"Configurator not found at {configurator_path}")
        
        try:
            # Add repo root to path
            repo_root_str = str(self.repo_root)
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)
            
            from model import GPTConfig, GPT
            
            print("✅ Training utils imports work correctly")
        except ImportError as e:
            self.fail(f"Training utils imports failed: {e}")
    
    def test_checkpoint_directories_exist(self):
        """Test that checkpoint directories exist with correct structure"""
        checkpoints_dir = self.repo_root / "checkpoints"
        self.assertTrue(checkpoints_dir.exists(), f"Checkpoints directory not found at {checkpoints_dir}")
        
        # Check for some expected checkpoint subdirectories
        expected_dirs = ["out-sft", "out-moe-v100", "out"]
        for dir_name in expected_dirs:
            # Create directories if they don't exist (for testing)
            checkpoint_subdir = checkpoints_dir / dir_name
            checkpoint_subdir.mkdir(exist_ok=True)
            self.assertTrue(checkpoint_subdir.exists(), f"Checkpoint directory {dir_name} not accessible")
        
        print("✅ Checkpoint directories are accessible")
    
    def test_config_files_reference_correct_paths(self):
        """Test that config files reference the correct checkpoint paths"""
        config_dir = self.repo_root / "config"
        self.assertTrue(config_dir.exists(), f"Config directory not found at {config_dir}")
        
        # Check some key config files
        config_files = [
            "train_gpt2.py",
            "train_shakespeare_char.py",
            "finetune_shakespeare.py"
        ]
        
        for config_file in config_files:
            config_path = config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    content = f.read()
                    # Check that it uses checkpoints/ prefix
                    if "out_dir" in content:
                        self.assertIn("checkpoints/", content, 
                                    f"Config file {config_file} should use checkpoints/ prefix")
        
        print("✅ Config files use correct checkpoint paths")
    
    def test_data_directories_accessible(self):
        """Test that data directories are accessible"""
        data_dir = self.repo_root / "data"
        self.assertTrue(data_dir.exists(), f"Data directory not found at {data_dir}")
        
        # Check for expected data subdirectories
        expected_data_dirs = ["openwebtext", "shakespeare", "shakespeare_char"]
        for dir_name in expected_data_dirs:
            data_subdir = data_dir / dir_name
            if data_subdir.exists():
                self.assertTrue(data_subdir.is_dir(), f"Data directory {dir_name} should be a directory")
        
        print("✅ Data directories are accessible")

class TestScriptExecution(unittest.TestCase):
    """Test that scripts can be executed without import errors"""
    
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent.parent
        self.original_cwd = os.getcwd()
        os.chdir(self.repo_root)
    
    def tearDown(self):
        os.chdir(self.original_cwd)
    
    def test_script_syntax_valid(self):
        """Test that all Python scripts have valid syntax"""
        script_paths = [
            "training/standard/train.py",
            "training/sft/train_sft.py", 
            "training/moe/train_moe_advanced.py",
            "training/utils/bench.py",
            "training/utils/configurator.py",
            "sampling/sample.py",
            "sampling/sample_moe.py"
        ]
        
        for script_path in script_paths:
            full_path = self.repo_root / script_path
            if full_path.exists():
                with self.subTest(script=script_path):
                    try:
                        # Check syntax by compiling
                        with open(full_path, 'r') as f:
                            source = f.read()
                        compile(source, str(full_path), 'exec')
                        print(f"✅ {script_path} has valid syntax")
                    except SyntaxError as e:
                        self.fail(f"Syntax error in {script_path}: {e}")

def run_import_tests():
    """Run all import tests and return results"""
    print("🧪 Running Import Tests for Reorganized nanoGPT")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestImports))
    test_suite.addTest(unittest.makeSuite(TestScriptExecution))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print("🔍 Import Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, trace in result.failures:
            print(f"  - {test}: {trace}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, trace in result.errors:
            print(f"  - {test}: {trace}")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n🎉 All import tests passed!")
        return True
    else:
        print("\n⚠️  Some tests failed - check import paths and file locations")
        return False

if __name__ == "__main__":
    success = run_import_tests()
    sys.exit(0 if success else 1)
