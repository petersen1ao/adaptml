"""
Test suite for AdaptML functionality
"""
import pytest
import sys
import os

# Add the parent directory to path to import adaptml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import adaptml
from adaptml import AdaptiveInference, AdaptiveConfig, ModelSize, DeviceType


class TestAdaptMLPackage:
    """Test AdaptML package metadata"""
    
    def test_package_metadata(self):
        """Test package has correct metadata"""
        assert adaptml.__version__ == "0.1.0"
        assert adaptml.__author__ == "AdaptML Team"
        assert adaptml.__email__ == "info2adaptml@gmail.com"
        assert adaptml.__website__ == "https://adaptml-web-showcase.lovable.app/"
    
    def test_imports(self):
        """Test all main classes can be imported"""
        assert AdaptiveInference is not None
        assert AdaptiveConfig is not None
        assert ModelSize is not None
        assert DeviceType is not None


class TestAdaptiveInference:
    """Test AdaptiveInference functionality"""
    
    def test_initialization(self):
        """Test AdaptiveInference can be initialized"""
        inference = AdaptiveInference()
        assert inference is not None
        assert inference.config is not None
        assert inference.device_profiler is not None
        assert inference.model_registry is not None
    
    def test_model_registration(self):
        """Test model registration"""
        inference = AdaptiveInference()
        model_id = inference.register_model(
            name="test_model",
            model_data={"type": "mock", "size": "small"},
            metadata={"description": "Test model"}
        )
        assert model_id is not None
        assert model_id.startswith("model_")
    
    def test_prediction(self):
        """Test basic prediction functionality"""
        inference = AdaptiveInference()
        model_id = inference.register_model(
            name="test_model",
            model_data={"type": "mock"},
            metadata={"description": "Test model"}
        )
        
        result = inference.predict(model_id, "test input")
        
        assert result is not None
        assert result.prediction is not None
        assert result.cost >= 0
        assert result.latency >= 0
        assert result.confidence >= 0
        assert result.model_used == "test_model"
    
    def test_stats(self):
        """Test system statistics"""
        inference = AdaptiveInference()
        inference.register_model("test", {"type": "mock"})
        
        stats = inference.get_stats()
        assert stats["registered_models"] >= 1
        assert "available_engines" in stats
        assert "device_info" in stats
        assert "mock" in stats["available_engines"]


class TestConfigurations:
    """Test configuration classes"""
    
    def test_adaptive_config(self):
        """Test AdaptiveConfig creation"""
        config = AdaptiveConfig(
            cost_threshold=0.005,
            prefer_cost=True
        )
        assert config.cost_threshold == 0.005
        assert config.prefer_cost == True
    
    def test_enums(self):
        """Test enum values"""
        assert ModelSize.SMALL.value == "small"
        assert DeviceType.CPU.value == "cpu"


class TestQuickstart:
    """Test quickstart functionality"""
    
    def test_quickstart_runs(self):
        """Test that quickstart demo runs without errors"""
        try:
            adaptml.quickstart()
            # If we get here, quickstart ran successfully
            assert True
        except Exception as e:
            pytest.fail(f"Quickstart failed with error: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
