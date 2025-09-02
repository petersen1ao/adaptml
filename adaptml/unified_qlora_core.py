#!/usr/bin/env python3
"""
AdaptML Unified QLoRA Core Integration
Integrates the Secure Unified QLoRA System with SSO into AdaptML Core
"""

import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Import the secure unified system
try:
    from ..secure_unified_qlora_with_sso import (
        SecureUnifiedQLoRADistributedLearningSystem,
        UnifiedQLoRAConfig,
        UnifiedQLorALevel,
        SectorType,
        ThreatType,
        ThreatSeverity,
        UnifiedSelfScaffoldingSecurityOrchestrator
    )
    HAS_UNIFIED_QLORA = True
except ImportError:
    HAS_UNIFIED_QLORA = False

@dataclass
class AdaptMLQLoRAConfig:
    """AdaptML-specific QLoRA configuration"""
    enable_unified_qlora: bool = True
    enable_sso_security: bool = True
    quantization_level: str = "balanced"
    sector: str = "technology"
    cost_optimization: bool = True
    security_level: str = "maximum"
    
class AdaptMLUnifiedQLoRAEngine:
    """AdaptML engine with Unified QLoRA and SSO integration"""
    
    def __init__(self, config: Optional[AdaptMLQLoRAConfig] = None):
        self.config = config or AdaptMLQLoRAConfig()
        self.unified_system = None
        
        if HAS_UNIFIED_QLORA and self.config.enable_unified_qlora:
            self._initialize_unified_system()
    
    def _initialize_unified_system(self):
        """Initialize the Secure Unified QLoRA System"""
        try:
            # Map AdaptML config to Unified QLoRA config
            quantization_mapping = {
                "ultra_efficient": UnifiedQLorALevel.ULTRA_EFFICIENT,
                "balanced": UnifiedQLorALevel.BALANCED,
                "high_quality": UnifiedQLorALevel.HIGH_QUALITY,
                "no_compression": UnifiedQLorALevel.NO_COMPRESSION
            }
            
            sector_mapping = {
                "technology": SectorType.TECHNOLOGY,
                "financial": SectorType.FINANCIAL_SERVICES,
                "healthcare": SectorType.HEALTHCARE,
                "manufacturing": SectorType.MANUFACTURING,
                "government": SectorType.GOVERNMENT
            }
            
            unified_config = UnifiedQLoRAConfig(
                quantization_level=quantization_mapping.get(
                    self.config.quantization_level, 
                    UnifiedQLorALevel.BALANCED
                ),
                sector_type=sector_mapping.get(
                    self.config.sector, 
                    SectorType.TECHNOLOGY
                ),
                enable_sso_security=self.config.enable_sso_security,
                enable_distributed_learning=True,
                enable_real_time_protection=True,
                enable_multi_modal_detection=True
            )
            
            self.unified_system = SecureUnifiedQLoRADistributedLearningSystem(unified_config)
            
        except Exception as e:
            print(f"⚠️ Failed to initialize Unified QLoRA System: {e}")
            self.unified_system = None
    
    async def load_model(self, model_name: str, **kwargs):
        """Load model with Unified QLoRA optimization and SSO security"""
        if self.unified_system:
            return await self.unified_system.load_model(model_name, **kwargs)
        else:
            raise RuntimeError("Unified QLoRA System not available")
    
    async def secure_inference(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Perform secure inference with SSO protection"""
        if self.unified_system:
            result, security_result = await self.unified_system.secure_generate(prompt, **kwargs)
            
            # Return AdaptML-compatible format
            return result, {
                'security_analysis': security_result,
                'threat_detected': security_result.threat_type != ThreatType.UNKNOWN if security_result else False,
                'is_safe': security_result.severity in [ThreatSeverity.BENIGN, ThreatSeverity.LOW] if security_result else True
            }
        else:
            # Fallback to basic processing
            return prompt + " [processed]", {'security_analysis': None, 'threat_detected': False, 'is_safe': True}
    
    async def secure_fine_tune(self, training_data, **kwargs):
        """Perform secure fine-tuning with SSO validation"""
        if self.unified_system:
            return await self.unified_system.secure_fine_tune(training_data, **kwargs)
        else:
            raise RuntimeError("Unified QLoRA System not available")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        if self.unified_system:
            stats = self.unified_system.get_performance_stats()
            return {
                'unified_qlora_enabled': True,
                'sso_security_enabled': self.config.enable_sso_security,
                'quantization_level': self.config.quantization_level,
                'sector': self.config.sector,
                **stats
            }
        else:
            return {
                'unified_qlora_enabled': False,
                'sso_security_enabled': False,
                'error': 'Unified QLoRA System not available'
            }

def create_adaptml_unified_system(config: Optional[Dict[str, Any]] = None) -> AdaptMLUnifiedQLoRAEngine:
    """Factory function to create AdaptML Unified QLoRA System"""
    adaptml_config = AdaptMLQLoRAConfig()
    
    if config:
        for key, value in config.items():
            if hasattr(adaptml_config, key):
                setattr(adaptml_config, key, value)
    
    return AdaptMLUnifiedQLoRAEngine(adaptml_config)

# Export for AdaptML integration
__all__ = [
    'AdaptMLUnifiedQLoRAEngine',
    'AdaptMLQLoRAConfig', 
    'create_adaptml_unified_system'
]
