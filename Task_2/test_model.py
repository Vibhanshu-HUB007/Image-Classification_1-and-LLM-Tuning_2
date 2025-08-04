"""
Standalone script to test your existing time extraction model
Run this directly without importing anything else
"""

import os
import json
import torch
import re
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class TimeExtractionResult:
    """Structure for time extraction results"""
    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None
    is_range: bool = False
    confidence: float = 1.0
    raw_text: str = ""

class SafeTimeExtractor:
    """Safe time extractor with fallback methods"""
    
    def __init__(self, model_path="./trained_time_extraction_model"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load model with comprehensive error handling"""
        print(f"🔄 Loading model from: {self.model_path}")
        print(f"📍 Using device: {self.device}")
        
        if not os.path.exists(self.model_path):
            print(f"❌ Model path does not exist: {self.model_path}")
            return False
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ Tokenizer loaded successfully")
            
            # Try different loading strategies
            loading_strategies = [
                ("float16 + auto device", {"torch_dtype": torch.float16, "device_map": "auto"}),
                ("float32 + manual device", {"torch_dtype": torch.float32, "device_map": None}),
                ("default", {})
            ]
            
            for strategy_name, kwargs in loading_strategies:
                try:
                    print(f"🔄 Trying {strategy_name}...")
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
                    
                    if kwargs.get("device_map") is None:
                        self.model = self.model.to(self.device)
                    
                    self.model.eval()
                    print(f"✅ Model loaded with {strategy_name}")
                    return True
                    
                except Exception as e:
                    print(f"⚠️ {strategy_name} failed: {str(e)[:100]}")
                    continue
            
            print("❌ All loading strategies failed")
            return False
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def extract_time_safe(self, query: str) -> TimeExtractionResult:
        """Extract time with multiple fallback methods"""
        print(f"\n🔍 Processing query: '{query}'")
        
        # Try model-based extraction first
        if self.model is not None and self.tokenizer is not None:
            try:
                result = self._model_extraction(query)
                if result.start_datetime:
                    print("✅ Model extraction successful")
                    return result
                else:
                    print("⚠️ Model extraction returned no result")
            except Exception as e:
                print(f"⚠️ Model extraction failed: {str(e)[:100]}")
        
        # Fallback to rule-based extraction
        print("🔄 Using rule-based fallback...")
        return self._rule_based_extraction(query)
    
    def _model_extraction(self, query: str) -> TimeExtractionResult:
        """Try to extract using the trained model"""
        input_text = f"Query: {query}\nResponse:"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Try different generation strategies
        generation_configs = [
            {"max_new_tokens": 50, "do_sample": False, "num_beams": 1},
            {"max_new_tokens": 30, "do_sample": False},
            {"max_new_tokens": 20, "temperature": 1.0, "do_sample": True}
        ]
        
        for config in generation_configs:
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        **config
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.replace(input_text, "").strip()
                
                # Clean response
                if response.endswith("<|endoftext|>"):
                    response = response[:-13].strip()
                
                # Try to parse JSON
                parsed = json.loads(response)
                return TimeExtractionResult(
                    start_datetime=parsed.get("start_datetime"),
                    end_datetime=parsed.get("end_datetime"),
                    is_range=parsed.get("is_range", False),
                    confidence=1.0,
                    raw_text=response
                )
                
            except Exception as e:
                continue
        
        return TimeExtractionResult(confidence=0.0, raw_text="Model extraction failed")
    
    def _rule_based_extraction(self, query: str) -> TimeExtractionResult:
        """Rule-based time extraction as fallback"""
        current_time = datetime.now()
        query_lower = query.lower()
        
        # Define patterns and their handlers
        patterns = [
            # Yesterday patterns
            (r'yesterday.*?(\d{1,2}):?(\d{2})?\s*(pm|am)?', self._parse_yesterday),
            (r'yesterday.*?(\d{1,2})\s*(pm|am)', self._parse_yesterday),
            
            # Today/morning patterns
            (r'today.*?(\d{1,2}):?(\d{2})?\s*(pm|am)?', self._parse_today),
            (r'this morning.*?(\d{1,2}):?(\d{2})?\s*(am)?', self._parse_today),
            
            # Tomorrow patterns
            (r'tomorrow.*?(\d{1,2}):?(\d{2})?\s*(pm|am)?', self._parse_tomorrow),
            
            # Time ranges
            (r'between.*?(\d{1,2})-(\d{1,2})\s*(pm|am)?.*?yesterday', self._parse_yesterday_range),
            (r'between.*?(\d{1,2})-(\d{1,2})\s*(pm|am)?', self._parse_today_range),
            
            # Last night
            (r'last night.*?(\d{1,2}):?(\d{2})?\s*(pm)?', self._parse_last_night),
        ]
        
        for pattern, parser in patterns:
            match = re.search(pattern, query_lower)
            if match:
                try:
                    return parser(match, current_time)
                except Exception as e:
                    print(f"⚠️ Pattern parsing failed: {e}")
                    continue
        
        return TimeExtractionResult(
            confidence=0.0,
            raw_text=f"Could not extract time from: {query}"
        )
    
    def _parse_yesterday(self, match, current_time):
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3) if len(match.groups()) > 2 and match.group(3) else ""
        
        if period.lower() == "pm" and hour < 12:
            hour += 12
        elif period.lower() == "am" and hour == 12:
            hour = 0
        
        yesterday = current_time - timedelta(days=1)
        result_dt = yesterday.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return TimeExtractionResult(
            start_datetime=result_dt.isoformat(),
            confidence=0.8,
            raw_text="Rule-based extraction"
        )
    
    def _parse_today(self, match, current_time):
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3) if len(match.groups()) > 2 and match.group(3) else ""
        
        if period.lower() == "pm" and hour < 12:
            hour += 12
        elif period.lower() == "am" and hour == 12:
            hour = 0
        
        result_dt = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return TimeExtractionResult(
            start_datetime=result_dt.isoformat(),
            confidence=0.8,
            raw_text="Rule-based extraction"
        )
    
    def _parse_tomorrow(self, match, current_time):
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3) if len(match.groups()) > 2 and match.group(3) else ""
        
        if period.lower() == "pm" and hour < 12:
            hour += 12
        elif period.lower() == "am" and hour == 12:
            hour = 0
        
        tomorrow = current_time + timedelta(days=1)
        result_dt = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return TimeExtractionResult(
            start_datetime=result_dt.isoformat(),
            confidence=0.8,
            raw_text="Rule-based extraction"
        )
    
    def _parse_last_night(self, match, current_time):
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        
        # Last night times are typically PM
        if hour < 12:
            hour += 12
        
        yesterday = current_time - timedelta(days=1)
        result_dt = yesterday.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return TimeExtractionResult(
            start_datetime=result_dt.isoformat(),
            confidence=0.8,
            raw_text="Rule-based extraction"
        )
    
    def _parse_yesterday_range(self, match, current_time):
        start_hour = int(match.group(1))
        end_hour = int(match.group(2))
        period = match.group(3) if len(match.groups()) > 2 and match.group(3) else ""
        
        if period.lower() == "pm":
            if start_hour < 12:
                start_hour += 12
            if end_hour < 12:
                end_hour += 12
        
        yesterday = current_time - timedelta(days=1)
        start_dt = yesterday.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        end_dt = yesterday.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        
        return TimeExtractionResult(
            start_datetime=start_dt.isoformat(),
            end_datetime=end_dt.isoformat(),
            is_range=True,
            confidence=0.8,
            raw_text="Rule-based extraction"
        )
    
    def _parse_today_range(self, match, current_time):
        start_hour = int(match.group(1))
        end_hour = int(match.group(2))
        period = match.group(3) if len(match.groups()) > 2 and match.group(3) else ""
        
        if period.lower() == "pm":
            if start_hour < 12:
                start_hour += 12
            if end_hour < 12:
                end_hour += 12
        
        start_dt = current_time.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        end_dt = current_time.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        
        return TimeExtractionResult(
            start_datetime=start_dt.isoformat(),
            end_datetime=end_dt.isoformat(),
            is_range=True,
            confidence=0.8,
            raw_text="Rule-based extraction"
        )

def test_time_extraction_model(model_path="./trained_time_extraction_model"):
    """Test the time extraction model safely"""
    print("🧪 Time Extraction Model Test")
    print("=" * 50)
    print(f"📁 Model path: {model_path}")
    print(f"🖥️ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    # Initialize extractor
    extractor = SafeTimeExtractor(model_path)
    
    # Try to load model
    model_loaded = extractor.load_model()
    if not model_loaded:
        print("⚠️ Model loading failed, will use rule-based extraction only")
    print()
    
    # Test queries
    test_queries = [
        "show me yesterday at 3pm",
        "what happened this morning at 9:30?",
        "between 2-4pm yesterday",
        "tomorrow at 10am",
        "last night around 11",
        "today at 2:15pm",
        "find videos from last night at 10:30"
    ]
    
    print("🔍 Testing Time Extraction:")
    print("-" * 40)
    
    for i, query in enumerate(test_queries, 1):
        result = extractor.extract_time_safe(query)
        
        print(f"\n{i}. Query: '{query}'")
        if result.start_datetime:
            print(f"   ✅ Start: {result.start_datetime}")
            if result.end_datetime:
                print(f"   📅 End: {result.end_datetime}")
            print(f"   🔄 Range: {result.is_range}")
            print(f"   📊 Confidence: {result.confidence:.2f}")
            if result.confidence < 1.0:
                print(f"   🛠️ Method: Fallback extraction")
        else:
            print(f"   ❌ No time extracted")
            print(f"   💬 Message: {result.raw_text}")
    
    print(f"\n🎉 Testing completed!")
    print(f"💡 Tip: Even if the model fails, rule-based extraction provides backup functionality")

if __name__ == "__main__":
    import sys
    
    # Allow custom model path as command line argument
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./trained_time_extraction_model"
    test_time_extraction_model(model_path)