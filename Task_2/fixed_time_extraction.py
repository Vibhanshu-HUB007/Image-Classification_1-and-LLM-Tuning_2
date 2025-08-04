import os
import json
import torch
import pandas as pd
from datetime import datetime, timedelta
import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
import dateparser
from dateutil.parser import parse as dateutil_parse
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TimeExtractionResult:
    """Structure for time extraction results"""
    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None
    is_range: bool = False
    confidence: float = 1.0
    raw_text: str = ""

class TimeExtractionDataGenerator:
    """Generate training data for time extraction"""
    
    def __init__(self):
        self.current_time = datetime.now()
        
    def generate_training_data(self, num_samples: int = 1000) -> List[Dict]:
        """Generate diverse training examples"""
        training_data = []
        
        # Template-based generation
        templates = [
            # Single time points
            ("yesterday evening around {time}", self._yesterday_evening),
            ("this morning at {time}", self._this_morning),
            ("last night {time}", self._last_night),
            ("show it on {date} at {time}", self._specific_date_time),
            ("last week {day} {time}", self._last_week_day),
            ("yesterday at {time}", self._yesterday_at),
            ("today at {time}", self._today_at),
            ("tomorrow at {time}", self._tomorrow_at),
            ("on {date} at {time}", self._on_date_at_time),
            
            # Time ranges
            ("yesterday between {start_time}-{end_time}", self._yesterday_between),
            ("this morning from {start_time} to {end_time}", self._this_morning_range),
            ("last week from {start_time}-{end_time}", self._last_week_range),
            ("on {date} between {start_time} and {end_time}", self._date_between),
            ("tomorrow from {start_time} to {end_time}", self._tomorrow_range),
        ]
        
        for i in range(num_samples):
            template_func = templates[i % len(templates)]
            template, func = template_func
            
            try:
                example = func(template)
                if example:
                    training_data.append(example)
            except Exception as e:
                continue
                
        # Add real-world examples
        real_examples = self._get_real_world_examples()
        training_data.extend(real_examples)
        
        return training_data
    
    def _yesterday_evening(self, template: str) -> Dict:
        times = ["8:30", "7:00", "9:15", "6:45", "8:00"]
        time_str = times[hash(str(self.current_time)) % len(times)]
        query = template.format(time=time_str)
        
        yesterday = self.current_time - timedelta(days=1)
        # Convert to 24-hour format for evening times (add 12 hours if < 12)
        hour, minute = map(int, time_str.split(':'))
        if hour < 12:
            hour += 12  # Convert PM times
        time_obj = datetime.strptime(f"{hour}:{minute:02d}", "%H:%M").time()
        result_dt = datetime.combine(yesterday.date(), time_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": result_dt.isoformat(),
                "end_datetime": None,
                "is_range": False
            })
        }
    
    def _this_morning(self, template: str) -> Dict:
        times = ["7:00", "8:30", "9:00", "6:30", "10:00"]
        time_str = times[hash(str(self.current_time)) % len(times)]
        query = template.format(time=time_str)
        
        today_morning = self.current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        time_obj = datetime.strptime(time_str, "%H:%M").time()
        result_dt = datetime.combine(today_morning.date(), time_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": result_dt.isoformat(),
                "end_datetime": None,
                "is_range": False
            })
        }
    
    def _last_night(self, template: str) -> Dict:
        times = ["11:30", "10:00", "11:00", "9:30", "10:30"]
        time_str = times[hash(str(self.current_time)) % len(times)]
        query = template.format(time=time_str)
        
        yesterday = self.current_time - timedelta(days=1)
        # Convert to 24-hour format for night times (add 12 hours if < 12)
        hour, minute = map(int, time_str.split(':'))
        if hour < 12:
            hour += 12  # Convert PM times
        time_obj = datetime.strptime(f"{hour}:{minute:02d}", "%H:%M").time()
        result_dt = datetime.combine(yesterday.date(), time_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": result_dt.isoformat(),
                "end_datetime": None,
                "is_range": False
            })
        }
    
    def _specific_date_time(self, template: str) -> Dict:
        dates = ["26th April", "15th March", "3rd May", "10th June", "22nd July"]
        times = ["10:00", "14:30", "16:00", "11:00", "15:30"]
        
        date_str = dates[hash(str(self.current_time)) % len(dates)]
        time_str = times[hash(str(self.current_time) + "time") % len(times)]
        
        query = template.format(date=date_str, time=time_str)
        
        # Parse the date
        try:
            parsed_date = dateparser.parse(f"{date_str} {self.current_time.year}")
            if parsed_date is None:
                # Fallback parsing
                parsed_date = self.current_time
        except:
            parsed_date = self.current_time
            
        time_obj = datetime.strptime(time_str, "%H:%M").time()
        result_dt = datetime.combine(parsed_date.date(), time_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": result_dt.isoformat(),
                "end_datetime": None,
                "is_range": False
            })
        }
    
    def _last_week_day(self, template: str) -> Dict:
        days = ["Saturday", "Sunday", "Monday", "Friday", "Thursday"]
        times = ["10:00", "14:00", "16:30", "11:30", "15:00"]
        
        day_str = days[hash(str(self.current_time)) % len(days)]
        time_str = times[hash(str(self.current_time) + day_str) % len(times)]
        
        query = template.format(day=day_str, time=time_str)
        
        # Calculate last week's date for the specified day
        days_back = 7 + self.current_time.weekday()
        last_week_date = self.current_time - timedelta(days=days_back)
        
        # Find the specific day in that week
        target_weekday = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"].index(day_str.lower())
        days_to_target = target_weekday - last_week_date.weekday()
        target_date = last_week_date + timedelta(days=days_to_target)
        
        time_obj = datetime.strptime(time_str, "%H:%M").time()
        result_dt = datetime.combine(target_date.date(), time_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": result_dt.isoformat(),
                "end_datetime": None,
                "is_range": False
            })
        }
    
    def _yesterday_at(self, template: str) -> Dict:
        times = ["15:30", "14:00", "13:30", "16:00", "17:30"]
        time_str = times[hash(str(self.current_time)) % len(times)]
        query = template.format(time=time_str)
        
        yesterday = self.current_time - timedelta(days=1)
        time_obj = datetime.strptime(time_str, "%H:%M").time()
        result_dt = datetime.combine(yesterday.date(), time_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": result_dt.isoformat(),
                "end_datetime": None,
                "is_range": False
            })
        }
    
    def _today_at(self, template: str) -> Dict:
        times = ["12:00", "13:30", "14:45", "16:30", "18:00"]
        time_str = times[hash(str(self.current_time)) % len(times)]
        query = template.format(time=time_str)
        
        time_obj = datetime.strptime(time_str, "%H:%M").time()
        result_dt = datetime.combine(self.current_time.date(), time_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": result_dt.isoformat(),
                "end_datetime": None,
                "is_range": False
            })
        }
    
    def _tomorrow_at(self, template: str) -> Dict:
        times = ["09:00", "10:30", "11:45", "13:30", "15:00"]
        time_str = times[hash(str(self.current_time)) % len(times)]
        query = template.format(time=time_str)
        
        tomorrow = self.current_time + timedelta(days=1)
        time_obj = datetime.strptime(time_str, "%H:%M").time()
        result_dt = datetime.combine(tomorrow.date(), time_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": result_dt.isoformat(),
                "end_datetime": None,
                "is_range": False
            })
        }
    
    def _on_date_at_time(self, template: str) -> Dict:
        dates = ["March 15th", "April 20th", "May 5th", "June 10th"]
        times = ["10:00", "14:30", "16:00", "11:30"]
        
        date_str = dates[hash(str(self.current_time)) % len(dates)]
        time_str = times[hash(str(self.current_time) + date_str) % len(times)]
        
        query = template.format(date=date_str, time=time_str)
        
        try:
            parsed_date = dateparser.parse(f"{date_str} {self.current_time.year}")
            if parsed_date is None:
                parsed_date = self.current_time
        except:
            parsed_date = self.current_time
            
        time_obj = datetime.strptime(time_str, "%H:%M").time()
        result_dt = datetime.combine(parsed_date.date(), time_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": result_dt.isoformat(),
                "end_datetime": None,
                "is_range": False
            })
        }
    
    def _yesterday_between(self, template: str) -> Dict:
        time_ranges = [("3:00", "4:00"), ("15:00", "16:00"), ("10:00", "11:00"), ("14:00", "15:00"), ("16:00", "17:00")]
        start_time, end_time = time_ranges[hash(str(self.current_time)) % len(time_ranges)]
        
        query = template.format(start_time=start_time.split(':')[0], end_time=end_time.split(':')[0])
        
        yesterday = self.current_time - timedelta(days=1)
        start_obj = datetime.strptime(start_time, "%H:%M").time()
        end_obj = datetime.strptime(end_time, "%H:%M").time()
        
        start_dt = datetime.combine(yesterday.date(), start_obj)
        end_dt = datetime.combine(yesterday.date(), end_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": start_dt.isoformat(),
                "end_datetime": end_dt.isoformat(),
                "is_range": True
            })
        }
    
    def _this_morning_range(self, template: str) -> Dict:
        time_ranges = [("7:00", "9:00"), ("8:00", "10:00"), ("6:30", "8:30")]
        start_time, end_time = time_ranges[hash(str(self.current_time)) % len(time_ranges)]
        
        query = template.format(start_time=start_time, end_time=end_time)
        
        today = self.current_time.date()
        start_obj = datetime.strptime(start_time, "%H:%M").time()
        end_obj = datetime.strptime(end_time, "%H:%M").time()
        
        start_dt = datetime.combine(today, start_obj)
        end_dt = datetime.combine(today, end_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": start_dt.isoformat(),
                "end_datetime": end_dt.isoformat(),
                "is_range": True
            })
        }
    
    def _last_week_range(self, template: str) -> Dict:
        time_ranges = [("9:00", "11:00"), ("14:00", "16:00"), ("10:30", "12:30")]
        start_time, end_time = time_ranges[hash(str(self.current_time)) % len(time_ranges)]
        
        query = template.format(start_time=start_time, end_time=end_time)
        
        last_week = self.current_time - timedelta(days=7)
        start_obj = datetime.strptime(start_time, "%H:%M").time()
        end_obj = datetime.strptime(end_time, "%H:%M").time()
        
        start_dt = datetime.combine(last_week.date(), start_obj)
        end_dt = datetime.combine(last_week.date(), end_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": start_dt.isoformat(),
                "end_datetime": end_dt.isoformat(),
                "is_range": True
            })
        }
    
    def _date_between(self, template: str) -> Dict:
        dates = ["March 15th", "April 20th", "May 5th"]
        time_ranges = [("10:00", "12:00"), ("14:00", "16:00"), ("9:30", "11:30")]
        
        date_str = dates[hash(str(self.current_time)) % len(dates)]
        start_time, end_time = time_ranges[hash(str(self.current_time) + date_str) % len(time_ranges)]
        
        query = template.format(date=date_str, start_time=start_time, end_time=end_time)
        
        try:
            parsed_date = dateparser.parse(f"{date_str} {self.current_time.year}")
            if parsed_date is None:
                parsed_date = self.current_time
        except:
            parsed_date = self.current_time
            
        start_obj = datetime.strptime(start_time, "%H:%M").time()
        end_obj = datetime.strptime(end_time, "%H:%M").time()
        
        start_dt = datetime.combine(parsed_date.date(), start_obj)
        end_dt = datetime.combine(parsed_date.date(), end_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": start_dt.isoformat(),
                "end_datetime": end_dt.isoformat(),
                "is_range": True
            })
        }
    
    def _tomorrow_range(self, template: str) -> Dict:
        time_ranges = [("9:00", "11:00"), ("13:00", "15:00"), ("10:30", "12:30")]
        start_time, end_time = time_ranges[hash(str(self.current_time)) % len(time_ranges)]
        
        query = template.format(start_time=start_time, end_time=end_time)
        
        tomorrow = self.current_time + timedelta(days=1)
        start_obj = datetime.strptime(start_time, "%H:%M").time()
        end_obj = datetime.strptime(end_time, "%H:%M").time()
        
        start_dt = datetime.combine(tomorrow.date(), start_obj)
        end_dt = datetime.combine(tomorrow.date(), end_obj)
        
        return {
            "query": query,
            "response": json.dumps({
                "start_datetime": start_dt.isoformat(),
                "end_datetime": end_dt.isoformat(),
                "is_range": True
            })
        }
    
    def _get_real_world_examples(self) -> List[Dict]:
        """Add some real-world examples"""
        examples = []
        
        # Calculate last Tuesday
        days_since_tuesday = (self.current_time.weekday() - 1) % 7
        if days_since_tuesday == 0:  # Today is Tuesday
            days_since_tuesday = 7
        last_tuesday = self.current_time - timedelta(days=days_since_tuesday)
        
        examples.append({
            "query": "show me the video from last Tuesday at 3pm",
            "response": json.dumps({
                "start_datetime": last_tuesday.replace(hour=15, minute=0, second=0, microsecond=0).isoformat(),
                "end_datetime": None,
                "is_range": False
            })
        })
        
        examples.append({
            "query": "what happened between 2-4pm yesterday?",
            "response": json.dumps({
                "start_datetime": (self.current_time - timedelta(days=1)).replace(hour=14, minute=0, second=0, microsecond=0).isoformat(),
                "end_datetime": (self.current_time - timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0).isoformat(),
                "is_range": True
            })
        })
        
        return examples

class BasicTimeExtractionTrainer:
    """Basic trainer with proper device handling"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_model(self):
        """Setup tokenizer and model with proper device handling and error recovery"""
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Try loading with mixed precision first
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
                print("✅ Model loaded with mixed precision")
            except Exception as e:
                print(f"⚠️ Mixed precision loading failed: {e}")
                print("🔄 Trying with float32...")
                
                # Fallback to float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map=None
                )
                self.model = self.model.to(self.device)
                print("✅ Model loaded with float32")
            
            # Resize token embeddings if needed
            if len(self.tokenizer) != self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"🔧 Resized token embeddings to {len(self.tokenizer)}")
            
            # Set model to evaluation mode
            self.model.eval()
            print("📊 Model set to evaluation mode")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("💡 The model might be corrupted or incompatible")
            raise e
        
    def prepare_dataset(self, training_data: List[Dict]):
        """Prepare dataset for training"""
        def tokenize_function(examples):
            texts = [
                f"Query: {q}\nResponse: {r}<|endoftext|>"
                for q, r in zip(examples["query"], examples["response"])
            ]
            
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        df = pd.DataFrame(training_data)
        dataset = Dataset.from_pandas(df)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        return tokenized_dataset
    
    def train(self, training_data: List[Dict], output_dir: str = "./time_extraction_model"):
        """Train the model"""
        if self.model is None:
            self.setup_model()
            
        train_dataset = self.prepare_dataset(training_data)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="no",
            save_strategy="steps",
            load_best_model_at_end=False,
            report_to=None,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        print("Starting training...")
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
        
    def extract_time(self, query: str) -> TimeExtractionResult:
        """Extract time from query using the trained model with robust inference"""
        if self.model is None:
            self.setup_model()
            
        input_text = f"Query: {query}\nResponse:"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                # Use more conservative generation parameters to avoid CUDA errors
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=50,  # Reduced from 100
                    do_sample=False,    # Use greedy decoding instead of sampling
                    num_beams=1,        # Simple beam search
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs.get("attention_mask", None)
                )
        except RuntimeError as e:
            print(f"CUDA generation failed, trying with different parameters: {e}")
            try:
                # Fallback: try with even more conservative settings
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=30,
                        do_sample=False,
                        temperature=None,  # Disable temperature for greedy
                        top_p=None,        # Disable top_p
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            except RuntimeError as e2:
                print(f"Second generation attempt failed: {e2}")
                # Fallback to rule-based extraction
                return self._fallback_time_extraction(query)
        
        try:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(input_text, "").strip()
            
            # Clean up response
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
        except (json.JSONDecodeError, Exception) as e:
            print(f"JSON parsing failed: {e}, trying fallback extraction")
            return self._fallback_time_extraction(query)
    
    def _fallback_time_extraction(self, query: str) -> TimeExtractionResult:
        """Fallback rule-based time extraction when model fails"""
        print("Using fallback rule-based extraction...")
        
        current_time = datetime.now()
        query_lower = query.lower()
        
        # Simple patterns for common time expressions
        patterns = {
            # Yesterday patterns
            r'yesterday.*?(\d{1,2}):?(\d{2})?\s*(pm|am)?': lambda m: self._parse_yesterday_time(m, current_time),
            r'yesterday.*?(\d{1,2})\s*(pm|am)': lambda m: self._parse_yesterday_time(m, current_time),
            
            # Today patterns  
            r'today.*?(\d{1,2}):?(\d{2})?\s*(pm|am)?': lambda m: self._parse_today_time(m, current_time),
            r'this morning.*?(\d{1,2}):?(\d{2})?\s*(am)?': lambda m: self._parse_morning_time(m, current_time),
            
            # Tomorrow patterns
            r'tomorrow.*?(\d{1,2}):?(\d{2})?\s*(pm|am)?': lambda m: self._parse_tomorrow_time(m, current_time),
            
            # Time ranges
            r'between.*?(\d{1,2})-(\d{1,2})\s*(pm|am)?': lambda m: self._parse_time_range(m, current_time, query_lower),
        }
        
        for pattern, parser in patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                try:
                    return parser(match)
                except:
                    continue
        
        # If no pattern matches, return empty result
        return TimeExtractionResult(
            confidence=0.0,
            raw_text=f"Could not extract time from: {query}"
        )
    
    def _parse_yesterday_time(self, match, current_time):
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3) if match.group(3) else ""
        
        if period.lower() == "pm" and hour < 12:
            hour += 12
        elif period.lower() == "am" and hour == 12:
            hour = 0
        
        yesterday = current_time - timedelta(days=1)
        result_dt = yesterday.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return TimeExtractionResult(
            start_datetime=result_dt.isoformat(),
            end_datetime=None,
            is_range=False,
            confidence=0.8,
            raw_text="Fallback extraction"
        )
    
    def _parse_today_time(self, match, current_time):
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3) if match.group(3) else ""
        
        if period.lower() == "pm" and hour < 12:
            hour += 12
        elif period.lower() == "am" and hour == 12:
            hour = 0
        
        result_dt = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return TimeExtractionResult(
            start_datetime=result_dt.isoformat(),
            end_datetime=None,
            is_range=False,
            confidence=0.8,
            raw_text="Fallback extraction"
        )
    
    def _parse_morning_time(self, match, current_time):
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        
        # Morning times are typically AM
        if hour == 12:
            hour = 0
        
        result_dt = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return TimeExtractionResult(
            start_datetime=result_dt.isoformat(),
            end_datetime=None,
            is_range=False,
            confidence=0.8,
            raw_text="Fallback extraction"
        )
    
    def _parse_tomorrow_time(self, match, current_time):
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3) if match.group(3) else ""
        
        if period.lower() == "pm" and hour < 12:
            hour += 12
        elif period.lower() == "am" and hour == 12:
            hour = 0
        
        tomorrow = current_time + timedelta(days=1)
        result_dt = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return TimeExtractionResult(
            start_datetime=result_dt.isoformat(),
            end_datetime=None,
            is_range=False,
            confidence=0.8,
            raw_text="Fallback extraction"
        )
    
    def _parse_time_range(self, match, current_time, query_lower):
        start_hour = int(match.group(1))
        end_hour = int(match.group(2))
        period = match.group(3) if match.group(3) else ""
        
        if period.lower() == "pm":
            if start_hour < 12:
                start_hour += 12
            if end_hour < 12:
                end_hour += 12
        
        # Determine the date based on query context
        target_date = current_time.date()
        if "yesterday" in query_lower:
            target_date = (current_time - timedelta(days=1)).date()
        elif "tomorrow" in query_lower:
            target_date = (current_time + timedelta(days=1)).date()
        
        start_dt = datetime.combine(target_date, datetime.min.time().replace(hour=start_hour))
        end_dt = datetime.combine(target_date, datetime.min.time().replace(hour=end_hour))
        
        return TimeExtractionResult(
            start_datetime=start_dt.isoformat(),
            end_datetime=end_dt.isoformat(),
            is_range=True,
            confidence=0.8,
            raw_text="Fallback extraction"
        )

# Enhanced usage example with better error handling
def main():
    """Complete training and testing workflow"""
    print("Time Extraction Model Training")
    print("=" * 50)
    
    # Configuration
    config = {
        "num_training_samples": 50,  # Increased for better training
        "model_name": "microsoft/DialoGPT-small",
        "output_dir": "./fixed_time_extraction_model",
        "num_epochs": 3,
        "batch_size": 2,
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Step 1: Generate training data
    print("📝 Step 1: Generating training data...")
    generator = TimeExtractionDataGenerator()
    training_data = generator.generate_training_data(config["num_training_samples"])
    
    print(f"Generated {len(training_data)} training examples")
    
    # Show some examples
    print("\n📋 Sample training examples:")
    for i, example in enumerate(training_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Query: {example['query']}")
        response = json.loads(example['response'])
        print(f"  Start: {response['start_datetime']}")
        if response['end_datetime']:
            print(f"  End: {response['end_datetime']}")
        print(f"  Is Range: {response['is_range']}")
    
    # Step 2: Initialize trainer
    print(f"\n🤖 Step 2: Initializing model ({config['model_name']})...")
    trainer = BasicTimeExtractionTrainer(model_name=config["model_name"])
    
    # Check device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Step 3: Train the model
    print(f"\n🏋️ Step 3: Training model...")
    print("This may take several minutes depending on your hardware...")
    
    try:
        trainer.train(
            training_data=training_data,
            output_dir=config["output_dir"]
        )
        print(f"✅ Training completed! Model saved to {config['output_dir']}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("💡 Try reducing batch_size or num_training_samples in config")
        return False
    
    # Step 4: Test the trained model
    print(f"\n🧪 Step 4: Testing the trained model...")
    
    test_queries = [
        "show me yesterday at 3pm",
        "what happened this morning at 9:30?",
        "find videos from last night around 11",
        "between 2-4pm yesterday",
        "tomorrow at 10am"
    ]
    
    print("Test queries and predictions:")
    for query in test_queries:
        try:
            result = trainer.extract_time(query)
            print(f"\n🔍 Query: '{query}'")
            if result.start_datetime:
                print(f"    ✅ Start: {result.start_datetime}")
                if result.end_datetime:
                    print(f"    📅 End: {result.end_datetime}")
                print(f"    🔄 Range: {result.is_range}")
                print(f"    📊 Confidence: {result.confidence:.2f}")
            else:
                print(f"    ❌ No time extracted")
                print(f"    🔍 Raw output: {result.raw_text[:100]}...")
        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
    
    print(f"\n🎉 Training and testing completed!")
    print(f"📁 Model saved in: {config['output_dir']}")
    print(f"🚀 You can now use this model for time extraction in your applications")
    
    return True

# Safe testing function
def test_existing_model(model_path="./trained_time_extraction_model"):
    """Safely test an existing trained model"""
    print("🧪 Testing Existing Time Extraction Model")
    print("=" * 50)
    
    try:
        # Initialize trainer with existing model
        trainer = BasicTimeExtractionTrainer(model_name=model_path)
        trainer.setup_model()
        
        test_queries = [
            "show me yesterday at 3pm",
            "what happened this morning at 9:30?",
            "between 2-4pm yesterday",
            "tomorrow at 10am",
            "last night around 11"
        ]
        
        print("🔍 Testing queries:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            try:
                result = trainer.extract_time(query)
                if result.start_datetime:
                    print(f"   ✅ Start: {result.start_datetime}")
                    if result.end_datetime:
                        print(f"   📅 End: {result.end_datetime}")
                    print(f"   🔄 Range: {result.is_range}")
                    print(f"   📊 Confidence: {result.confidence:.2f}")
                    if result.confidence < 1.0:
                        print(f"   ℹ️ Method: Fallback extraction")
                else:
                    print(f"   ❌ No time extracted")
                    print(f"   🔍 Raw: {result.raw_text[:50]}...")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Suggestions:")
        print("   - Check if model path is correct")
        print("   - Try retraining the model")
        print("   - Use CPU instead of GPU")
        
        # Offer rule-based fallback
        print("\n🔄 Trying rule-based extraction instead...")
        trainer = BasicTimeExtractionTrainer()
        for query in ["show me yesterday at 3pm", "between 2-4pm yesterday"]:
            result = trainer._fallback_time_extraction(query)
            print(f"Query: '{query}' -> Start: {result.start_datetime}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Quick test mode
            quick_test()
        elif sys.argv[1] == "--test-existing":
            # Test existing model
            model_path = sys.argv[2] if len(sys.argv) > 2 else "./trained_time_extraction_model"
            test_existing_model(model_path)
        else:
            print("Usage:")
            print("  python script.py                    # Full training")
            print("  python script.py --test             # Quick data test")
            print("  python script.py --test-existing    # Test existing model")
    else:
        # Full training mode
        success = main()
        if not success:
            print("\n💡 If training failed, try:")
            print("   - python script.py --test")
            print("   - python script.py --test-existing")