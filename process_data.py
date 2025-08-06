"""
Complete Agricultural Data Processing Pipeline
Step-by-step processing of 150 crop files with 2.78M+ rows each
Filters data from 2014 onwards, removes duplicates, formats for Gemma-3n training
"""

import pandas as pd
import numpy as np
import json
import hashlib
import gc
import time
import re
from pathlib import Path
from datetime import datetime
import logging



# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x          # fallback if tqdm not installed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AgriculturalDataProcessor:
    def __init__(self, input_dir, output_dir="processed_agri_data"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        (self.output_dir / "individual_crops").mkdir(exist_ok=True)
        (self.output_dir / "final_dataset").mkdir(exist_ok=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True)

        self.global_stats = {
            'total_files_processed': 0,
            'total_raw_rows': 0,
            'total_after_date_filter': 0,
            'total_after_quality_filter': 0,
            'total_duplicates_removed': 0,
            'total_final_conversations': 0,
            'processing_start_time': None,
            'processing_end_time': None,
            'crops_processed': [],
            'failed_files': []
        }

        self.config = {
            'min_year': 2014,
            'min_question_length': 15,
            'min_answer_length': 25,
            'max_samples_per_crop': 50000,
            'date_columns':    ['date', 'Date', 'year', 'Year', 'timestamp', 'Year'],
            'question_columns':['question', 'Query', 'problem', 'QueryText'],
            'answer_columns':  ['answer', 'Solution', 'response', 'KccAns'],
            'blacklist_answers': {'nan', 'answer given detail to farmer'}
        }

    # ------------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------------
    def detect_columns(self, df):
        columns = {}
        for col in self.config['date_columns']:
            if col in df.columns:
                columns['date'] = col
                break
        for col in self.config['question_columns']:
            if col in df.columns:
                columns['question'] = col
                break
        for col in self.config['answer_columns']:
            if col in df.columns:
                columns['answer'] = col
                break
        for col in ['crop', 'Crop', 'crop_name', 'cropname']:
            if col in df.columns:
                columns['crop'] = col
                break
        logger.info(f"Detected columns: {columns}")
        return columns

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------
    def extract_year_from_date(self, date_value):
        if pd.isna(date_value):
            return None
        date_str = str(date_value).strip()
        for fmt in ('%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%Y/%m/%d', '%Y'):
            try:
                return datetime.strptime(date_str, fmt).year
            except ValueError:
                continue
        match = re.search(r'(20\d{2}|19\d{2})', date_str)
        return int(match.group(1)) if match else None

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------
    def clean_text(self, text):
        if pd.isna(text) or text == '':
            return ""
        text = str(text).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,?!:;-]', '', text)
        return text

    # ------------------------------------------------------------------
    # Strict normalisation for duplicate detection
    # ------------------------------------------------------------------
    def normalize_for_deduplication(self, text):
        if pd.isna(text) or text == '':
            return ""
        text = str(text).strip().lower()
        text = re.sub(r'[^\w\s]', '', text)        # remove punctuation
        text = re.sub(r'\s+', ' ', text)           # collapse whitespace
        return text.strip()

    # ------------------------------------------------------------------
    # Duplicate hashing
    # ------------------------------------------------------------------
    def create_duplicate_hash(self, q_norm, a_norm):
        combined = f"{q_norm}|||{a_norm}"
        return hashlib.md5(combined.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def filter_by_date(self, df, columns):
        if 'date' not in columns:
            logger.warning("No date column found, skipping date filter")
            return df
        date_col = columns['date']
        logger.info(f"Applying date filter (≥2014) on column: {date_col}")
        df['extracted_year'] = df[date_col].apply(self.extract_year_from_date)
        before = len(df)
        df = df[df['extracted_year'] >= self.config['min_year']].copy()
        logger.info(f"Date filter: {before} -> {len(df)} rows")
        return df.drop('extracted_year', axis=1)

    def quality_filter(self, df, columns):
        q_col, a_col = columns['question'], columns['answer']
        before = len(df)

        # basic emptiness
        df = df.dropna(subset=[q_col, a_col])
        df = df[df[q_col].str.strip().ne('') & df[a_col].str.strip().ne('')]

        # length thresholds
        df = df[df[q_col].str.len() >= self.config['min_question_length']]
        df = df[df[a_col].str.len() >= self.config['min_answer_length']]

        # --- weather-related questions filter ---
        weather_keywords = {
                    'weather', 'forecast', 'rainfall', 'temperature', 'humidity',
                   'climate', 'monsoon', 'drought', 'flood'
                     }
        df = df[~df[q_col].str.lower().str.contains('|'.join(weather_keywords),
                                            case=False, na=False)]        


        # junk / numeric-only filter
        def is_junk(text):
            if pd.isna(text):
                return True
            txt = str(text).strip().lower()
            if len(txt.split()) <= 3:
                return True
            alpha = re.sub(r'[^a-z]', '', txt)
            return len(alpha) < max(1, len(txt) * 0.2)

        df = df[~df[a_col].apply(is_junk)]

        # blacklist literal answers
        bad = self.config['blacklist_answers']
        df = df[~df[a_col].str.strip().str.lower().isin(bad)]

        logger.info(f"Quality filter: {before} -> {len(df)} rows")
        return df

    def remove_duplicates(self, df, columns):
        q_col, a_col = columns['question'], columns['answer']
        before = len(df)

        # strict normalised columns for hashing
        df['norm_q'] = df[q_col].apply(self.normalize_for_deduplication)
        df['norm_a'] = df[a_col].apply(self.normalize_for_deduplication)
        df['dup_hash'] = df.apply(
            lambda r: self.create_duplicate_hash(r['norm_q'], r['norm_a']), axis=1
        )

        df = df.drop_duplicates(subset=['dup_hash'], keep='first')
        removed = before - len(df)
        logger.info(f"Duplicate removal: {before} -> {len(df)} rows ({removed})")
        return df.drop(columns=['norm_q', 'norm_a', 'dup_hash'])

    def sample_conversations(self, df, crop_name, columns):
        if len(df) <= self.config['max_samples_per_crop']:
            return df
        sampled = df.sample(n=self.config['max_samples_per_crop'], random_state=42)
        logger.info(f"{crop_name}: sampled {len(sampled)} from {len(df)}")
        return sampled

    # ------------------------------------------------------------------
    # Training format
    # ------------------------------------------------------------------
    def convert_to_training_format(self, df, crop_name, columns):
        q_col, a_col = columns['question'], columns['answer']
        conversations = []
        for idx, row in df.iterrows():
            q = self.clean_text(row[q_col])
            a = self.clean_text(row[a_col])
            if not q or not a:
                continue
            conversations.append({
                "conversations": [
                    {"role": "user", "content": [{"type": "text", "text": q}]},
                    {"role": "assistant", "content": [{"type": "text", "text": a}]}
                ],
                "metadata": {
                    "crop_name": crop_name,
                    "source_file": f"{crop_name}.csv",
                    "original_row": int(idx)
                }
            })
        return conversations

    # ------------------------------------------------------------------
    # Core per-file processing
    # ------------------------------------------------------------------
    def process_single_crop(self, file_path):
        crop_name = file_path.stem.lower()
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING: {crop_name.upper()}")
        logger.info(f"{'='*60}")

        start = time.time()
        try:
            df = pd.read_csv(file_path, low_memory=False, dtype=str)  # memory-friendly
            self.global_stats['total_raw_rows'] += len(df)

            columns = self.detect_columns(df)
            if 'question' not in columns or 'answer' not in columns:
                logger.error(f"Essential columns not found in {crop_name}")
                self.global_stats['failed_files'].append(crop_name)
                return []

            df = self.filter_by_date(df, columns)
            self.global_stats['total_after_date_filter'] += len(df)

            df = self.quality_filter(df, columns)
            self.global_stats['total_after_quality_filter'] += len(df)

            df = self.remove_duplicates(df, columns)
            self.global_stats['total_duplicates_removed'] += len(df)

            df = self.sample_conversations(df, crop_name, columns)

            conversations = self.convert_to_training_format(df, crop_name, columns)
            self.global_stats['total_final_conversations'] += len(conversations)

            out_file = self.output_dir / "individual_crops" / f"{crop_name}_processed.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ {crop_name.upper()} done – {len(conversations)} conversations, "
                        f"{time.time() - start:.1f}s")
            self.global_stats['crops_processed'].append(crop_name)
            del df
            gc.collect()
            return conversations

        except Exception as e:
            logger.error(f"❌ Error processing {crop_name}: {e}")
            self.global_stats['failed_files'].append(crop_name)
            return []

    # ------------------------------------------------------------------
    # Cross-crop deduplication
    # ------------------------------------------------------------------
    def cross_crop_deduplication(self, all_conversations):
        logger.info("Cross-crop deduplication …")
        before = len(all_conversations)
        seen = set()
        unique = []
        for conv in all_conversations:
            try:
                q = self.normalize_for_deduplication(conv['conversations'][0]['content'][0]['text'])
                a = self.normalize_for_deduplication(conv['conversations'][1]['content'][0]['text'])
                h = self.create_duplicate_hash(q, a)
                if h not in seen:
                    seen.add(h)
                    unique.append(conv)
            except Exception as e:
                logger.error(f"Cross-crop dedup error: {e}")
        removed = before - len(unique)
        pct = (removed / max(before, 1)) * 100
        logger.info(f"Cross-crop dedup: {before} -> {len(unique)} ({pct:.1f}%)")
        return unique

    # ------------------------------------------------------------------
    # Splits & stats
    # ------------------------------------------------------------------
    def create_training_splits(self, conversations):
        np.random.seed(42)
        np.random.shuffle(conversations)
        total = len(conversations)
        splits = {
            'train': conversations[:int(total * 0.8)],
            'validation': conversations[int(total * 0.8):int(total * 0.9)],
            'test': conversations[int(total * 0.9):]
        }
        for name, data in splits.items():
            with open(self.output_dir / "final_dataset" / f"{name}.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Splits saved.")

    def save_statistics(self):
        stats_path = self.output_dir / "statistics" / "processing_stats.json"
        elapsed = self.global_stats['processing_end_time'] - self.global_stats['processing_start_time']
        detailed = {**self.global_stats,
                    'processing_duration_seconds': elapsed,
                    'processing_duration_hours': elapsed / 3600}
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(detailed, f, indent=2)
        return detailed

    def generate_report(self, stats):
        report = f"""
🌾 AGRICULTURAL DATA PROCESSING REPORT
{'='*60}
• Files processed: {len(stats['crops_processed'])}/{stats['total_files_processed']}
• Failed files: {len(stats['failed_files'])}
• Raw rows: {stats['total_raw_rows']:,}
• Final conversations: {stats['total_final_conversations']:,}
• Duration: {stats['processing_duration_hours']:.1f} h
"""
        with open(self.output_dir / "statistics" / "processing_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        return report

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------
    def process_all_crops(self):
        logger.info("🚀 STARTING AGRICULTURAL DATA PROCESSING")
        self.global_stats['processing_start_time'] = time.time()

        csv_files = list(self.input_dir.glob("*.csv"))
        if not csv_files:
            logger.error("No CSV files found")
            return []

        self.global_stats['total_files_processed'] = len(csv_files)
        logger.info(f"Found {len(csv_files)} CSV files")

        all_conversations = []
        for file_path in tqdm(csv_files, desc="Processing"):
            all_conversations.extend(self.process_single_crop(file_path))

        all_conversations = self.cross_crop_deduplication(all_conversations)
        self.create_training_splits(all_conversations)

        self.global_stats['processing_end_time'] = time.time()
        stats = self.save_statistics()
        self.generate_report(stats)

        logger.info("🎉 DONE")
        return all_conversations


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    input_dir = "raw_data"
    output_dir = "processed_agri_data"
    processor = AgriculturalDataProcessor(input_dir, output_dir)
    processor.process_all_crops()


if __name__ == "__main__":
    main()