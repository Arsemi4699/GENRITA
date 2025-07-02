import pandas as pd
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns


class DataProcessor:
    """
    A class to preprocess a text dataset for a multi-task classification model.
    It handles cleaning, normalization, filtering, and provides statistical analysis.
    """

    def __init__(self, sense_classes, age_classes):
        """
        Initializes the processor with the class mappings.

        Args:
            sense_classes (dict): A dictionary mapping sense class names to IDs.
            age_classes (dict): A dictionary mapping age class names to IDs.
        """
        self.sense_classes = sense_classes
        self.age_classes = age_classes
        self.sense_id_to_name = {v: k for k, v in sense_classes.items()}
        self.age_id_to_name = {v: k for k, v in age_classes.items()}
        self.stats = {}
        print("DataProcessor initialized.")

    @staticmethod
    def _clean_text(text):
        """
        Performs basic text cleaning operations, including Markdown removal.
        - Converts to lowercase.
        - Removes URLs, mentions, and hashtags.
        - Removes common Markdown syntax.
        - Removes special characters, keeping essential punctuation.
        - Normalizes whitespace.
        This is a static method, so it can be called directly: DataProcessor._clean_text(your_text)
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # --- NEW: Remove Markdown syntax ---
        # Remove images and links, keeping the alt/link text
        text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        # Remove headers
        text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Remove blockquotes
        text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
        # Remove horizontal rules
        text = re.sub(r'^\s*[-\*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        # Remove bold, italic, strikethrough, and inline code markers
        text = re.sub(r'(\*\*|__|\*|_|~~|`)', '', text)
        # --- END NEW ---

        # Remove non-alphanumeric characters, but keep spaces and basic punctuation
        # This will also clean up any leftover markdown characters like `[` or `]`
        text = re.sub(r'[^\w\s\'-]', '', text)
        # Normalize whitespace to a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _randomly_truncate_start(text, max_words_to_remove=10):
        """
        Randomly removes 0 to max_words_to_remove words from the beginning of a text.
        Note: This is typically used for training data augmentation and might not be desired for inference.
        """
        words = text.split()
        if not words:
            return ""

        num_words_to_remove = random.randint(0, min(max_words_to_remove, len(words) - 1))
        return ' '.join(words[num_words_to_remove:])

    def _collect_stats(self, df, stage):
        """
        Collects and stores statistics about the dataframe at a given stage.
        """
        self.stats[stage] = {
            'total_samples': len(df),
            'sense_distribution': df['sense_class_id'].value_counts().sort_index(),
            'age_distribution': df['age_class_id'].value_counts().sort_index(),
        }
        print(f"Collected statistics for stage: '{stage}'")

    def run_pipeline(self, input_filepath, output_filepath, min_word_count=10):
        """
        Executes the full preprocessing pipeline.

        Args:
            input_filepath (str): Path to the input CSV file.
            output_filepath (str): Path to save the cleaned CSV file.
            min_word_count (int): Minimum number of words a text must have to be kept.

        Returns:
            pandas.DataFrame: The cleaned and preprocessed DataFrame.
        """
        print(f"Starting pipeline with input: {input_filepath}")

        # 1. Load Data
        try:
            df = pd.read_csv(input_filepath)
            print(f"Successfully loaded {len(df)} rows from {input_filepath}")
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_filepath}")
            return None

        # Validate that required columns exist
        print(f"CSV columns found: {df.columns.tolist()}")
        required_columns = ['text', 'sense_class_id', 'age_class_id']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"\n--- ERROR: Missing required columns in the input CSV file. ---")
            print(f"Missing column(s): {missing_columns}")
            print(f"Please ensure your CSV file contains the columns: {required_columns}")
            print("-----------------------------------------------------------------")
            return None

        self._collect_stats(df, 'initial')

        # 2. Remove duplicate texts
        initial_rows = len(df)
        # keep=False marks all duplicates as True
        df = df.drop_duplicates(subset=['text'], keep=False)
        rows_after_duplicates = len(df)
        self.stats['duplicates_removed'] = initial_rows - rows_after_duplicates
        print(f"Removed {self.stats['duplicates_removed']} rows due to duplicate text.")

        # 3. Apply basic text cleaning
        print("Applying text cleaning...")
        df['text'] = df['text'].apply(self._clean_text)

        # 4. Apply random start truncation (for training data augmentation)
        print("Applying random start truncation...")
        df['text'] = df['text'].apply(self._randomly_truncate_start)

        # 5. Filter by length
        initial_rows = len(df)
        df['word_count'] = df['text'].apply(lambda x: len(x.split()))
        df = df[df['word_count'] >= min_word_count]
        rows_after_length_filter = len(df)
        self.stats['short_texts_removed'] = initial_rows - rows_after_length_filter
        print(f"Removed {self.stats['short_texts_removed']} rows with less than {min_word_count} words.")

        # 6. Normalize class names based on IDs
        print("Normalizing class names...")
        df['sense_class_name'] = df['sense_class_id'].map(self.sense_id_to_name)
        df['age_class_name'] = df['age_class_id'].map(self.age_id_to_name)
        # Drop rows where mapping failed (if any invalid IDs were present)
        df.dropna(subset=['sense_class_name', 'age_class_name'], inplace=True)

        # 7. Final cleanup
        df_cleaned = df[['text', 'sense_class_name', 'sense_class_id', 'age_class_name', 'age_class_id']]

        # 8. Collect final stats
        self._collect_stats(df_cleaned, 'final')

        # 9. Save cleaned data
        print(f"Saving cleaned data to {output_filepath}")
        df_cleaned.to_csv(output_filepath, index=False, encoding='utf-8')

        print("Pipeline finished successfully.")
        return df_cleaned

    def display_statistics(self):
        """
        Prints a summary of the preprocessing statistics and generates plots.
        """
        print("\n" + "=" * 50)
        print("           Preprocessing Statistics Summary")
        print("=" * 50)

        initial_stats = self.stats.get('initial', {})
        final_stats = self.stats.get('final', {})

        print(f"\nTotal Samples (Initial): {initial_stats.get('total_samples', 0)}")
        print(f"Total Samples (Final):   {final_stats.get('total_samples', 0)}")
        print("-" * 50)
        print(f"Duplicate Text Rows Removed: {self.stats.get('duplicates_removed', 0)}")
        print(f"Short Text Rows Removed:     {self.stats.get('short_texts_removed', 0)}")

        if not initial_stats or not final_stats:
            print("\nStatistics not available for comparison.")
            return

        # Plotting distributions
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Class Distribution Comparison', fontsize=18)
        sns.set_style("whitegrid")

        # Sense Class Distribution
        initial_sense_dist = initial_stats['sense_distribution'].rename(self.sense_id_to_name)
        final_sense_dist = final_stats['sense_distribution'].rename(self.sense_id_to_name)

        sns.barplot(ax=axes[0, 0], x=initial_sense_dist.index, y=initial_sense_dist.values,
                    hue=initial_sense_dist.index, palette="viridis", legend=False)
        axes[0, 0].set_title('Initial Sense Class Distribution')
        axes[0, 0].tick_params(axis='x', rotation=45, labelsize=8)
        axes[0, 0].set_ylabel('Count')

        sns.barplot(ax=axes[0, 1], x=final_sense_dist.index, y=final_sense_dist.values, hue=final_sense_dist.index,
                    palette="viridis", legend=False)
        axes[0, 1].set_title('Final Sense Class Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45, labelsize=8)
        axes[0, 1].set_ylabel('Count')

        # Age Class Distribution
        initial_age_dist = initial_stats['age_distribution'].rename(self.age_id_to_name)
        final_age_dist = final_stats['age_distribution'].rename(self.age_id_to_name)

        sns.barplot(ax=axes[1, 0], x=initial_age_dist.index, y=initial_age_dist.values, hue=initial_age_dist.index,
                    palette="plasma", legend=False)
        axes[1, 0].set_title('Initial Age Class Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45, labelsize=8)
        axes[1, 0].set_ylabel('Count')

        sns.barplot(ax=axes[1, 1], x=final_age_dist.index, y=final_age_dist.values, hue=final_age_dist.index,
                    palette="plasma", legend=False)
        axes[1, 1].set_title('Final Sense Class Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45, labelsize=8)
        axes[1, 1].set_ylabel('Count')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_filename = 'data/class_distribution_comparison.png'
        plt.savefig(plot_filename)
        print(f"\nSaved class distribution plots to '{plot_filename}'")
        plt.show()


if __name__ == '__main__':
    # Define the class structures as provided
    SENSE_CLASSES = {
        "Normal and neutral": 0, "Love and romantic": 1, "War and combat": 2,
        "Fantasy and mythology": 3, "Honor and respect": 4, "Drama and tragedy": 5,
        "City and Crowd": 6, "Mountain and the heights": 7, "Desert and dunes": 8,
        "Sea and tides": 9, "Forest and tress": 10,
    }
    AGE_CLASSES = {
        "ancient and old age": 0,
        "neutral and not special age (non-ancient, non technology)": 1,
        "technology modern age": 2,
    }

    # --- IMPORTANT ---
    # Please update the INPUT_CSV path to your actual data file.
    # For example: INPUT_CSV = './data/raw_dataset.csv'
    INPUT_CSV = 'data/dataset3.csv'
    OUTPUT_CSV = 'data/cleaned_data3.csv'

    # Initialize and run the processor
    processor = DataProcessor(sense_classes=SENSE_CLASSES, age_classes=AGE_CLASSES)
    # Make sure to replace the path in the line below with your actual file path
    cleaned_df = processor.run_pipeline(input_filepath=INPUT_CSV, output_filepath=OUTPUT_CSV)

    # Display the final statistics and visualizations
    if cleaned_df is not None:
        processor.display_statistics()
        print(f"\n--- First 5 rows of cleaned data in {OUTPUT_CSV} ---")
        print(cleaned_df.head())
        print("\n--- End of script ---")


