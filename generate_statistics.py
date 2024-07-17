import argparse
import numpy as np

def generate(transcriptions_folder):
    """
        Generates descriptive statistics.

        Args:
            transcriptions_folder (str): Folder that stores your English and non-English text.

        Returns:
            List (int): List of ints where List[0] is statistics on number of words,
                        List[1] is statistics on number of IUs.
                        Each sublist is in the order: [mean, standard deviation, min, max]
    """
    # Open detected IU boundaries
    with open(transcriptions_folder+'boundaries.txt', 'r') as f:
        detected_boundaries = f.readlines()
        detected_boundaries = list(map(lambda x: x.rstrip(), detected_boundaries))
    
    # Store source (==English) transcriptions
    with open(transcriptions_folder+'en.txt', 'r') as f:
        source_transcriptions = f.readlines()
        source_transcriptions = list(map(lambda x: x.rstrip(), source_transcriptions))
    
    # Parse boundaries
    count_boundaries = []
    separator = '<|IU_Boundary|>'
    for detected in detected_boundaries:
        count_boundaries.append(detected.count(separator))
    
    # Parse transcriptions
    count_words = []
    for transcription in source_transcriptions:
        count_words.append(len(transcription.split()))

    return [[round(np.mean(count_boundaries), 1), round(np.std(count_boundaries), 1), min(count_boundaries), max(count_boundaries)],
            [round(np.mean(count_words), 1), round(np.std(count_words), 1), min(count_words), max(count_words)]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate descriptive statistics")
    parser.add_argument("--transcriptionsfolder", required=True, type=str, help="Folder that stores your PSST transcriptions.")

    args = parser.parse_args()

    print(generate(args.transcriptionsfolder))
