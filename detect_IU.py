import tarfile
import argparse
import librosa
import os
from google.colab import drive
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

"""
Part of this code is drawn from the PSST Tutorial in https://github.com/nathan-roll1/psst.
Running this script creates a English transcript of Common Voice 4.0 files.
PSST performs two functionalities: 1) transcribing and 2) marking IU Boundaries. See Table 1 in paper for number of utterances in each step.
You will need 1 text file that stores all the filenames of Common Voice 4.0 and feed it as the argument for '--englishfilename'.
"""

def extract_files_with_progress(tar_file_path, destination_folder):
    """ Extracts each mp3 file from tar file to destination folder.

    Returns: 
        void
    """
    # Open the tar file
    with tarfile.open(tar_file_path, 'r') as tar:
        # Create a tqdm progress bar with total size equal to number of members in tar file
        progress_bar = tqdm(total=len(tar.getmembers()), desc="Extracting")

        # Iterate over each member (file/directory) in the tar file
        for member in tar:
            # Extract the member to the specified destination folder
            tar.extract(member, destination_folder)
            # Update the progress bar
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

## Init models
def init_model_processor(gpu=False):
    """ Initializes the model and processor with the pre-trained weights.

    Returns:
      model (AutoModelForSpeechSeq2Seq): A model with the pre-trained weights.
      processor (AutoProcessor): Processes audio data.
    """
    # Initialize the processor with the pre-trained weights
    processor = AutoProcessor.from_pretrained("NathanRoll/psst-medium-en")

    if gpu:
        # Initialize the model with the pre-trained weights and move it to the gpu
        model = AutoModelForSpeechSeq2Seq.from_pretrained("NathanRoll/psst-medium-en").to("cuda")
    else:
        # Initialize the model with the pre-trained weights
        model = AutoModelForSpeechSeq2Seq.from_pretrained("NathanRoll/psst-medium-en")

    return model, processor

def generate_transcription(audio, model, processor, gpu=False):
    """Generate a transcription from audio using a pre-trained model

    Args:
      audio: The audio to be transcribed
      model (AutoModelForSpeechSeq2Seq): Pre-trained model.
      processor (AutoProcessor): Processor for audio data.
      gpu: Whether to use GPU or not. Defaults to False.

    Returns:
      transcription: The transcribed text
    """
    # Preprocess audio and return tensors
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)

    # Assign inputs to GPU or CPU based on argument
    if gpu:
        input_features = inputs.input_features.to("cuda")
    else:
        input_features = inputs.input_features

    # Generate transcribed ids
    generated_ids = model.generate(input_features=input_features, max_length=250)

    # Decode generated ids and replace special tokens
    transcription = processor.batch_decode(
        generated_ids, skip_special_tokens=True, output_word_offsets=True)[0].replace('!!!!!', '<|IU_Boundary|>')

    return transcription
    
if __name__ == "__main__":
    # Mount Google Drive
    drive.mount('/content/drive')

    parser = argparse.ArgumentParser(description="Detect Intonation Units using PSST")
    # /content/drive/MyDrive/validation-files.tar
    parser.add_argument('--tarfilepath', type=str, default="", help="Path to tarfile")
    # /content/drive/MyDrive/validtest/validextracted
    parser.add_argument('--destinationfolder', type=str, default="", help="Path to folder that will unzip tar files.")
    parser.add_argument('--englishfilename', required=True, type=str, help="Name for text file that stores English Common Voice 4.0 filenames.")
    parser.add_argument('--boundaryfilename', required=True, type=str, help="Name for text file that will contain IU transcripts.")
    args = parser.parse_args()

    if args.tarfilepath and args.destinationfolder:
        extract_files_with_progress(args.tarfilepath, args.destinationfolder)

    # Init model and processor
    model, processor = init_model_processor(gpu=True)

    # Read each filename of Common Voice 4.0
    with open(args.englishfilepath) as f:
        filenames = f.readlines()
        filenames = list(map(str.rstrip, filenames))

    # For each mp3 file, prompt PSST for IU detected transcript
    with open(args.boundaryfilepath, "w") as f:
        for fname in tqdm(filenames):
            y, sr = librosa.load(fname)
            audio = librosa.resample(y, orig_sr=sr, target_sr=16000)

            transcript = generate_transcription(audio, model, processor, gpu=True)
            f.write(transcript + '\n')

