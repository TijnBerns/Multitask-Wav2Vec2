
from numpy import NaN
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import torch
from config import Config
import data
from tqdm import tqdm
import math
import jiwer
from collections import Counter
from utils import mean_measures, remove_speaker_change_symbol


def train(train_loader, val_loader, test_loader, model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor, num_epochs: int, device: str):
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, num_epochs)

    for i in range(num_epochs):
        train_measures = epoch(train_loader, model, processor, optim, device)
        train_measures = {f"train_{k}": v for (k, v) in train_measures.items()}
        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_measures = epoch(test_loader, model, processor, None, device)
            test_measures = {f"test_{k}": v for (
                k, v) in test_measures.items()}

        print(f"{i}\t{train_measures}\t{test_measures}")


def epoch(dataloader, model, processor, optim, device):
    total_loss = 0
    num_batches = 0
    measures = Counter({})
    measures_no_sep = Counter({})

    for sample in tqdm(dataloader):
        waveform = sample['waveform'].to(device)
        ground_truth = sample['transcription']
        ground_truth_no_sep = remove_speaker_change_symbol(ground_truth)

        # Retrieve input values
        input_values = processor(
            waveform, sampling_rate=16_000, return_tensors="pt", padding="longest").input_values
        input_values = torch.reshape(
            input_values, (Config.batch_size, -1)).to(device)

        # Retrieve target labels
        with processor.as_target_processor():
            labels = processor(ground_truth, padding=True,
                               return_tensors='pt').input_ids
            labels = labels.to(device)

        # Compute loss by passing labels
        output = model(input_values, labels=labels)
        loss = output.loss
        total_loss += loss.item()

        if math.isnan(loss):
            exit("LOSS REACHED NAN VALUE")

        # Backpropagation
        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Construct hypothesis
        logits = output.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        hypothesis = processor.batch_decode(predicted_ids)
        hypothesis_no_sep = remove_speaker_change_symbol(hypothesis)

        # Add measures
        measures = Counter(jiwer.compute_measures(
            ground_truth, hypothesis)) + measures
        measures_no_sep = Counter(jiwer.compute_measures(
            ground_truth_no_sep, hypothesis_no_sep)) + measures_no_sep
        
        num_batches += 1
    
    # Construct measures dict and return
    measures_no_sep = {f"no_sep_{k}": v for (k, v) in measures_no_sep.items()}
    measures_no_sep = mean_measures(measures_no_sep, num_batches)
    measures = mean_measures(measures, num_batches)
    measures.update(measures_no_sep)
    measures['CTCloss'] = total_loss / len(dataloader.dataset)
    return measures


def main(jobid=None):
    # Set device
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    # device = 'cpu'

    # Initialize processor
    tokenizer = Wav2Vec2CTCTokenizer(
        "models/vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=Config.sample_rate, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

    """
    processor = Wav2Vec2Processor.from_pretrained(
        pretrained_model_name_or_path="facebook/wav2vec2-base-960h")
    """

    # Load model
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h", ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id).to(device)
    model.freeze_feature_encoder()

    # Load datasets
    train_set = data.CustomLibriSpeechDataset(
        Config.datapath + '/train-clean-no-rep')
    test_set = data.CustomLibriSpeechDataset(
        Config.datapath + '/test-clean-no-rep')
    dev_set = data.CustomLibriSpeechDataset(
        Config.datapath + '/dev-clean-no-rep')

    # Initialize dataloaders
    train_loader = data.initialize_loader(train_set)
    test_loader = data.initialize_loader(test_set)
    dev_loader = data.initialize_loader(dev_set)
    val_loader = None

    # Perform training
    train(train_loader, val_loader, test_loader, model,
          processor, num_epochs=10, device=device)


if __name__ == "__main__":
    main()


"""
for sample in dev_clean_loader:

    input_values = processor(sample['waveform'], sampling_rate=16_000, return_tensors="pt", padding="longest").input_values  
    
    # retrieve logits
    input_values = torch.reshape(input_values, (1,-1))
    logits = model(input_values).logits
    

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    breakpoint()
        
"""
