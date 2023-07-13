import evaluate
from tqdm.auto import tqdm
from transformers import get_scheduler
from accelerate import Accelerator
from torch.optim import AdamW
import torch
import torch.nn as nn
from transformers import DataCollatorForTokenClassification
import time
from torch.utils.data import DataLoader
from EventExtratorClassifer import EventExtractorClassifer
from event_dataset2 import EventDataset
import argparse
import os
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    AutoConfig,
)

# get current timestamp and convert to string
timestamp = str(int(time.time()))

parser = argparse.ArgumentParser(description="Train model on the FNDEE dataset")
parser.add_argument(
    "--path",
    type=str,
    default="../result/models-event",
    help="The file path will save the trained model (default: ../result/models)",
)
# parser.add_argument('--model_name', type=str, default=f'{timestamp}',
#                     help='The file name of the trained model will be saved (default: timestamp)')

parser.add_argument(
    "--train_file",
    type=str,
    default="./data/FNDEE_train1.json",
    help="The file name of the train dateset file",
)
parser.add_argument(
    "--valid_file",
    type=str,
    default="./data/FNDEE_valid.json",
    help="The file name of the train dateset file",
)

parser.add_argument(
    "--batch_size", type=int, help="Train Batch size, default 4", default=4
)
parser.add_argument(
    "--bert_model",
    type=str,
    default="../models/xlm-roberta-base-ner-hrl",
    help='Pretrained bert model name (default: "../models/xlm-roberta-base-ner-hrl")',
)
parser.add_argument(
    "--num_epochs", type=int, default=20, help="Number of epochs to train (default: 20)"
)

parser.add_argument(
    "--disable_short_circle",
    default=False,
    action="store_true",
    help="Short circle the bert model last two hidden states (default: True)",
)

parser.add_argument(
    "--mask",
    type=bool,
    default=True,
    help="Mask the dataset no event and arguments words (default: True)",
)

parser.add_argument(
    "--disable_replace",
    action="store_true",
    default=False,
    help="Replace the argument's text with the same type event and same argument role text  (default: True)",
)

parser.add_argument(
    "--mask_ratio",
    type=float,
    default=0.3,
    help="Mask ratio (default: 0.3)",
)

parser.add_argument(
    "--replace_ratio",
    type=float,
    default=0.3,
    help="Replace ratio  (default: 0.3)",
)

parser.add_argument(
    "--max_mask_num",
    type=int,
    default=10,
    help="Max mask token number (default: 10)",
)

parser.add_argument(
    "--max_replace_num",
    type=int,
    default=3,
    help="Max Replaced arguments number  (default: 3)",
)

parser.add_argument(
    "--refine",
    type=bool,
    default=False,
    help="Refine the model or train the model (default: False)",
)

parser.add_argument(
    "--base_model",
    type=str,
    default=None,
    help="If Refine the model, the base model name to refine (default: None)",
)

args = parser.parse_args()

train_file = args.train_file
valid_file = args.valid_file
batch_size = args.batch_size
bert_name = args.bert_model
num_epochs = args.num_epochs
disable_short_circle = args.disable_short_circle
mask = args.mask
disable_replace = args.disable_replace
mask_ratio = args.mask_ratio
replace_ratio = args.replace_ratio
max_replace_num = args.max_replace_num
max_mask_num = args.max_mask_num
refine = args.refine
base_model = args.base_model

if refine and base_model is None:
    print("Please input the base model name to refine.")
    exit(-1)
if refine:
    path = f"{ args.path}_{disable_short_circle}_m.{mask}.{mask_ratio}_r.{disable_replace}.{max_replace_num}_refine_{base_model}_{timestamp}"
else:
    path = f"{ args.path}_{disable_short_circle}_m.{mask}.{mask_ratio}_r.{disable_replace}.{max_replace_num}_{timestamp}"


# create folder to save model
if not os.path.exists(path):
    os.makedirs(path)

tokenizer = AutoTokenizer.from_pretrained(bert_name)
event_dataset = EventDataset(
    train_file,
    valid_file,
    tokenizer,
    mask=True,
    replace=False,
    mask_ratio=mask_ratio,
    replace_ratio=replace_ratio,
    max_replace_num=max_replace_num,
    max_mask_num=max_mask_num,
)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

train_dataloader = DataLoader(
    event_dataset["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    event_dataset["test"], collate_fn=data_collator, batch_size=batch_size
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading pretrained model...")
config = AutoConfig.from_pretrained(bert_name, output_hidden_states=True)
bert_model = AutoModelForTokenClassification.from_pretrained(bert_name, config=config)
model = EventExtractorClassifer(
    bert_model,
    num_labels=event_dataset.num_event_types,
    short_circle=False,
).to(device)

if refine:
    model_data = torch.load(base_model)
    model.load_state_dict(model_data, map_location=device)

optimizer = AdamW(model.parameters(), lr=2e-5)

print("Warming up...")
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)


num_train_epochs = num_epochs
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [
        [event_dataset.label_names[l] for l in label if l != -100] for label in labels
    ]
    true_predictions = [
        [event_dataset.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


metric = evaluate.load("seqeval")
progress_bar = tqdm(range(num_training_steps))
loss_fn = nn.CrossEntropyLoss(torch.tensor(event_dataset.class_weights).to(device))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(batch["input_ids"], batch["attention_mask"])
        loss = loss_fn(
            outputs.view(-1, event_dataset.num_event_types), batch["labels"].view(-1)
        )
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(batch["input_ids"], batch["attention_mask"])

        predictions = outputs.argmax(dim=-1)
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        predictions = accelerator.pad_across_processes(
            predictions, dim=1, pad_index=-100
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(
            predictions_gathered, labels_gathered
        )
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )

    # Save and upload
    event_filename = f"{path}/model_{epoch}_p{results['overall_precision']:.4f}_r{results['overall_recall']:.4f}_f{results['overall_f1']:.4f}.pt"
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(event_filename, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(event_filename)

    print("Model saved to:", event_filename)
