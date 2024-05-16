import pandas as pd
import paramiko
import os
from transformers import (
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Value, ClassLabel, Features
import time

output_path = '/users/arwill/SAVSNET_test/'

# Create command string to download data from savsnet to pickles
remove_string = "rm /users/arwill/pickles/open_source_data.xlsx"
report_string = "SAVSNET HPC Adventures\n"

# Details for creating ssh connection 
hostname = "savsnetjupyter.liv.ac.uk"
username = "adam"
source_path = "/home/adam/pickles/open_source_data.xlsx"
destination_path = "/users/arwill/pickles/open_source_data.xlsx"  
key_filename = "/users/arwill/.ssh/id_rsa"


def create_ssh_connection(hostname, username, password=None, key_filename=None):
    """
    Function to create an SSH connection. This can be done using a password or a private key
    hostname: domain to be connected too (e.g foobar.co.uk)
    username: e.g. eric
    password: users password to access remote server
    key_filename: full filepath to the file containing the private key in the connection 
    """
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        if password:
            ssh_client.connect(hostname=hostname, username=username, password=password)
        elif key_filename:
            pkey = paramiko.RSAKey.from_private_key_file(key_filename)
            ssh_client.connect(hostname=hostname, username=username, pkey=pkey)
        else:
            raise ValueError("Either password or key_filename is required for authentication.")

        return ssh_client.open_sftp()
    except paramiko.SSHException as e:
        print(f"SSH connection failed: {e}")
        return None
    
def copy_file(sftp, source_path, destination_path):
    """
    Copies file from remote server to local server using sftp
    source_path = path to file on remote server
    destination path = desired path to folder on local server
    """
    try:
        sftp.get(source_path, destination_path)
        print(f"File copied: {source_path} -> {destination_path}")
    except Exception as e:
        print(f"File copy failed: {e}")

# Crete an ssh connection and copy file accross
with create_ssh_connection(hostname, username, key_filename=key_filename) as sftp:
    if sftp:
        copy_file(sftp, source_path, destination_path)


# Attempt to load pickles 
try:
    df = pd.read_excel('/users/arwill/pickles/open_source_data.xlsx', engine="openpyxl")
    with open(output_path + 'test_result_manual.txt',"w") as result_file:
        result_file.write("DataFrame Successfully Created")
    os.system(remove_string)
    with open(output_path + 'test_result_manual.txt',"a") as result_file:
            result_file.write("\nTemporary File Successfully Removed")
except:
    os.system(remove_string)
    with open(output_path + 'test_result_manual.txt',"w") as result_file:
        result_file.write("DataFrame Creation Failed\nTemporary File Successfully Removed")

df = df.dropna(subset=["Narrative", "SAVSNET MPC"]).rename(columns={"Narrative":"text", "SAVSNET MPC":"labels"})
df.to_csv(output_path + "fixed_narratives.csv", index=False)

time_start = time.time()

mpc_features = Features(
    {
       	"text": Value("string"),
       	"labels": ClassLabel(names=list(df["labels"].unique())),

    }
)

dataset = load_dataset(
    "csv",
    data_files={output_path + "fixed_narratives.csv"},
    delimiter=",",
    usecols=["text", "labels"],
    features=mpc_features,
    keep_in_memory=True,
    memory_map=True,
    split="train",
)

#train test split
dataset = dataset.train_test_split(test_size=0.2, seed=42)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

config = BertConfig.from_pretrained(
    "bert-base-uncased", num_labels=df["labels"].nunique()
)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", config=config
)


def tokenize(batch):
    tokenized_batch = tokenizer(
        batch["text"], padding='max_length', truncation=True, max_length=512
    )
    return tokenized_batch


tokenized_datasets = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized_datasets.set_format(
    "torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True
)

train_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    output_dir= output_path + "results",
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()
time_end = time.time()
print(str(time_end - time_start))



# save time_end to a file
with open("SAVSNET_test/time.txt", "w") as f:
    f.write(str(time_end - time_start))


with open(output_path + 'test_result_manual.txt',"w") as result_file:
    result_file.write(report_string)







