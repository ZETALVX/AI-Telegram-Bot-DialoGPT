# -*- coding: utf-8 -*- 

import re
import warnings
import torch
import threading
import logging
import json
import numpy as np
from gtts import gTTS
from playsound import playsound
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_dataset, concatenate_datasets, Dataset
from datetime import datetime
import os
import shutil
import pandas as pd
import traceback
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import asyncio  # Aggiunto per gestire asyncio in avvia_bot_telegram

# Configure the logging
logging.basicConfig(
    filename='chatbot_debug.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
# Used to not show warnings
warnings.filterwarnings("ignore")

# Path to save the models, the current state e the dataset
PERCORSO_MODELLO = '//192.168.1.253/Vol 2/Test/modello_salvato/'   # model data file path 
PERCORSO_STATO = './stato_salvato/'       # current state data file path
PERCORSO_DATASET = './dataset/'           # dataset path
# Token of the Telegram's bot
TELEGRAM_TOKEN = 'Your Telegram Token'  # token of the bot

# Set device (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Corretto 'gpu' in 'cpu'
print(f"Device: {device}")

# Check if CUDA is evailable
print("CUDA:", torch.cuda.is_available())

# Numbers of GPU's evailable
print("GPU's evailable:", torch.cuda.device_count())

# GPU Name
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Function to sanitize text while keeping valid characters
def sanitize_text(text):
    """
    Remove invalid characters from text to avoid encoding errors,
    while maintaining valid accented characters in supported languages.
    """
    # Keep printable characters and replace others with a space
    sanitized = re.sub(r'[^\w\s.,!?\'’"-]', ' ', text, flags=re.UNICODE)
    return sanitized.strip()

# Function to save conversation log to file
def salva_conversazione(sorgente, messaggio, file_log="conversazione_log.txt"):
    try:
        with open(file_log, "a", encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {sorgente}: {messaggio}\n")
    except Exception as e:
        logging.error(f"Error saving conversation: {e}")

# Function to save trained knowledge (consciousness)
def salva_conoscenza(model, tokenizer, path=PERCORSO_MODELLO):
    try:
        # Delete the folder if it exists to avoid conflicts
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        # Save the model in standard format to avoid problems with safetensors
        model.save_pretrained(path, safe_serialization=False)
        tokenizer.save_pretrained(path)
        print(f"Model and tokenizer saved in {path}")
        logging.info(f"Model and tokenizer saved in {path}")
    except Exception as e:
        logging.error(f"Error saving model and tokenizer: {e}")

# Function to save the conversation state
def salva_stato(history, path=PERCORSO_STATO):
    try:
        os.makedirs(path, exist_ok=True)
        # Save conversation history
        with open(os.path.join(path, 'history.pt'), 'wb') as f:
            torch.save(history, f)
        print(f"Conversation history saved in{path}")
        logging.info(f"Conversation history saved in {path}")
    except Exception as e:
        logging.error(f"Error saving conversation state: {e}")

# Function to load conversation status
def carica_stato(path=PERCORSO_STATO):
    history = []
    try:
        if os.path.exists(os.path.join(path, 'history.pt')):
            with open(os.path.join(path, 'history.pt'), 'rb') as f:
                history = torch.load(f)
            print(f"Conversation history uploaded by{path}")
            logging.info(f"Conversation history uploaded by {path}")
    except Exception as e:
        logging.error(f"Error loading conversation state: {e}")
    return history

# Function to remove duplicates from the dataset
def rimuovi_duplicati(dataset):
    try:
        unique_dataset = []
        visto = set()
        for esempio in dataset:
            chiave = (esempio['instruction'], esempio['input'], esempio['output'])
            if chiave not in visto:
                unique_dataset.append(esempio)
                visto.add(chiave)
        logging.info(f"Duplicates removed: {len(dataset) - len(unique_dataset)}")
        return unique_dataset
    except Exception as e:
        logging.error(f"Error removing duplicates: {e}")
        return dataset

# Function to validate the dataset
def valida_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        validi = 0
        non_validi = 0
        for esempio in data:
            if all(key in esempio for key in ['instruction', 'input', 'output']):
                validi += 1
            else:
                non_validi += 1
                logging.warning(f"Example not valid: {esempio}")
        print(f"Dataset valid: {validi} examples")
        print(f"Dataset not valid: {non_validi} examples")
        logging.info(f"Dataset valid: {validi} examples, not valid: {non_validi} example")
        return validi, non_validi
    except Exception as e:
        logging.error(f"Dataset validation error: {e}")
        return 0, 0

# Function to remove duplicates from the dataset and save
def rimuovi_duplicati_dataset(file_path, output_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        unique_data = rimuovi_duplicati(data)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=4)
        print(f"Dataset without duplicates saved in {output_path}")
        logging.info(f"Dataset without duplicates saved in {output_path}")
    except Exception as e:
        logging.error(f"Error removing duplicates from dataset: {e}")

# Function to load the model and tokenizer
def carica_modello(modello_salvato=None):
    try:
        if modello_salvato and os.path.exists(modello_salvato):
            print(f"Loading the model from {modello_salvato}")
            model = AutoModelForCausalLM.from_pretrained(modello_salvato)
            tokenizer = AutoTokenizer.from_pretrained(modello_salvato)
            logging.info(f"Model uploaded by {modello_salvato}")
        else:
            print("Loading the pre-trained model DialoGPT-large")
            model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-large')
            tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
            logging.info("Model DialoGPT-large pre-trained loaded")
    
        # Add a padding token if none is present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            logging.info("Padding token added to tokenizer")
    
        tokenizer.clean_up_tokenization_spaces = True
    
        model.to(device)
    
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e

# Function to generate responses from the chatbot with error handling
def genera_risposta(input_text, model, tokenizer, history=None, max_length=100, temperature=0.7):
    try:
        if history is None:
            history = []
    
        # Tokenize user input
        new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt').to(device)
        history.append(new_user_input_ids)
        history = history[-5:]  # Mantieni solo gli ultimi 5 scambi
    
        # Chain the story
        bot_input_ids = torch.cat(history, dim=-1).to(device)
        attention_mask = torch.ones_like(bot_input_ids).to(device)
    
        # Generate the response passing the `attention_mask`
        output_ids = model.generate(
            bot_input_ids,
            attention_mask=attention_mask,
            max_length=bot_input_ids.shape[-1] + max_length,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            no_repeat_ngram_size=2,
            do_sample=True,
            length_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
        # Get the generated response
        response_ids = output_ids[:, bot_input_ids.shape[-1]:]
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        history.append(response_ids)
    
        # Sanitize the response text
        sanitized_response = sanitize_text(response)
        logging.info(f"Response generated: '{response}'")
        logging.info(f"Sanitized response: '{sanitized_response}'")
        print(f"Response generated: '{response}'")  # for debug
        print(f"Sanitized response: '{sanitized_response}'")  # fordebug
    
        # Check if the response is empty after sanitization
        if not sanitized_response.strip():
            print("The generated response is empty after sanitization. No speech synthesis performed.")
            logging.warning("Empty response after sanitization.")
            return response, history
    
        # Play bot's response
        text_to_speech(sanitized_response, lang='en')
    
        return response, history
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        print(f"Error generating response: {e}")
        return "Sorry, there was an error generating the response.", history

def verifica_tokenizzazione(modello_salvato=PERCORSO_MODELLO, path_dataset=PERCORSO_DATASET, num_esempi=50):
    try:
        # Carica il modello e il tokenizer
        model, tokenizer = carica_modello(modello_salvato=modello_salvato)
        
        # Carica un piccolo subset del dataset
        dataset_paths = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset) if f.endswith('.json')]
        if not dataset_paths:
            print("Nessun dataset trovato per la verifica.")
            return
        
        # Carica solo i primi 'num_esempi' esempi da ogni file JSON
        sample_texts = []
        for path in dataset_paths:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for esempio in data[:num_esempi]:
                    instruction = esempio.get('instruction', '').strip()
                    input_text = esempio.get('input', '').strip()
                    output_text = esempio.get('output', '').strip()
                    
                    if input_text:
                        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"
                    else:
                        prompt = f"Instruction: {instruction}\nResponse: {output_text}"
                    
                    # Sanitize the text
                    sanitized_prompt = sanitize_text(prompt)
                    sample_texts.append(sanitized_prompt)
        
        # Tokenizza gli esempi
        tokenized = tokenizer(sample_texts, truncation=True, padding='max_length', max_length=100, return_tensors='pt')
        
        # Stampa gli esempi originali e i relativi token
        for i in range(len(sample_texts)):
            print(f"--- Esempio {i+1} ---")
            print("Testo Originale:")
            print(sample_texts[i])
            print("\nToken ID:")
            print(tokenized['input_ids'][i])
            print("\nTokenizzatore Decode:")
            decoded_text = tokenizer.decode(tokenized['input_ids'][i], skip_special_tokens=True)
            print(decoded_text)
            print("\n----------------------\n")
    
    except Exception as e:
        logging.error(f"Errore durante la verifica della tokenizzazione: {e}")
        print(f"Errore durante la verifica della tokenizzazione: {e}")

def analizza_lunghezza_dataset(path_dataset=PERCORSO_DATASET):
    tokenizer = AutoTokenizer.from_pretrained('//192.168.1.253/Vol 2/Test/modello_salvato/')
    token_lengths = []
    dataset_paths = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset) if f.endswith('.json')]
    for path in dataset_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for esempio in data:
                instruction = esempio.get('instruction', '').strip()
                input_text = esempio.get('input', '').strip()
                output_text = esempio.get('output', '').strip()
                if input_text:
                    prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"
                else:
                    prompt = f"Instruction: {instruction}\nResponse: {output_text}"
                tokenized = tokenizer.encode(prompt, truncation=True, max_length=None)
                token_lengths.append(len(tokenized))
    # Calcola la lunghezza media e la distribuzione
    avg_length = sum(token_lengths) / len(token_lengths)
    print(f"Lunghezza media delle sequenze: {avg_length}")
    print(f"Lunghezza massima delle sequenze: {max(token_lengths)}")
    return token_lengths

# Speech synthesis function
def text_to_speech(text, lang='en'):
    """
    Converts text to speech using gTTS and plays audio.
    """
    print(f"Call text_to_speech with text: '{text}' e lingua: '{lang}'")  # for debug

    try:
        # Sanitize the text
        sanitized_text = sanitize_text(text)
        print(f"Sanitized text for TTS: '{sanitized_text}'")  # for debug

        if not sanitized_text:
            print("The provided text is empty after sanitization. No speech synthesis performed.")
            logging.warning("Empty text after sanitization for TTS.")
            return

        # Generate audio in MP3
        tts = gTTS(text=sanitized_text, lang=lang)
        mp3_filename = "response.mp3"
        tts.save(mp3_filename)
        print(f"MP3 audio saved as{mp3_filename}")

        # Play audio
        playsound(mp3_filename)
        print(f"Audio {mp3_filename} successfully reproduced.")

        # Remove audio file
        os.remove(mp3_filename)
        print(f"File {mp3_filename} removed.")

    except Exception as e:
        logging.error(f"Speech synthesis error: {e}")
        print(f"Speech synthesis error: {e}")

#Function to train the model on JSON datasets according to the specified standard
def addestra_modello(model, tokenizer, path_dataset=PERCORSO_DATASET):
    #try:
        # Load existing model to continue training
        if os.path.exists('//192.168.1.253/Vol 2/Test/temp_model'):
            print("Loading existing model from //192.168.1.253/Vol 2/Test/temp_model to continue training.")
            model = AutoModelForCausalLM.from_pretrained('//192.168.1.253/Vol 2/Test/temp_model')
            tokenizer = AutoTokenizer.from_pretrained('//192.168.1.253/Vol 2/Test/temp_model')
            model.to(device)
            logging.info("Existing model loaded for continued training.")
        else:
            print("No existing models found. Training from scratch.")
            logging.info("No existing models found. Training from scratch.")

        # List of paths to your JSON datasets
        dataset_paths = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset) if f.endswith('.json')]

        if not dataset_paths:
            print("No dataset was found for training.")
            logging.warning("No dataset was found for training..")
            return model, tokenizer  # Returns the original model and tokenizer

        datasets = []

        for path in dataset_paths:
            print(f"Loading the dataset: {path}")
            try:
                # Load JSON dataset
                dataset = load_dataset('json', data_files=path, split='train')
                # Preprocess the dataset to create the text to train
                def preprocess_examples(examples):
                    texts = []
                    for instruction, input_text, output_text in zip(examples['instruction'], examples['input'], examples['output']):
                        # Let's create the prompt
                        if input_text.strip():
                            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"
                        else:
                            prompt = f"Instruction: {instruction}\nResponse: {output_text}"
                        texts.append(prompt)
                    return {'text': texts}
                dataset = dataset.map(preprocess_examples, batched=True, remove_columns=['instruction', 'input', 'output'])
                datasets.append(dataset)
                print(f"Dataset {path} loaded successfully.")
                logging.info(f"Dataset {path} loaded successfully.")
            except Exception as e:
                print(f"Error loading dataset {path}: {e}")
                logging.error(f"Error loading dataset {path}: {e}")
                continue

        if not datasets:
            print("No dataset was successfully loaded for training.")
            logging.warning("No dataset was successfully loaded for training.")
            return model, tokenizer

        # Concatenate all datasets
        try:
            combined_dataset = concatenate_datasets(datasets)
            logging.info("Tutti i dataset caricati e concatenati.")
        except Exception as e:
            logging.error(f"Errore nella concatenazione dei dataset: {e}")
            return model, tokenizer

        # Convert combined dataset to pandas DataFrame for deduplication
        try:
            df = combined_dataset.to_pandas()
            logging.info("Dataset converted to pandas DataFrame for deduplication.")
        except Exception as e:
            logging.error(f"Error converting Dataset to DataFrame: {e}")
            return model, tokenizer

        # Remove duplicates based on 'text' column
        try:
            df_unique = df.drop_duplicates(subset=['text'])
            logging.info("Duplicates removed from concatenated dataset.")
        except Exception as e:
            logging.error(f"Error removing duplicates: {e}")
            return model, tokenizer

        # Convert back to Dataset
        try:
            combined_dataset = Dataset.from_pandas(df_unique)
            logging.info("Deduplicated dataset converted back to Dataset object.")
        except Exception as e:
            logging.error(f"Error converting deduplicated DataFrame to Dataset: {e}")
            return model, tokenizer

        # Split the dataset into train and validation
        try:
            train_val = combined_dataset.train_test_split(test_size=0.1)
            train_dataset = train_val['train']
            eval_dataset = train_val['test']
            logging.info("Dataset split into train and validation.")
        except Exception as e:
            logging.error(f"Error in splitting the dataset: {e}")
            return model, tokenizer

        # Tokenize the dataset and include labels
        def tokenize_function(examples):
            tokenized_inputs = tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=100
            )
            # Set labels equal to input_ids
            tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
            return tokenized_inputs

        try:
            tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
            tokenized_val = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
            logging.info("Tokenized datasets.")
        except Exception as e:
            logging.error(f"Dataset tokenization error: {e}")
            return model, tokenizer

        # Add DataCollator to handle padding and `attention_mask`
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Set training topics
        training_args = TrainingArguments(
            output_dir='//192.168.1.253/Vol 2/Test/temp_model_new',
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            save_total_limit=2,
            save_strategy='steps',
            save_steps=500,
            evaluation_strategy='steps',
            eval_steps=500,
            learning_rate=5e-6,
            weight_decay=0.01,
            report_to="tensorboard",
            load_best_model_at_end=True
        )

        # Implement Early Stopping
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,  # Provide the evaluation dataset
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        # Start training
        try:
            print("Start of training...")
            logging.info("Start of training.")
            trainer.train()
            print("Training completed.")
            logging.info("Training completed.")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            print(f"Error during training: {e}")
            return model, tokenizer

        # Save the updated model in the new directory
        try:
            trainer.save_model('//192.168.1.253/Vol 2/Test/temp_model_new')
            tokenizer.save_pretrained('//192.168.1.253/Vol 2/Test/temp_model_new')
            print(f"Updated and saved model in //192.168.1.253/Vol 2/Test/temp_model_new")
            logging.info("Model and tokenizer updated and saved in //192.168.1.253/Vol 2/Test/temp_model_new.")
        except Exception as e:
            logging.error(f"Error saving updated model: {e}")

        # Delete the old directory and rename the new one
        try:
            if os.path.exists('//192.168.1.253/Vol 2/Test/temp_model'):
                shutil.rmtree('//192.168.1.253/Vol 2/Test/temp_model')
            os.rename('//192.168.1.253/Vol 2/Test/temp_model_new', '//192.168.1.253/Vol 2/Test/temp_model')
            print("Updated and saved model in //192.168.1.253/Vol 2/Test/temp_model")
            logging.info("Updated and saved model in //192.168.1.253/Vol 2/Test/temp_model.")
        except Exception as e:
            print(f"Error replacing temp_model directory: {e}")
            logging.error(f"Error replacing temp_model directory: {e}")

        # Return the updated model and tokenizer
        return trainer.model, tokenizer

# Function to prepare the dataset: validate, remove duplicates and save
def prepara_dataset(file_path, output_path):
    valida_dataset(file_path)
    rimuovi_duplicati_dataset(file_path, output_path)

# Add dictionary to keep chat history for each Telegram user
telegram_chat_histories = {}

#Handler for the /start command
async def telegram_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to the chatbot! Write something to start the conversation.")

# Handler for Telegram messages
async def telegram_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_input = update.message.text

    # Get chat history for current user
    history = telegram_chat_histories.get(user_id, [])

    # Generate the response using the existing function
    risposta, history = genera_risposta(user_input, model, tokenizer, history)

    # Send reply to user on Telegram
    await update.message.reply_text(risposta)

    #Update chat history for current user
    telegram_chat_histories[user_id] = history

    # Save the conversation
    salva_conversazione(f"Telegram User {user_id}", user_input)
    salva_conversazione("Chatbot", risposta)

def avvia_bot_telegram():
    # Configure the event loop for the current thread
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Define handlers
    application.add_handler(CommandHandler("start", telegram_start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, telegram_handle_message))

    # Start the bot
    print("Telegram Bot Started and Waiting for Messages.")
    loop.run_until_complete(application.run_polling())

# Main: Example of use
if __name__ == "__main__":
    # Prepare the dataset: validate and remove duplicates
    #input_dataset_path = 'human_evolution_dataset.json'  # Replace with your file
    #output_dataset_path = 'human_evolution_dataset_unici.json'
    #prepara_dataset(input_dataset_path, output_dataset_path)

    # Load the pre-trained or saved model
    model, tokenizer = carica_modello(modello_salvato=PERCORSO_MODELLO)

    # Load conversation state if it exists
    history = carica_stato(path=PERCORSO_STATO)

    print("Welcome back sir! Write 'esci' to close.")
    print("Commands available: 'salva stato', 'salva coscienza', 'addestra'") #'save state', 'save conscience', 'train'"
    text_to_speech("Welcome back sir!", lang='en')  # You can change the language

    # Start the Telegram bot in a separate thread
    telegram_thread = threading.Thread(target=avvia_bot_telegram)
    telegram_thread.start()

    while True:
        user_input = input("You: ")
        salva_conversazione("User", user_input)

        if user_input.lower() == "esci": #exit
            print("Chatbot: Arrivederci!")
            salva_conversazione("Chatbot", "Arrivederci!")
            text_to_speech("Arrivederci!", lang='en')  # Cambia la lingua se necessario
            break
        elif user_input.lower() == "salva coscienza": #'save conscience'
            salva_conoscenza(model, tokenizer, path=PERCORSO_MODELLO)
        elif user_input.lower() == "salva stato": #'save state'
            salva_stato(history, path=PERCORSO_STATO)
        elif user_input.lower() == "verifica": # check toneniker
            verifica_tokenizzazione()
        elif user_input.lower() == "analizza": # check dataset length
            analizza_lunghezza_dataset() 
        elif user_input.lower() == "addestra":  # 'train'
            # Chiedi conferma all'utente
            conferma = input("Are you sure you want to start training? This may take some time. (Y/N): ")
            if conferma.lower() == "y":
                model, tokenizer = addestra_modello(model, tokenizer, path_dataset=PERCORSO_DATASET)
                # Salva il modello aggiornato
                salva_conoscenza(model, tokenizer, path=PERCORSO_MODELLO)
            else:
                print("Training cancelled.")
        else:
            risposta, history = genera_risposta(user_input, model, tokenizer, history)
            print("Chatbot:", risposta)
            salva_conversazione("Chatbot", risposta)




