# AI-Telegram-Bot-DialoGPT
Basic GPT AI Chatbot with DIalo-GPT Large and Telegram integration using Python.

Video: https://www.youtube.com/watch?v=t8VHjuGTduE&t

******************
**The code is made by me, and you are free to use it. I only ask you, if you appreciate my work, to cite me in case you will use it for any purpose.
Also consider following me on my channels and feel free to ask anything.**
******************

The code implements a chatbot that can interact with users both through the console and through Telegram. It uses a pre-trained language model (specifically, Microsoft's DialoGPT-large) to generate responses based on user input. Key features of the code include:

- **Text-to-Speech (TTS):** Converts bot responses to speech using Google Text-to-Speech (gTTS) and plays the audio.
- **Conversation Recording:** Saves all conversations to a timestamped log file.
- **Model Training:** Allows additional training of the model using custom datasets.
- **State Management:** Saves and loads conversation history and model state.
- **Telegram Integration:** Runs a Telegram bot that users can interact with in real time.

**How ​​the code works:**

1. **Imports and Configuration:**

- Import the libraries needed for natural language processing, model management, data manipulation, and Telegram bot functionality.
- Configure logging to capture debugging information and errors.
- Set the device configuration to use the GPU if available; otherwise, defaults to CPU.
- Print device and GPU information for verification.

2. **Utilities:**

- `sanitize_text(text)`: Cleans up the input text by removing invalid characters to prevent encoding errors, while preserving valid accented characters.
- `save_conversation(source, message, log_file)`: Saves conversation logs to a timestamped file, capturing both user input and bot responses.
- `save_knowledge(model, tokenizer, path)`: Saves the trained model and tokenizer to a specified path, ensuring the model can be reloaded later.
- `save_state(history, path)`: Saves the conversation history to maintain context between sessions.
- `load_state(path)`: Loads the conversation history if it exists, allowing the bot to continue where it left off.
- `remove_duplicates(dataset)`: Removes duplicate entries from the dataset to prevent redundant training data.
- `validate_dataset(file_path)`: Validates the dataset to ensure it contains the required keys (`'instruction'`, `'input'`, `'output'`).
- `remove_duplicates_dataset(file_path, output_path)`: Removes duplicates from the dataset and saves the clean version.

3. **Model Functions:**

- `load_model(saved_model)`: Loads the pre-trained model and tokenizer. If a saved model path is provided and exists, loads the model from that path; otherwise, load the default DialoGPT-large model.
- `generate_response(input_text, model, tokenizer, history, max_length, temperature)`: Generates a response using the model and tokenizer. Maintains conversation history, handles text sanitization, and converts the bot's response to speech.
- `text_to_speech(text, lang)`: Converts text to speech using gTTS and plays it back using `playsound`. Sanitizes text before conversion.

4. **Training Functions:**

- `train_model(model, tokenizer, path_dataset)`: Allows additional model training using custom datasets. Handles dataset loading, preprocessing, tokenization, and model training with early stopping to prevent overfitting.

5. **Dataset Preparation:**

- `prepare_dataset(file_path, output_path)`: Validates and cleans the dataset by removing duplicates and ensuring data integrity.

6. **Telegram Bot Integration:**

- **Conversation Histories:** Maintains a `telegram_chat_histories` dictionary to store the conversation history for each Telegram user, ensuring personalized interactions.
- **Handlers:**
- `telegram_start(update, context)`: Responds to the `/start` command on Telegram, welcoming the user.
- `telegram_handle_message(update, context)`: Handles incoming messages from users, generates replies, updates the conversation history, and records the interaction.
- `start_bot_telegram()`: Initializes and runs the Telegram bot in a separate thread, setting up the event loop and registering handlers.

7. **Main Execution Block:**

- **Dataset Preparation:** Validates and cleans the dataset to be used for training or generating responses.
- **Model Loading:** Loads the pre-trained or saved model and tokenizer to be used to generate responses.
- **Conversation History Loading:** Loads any existing conversation history to maintain context.
- **Welcome Message:** Provides a welcome message.
nuto in both text and speech.
- **Telegram Bot Thread:** Starts the Telegram bot in a separate thread to handle Telegram interactions at the same time as console interactions.
- **Console Interaction Loop:** Provides a loop for console-based interaction, handling special commands (`'save state'`, `'save consciousness'`, `'train'`) and generating responses to user inputs.
  
Latest additions:
- **verification_tokenization** to do some tests by taking a number of examples of the dataset based on those indicated in the code, and have the bot analyze them as it would do during training, so as to see if the generated response and the related token contain errors,
- **analysis_dataset_length** analyzes the length of the dataset so as to understand its weight and be able to customize the bot with the parameters you need.

**Language used:**
Python

**Imported Libraries:**

- **Standard Libraries:**
- `re`, `warnings`, `torch`, `threading`, `logging`, `json`, `numpy`, `datetime`, `os`, `shutil`, `pandas`, `traceback`, `asyncio`
- **Third Party Libraries:**
- `gtts`, `playsound`, `transformers`, `datasets`, `telegram`, `telegram.ext`

**Important Considerations for Rebuilding Code:**

1. **Installing Dependencies:**
- Make sure you install all required libraries using `pip`.
2. **GPU Availability:**
- Make sure your system has a compatible GPU and the necessary drivers installed.
3. **Telegram Bot Token Security:**
- Keep your Telegram bot token private and do not share it publicly.
4. **Dataset Preparation:**
- Make sure your datasets are formatted correctly and placed in the specified directory.
5. **File Paths and Permissions:**
- Update the paths to match your system's directory structure and make sure your script has read/write permissions.
6. **Audio Playback Compatibility:**
- Make sure your system can play audio files for the text-to-speech functionality to work.
7. **Asynchronous Programming with the Telegram Bot:**
- The Telegram bot runs in an asynchronous event loop, started in a separate thread.
8. **Model Size and Performance:**
- The DialoGPT-large model is quite large and may require a lot of RAM and VRAM.
9. **Error Handling and Logging:**
- The code includes extensive logging for debugging purposes.
10. **API Licenses and Policies:**
- Comply with the licenses associated with the libraries and models you are using.

***Fine tuning***
The dataset file for the training (example redcity.json) must be inside the ***dataset*** folder in the same path as the Python code file.

**In Summary:**

This code provides a solid foundation for a conversational AI chatbot that can interact both via the console interface and Telegram. It integrates natural language processing, text-to-speech, and real-time messaging. By following the above considerations, you can recreate the code, customize it to your needs, and ensure it operates effectively and securely.

If you have any additional questions or need assistance with specific parts of the code, feel free to ask!

Consider that this is my first approach to the world of AI, I am autonomous and I do everything just for fun and interest. If there are errors, or changes that you think should be made, please write to me, I will be happy to listen and learn.

-- **ZetaLvX** --
