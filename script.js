// --- DOM Element References ---
const chatHistory = document.getElementById('chat-history');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const loadingIndicator = document.getElementById('loading-indicator');
const symptomForm = document.getElementById('symptom-form');
const predictionResultDiv = document.getElementById('prediction-result');
const skipFormButton = document.getElementById('skip-form-button');
const backToAssessmentButton = document.getElementById('back-to-assessment-button');
const assessmentFormContainer = document.getElementById('assessment-form-container');
const chatContainer = document.getElementById('chat-container');
const mainContainer = document.getElementById('main-container');

// --- Configuration ---
const API_KEY = "AIzaSyCdhUj6MjNBQYxF63A8bN71Oa5UO25Vbsw"; 
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${API_KEY}`;
const PREDICTION_API_URL = 'http://127.0.0.1:5000/predict'; 

// --- Chat History Management ---
let chatContext = []; 

// --- UI State Management ---
function enableChatbot() {
    userInput.disabled = false;
    sendButton.disabled = false;
    userInput.focus();
    
    // Hide assessment form and make it zero width to give space to chat
    assessmentFormContainer.classList.add('hidden', 'lg:w-0');
    assessmentFormContainer.classList.remove('lg:w-3/4'); 
    
    // Adjust main container layout (remove horizontal spacing)
    mainContainer.classList.remove('lg:space-x-6'); 
    // If you want the chat to center horizontally when full page:
    // mainContainer.classList.add('lg:justify-center');

    // Make chat container full width
    chatContainer.classList.remove('lg:w-1/4'); 
    chatContainer.classList.add('w-full', 'lg:w-full', 'max-w-4xl'); 

    // Make chat history fill available height
    chatHistory.classList.remove('h-[35vh]', 'min-h-[35vh]'); 
    // Using a calc to fill height, adjust 150px as needed for header/footer elements
    chatHistory.classList.add('h-[calc(100vh-200px)]'); 

    predictionResultDiv.classList.add('hidden'); 
    backToAssessmentButton.classList.remove('hidden'); 
}

function disableChatbotAndShowForm() {
    userInput.disabled = true;
    sendButton.disabled = true;
    
    // Show assessment form and revert its width
    assessmentFormContainer.classList.remove('hidden', 'lg:w-0');
    assessmentFormContainer.classList.add('lg:w-3/4'); 
    
    // Revert main container layout (add horizontal spacing)
    mainContainer.classList.add('lg:space-x-6');
    // If you added centering, remove it:
    // mainContainer.classList.remove('lg:justify-center');

    // Revert chat container to its smaller width
    chatContainer.classList.remove('w-full', 'lg:w-full', 'max-w-4xl'); 
    chatContainer.classList.add('lg:w-1/4'); 

    // Revert chat history height
    chatHistory.classList.remove('h-[calc(100vh-200px)]'); 
    chatHistory.classList.add('h-[35vh]', 'min-h-[35vh]'); 

    backToAssessmentButton.classList.add('hidden'); 
    predictionResultDiv.classList.add('hidden'); 
    
    chatContext = []; 
    chatHistory.innerHTML = `
        <div class="flex justify-start">
            <div class="p-3 rounded-lg max-w-[80%] shadow-sm bg-gray-200 text-gray-800 bot-message">
                Hello! I'm MindConnect AI. Please answer the questions on the left or skip them to start chatting with me. I'm here to support you.
            </div>
        </div>
    `; 
}


// --- Chat UI Functions ---
function appendMessage(message, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('flex', sender === 'user' ? 'justify-end' : 'justify-start');

    const messageBubble = document.createElement('div');
    messageBubble.classList.add(
        'p-3',
        'rounded-lg',
        'max-w-[80%]',
        'shadow-sm',
        sender === 'user' ? 'bg-blue-500' : 'bg-gray-200',
        sender === 'user' ? 'text-white' : 'text-gray-800'
    );
    if (sender === 'bot') {
        messageBubble.classList.add('bot-message');
    }
    messageBubble.innerHTML = sender === 'user' ? message : formatBotResponse(message);
    messageDiv.appendChild(messageBubble);
    chatHistory.appendChild(messageDiv);

    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function formatBotResponse(text) {
    let formattedText = text.replace(/```(.*?)```/gs, (match, code) => {
        code = code.startsWith('\n') ? code.substring(1) : code;
        return `<pre>${code}</pre>`;
    });
    formattedText = formattedText.replace(/`(.*?)`/g, (match, code) => {
        return `<code>${code}</code>`;
    });
    formattedText = formattedText.replace(/\n/g, '<br>');
    return formattedText;
}

// --- Gemini Chatbot Logic ---
async function sendMessage() {
    const prompt = userInput.value.trim();
    if (!prompt) return;

    if (userInput.disabled) {
        appendMessage("Please complete the assessment or skip the questions to use the chatbot.", 'bot');
        return;
    }

    appendMessage(prompt, 'user');
    userInput.value = '';

    loadingIndicator.classList.remove('hidden');
    sendButton.disabled = true;

    try {
        chatContext.push({ role: "user", parts: [{ text: prompt }] });

        const payload = {
            contents: chatContext
        };

        const response = await fetch(GEMINI_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`API error: ${response.status} ${response.statusText} - ${errorData.error.message}`);
        }

        const result = await response.json();

        if (result.candidates && result.candidates.length > 0 &&
            result.candidates[0].content && result.candidates[0].content.parts &&
            result.candidates[0].content.parts.length > 0) {
            const botResponse = result.candidates[0].content.parts[0].text;
            appendMessage(botResponse, 'bot');
            chatContext.push({ role: "model", parts: [{ text: botResponse }] });
        } else {
            appendMessage("Sorry, I couldn't get a response from the bot.", 'bot');
        }
    } catch (error) {
        console.error("Error sending message to Gemini API:", error);
        appendMessage(`Error: ${error.message}. Please try again later.`, 'bot');
    } finally {
        loadingIndicator.classList.add('hidden');
        sendButton.disabled = false;
        userInput.focus();
    }
}

// --- Prediction Logic ---
async function sendPredictionRequest(userInputs) {
    predictionResultDiv.classList.add('hidden'); 
    predictionResultDiv.classList.remove('bg-green-100', 'text-green-800', 'bg-red-100', 'text-red-800'); 

    loadingIndicator.classList.remove('hidden');
    symptomForm.querySelector('button[type="submit"]').disabled = true; 
    skipFormButton.disabled = true; 

    try {
        const response = await fetch(PREDICTION_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(userInputs),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`Prediction API error: ${response.status} ${response.statusText} - ${errorData.error || 'Unknown error'}`);
        }

        const result = await response.json();
        const predictionText = result.prediction;
        const explanationText = result.explanation || "No detailed explanation available."; 

        appendMessage(`**Mental Health Prediction:**<br>${predictionText}<br><br>**Explanation:**<br>${explanationText}`, 'bot');

        predictionResultDiv.textContent = `Prediction Result: ${predictionText.split('!')[0]}`; 
        predictionResultDiv.classList.remove('hidden');
        predictionResultDiv.classList.add('bg-green-100', 'text-green-800'); 

        enableChatbot(); 
    } catch (error) {
        console.error("Error fetching prediction:", error);
        appendMessage(`Error getting prediction: ${error.message}. Please check the server and try again.`, 'bot');

        predictionResultDiv.textContent = `Prediction Error: ${error.message}`;
        predictionResultDiv.classList.remove('hidden');
        predictionResultDiv.classList.add('bg-red-100', 'text-red-800'); 
    } finally {
        loadingIndicator.classList.add('hidden');
        symptomForm.querySelector('button[type="submit"]').disabled = false;
        skipFormButton.disabled = false;
        chatHistory.scrollTop = chatHistory.scrollHeight; 
    }
}

// --- Event Listeners and Initial Setup ---
sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

document.getElementById('optimism').addEventListener('input', (e) => {
    document.getElementById('optimism-value').textContent = e.target.value;
});
document.getElementById('optimism-value').textContent = document.getElementById('optimism').value;

document.getElementById('sexual_activity').addEventListener('input', (e) => {
    document.getElementById('sexual-activity-value').textContent = e.target.value;
});
document.getElementById('sexual-activity-value').textContent = document.getElementById('sexual_activity').value;

document.getElementById('concentration').addEventListener('input', (e) => {
    document.getElementById('concentration-value').textContent = e.target.value;
});
document.getElementById('concentration-value').textContent = document.getElementById('concentration').value;

symptomForm.addEventListener('submit', (event) => {
    event.preventDefault(); 

    const userInputs = {};
    userInputs['mood_swing'] = document.querySelector('input[name="mood_swing"]:checked').value;
    userInputs['authority_respect'] = document.querySelector('input[name="authority_respect"]:checked').value;
    userInputs['suicidal_thoughts'] = document.querySelector('input[name="suicidal_thoughts"]:checked').value;

    userInputs['optimism'] = parseInt(document.getElementById('optimism').value);
    userInputs['sexual_activity'] = parseInt(document.getElementById('sexual_activity').value);
    userInputs['concentration'] = parseInt(document.getElementById('concentration').value);

    userInputs['sadness'] = document.getElementById('sadness').value;
    userInputs['exhausted'] = document.getElementById('exhausted').value;
    userInputs['euphoric'] = document.getElementById('euphoric').value;
    userInputs['sleep_disorder'] = document.getElementById('sleep_disorder').value;

    sendPredictionRequest(userInputs);
});

skipFormButton.addEventListener('click', () => {
    enableChatbot();
    appendMessage("You have skipped the assessment. Feel free to chat with me!", 'bot');
});

backToAssessmentButton.addEventListener('click', () => {
    disableChatbotAndShowForm();
});

// window.onload is no longer strictly necessary because initial state is in HTML
// But keeping it ensures consistency on refresh/re-render if JS loads after content.
window.onload = () => {
    // This call is now primarily to ensure all JS initializations are done
    // and to set the chatbot disabled state, which is also a "reset" to the form view.
    disableChatbotAndShowForm(); 
};