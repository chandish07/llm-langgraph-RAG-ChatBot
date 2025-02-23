import React, { useState, useRef, useEffect } from 'react';

function ChatInterface({ userDetails, onEndChat }) {
    const [message, setMessage] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const chatEndRef = useRef(null);

    const scrollToBottom = () => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [chatHistory]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!message.trim()) return;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message,
                    email: userDetails.email,
                    name: userDetails.name,
                    chatHistory: chatHistory
                }),
            });

            const data = await response.json();
            setChatHistory([
                ...chatHistory,
                { type: 'user', content: message, user: userDetails },
                { type: 'ai', content: data.response, user: userDetails }
            ]);
            setMessage('');
        } catch (error) {
            console.error('Error:', error);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    const handleEndChat = async () => {
        try {
            await fetch('/end-chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: userDetails.email,
                    name: userDetails.name,
                    messages: chatHistory,
                }),
            });
            onEndChat();
        } catch (error) {
            console.error('Error ending chat:', error);
        }
    };

    return (
        <div className="chat-interface">
            <div className="chat-header">
                <h2>Chat with GovSearchAI</h2>
            </div>
            <div className="chat-history">
                {chatHistory.map((msg, index) => (
                    <div key={index} className={`message ${msg.type}`}>
                        {msg.content}
                    </div>
                ))}
                <div ref={chatEndRef} />
            </div>
            <div className="chat-input">
                <form onSubmit={handleSubmit}>
                    <input
                        type="text"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Ask about contracts..."
                    />
                    <button type="submit">Send</button>
                    <button type="button" onClick={handleEndChat} className="end-chat-button">
                        End Chat
                    </button>
                </form>
            </div>
        </div>
    );
}

export default ChatInterface; 