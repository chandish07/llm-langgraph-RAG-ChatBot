import React, { useState } from 'react';
import UserForm from './components/UserForm';
import ChatInterface from './components/ChatInterface';
import './styles.css';

function App() {
    const [userDetails, setUserDetails] = useState(null);
    const [showChat, setShowChat] = useState(false);

    const handleUserSubmit = (details) => {
        setUserDetails(details);
        setShowChat(true);
    };

    const handleEndChat = () => {
        setShowChat(false);
        setUserDetails(null);
    };

    return (
        <div className="app-container">
            {!showChat ? (
                <UserForm onSubmit={handleUserSubmit} />
            ) : (
                <ChatInterface 
                    userDetails={userDetails} 
                    onEndChat={handleEndChat} 
                />
            )}
        </div>
    );
}

export default App; 