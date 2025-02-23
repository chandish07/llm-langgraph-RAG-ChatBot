import React, { useState } from 'react';

function UserForm({ onSubmit }) {
    const [email, setEmail] = useState('');
    const [name, setName] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit({ email, name });
    };

    return (
        <div className="user-form">
            <h2>Welcome to GovSearchAI Chat</h2>
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label>Name:</label>
                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        required
                        placeholder="Enter your name"
                    />
                </div>
                <div className="form-group">
                    <label>Email:</label>
                    <input
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                        placeholder="Enter your email"
                    />
                </div>
                <button type="submit">Start Chat</button>
            </form>
        </div>
    );
}

export default UserForm; 