// --- FILE: src/author/AuthorProfilePage.jsx (NEW FILE) ---
// This is the new profile page specifically for authors.
import React from 'react';
import SettingsComponent from '../components/common/SettingsComponent';

const AuthorProfilePage = ({ onLogout }) => {
    return (
        <div>
            <SettingsComponent onLogout={onLogout} />
        </div>
    );
};

export default AuthorProfilePage;