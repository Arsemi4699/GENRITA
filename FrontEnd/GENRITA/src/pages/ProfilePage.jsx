// --- FILE: src/pages/ProfilePage.jsx (REVISED) ---
import React from 'react';
import SettingsComponent from '../components/common/SettingsComponent';

const ProfilePage = ({ onLogout, setPage }) => {
    return (
        <div>
            <SettingsComponent onLogout={onLogout} setPage={setPage} />
        </div>
    );
};

export default ProfilePage;