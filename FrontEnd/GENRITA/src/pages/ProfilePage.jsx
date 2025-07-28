// --- FILE: src/pages/ProfilePage.js ---
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { SunIcon, MoonIcon, BookIcon } from '../components/common/icons/index';

const ProfilePage = ({ onLogout }) => {
    const { user } = useAuth();
    const { theme, setTheme } = useTheme();
    return (
        <div className="p-4 md:p-6 text-[var(--color-text-primary)]">
            <div className="max-w-2xl mx-auto">
                <div className="flex items-center space-x-4 mb-8">
                    <div className="w-20 h-20 bg-[var(--color-accent)] rounded-full flex items-center justify-center text-white text-3xl font-bold">{user.name.charAt(0)}</div>
                    <div><h1 className="text-2xl font-bold">{user.name}</h1><p className="text-[var(--color-text-muted)]">{user.email}</p></div>
                </div>
                <div className="bg-[var(--color-background-secondary)] p-4 rounded-lg shadow mb-6">
                    <h2 className="text-lg font-semibold mb-2">Subscription Status</h2>
                    <p>Your subscription is <span className="font-bold capitalize">{user.subscription.status}</span>.</p>
                    {user.subscription.status === 'active' && <p className="text-sm text-[var(--color-text-muted)]">Expires on: {user.subscription.expiryDate}</p>}
                    <button className="w-full mt-2 py-2 rounded-lg bg-[var(--color-accent)] text-white font-bold">Manage Subscription</button>
                </div>
                <div className="bg-[var(--color-background-secondary)] p-4 rounded-lg shadow mb-6">
                    <h2 className="text-lg font-semibold mb-2">Settings</h2>
                    <div className="flex items-center justify-between">
                        <p>Theme</p>
                        <div className="flex items-center space-x-2 p-1 bg-gray-200 dark:bg-gray-700 rounded-full">
                            <button onClick={() => setTheme('light')} className={`p-1.5 rounded-full ${theme === 'light' && 'bg-white shadow'}`}><SunIcon /></button>
                            <button onClick={() => setTheme('dark')} className={`p-1.5 rounded-full ${theme === 'dark' && 'bg-gray-800 text-white shadow'}`}><MoonIcon /></button>
                            <button onClick={() => setTheme('sepia')} className={`p-1.5 rounded-full ${theme === 'sepia' && 'bg-yellow-100 shadow'}`}><BookIcon /></button>
                        </div>
                    </div>
                </div>
                <button onClick={onLogout} className="w-full text-center py-2 text-red-500 bg-red-100 dark:bg-red-900/50 rounded-lg">Log Out</button>
            </div>
        </div>
    );
}
export default ProfilePage;