// --- FILE: src/pages/AuthPage.js ---
import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
const AuthPage = () => {
    const { login } = useAuth();
    const [isLoading, setIsLoading] = useState(false);
    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        try { await login(e.target.username.value, e.target.password.value) }
        catch (e) { console.error(e) }
        finally { setIsLoading(false) }
    };
    return (
        <div className="flex items-center justify-center min-h-screen bg-[var(--color-background)]">
            <div className="w-full max-w-sm p-8 space-y-6 bg-[var(--color-background-secondary)] rounded-lg shadow-md">
                <h2 className="text-2xl font-bold text-center text-[var(--color-text-primary)]">Bookworm Login</h2>
                <form className="space-y-6" onSubmit={handleSubmit}>
                    <div>
                        <label className="text-sm font-medium text-[var(--color-text-primary)]">Username</label>
                        <input name="username" type="text" defaultValue="user" required className="w-full px-3 py-2 mt-1 bg-[var(--color-background)] border border-[var(--color-border)] rounded-md" />
                    </div>
                    <div>
                        <label className="text-sm font-medium text-[var(--color-text-primary)]">Password</label>
                        <input name="password" type="password" defaultValue="pass" required className="w-full px-3 py-2 mt-1 bg-[var(--color-background)] border border-[var(--color-border)] rounded-md" />
                    </div>
                    <button type="submit" disabled={isLoading} className="w-full px-4 py-2 font-medium text-[var(--color-accent-text)] bg-[var(--color-accent)] rounded-md hover:bg-[var(--color-accent-hover)] disabled:opacity-50">
                        {isLoading ? 'Logging in...' : 'Login'}
                    </button>
                </form>
            </div>
        </div>
    );
}
export default AuthPage