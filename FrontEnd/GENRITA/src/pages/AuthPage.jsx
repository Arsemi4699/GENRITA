// --- FILE: src/pages/AuthPage.jsx (REVISED) ---
import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

const AuthPage = () => {
    const { login, register } = useAuth();
    const [isLoginView, setIsLoginView] = useState(true);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    // Role selection is now present in BOTH login and registration
    const [selectedRole, setSelectedRole] = useState('reader');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);
        setSuccess(null);

        const form = e.target;
        const email = form.email.value;
        const password = form.password.value;

        try {
            if (isLoginView) {
                // Pass the selected role during login for backend validation
                await login(email, password, selectedRole);
            } else {
                const name = form.name.value;
                // Pass the selected role during registration to be stored permanently
                await register(name, email, password, selectedRole);
                setSuccess("Registration successful! Please log in.");
                setIsLoginView(true); // Switch to login view after successful registration
            }
        } catch (err) {
            setError(err.response?.data?.detail || (isLoginView ? "Login failed. Please check your credentials or selected role." : "Registration failed."));
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-[var(--color-background)]">
            <div className="w-full max-w-sm p-8 space-y-6 bg-[var(--color-background-secondary)] rounded-lg shadow-md">
                <h2 className="text-2xl font-bold text-center text-[var(--color-text-primary)]">
                    {isLoginView ? 'Welcome Back' : 'Create Your Account'}
                </h2>

                <form className="space-y-4" onSubmit={handleSubmit}>
                    {/* --- ROLE SELECTION IS NOW IN BOTH VIEWS --- */}
                    <div>
                        <label className="text-sm font-medium text-[var(--color-text-primary)]">
                            {isLoginView ? 'Log in as:' : 'Register as:'}
                        </label>
                        <div className="grid grid-cols-2 gap-2 p-1 mt-1 rounded-lg bg-[var(--color-background)]">
                            <button type="button" onClick={() => setSelectedRole('reader')} className={`px-4 py-2 text-sm font-semibold rounded-md transition-colors ${selectedRole === 'reader' ? 'bg-[var(--color-accent)] text-white shadow' : 'text-[var(--color-text-muted)]'}`}>
                                Reader
                            </button>
                            <button type="button" onClick={() => setSelectedRole('author')} className={`px-4 py-2 text-sm font-semibold rounded-md transition-colors ${selectedRole === 'author' ? 'bg-[var(--color-accent)] text-white shadow' : 'text-[var(--color-text-muted)]'}`}>
                                Author
                            </button>
                        </div>
                    </div>

                    {!isLoginView && (
                        <div>
                            <label className="text-sm font-medium text-[var(--color-text-primary)]">Name</label>
                            <input name="name" type="text" required className="w-full px-3 py-2 mt-1 input-field" />
                        </div>
                    )}
                    <div>
                        <label className="text-sm font-medium text-[var(--color-text-primary)]">Email</label>
                        <input name="email" type="email" required className="w-full px-3 py-2 mt-1 input-field" />
                    </div>
                    <div>
                        <label className="text-sm font-medium text-[var(--color-text-primary)]">Password</label>
                        <input name="password" type="password" required className="w-full px-3 py-2 mt-1 input-field" />
                    </div>

                    <button type="submit" disabled={isLoading} className="w-full py-2.5 font-medium text-white bg-[var(--color-accent)] rounded-md hover:bg-[var(--color-accent-hover)] disabled:opacity-50">
                        {isLoading ? 'Please wait...' : (isLoginView ? 'Login' : 'Create Account')}
                    </button>
                </form>

                {error && <p className="text-sm text-center text-red-500 mt-2">{error}</p>}
                {success && <p className="text-sm text-center text-green-500 mt-2">{success}</p>}

                <p className="text-sm text-center text-[var(--color-text-muted)]">
                    {isLoginView ? "Don't have an account?" : "Already have an account?"}
                    <button onClick={() => { setIsLoginView(!isLoginView); setError(null); setSuccess(null); }} className="font-medium text-[var(--color-accent)] hover:underline ml-1">
                        {isLoginView ? 'Register' : 'Login'}
                    </button>
                </p>
            </div>
        </div>
    );
};
export default AuthPage;