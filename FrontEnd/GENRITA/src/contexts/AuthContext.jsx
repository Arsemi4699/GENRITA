// --- FILE: src/contexts/AuthContext.jsx (REVISED) ---
import React, { useState, useEffect, createContext, useContext } from 'react';
import { api } from '../services/api';

const AuthContext = createContext();
export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const tryLogin = async () => {
            const token = localStorage.getItem('accessToken');
            if (token) {
                try {
                    const userData = await api.fetchMe();
                    setUser(userData);
                } catch (error) {
                    console.error("Session expired or invalid.", error);
                    api.logout();
                }
            }
            setIsLoading(false);
        };
        tryLogin();
    }, []);

    // Login now accepts the role for backend validation
    const login = async (email, password, role) => {
        await api.login(email, password, role);
        const userData = await api.fetchMe();
        setUser(userData);
    };

    // Register also accepts the role to be stored permanently
    const register = async (name, email, password, role) => {
        return await api.register(name, email, password, role);
    };

    const logout = () => {
        api.logout();
        setUser(null);
    };

    const updateUserLibrary = (newLibraryIds) => {
        setUser(prev => ({ ...prev, libraryBookIds: newLibraryIds }));
    };

    const refetchUser = async () => {
        try {
            const userData = await api.fetchMe();
            setUser(userData);
        } catch (error) {
            console.error("Failed to refetch user.", error);
            logout(); // If refetch fails, token might be invalid, so log out
        }
    };

    // The 'user' object now contains the persistent 'role' from the server
    return (
        <AuthContext.Provider value={{ user, login, register, logout, isAuthenticated: !!user, updateUserLibrary, refetchUser }}>
            {!isLoading && children}
        </AuthContext.Provider>
    );
};