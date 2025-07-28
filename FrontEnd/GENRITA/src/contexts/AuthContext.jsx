import { useState, useEffect, createContext, useContext } from 'react';
import { mockApi } from '../data/mockApi'
// --- FILE: src/contexts/AuthContext.js ---
const AuthContext = createContext();
export const useAuth = () => useContext(AuthContext);
export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    useEffect(() => { setTimeout(() => setIsLoading(false), 1000) }, []);
    const login = async (u, p) => setUser(await mockApi.login(u, p));
    const logout = () => setUser(null);
    const addToLibrary = (bookId) => {
        setUser(prev => ({ ...prev, libraryBookIds: [...prev.libraryBookIds, bookId] }));
    };
    return <AuthContext.Provider value={{ user, login, logout, isAuthenticated: !!user, addToLibrary }}>{!isLoading && children}</AuthContext.Provider>;
};