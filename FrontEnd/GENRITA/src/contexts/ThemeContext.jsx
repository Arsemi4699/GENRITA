import { useState, useEffect, createContext, useContext } from 'react';
// --- FILE: src/contexts/ThemeContext.js ---
const ThemeContext = createContext();
export const useTheme = () => useContext(ThemeContext);
export const ThemeProvider = ({ children }) => {
    const [theme, setTheme] = useState('light');
    useEffect(() => {
        const root = window.document.documentElement;
        root.classList.remove('dark', 'sepia');
        if (theme !== 'light') root.classList.add(theme);
    }, [theme]);
    return <ThemeContext.Provider value={{ theme, setTheme }}>{children}</ThemeContext.Provider>;
};